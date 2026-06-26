# duckdb-rs Zero-Copy Arrow Registration (Upstream) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `Connection::register_arrow(name, batches) -> ArrowView` in duckdb-rs: zero-copy Arrow registration over the built-in `arrow_scan`, with correct projection **and** filter pushdown (DuckDB's `TableFilterSet` translated and applied to the Arrow in the factory) — no global `SET` footgun — built upstream-clean on a new branch.

**Architecture:** A C++ shim (in libduckdb-sys) registers `arrow_scan` with a Rust factory and, per scan, **flattens** `ArrowStreamParameters.projected_columns` + `filters` (the `TableFilterSet` tree) into a length-prefixed C-ABI buffer. A Rust evaluator decodes it to a `FilterNode` tree and applies it to the projected `RecordBatch`es via arrow-compute (a `BooleanArray` mask → `filter_record_batch`), exporting a fresh `FFI_ArrowArrayStream` per scan (replayable). Unhandled filter kinds/types produce a clean error stream (never silently wrong).

**Tech Stack:** Rust (edition 2024) + C++ (FFI), DuckDB bundled amalgamation v1.10504.0, arrow-rs v58 (`ffi`, `ffi_stream`, `compute`), `cc` build.

## Global Constraints

- Work in `/Volumes/Work/Code/duckdb-rs` on a **new branch `arrow-zerocopy` off `main`**. The `zerocopy-arrow-shim-spike` branch stays untouched; its committed files are the **starting point to copy/extend** (commits `6a48b94`..`8182c1d`).
- duckdb-rs **1.10504.0**, `bundled` feature; arrow-rs **v58**; edition **2024** (`unsafe extern "C"` blocks).
- All Arrow + filter-evaluation logic in **Rust**; C++ is trampoline + `TableFilterSet` flatten + `TableFunction("arrow_scan",…)->CreateView` registration only.
- Register over the built-in **`arrow_scan`** (projection + filter pushdown ON). The factory MUST pre-apply pushed filters (DuckDB removed them from the plan).
- **Correctness policy** (grounded in duckdb-python `filter_pushdown_visitor`): translate `CONSTANT_COMPARISON`, `IS_NULL`, `IS_NOT_NULL`, `CONJUNCTION_AND`, `CONJUNCTION_OR`, `IN_FILTER`; treat `OPTIONAL_FILTER` (apply child if handled, else skip) and join-reduction kinds `DYNAMIC_FILTER`/`BLOOM_FILTER` as non-required (skip safely — the join above re-checks); **any other kind or any scalar/column type the evaluator can't handle → the factory exports an `error_stream` (clean scan failure naming the kind), never silently wrong.**
- `#[repr(C)] struct ArrowFactory` with the two Rust callback fn pointers FIRST (C++ `RustCallbacks` prefix aliases them).
- API: `Connection::register_arrow(&self, name: &str, batches: Vec<RecordBatch>) -> Result<ArrowView>`; `ArrowView` owns the factory box, frees on `Drop`, must outlive the view, is `!Send`/`!Sync` (documented).
- Feature gate: `#[cfg(all(feature = "bundled", feature = "vtab-arrow"))]`; tests run with `--features "bundled appender-arrow"`.
- Commit messages: conventional-commits ending with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## Verified DuckDB/arrow facts (use these; do not re-derive)

- `arrow_scan` is registered with `projection_pushdown=true, filter_pushdown=true, filter_prune=true, supports_pushdown_type=ArrowPushdownType` (so filters only reach the factory for filterable scalar column types).
- `ArrowStreamParameters { ArrowProjectedColumns projected_columns; TableFilterSet *filters; }`; `ArrowProjectedColumns { unordered_map<idx_t,string> projection_map; vector<string> columns; unordered_map<idx_t,idx_t> filter_to_col; }`.
- `TableFilterSet { map<idx_t, unique_ptr<TableFilter>> filters; }` (col_idx → filter). `TableFilter` has `filter_type` (`TableFilterType`).
- `ConstantFilter { ExpressionType comparison_type; Value constant; }`. `ExpressionType`: `COMPARE_EQUAL=25, COMPARE_NOTEQUAL=26, COMPARE_LESSTHAN=27, COMPARE_GREATERTHAN=28, COMPARE_LESSTHANOREQUALTO=29, COMPARE_GREATERTHANOREQUALTO=30`.
- `ConjunctionAndFilter`/`ConjunctionOrFilter` : base `ConjunctionFilter { vector<unique_ptr<TableFilter>> child_filters; }`. `InFilter { vector<Value> values; }`. `OptionalFilter { unique_ptr<TableFilter> child_filter; }`. `IsNullFilter`/`IsNotNullFilter`: no extra fields.
- `Value`: `value.type()` → `LogicalType` (`.id()` → `LogicalTypeId`), `value.IsNull()`, templated `value.GetValue<int64_t>()` / `<double>()` / `<bool>()`, `value.ToString()`.
- arrow-rs v58 compute (the duckdb crate already uses `arrow::compute::cast`): scalar comparisons via `arrow::compute::kernels::cmp::{eq,neq,lt,lt_eq,gt,gt_eq}` taking `Datum` (use `arrow::array::Scalar::new(<one-element array>)`); `arrow::compute::filter_record_batch(&batch, &mask)`; `arrow::compute::kernels::boolean::{and,or}`; `arrow::compute::is_null`/`is_not_null`.
- Reuse from the spike (already correct): the `ArrowArrayStreamWrapper` ownership pattern, `error_stream(schema, msg)`, `rust_get_schema`, the `build_bundled_cc.rs` wiring, the `ddb_rs_arrow_register` entrypoint, and the lifetime/`#[repr(C)]` contract.

---

## File Structure

- **Modify** `crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp` — add `TableFilterSet` flattening in `ShimProduce`; pass the filter buffer to the extended Rust produce callback.
- **Modify** `crates/libduckdb-sys/build_bundled_cc.rs` — already wires the shim (copied from spike); no change unless the shim adds includes.
- **Create** `crates/duckdb/src/arrow_zerocopy/mod.rs` — `ArrowFactory`, `rust_produce`/`rust_get_schema`, `Connection::register_arrow`, `ArrowView`, the extern decl, `error_stream`, `duckdb_err`.
- **Create** `crates/duckdb/src/arrow_zerocopy/filter.rs` — the C-ABI buffer decoder (`FilterNode`) + `evaluate(&FilterNode, &RecordBatch) -> Result<BooleanArray, FilterError>`.
- **Modify** `crates/duckdb/src/lib.rs` — `#[cfg(all(feature="bundled",feature="vtab-arrow"))] mod arrow_zerocopy; pub use arrow_zerocopy::ArrowView;`

## Filter buffer wire format (the C++↔Rust ABI — implement exactly)

Little-endian, pre-order, single root. The C++ flatten emits ONE root node: an `AND` over one child per `TableFilterSet.filters` entry (an empty filter set → a 0-child `AND`, i.e. "keep all"). Each node:

```
node      := tag:u8  body
tag values: 0=COMPARE 1=IS_NULL 2=IS_NOT_NULL 3=AND 4=OR 5=IN 255=UNHANDLED
COMPARE   := col:u32  op:u8(0=EQ,1=NE,2=LT,3=GT,4=LE,5=GE)  scalar
IS_NULL   := col:u32
IS_NOT_NULL := col:u32
AND/OR    := child_count:u32  child[child_count]
IN        := col:u32  value_count:u32  scalar[value_count]
UNHANDLED := kind:u32                       (the raw TableFilterType value)
scalar    := stype:u8  payload
  stype: 0=i64(payload i64), 1=u64(u64), 2=f64(f64), 3=bool(u8), 4=utf8(len:u32 + bytes), 5=null(no payload), 255=unsupported(no payload)
```

`col` is the column's position **in the produced (projected) batch** — the C++ side resolves the `TableFilterSet` key (table col idx) to the projected position via `projected_columns` (`filter_to_col` / projection order), so Rust indexes the projected `RecordBatch` directly. (Verify mapping against duckdb-python's factory usage of `filter_to_col`.) A `255=unsupported` scalar or `255=UNHANDLED` node makes the Rust evaluator return `FilterError` → `error_stream`.

---

### Task 1: Branch + module skeleton + filter pipeline for CONSTANT_COMPARISON (always-correct end-to-end slice)

This is the irreducible slice: registration + projection + the full C++-flatten→Rust-decode→evaluate→filter pipeline, handling `CONSTANT_COMPARISON` and erroring (cleanly) on every other pushed kind.

**Files:**
- Create branch; copy spike files; restructure into `crates/duckdb/src/arrow_zerocopy/{mod.rs,filter.rs}`; modify `crates/duckdb/src/lib.rs`; extend `crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp`.

**Interfaces (produced; used by Tasks 2–4):**
- `Connection::register_arrow(&self, name: &str, batches: Vec<RecordBatch>) -> duckdb::Result<ArrowView>`
- `pub struct ArrowView` (owns factory box; `Drop` frees; `!Send`).
- `filter::FilterNode` enum + `filter::decode(buf: &[u8]) -> FilterNode` + `filter::evaluate(node: &FilterNode, batch: &RecordBatch) -> Result<arrow::array::BooleanArray, filter::FilterError>`.
- Extended `rust_produce(factory, names, n, filter_ptr, filter_len, out)` and C++ `RustProduceFn` signature with the filter buffer.

- [ ] **Step 1: Create the branch and seed it from the spike**

```bash
cd /Volumes/Work/Code/duckdb-rs
git checkout main
git checkout -b arrow-zerocopy
git checkout zerocopy-arrow-shim-spike -- crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp crates/libduckdb-sys/build_bundled_cc.rs crates/duckdb/src/arrow_zerocopy.rs crates/duckdb/src/lib.rs
git status
```
Expected: the spike's shim/build/module/lib changes staged on `arrow-zerocopy`. (If `lib.rs` conflicts, re-apply just the `mod arrow_zerocopy` lines.)

- [ ] **Step 2: Restructure the Rust module into a directory + rename the API**

Move `crates/duckdb/src/arrow_zerocopy.rs` → `crates/duckdb/src/arrow_zerocopy/mod.rs`, create empty `crates/duckdb/src/arrow_zerocopy/filter.rs`, and in `mod.rs` add `mod filter;`. Rename the public API for upstream: `register_arrow_zerocopy` → `register_arrow`, `ArrowRegistration` → `ArrowView` (update `lib.rs` `pub use` to `pub use arrow_zerocopy::ArrowView;`). Keep the spike's `error_stream`, `rust_get_schema`, `duckdb_err`, `#[repr(C)] ArrowFactory`, `ddb_rs_arrow_register` extern, lifetime/ownership.

```bash
mkdir -p crates/duckdb/src/arrow_zerocopy
git mv crates/duckdb/src/arrow_zerocopy.rs crates/duckdb/src/arrow_zerocopy/mod.rs
: > crates/duckdb/src/arrow_zerocopy/filter.rs
```

- [ ] **Step 3: Write the failing CONSTANT_COMPARISON filter test**

In `crates/duckdb/src/arrow_zerocopy/mod.rs` `#[cfg(test)] mod test`, reuse/define `sample()` (k: Int64 [1,2,3,4]; region: Utf8 [us,eu,NULL,us]) and add:

```rust
    #[test]
    fn constant_comparison_filter_applied_no_global_setting() {
        // No `SET disabled_optimizers='filter_pushdown'` — the factory must honor the pushed filter.
        let conn = Connection::open_in_memory().unwrap();
        let reg = conn.register_arrow("v", vec![sample()]).unwrap();
        let sum_gt2: i64 = conn.query_row("SELECT sum(k) FROM v WHERE k > 2", [], |r| r.get(0)).unwrap();
        assert_eq!(sum_gt2, 7); // 3 + 4
        let cnt_eq: i64 = conn.query_row("SELECT count(*) FROM v WHERE k = 2", [], |r| r.get(0)).unwrap();
        assert_eq!(cnt_eq, 1);
        drop(reg);
    }

    #[test]
    fn unhandled_pushed_filter_fails_cleanly() {
        // An IN filter is not handled in Task 1 → clean query error, NOT a wrong result or crash.
        let conn = Connection::open_in_memory().unwrap();
        let reg = conn.register_arrow("v", vec![sample()]).unwrap();
        let err = conn.query_row("SELECT count(*) FROM v WHERE k IN (1, 3)", [], |r| r.get::<_, i64>(0));
        assert!(err.is_err(), "expected a clean error for an unhandled pushed filter, got {err:?}");
        drop(reg);
    }
```

- [ ] **Step 4: Run the tests to confirm they fail**

Run: `cargo test -p duckdb --features "bundled appender-arrow" arrow_zerocopy 2>&1 | tail -20`
Expected: compile errors / wrong results — `register_arrow` not yet present and the filter pipeline not built. (RED.)

- [ ] **Step 5: Extend the C++ shim to flatten the TableFilterSet**

In `crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp`: add includes
```cpp
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/in_filter.hpp"
#include "duckdb/planner/filter/null_filter.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"
#include "duckdb/common/enums/expression_type.hpp"
```
Change the Rust produce callback type to carry the filter buffer:
```cpp
typedef void (*RustProduceFn)(void *factory, const char *const *names, size_t n,
                              const uint8_t *filter_buf, size_t filter_len, ArrowArrayStream *out);
```
In `ShimProduce`, build a `std::vector<uint8_t> buf` per the wire format above: emit a root `AND` (tag 3) with `child_count` = number of entries in `params.filters->filters`, then for each entry serialize the column's filter via a recursive `void emit(const TableFilter&, idx_t projected_col, std::vector<uint8_t>&)`:
- `CONSTANT_COMPARISON`: `auto &c = filter.Cast<ConstantFilter>();` → tag 0, `projected_col` (u32), op from `c.comparison_type` (map the 6 ExpressionType values to 0..5; any other → emit UNHANDLED), then `emit_scalar(c.constant, buf)`.
- Every other `filter.filter_type` (Task 1): tag 255 UNHANDLED + the raw `(uint32_t)filter.filter_type`.
- `emit_scalar(const Value&, buf)`: if `v.IsNull()` → stype 5; else by `v.type().id()`: `BIGINT/INTEGER/SMALLINT/TINYINT` → stype 0 + `v.GetValue<int64_t>()`; `DOUBLE/FLOAT` → stype 2 + `v.GetValue<double>()`; `BOOLEAN` → stype 3 + `v.GetValue<bool>()`; `VARCHAR` → stype 4 + len+bytes of `v.ToString()`; else stype 255. (Task 3 widens types.)
- Resolve `projected_col` from the `TableFilterSet` key using `params.projected_columns` (`filter_to_col`/projection order) — the position in the produced projected batch. Reference duckdb-python's factory for the exact mapping.
Then `cbs->produce((void*)factory_ptr, names.data(), names.size(), buf.data(), buf.size(), &wrapper->arrow_array_stream);`

- [ ] **Step 6: Implement the Rust filter decoder + evaluator (`filter.rs`)**

```rust
use arrow::array::{ArrayRef, BooleanArray, Int64Array, RecordBatch, Scalar, StringArray};
use arrow::compute::kernels::{boolean, cmp};

#[derive(Debug)]
pub enum FilterError { Unhandled(String) }

#[derive(Debug)]
pub enum Scalar_ { I64(i64), U64(u64), F64(f64), Bool(bool), Utf8(String), Null, Unsupported }

#[derive(Debug)]
pub enum CmpOp { Eq, Ne, Lt, Gt, Le, Ge }

#[derive(Debug)]
pub enum FilterNode {
    Compare { col: usize, op: CmpOp, val: Scalar_ },
    IsNull { col: usize }, IsNotNull { col: usize },
    And(Vec<FilterNode>), Or(Vec<FilterNode>),
    In { col: usize, vals: Vec<Scalar_> },
    Unhandled(u32),
}

pub fn decode(buf: &[u8]) -> FilterNode { /* a cursor over `buf` per the wire format; returns the root */ }

pub fn evaluate(node: &FilterNode, batch: &RecordBatch) -> Result<BooleanArray, FilterError> {
    match node {
        FilterNode::Unhandled(k) => Err(FilterError::Unhandled(format!("TableFilterType {k}"))),
        FilterNode::And(children) => {
            // start all-true; AND each child mask
            let mut acc = BooleanArray::from(vec![true; batch.num_rows()]);
            for c in children { acc = boolean::and(&acc, &evaluate(c, batch)?).unwrap(); }
            Ok(acc)
        }
        FilterNode::Or(children) => { /* start all-false; OR each */ unimplemented_in_task1() }
        FilterNode::IsNull { .. } | FilterNode::IsNotNull { .. } | FilterNode::In { .. } => {
            Err(FilterError::Unhandled("kind not in Task 1".into()))
        }
        FilterNode::Compare { col, op, val } => compare(batch.column(*col), op, val),
    }
}

fn compare(col: &ArrayRef, op: &CmpOp, val: &Scalar_) -> Result<BooleanArray, FilterError> {
    // Build a one-element Datum scalar matching the column type, then dispatch the cmp kernel.
    // Task 1 needs i64 + utf8 for the tests; other scalar/type combos → FilterError (Task 3 widens).
    match val {
        Scalar_::I64(v) => {
            let s = Scalar::new(Int64Array::from(vec![*v]));
            dispatch(op, col, &s)
        }
        Scalar_::Utf8(v) => {
            let s = Scalar::new(StringArray::from(vec![v.clone()]));
            dispatch(op, col, &s)
        }
        _ => Err(FilterError::Unhandled("scalar type not in Task 1".into())),
    }
}

fn dispatch(op: &CmpOp, l: &dyn arrow::array::Datum, r: &dyn arrow::array::Datum) -> Result<BooleanArray, FilterError> {
    let out = match op {
        CmpOp::Eq => cmp::eq(l, r), CmpOp::Ne => cmp::neq(l, r),
        CmpOp::Lt => cmp::lt(l, r), CmpOp::Gt => cmp::gt(l, r),
        CmpOp::Le => cmp::lt_eq(l, r), CmpOp::Ge => cmp::gt_eq(l, r),
    };
    out.map_err(|e| FilterError::Unhandled(e.to_string()))
}
```
Adapt exact arrow-v58 signatures (`Scalar::new`, `Datum`, `cmp::*`, `boolean::and`) against `crates/duckdb/src/vtab/arrow.rs` (which uses `arrow::compute`). Implement `decode` as a simple byte cursor matching the wire format exactly.

- [ ] **Step 7: Wire the evaluator into `rust_produce` (`mod.rs`)**

Extend the produce callback signature to `(factory, names, n, filter_ptr: *const u8, filter_len: usize, out)`. After projecting the batches (existing spike logic), decode + evaluate the filter and apply it:
```rust
let filter_buf = if filter_ptr.is_null() || filter_len == 0 { &[][..] }
                 else { unsafe { std::slice::from_raw_parts(filter_ptr, filter_len) } };
let node = filter::decode(filter_buf);
let filtered: Vec<RecordBatch> = projected.into_iter()
    .map(|b| {
        let mask = filter::evaluate(&node, &b).map_err(|e| e)?;          // FilterError -> error path
        arrow::compute::filter_record_batch(&b, &mask).map_err(|e| /* -> error */ )
    })
    .collect::<Result<_, String>>()?;                                    // any err -> error_stream(schema, msg)
```
On any `FilterError`/compute error, fall through to `error_stream(f.schema.clone(), msg)` (the spike's panic/err path already does this — route filter errors into the same `error_stream`). Update the `extern "C" fn rust_produce` Rust signature + the C++ `RustProduceFn` typedef to match (filter ptr+len before `out`).

- [ ] **Step 8: Run the tests to confirm they pass**

Run: `cargo test -p duckdb --features "bundled appender-arrow" arrow_zerocopy 2>&1 | tail -20`
Expected: `constant_comparison_filter_applied_no_global_setting` PASS (7 and 1) and `unhandled_pushed_filter_fails_cleanly` PASS (IN → clean error). Also re-run the carried-over spike tests (round-trip, projection, self-join) — all pass. (First build recompiles the amalgamation + shim, several minutes.)

- [ ] **Step 9: Commit**

```bash
cd /Volumes/Work/Code/duckdb-rs
git add -A
git commit -m "feat(arrow): register_arrow with filter pushdown (CONSTANT_COMPARISON)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Expand filter kinds — null checks, conjunctions, IN; skip optional/join-reduction

**Files:** Modify `crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp` (emit the new kinds), `crates/duckdb/src/arrow_zerocopy/filter.rs` (decode + evaluate them), tests in `mod.rs`.

**Interfaces:** Consumes Task 1's `FilterNode`/`decode`/`evaluate`/wire format.

- [ ] **Step 1: Write failing tests for the new kinds**

Add to `mod test` (reuse `sample()`):
```rust
    #[test]
    fn null_and_conjunction_and_in_filters() {
        let conn = Connection::open_in_memory().unwrap();
        let reg = conn.register_arrow("v", vec![sample()]).unwrap();
        let is_null: i64 = conn.query_row("SELECT count(*) FROM v WHERE region IS NULL", [], |r| r.get(0)).unwrap();
        assert_eq!(is_null, 1);
        let not_null: i64 = conn.query_row("SELECT count(*) FROM v WHERE region IS NOT NULL", [], |r| r.get(0)).unwrap();
        assert_eq!(not_null, 3);
        let conj: i64 = conn.query_row("SELECT sum(k) FROM v WHERE k > 1 AND k < 4", [], |r| r.get(0)).unwrap();
        assert_eq!(conj, 5); // 2 + 3
        let in_list: i64 = conn.query_row("SELECT sum(k) FROM v WHERE k IN (1, 3)", [], |r| r.get(0)).unwrap();
        assert_eq!(in_list, 4); // 1 + 3
        drop(reg);
    }

    #[test]
    fn inner_join_with_filter_matches_appender() {
        // A join can push a dynamic/bloom filter to the arrow scan; skipping those must stay correct
        // because the join re-checks. Compare the zero-copy result to the Appender's.
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE t AS SELECT * FROM (VALUES (2),(3),(9)) AS x(k)").unwrap();
        let reg = conn.register_arrow("v", vec![sample()]).unwrap();
        let joined: i64 = conn.query_row(
            "SELECT sum(v.k) FROM v JOIN t ON v.k = t.k WHERE v.k > 1", [], |r| r.get(0)).unwrap();
        assert_eq!(joined, 5); // v.k in {2,3} match t, >1 → 2+3
        drop(reg);
    }
```

- [ ] **Step 2: Run to confirm they fail**

Run: `cargo test -p duckdb --features "bundled appender-arrow" arrow_zerocopy::test::null_and_conjunction_and_in_filters arrow_zerocopy::test::inner_join_with_filter_matches_appender 2>&1 | tail -20`
Expected: FAIL (these kinds currently hit the `error_stream` path → query errors). (RED.)

- [ ] **Step 3: C++ — emit the new kinds**

In `emit(...)`: handle `IS_NULL` (tag 1 + col), `IS_NOT_NULL` (tag 2 + col), `CONJUNCTION_AND` (tag 3) / `CONJUNCTION_OR` (tag 4) via `filter.Cast<ConjunctionAndFilter>()`/`ConjunctionOrFilter>()` recursing over `child_filters`, `IN_FILTER` (tag 5 + col + `values.size()` + `emit_scalar` each). For `OPTIONAL_FILTER` (`filter.Cast<OptionalFilter>().child_filter`): emit the child if it is a handled kind, else emit nothing (skip). For `DYNAMIC_FILTER`/`BLOOM_FILTER`: emit nothing (skip — join re-checks). Keep tag 255 UNHANDLED for anything else.

- [ ] **Step 4: Rust — decode + evaluate the new kinds**

In `filter.rs`: `decode` already covers all tags (Task 1 wire format) — verify IN/AND/OR/IS_NULL paths. In `evaluate`: implement `Or` (start all-false, OR each child via `boolean::or`), `IsNull` (`arrow::compute::is_null(batch.column(col))`), `IsNotNull` (`is_not_null`), and `In { col, vals }` (OR of `eq` against each scalar, or `arrow::compute::kernels::cmp::eq` reduced). A `FilterNode` whose root `AND` has an empty child list → all-true mask (no filter). Skipped kinds never reach Rust (C++ emits nothing), so no Rust change needed for them.

- [ ] **Step 5: Run to confirm they pass**

Run: `cargo test -p duckdb --features "bundled appender-arrow" arrow_zerocopy 2>&1 | tail -20`
Expected: the two new tests PASS plus all earlier tests still pass.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(arrow): filter pushdown for null/conjunction/IN; skip optional+join-reduction

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Type breadth for filter scalars + null semantics

**Files:** Modify `arrow_zerocopy_shim.cpp` (`emit_scalar` widen), `filter.rs` (`compare` widen), tests in `mod.rs`.

**Interfaces:** Consumes Task 1–2 pipeline; widens the scalar/column types `compare`/`emit_scalar` accept.

- [ ] **Step 1: Write failing type-coverage tests**

Add a helper `sample_types()` building a 5-row batch with this exact data, then the test:
```rust
    fn sample_types() -> RecordBatch {
        use arrow::array::{BooleanArray, Float64Array, Int32Array, StringArray, UInt32Array};
        let schema = Arc::new(Schema::new(vec![
            Field::new("i32", DataType::Int32, false),
            Field::new("u32", DataType::UInt32, false),
            Field::new("f64", DataType::Float64, false),
            Field::new("b", DataType::Boolean, false),
            Field::new("s", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(schema, vec![
            Arc::new(Int32Array::from(vec![-1, 0, 1, 2, 3])),
            Arc::new(UInt32Array::from(vec![10u32, 50, 99, 100, 200])),
            Arc::new(Float64Array::from(vec![0.5, 1.0, 1.5, 2.0, 2.5])),
            Arc::new(BooleanArray::from(vec![true, false, true, false, true])),
            Arc::new(StringArray::from(vec!["x", "y", "x", "z", "x"])),
        ]).unwrap()
    }

    #[test]
    fn filter_pushdown_across_scalar_types() {
        let conn = Connection::open_in_memory().unwrap();
        let reg = conn.register_arrow("v", vec![sample_types()]).unwrap();
        let q = |sql: &str| conn.query_row::<i64, _, _>(sql, [], |r| r.get(0)).unwrap();
        assert_eq!(q("SELECT count(*) FROM v WHERE i32 >= 0"), 4);  // 0,1,2,3
        assert_eq!(q("SELECT count(*) FROM v WHERE u32 < 100"), 3); // 10,50,99
        assert_eq!(q("SELECT count(*) FROM v WHERE f64 > 1.5"), 2); // 2.0,2.5
        assert_eq!(q("SELECT count(*) FROM v WHERE b = true"), 3);  // 3 trues
        assert_eq!(q("SELECT count(*) FROM v WHERE s = 'x'"), 3);   // 3 x's
        drop(reg);
    }
```

- [ ] **Step 2: Run to confirm failures** (types beyond i64/utf8 currently `FilterError`). RED.

Run: `cargo test -p duckdb --features "bundled appender-arrow" arrow_zerocopy::test::filter_pushdown_across_scalar_types 2>&1 | tail -20`

- [ ] **Step 3: C++ — widen `emit_scalar`**

Map more `Value` types: `INTEGER/SMALLINT/TINYINT` already i64; add `UINTEGER/USMALLINT/UTINYINT/UBIGINT` → stype 1 (u64, `v.GetValue<uint64_t>()`); `FLOAT` → stype 2 (f64); `DATE` → stype 0 (i64 days) ; `TIMESTAMP` → stype 0 (i64 micros) — only if you add matching column handling in Rust; otherwise leave as stype 255. Keep stype 255 for anything unmapped (clean error).

- [ ] **Step 4: Rust — widen `compare` to build the right Datum per column type**

In `compare`, dispatch on the **column's** `DataType` and the scalar: build a one-element array of the column's type from the scalar (e.g. `Int32Array::from(vec![v as i32])` when the column is `Int32` and the scalar is `I64`), then `dispatch(op, col, &Scalar::new(arr))`. Cover Int8/16/32/64, UInt8/16/32/64, Float32/64, Boolean, Utf8/LargeUtf8 (and Date32/Timestamp if emitted). Mismatched/uncovered combos → `FilterError`. Reuse arrow cast where convenient (`arrow::compute::cast`).

- [ ] **Step 5: Run to confirm pass.**

Run: `cargo test -p duckdb --features "bundled appender-arrow" arrow_zerocopy 2>&1 | tail -20`
Expected: type-coverage test passes; all prior tests pass.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(arrow): filter pushdown across scalar types + null semantics

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: API docs, benchmark, conventions (upstream polish)

**Files:** `crates/duckdb/src/arrow_zerocopy/mod.rs` (doc comments + example), `crates/duckdb/examples/arrow_zerocopy_bench.rs` (carry from spike, add a filtered query), `CHANGELOG`/PR notes; rustfmt + clippy.

**Interfaces:** Consumes the finished pipeline.

- [ ] **Step 1: Doc comments + a runnable example on `register_arrow`/`ArrowView`**

Add `///` docs: what it does (zero-copy view; projection + filter pushdown; no global setting), the lifetime rule (`ArrowView` must outlive the view), `!Send`, and a `# Examples` block that registers a batch and runs a filtered query. Ensure the doc example compiles (`cargo test --doc`).

- [ ] **Step 2: Benchmark example with a filtered, projected query**

Copy `examples/arrow_zerocopy_bench.rs` from the spike branch; add a second measured query that is filtered + projected: `format!("SELECT sum(c0) FROM v WHERE c0 > {}", n / 2)` (and the Appender equivalent `format!("SELECT sum(c0) FROM t WHERE c0 > {}", n / 2)` over the real table), selecting ~half the rows so the bench exercises both projection and filter pushdown. Keep the in-run `assert_eq!` correctness gates (zero-copy == Appender) for both the unfiltered and filtered queries.

```bash
git checkout zerocopy-arrow-shim-spike -- crates/duckdb/examples/arrow_zerocopy_bench.rs
```
Then edit it to add the filtered query + assert, and fix the API names (`register_arrow`/`ArrowView`).

- [ ] **Step 3: Build + run the benchmark (release), record numbers**

Run: `cargo run -p duckdb --features "bundled appender-arrow" --release --example arrow_zerocopy_bench 2>&1 | tail -10`
Expected: a table; in-run asserts hold. Record the numbers.

- [ ] **Step 4: rustfmt + clippy clean**

Run: `cargo fmt -p duckdb && cargo clippy -p duckdb --features "bundled appender-arrow" --all-targets 2>&1 | tail -20`
Expected: no fmt diff; no clippy warnings in `arrow_zerocopy`/the shim-facing Rust. Fix any.

- [ ] **Step 5: CHANGELOG / PR notes**

Add a short entry (per the repo's CHANGELOG/PR conventions — check `CONTRIBUTING.md`) describing `Connection::register_arrow` (zero-copy Arrow registration with projection + filter pushdown). If the repo has no CHANGELOG, write the PR description text into a `docs/` note instead.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "docs(arrow): register_arrow docs, filtered benchmark, lint clean

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Post-implementation

Record the upstream-version outcome (final API, filter coverage, benchmark with filter pushdown) as a short addendum to the Daft follow-ups doc (`docs/superpowers/specs/2026-06-25-duckdb-poc-performance-followups.md`), and note the branch is ready to open as a duckdb-rs PR (or list what remains before submitting: e.g. linked/loadable-build support if the maintainers require non-bundled).

## Self-Review notes (for the implementer)

- **Arrow v58 API drift:** `Scalar::new`, `Datum`, `cmp::{eq,neq,lt,lt_eq,gt,gt_eq}`, `boolean::{and,or}`, `is_null`/`is_not_null`, `filter_record_batch`, `cast` — adapt exact signatures against `crates/duckdb/src/vtab/arrow.rs` and arrow-58 docs; do not invent.
- **Column index mapping** (`filter_to_col` → projected position) is the subtlest C++ point: verify against duckdb-python's `arrow_array_stream.cpp` factory and add a focused test if a projected+filtered query disagrees with the Appender.
- **Correctness invariant:** the factory must NEVER return a row set wider or narrower than the pushed filter requires for a *handled* kind, and must `error_stream` (not guess) for unhandled kinds/types. The `unhandled_pushed_filter_fails_cleanly` test guards this; keep an equivalent guard as coverage widens.
- **First build is slow** (amalgamation + shim); subsequent Rust/shim edits are fast.

# duckdb-rs Arrow EXPRESSION_FILTER Port (Phase B) Implementation Plan

> **⛔ SUPERSEDED (2026-06-26)** — do not execute. The DuckDB-`main` bump this plan targets is abandoned. See [v1.5.4 full-coverage design](../specs/2026-06-26-duckdb-rs-arrow-v154-fullfilter-design.md); a new plan supersedes this one. Kept for the record only.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `arrow_zerocopy`'s filter pushdown to DuckDB **main**'s `EXPRESSION_FILTER` model (structured `TableFilter` kinds are gone), porting duckdb-python's `TransformExpression` — restoring + extending filter-pushdown coverage on the bumped foundation.

**Architecture:** The shim's `ShimProduce` walks each `EXPRESSION_FILTER`'s `BoundExpression` tree (port of `filter_pushdown_visitor.cpp::TransformExpression`) and emits the existing little-endian wire format, extended so a column reference is a **path** (projected root column + struct-extract child indices). The Rust evaluator resolves the path to a leaf array and reuses the existing mask logic (`compare`/`is_null`/`is_not_null`/`in`/`and`/`or`). Still registers over `arrow_scan`; producer pre-applies filters; unhandled → `error_stream` (fail-loud).

**Tech Stack:** Rust (edition 2024) + C++ (FFI), DuckDB **main** (`06eb6b6`, bundled), arrow-rs v58 compute, `cc`.

## Global Constraints

- Branch `arrow-zerocopy-duckdb-main`; foundation already committed at `911fc15` (DuckDB bumped to main, C++17, jemalloc-excluded). The shim's structured-filter code does NOT compile against main and is replaced in Task 1.
- **Reference (port its coverage exactly):** `/Volumes/Work/Code/duckdb-python/src/duckdb_py/arrow/filter_pushdown_visitor.cpp` — `TransformFilter` (only `EXPRESSION_FILTER`) → `TransformExpression`. Mirror its node handling and its AND-drop / OR-fail / optional-skip / table-filter-function-skip policy.
- All pushed filters arrive as `EXPRESSION_FILTER` wrapping a `BoundExpression`. Register over the built-in `arrow_scan` (projection + filter pushdown ON); the producer MUST pre-apply pushed filters.
- **Correctness policy** (verbatim from the spec): translate constant comparisons, `IS [NOT] NULL`, `IN`, `AND`/`OR`, and `struct_extract` column paths; `AND` may drop an untranslatable child (engine re-checks above); `OR` with any untranslatable child → emit UNHANDLED for the whole OR; optional/selectivity-optional → emit child if translatable else skip; dynamic/bloom/etc. (`TableFilterFunctions::IsTableFilterFunction`) → skip; anything else → UNHANDLED → `error_stream`. Never an over/under-inclusive row set.
- Main DuckDB API: `expr.GetExpressionClass()` (`ExpressionClass::{BOUND_FUNCTION,BOUND_OPERATOR,BOUND_CONJUNCTION,BOUND_CONSTANT,BOUND_REF}`), `expr.GetExpressionType()`, `BoundComparisonExpression::IsComparison(type)`/`::Left(func)`/`::Right(func)`, `FlipComparisonExpression(type)`, `BoundReferenceExpression::Index()`, `BoundConstantExpression::GetValue()`, `BoundFunctionExpression::{Function().GetName(),GetChildren(),BindInfo()}`, `TryGetStructExtractChildIndex(func, idx)` (`duckdb/function/scalar/struct_utils.hpp`), `TableFilterFunctions::IsTableFilterFunction(name)`, `OptionalFilterScalarFun::NAME`/`SelectivityOptionalFilterScalarFun::NAME` (`duckdb/planner/filter/table_filter_functions.hpp`), `ExpressionFilter::expr` (`duckdb/planner/filter/expression_filter.hpp`).
- Keep `Connection::register_arrow(...) -> ArrowView<'conn>` + auto-unregister-in-Drop + projection pushdown unchanged.
- Edition 2024 (`unsafe extern "C"`). Build/test with `--features "bundled appender-arrow"`; first build recompiles the main amalgamation (slow). Commits end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## Wire format (extended; implement exactly)

Same little-endian pre-order tree as before, with the column reference now a **path** and new scalar types:
```
node      := tag:u8 body
tag: 0=COMPARE 1=IS_NULL 2=IS_NOT_NULL 3=AND 4=OR 5=IN 255=UNHANDLED
path      := depth:u32  index:u32 × depth      (depth≥1; index[0]=projected root column, index[1..]=struct child indices)
COMPARE   := path  op:u8(0..5 EQ,NE,LT,GT,LE,GE)  nan:u8(0|1)  scalar
IS_NULL/IS_NOT_NULL := path
AND/OR    := child_count:u32  child×n
IN        := path  value_count:u32  scalar×n
UNHANDLED := kind:u32
scalar    := stype:u8 payload
  0=i64(i64) 1=u64(u64) 2=f64(f64) 3=bool(u8) 4=utf8(len:u32+bytes) 5=null
  6=decimal128(i128 16 LE bytes + scale:u8)  7=date32(i32)  8=ts(i64 + unit:u8: 0=s,1=ms,2=us,3=ns)  255=unsupported
```
`nan:u8` on COMPARE marks a NaN constant (duckdb-python special-cases NaN). The root is an `AND` over one child per `*params.filters` entry.

---

## File Structure
- **Modify** `crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp` — replace the structured-`TableFilterSet` flattening with `emit_expression` (the `TransformExpression` port) + `emit_column_path` (`ResolveColumn` port) + widened `emit_scalar`.
- **Modify** `crates/duckdb/src/arrow_zerocopy/filter.rs` — decode `path`; `resolve_path`; new scalar arms (Date/Time/Timestamp/Decimal128) + NaN; reuse mask logic.
- **Modify** `crates/duckdb/src/arrow_zerocopy/mod.rs` — tests (and keep `rust_produce` decode/evaluate wiring).
- **Modify** `crates/duckdb/examples/arrow_zerocopy_bench.rs` — Task 5 only.

---

### Task 1: Expression-walker skeleton — compiles + greens on main (comparison on a plain column)

Rewrite the shim to the EXPRESSION_FILTER model handling only constant comparisons on a `BOUND_REF` column; UNHANDLED for everything else. This deletes the main-incompatible structured code and gets the branch building + passing on main.

**Files:** Modify `arrow_zerocopy_shim.cpp`, `filter.rs`, `mod.rs`.

**Interfaces (produced; later tasks extend):**
- C++ `static void emit_expression(const duckdb::Expression&, std::vector<duckdb::idx_t> path, std::vector<uint8_t>& buf)` and `static bool emit_column_path(const duckdb::Expression&, std::vector<duckdb::idx_t>& out)` (returns false = NotImplemented).
- Wire `path` encoding (above). Rust `FilterNode::{Compare{path:Vec<usize>,op,nan,val}, IsNull{path}, IsNotNull{path}, And, Or, In{path,vals}, Unhandled}` and `resolve_path(batch, &[usize]) -> Result<ArrayRef, FilterError>`.

- [ ] **Step 1: Read the reference + delete the structured walker**

Read `/Volumes/Work/Code/duckdb-python/src/duckdb_py/arrow/filter_pushdown_visitor.cpp` (the port target). In `arrow_zerocopy_shim.cpp`, remove the structured includes (`constant_filter.hpp`, `conjunction_filter.hpp`, `in_filter.hpp`, `null_filter.hpp`, `optional_filter.hpp`) and the old `emit_filter`/`is_handled_filter_type` + the `ShimProduce` TableFilterSet-structured loop body. Add includes:
```cpp
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/common/enums/expression_type.hpp"
```

- [ ] **Step 2: Write the failing test (constant comparison still works on main)**

In `mod.rs` test module, keep the existing `sample()` and `constant_comparison_filter_applied_no_global_setting` (sum k>2 == 7; count k=2 == 1). It currently can't build (shim broken). This is the RED gate.

- [ ] **Step 3: Implement `emit_column_path` (BOUND_REF only) + `emit_scalar` (core types) + `emit_expression` (comparison)**

```cpp
// Resolve a column-side expression to a projected-batch path. Task 1: BOUND_REF only.
static bool emit_column_path(const duckdb::Expression &expr, std::vector<duckdb::idx_t> &out) {
    if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
        out.push_back(expr.Cast<duckdb::BoundReferenceExpression>().Index());
        return true;
    }
    return false; // struct_extract chains handled in Task 3
}

static void push_path(std::vector<uint8_t> &buf, const std::vector<duckdb::idx_t> &path) {
    push_u32(buf, (uint32_t)path.size());
    for (auto p : path) push_u32(buf, (uint32_t)p);
}

// op byte from ExpressionType comparison; returns 255 for non-handled.
static uint8_t cmp_op_byte(duckdb::ExpressionType t) {
    using E = duckdb::ExpressionType;
    switch (t) { case E::COMPARE_EQUAL: return 0; case E::COMPARE_NOTEQUAL: return 1;
        case E::COMPARE_LESSTHAN: return 2; case E::COMPARE_GREATERTHAN: return 3;
        case E::COMPARE_LESSTHANOREQUALTO: return 4; case E::COMPARE_GREATERTHANOREQUALTO: return 5;
        default: return 255; }
}

static void emit_unhandled(std::vector<uint8_t> &buf, uint32_t kind) { push_u8(buf, 255); push_u32(buf, kind); }

static void emit_expression(const duckdb::Expression &expr, std::vector<duckdb::idx_t> path,
                            std::vector<uint8_t> &buf) {
    using duckdb::ExpressionClass;
    auto cls = expr.GetExpressionClass();
    auto etype = expr.GetExpressionType();
    if (cls == ExpressionClass::BOUND_FUNCTION &&
        duckdb::BoundComparisonExpression::IsComparison(etype)) {
        auto &func = expr.Cast<duckdb::BoundFunctionExpression>();
        auto &left = duckdb::BoundComparisonExpression::Left(func);
        auto &right = duckdb::BoundComparisonExpression::Right(func);
        const duckdb::Expression *col_side; const duckdb::BoundConstantExpression *const_side;
        if (right.GetExpressionType() == duckdb::ExpressionType::VALUE_CONSTANT) {
            col_side = &left; const_side = &right.Cast<duckdb::BoundConstantExpression>();
        } else if (left.GetExpressionType() == duckdb::ExpressionType::VALUE_CONSTANT) {
            col_side = &right; const_side = &left.Cast<duckdb::BoundConstantExpression>();
            etype = duckdb::FlipComparisonExpression(etype);
        } else { emit_unhandled(buf, (uint32_t)cls); return; }
        uint8_t op = cmp_op_byte(etype);
        std::vector<duckdb::idx_t> col_path = path;
        if (op == 255 || !emit_column_path(*col_side, col_path)) { emit_unhandled(buf, (uint32_t)cls); return; }
        push_u8(buf, 0); push_path(buf, col_path); push_u8(buf, op);
        push_u8(buf, 0 /* nan flag; Task 4 */); emit_scalar(const_side->GetValue(), buf);
        return;
    }
    emit_unhandled(buf, (uint32_t)cls);
}
```
Keep `emit_scalar` (from the prior branch) for i64/u64/f64/bool/utf8/null (Task 4 widens). In `ShimProduce`, replace the loop with: root `AND` over `*params.filters`; for each `kv`, `auto &ef = kv.second->Cast<duckdb::ExpressionFilter>(); emit_expression(*ef.expr, {kv.first}, filter_buf);` (seed the path with the projected key — same projected-position semantics as the prior fix; the multi-column test guards it).

- [ ] **Step 4: Rust — `path` decode + `resolve_path` + reuse `compare`**

In `filter.rs`: change `FilterNode::Compare/IsNull/IsNotNull/In` to carry `path: Vec<usize>` instead of `col: usize`; add `nan: bool` to `Compare`. Decode `path` (depth + indices). Add:
```rust
fn resolve_path(batch: &RecordBatch, path: &[usize]) -> Result<arrow::array::ArrayRef, FilterError> {
    let mut arr = batch.column(*path.first().ok_or_else(|| FilterError::Unhandled("empty path".into()))?).clone();
    for &child in &path[1..] {
        let s = arr.as_any().downcast_ref::<arrow::array::StructArray>()
            .ok_or_else(|| FilterError::Unhandled("struct-extract path on non-struct column".into()))?;
        arr = s.column(child).clone(); // Task 3 exercises depth>1
    }
    Ok(arr)
}
```
In `evaluate`, for `Compare/IsNull/IsNotNull/In`, call `let col = resolve_path(batch, path)?;` then the existing per-arm logic on `&col`. (Task 1: `nan` ignored; Task 4 uses it.)

- [ ] **Step 5: Build + run (RED→GREEN)**

Run: `cargo test -p duckdb --features "bundled appender-arrow" arrow_zerocopy 2>&1 | tail -20`
Expected: compiles on main; `constant_comparison_filter_applied_no_global_setting`, `round_trip_select_star`, `projection_subset`, `self_join_is_replayable`, `multi_column_projection_and_filter_matches_appender` (comparison-only) PASS. Tests that need null/IN/conjunction/struct/other-types may fail or error — that's expected until later tasks; temporarily `#[ignore]` them with a `// Task N` note (re-enabled in their task) so this task is green.

- [ ] **Step 6: Commit**
```bash
cd /Volumes/Work/Code/duckdb-rs && git add -A
git commit -m "feat(arrow): EXPRESSION_FILTER walker skeleton (comparison) on DuckDB main

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Null / IN / conjunction + optional/dynamic/bloom skips

**Files:** Modify `arrow_zerocopy_shim.cpp` (`emit_expression`), `mod.rs` (un-ignore + tests).

**Interfaces:** Consumes Task 1's `emit_expression`/wire/decode/`resolve_path`.

- [ ] **Step 1: Un-ignore the null/IN/conjunction tests** (`null_and_conjunction_and_in_filters`, the `evaluate_*` unit tests). RED.

- [ ] **Step 2: Extend `emit_expression`** (mirror `TransformExpression`):
  - `BOUND_OPERATOR`: `OPERATOR_IS_NULL` → resolve path on child[0], emit tag 1; `OPERATOR_IS_NOT_NULL` → tag 2; `COMPARE_IN` → resolve path on child[0], emit tag 5 + (n-1) scalars from `child[1..].Cast<BoundConstantExpression>().GetValue()`.
  - `BOUND_CONJUNCTION`: `CONJUNCTION_AND` → emit tag 3; recurse children, **dropping** a child that yields UNHANDLED (re-walk into a temp buffer; if it emitted UNHANDLED, skip it); `CONJUNCTION_OR` → tag 4; if ANY child is UNHANDLED, emit a single UNHANDLED for the whole OR instead. (Mirror the visitor's is_and drop vs OR-fail logic.)
  - `BOUND_FUNCTION` non-comparison: if `func.Function().GetName()` is `OptionalFilterScalarFun::NAME`/`SelectivityOptionalFilterScalarFun::NAME` → recurse into the `BindInfo` child if present + translatable, else emit empty `AND`(0) (skip); if `TableFilterFunctions::IsTableFilterFunction(name)` → emit empty `AND`(0) (skip). Else UNHANDLED.

- [ ] **Step 3: Rust `evaluate`** — ensure `Or`, `IsNull`, `IsNotNull`, `In` arms operate on `resolve_path(...)` (carried from Task 1). No new Rust beyond the path adaptation.

- [ ] **Step 4: Build + test.** Run the `arrow_zerocopy` suite; the un-ignored tests + `inner_join_with_filter_matches_appender` PASS.

- [ ] **Step 5: Commit** (`feat(arrow): null/IN/conjunction + optional/dynamic/bloom skips (expr model)`).

---

### Task 3: Struct-extract column paths

**Files:** Modify `arrow_zerocopy_shim.cpp` (`emit_column_path`), `mod.rs` (tests). (`resolve_path` already handles depth>1.)

**Interfaces:** Consumes Task 1 path encoding + `resolve_path`.

- [ ] **Step 1: Write a failing struct-filter test** in `mod.rs`:
```rust
#[test]
fn filter_on_struct_field_matches_appender() {
    use arrow::array::{ArrayRef, Int64Array, StructArray};
    use arrow::datatypes::{DataType, Field, Fields};
    // one column `s` STRUCT(a BIGINT, b BIGINT)
    let a = Arc::new(Int64Array::from(vec![1,2,3,4])) as ArrayRef;
    let b = Arc::new(Int64Array::from(vec![10,20,30,40])) as ArrayRef;
    let fields: Fields = vec![Field::new("a", DataType::Int64, false), Field::new("b", DataType::Int64, false)].into();
    let s = Arc::new(StructArray::new(fields.clone(), vec![a, b], None)) as ArrayRef;
    let schema = Arc::new(arrow::datatypes::Schema::new(vec![Field::new("s", DataType::Struct(fields), false)]));
    let batch = RecordBatch::try_new(schema, vec![s]).unwrap();
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch("CREATE TABLE t (s STRUCT(a BIGINT, b BIGINT))").unwrap();
    { let mut app = conn.appender("t").unwrap(); app.append_record_batch(batch.clone()).unwrap(); }
    let reg = conn.register_arrow("v", vec![batch]).unwrap();
    let q = |sql: String| conn.query_row::<i64,_,_>(&sql, [], |r| r.get(0)).unwrap();
    let w = "WHERE s.a > 2";
    assert_eq!(q(format!("SELECT count(*) FROM v {w}")), q(format!("SELECT count(*) FROM t {w}")));
    drop(reg);
}
```

- [ ] **Step 2: Extend `emit_column_path`** to port `ResolveColumn`: if not BOUND_REF, require BOUND_FUNCTION with `TryGetStructExtractChildIndex(func, child_idx)`; recurse `emit_column_path(func.GetChildren()[0], out)` then `out.push_back(child_idx)`. Return false otherwise (→ UNHANDLED via the caller).

- [ ] **Step 3: Build + test.** The struct test passes; `resolve_path` walks the `StructArray` child. All prior tests still pass.

- [ ] **Step 4: Commit** (`feat(arrow): struct_extract column paths in filter pushdown`).

---

### Task 4: Scalar type breadth (Date/Time/Timestamp/Decimal128) + NaN

**Files:** Modify `arrow_zerocopy_shim.cpp` (`emit_scalar` + NaN flag), `filter.rs` (`compare` scalar arms + NaN), `mod.rs` (tests).

**Interfaces:** Consumes the wire scalar types (stype 6/7/8 + nan).

- [ ] **Step 1: Failing tests** for Date32, Timestamp(us), Decimal128, and a NaN comparison (`WHERE f != 'NaN'`), each vs the Appender oracle (build typed `sample_*` batches; assert count/sum equals `t`).

- [ ] **Step 2: C++ `emit_scalar`** — add: `DATE` → stype 7 + `v.GetValue<duckdb::date_t>().days` (i32); `TIMESTAMP*`/`TIME*` → stype 8 + i64 value + unit byte (us=2 default; map MS=1/NS=3/SEC=0); `DECIMAL` (INT128 only per CanPushdown) → stype 6 + i128 bytes + `DecimalType::GetScale(v.type())`. For FLOAT/DOUBLE constants, set the COMPARE `nan` byte when `duckdb::Value::IsNan(...)`.

- [ ] **Step 3: Rust `compare`** — add column-type arms: Date32 (from stype7 i32), Time32/Time64 + Timestamp(unit) (from stype8, match the column's `TimeUnit`), Decimal128 (from stype6 i128 with `.with_precision_and_scale(p, scale)` matching the column). When `nan` is set, apply NaN comparison semantics (`= NaN`→all-false; `!= NaN`→all-true-for-non-null; `<`/`>` per IEEE) using `f64::is_nan`. Mismatch/unsupported → `FilterError`.

- [ ] **Step 4: Build + test.** Type + NaN tests pass; prior tests pass.

- [ ] **Step 5: Commit** (`feat(arrow): Date/Time/Timestamp/Decimal128 + NaN filter scalars`).

---

### Task 5: Docs, benchmark, lint, full-suite green

**Files:** Modify `mod.rs` (module doc), `examples/arrow_zerocopy_bench.rs`, CHANGELOG.

- [ ] **Step 1: Update the module doc** in `mod.rs` to describe the EXPRESSION_FILTER model + the handled set (comparison/null/in/and-or/struct-extract; skip optional/dynamic/bloom; fail-loud rest) and that it targets DuckDB main.

- [ ] **Step 2: Benchmark** — confirm `examples/arrow_zerocopy_bench.rs` still builds/runs on main (filtered + unfiltered queries, assert == Appender). Run release; record numbers.
Run: `cargo run -p duckdb --features "bundled appender-arrow" --release --example arrow_zerocopy_bench 2>&1 | tail -8`

- [ ] **Step 3: fmt + clippy + FULL suite on main.**
Run: `cargo fmt -p duckdb && cargo clippy -p duckdb --features "bundled appender-arrow" --all-targets 2>&1 | tail -20`
Run: `cargo test -p duckdb --features "bundled appender-arrow" 2>&1 | tail -25` — confirm the bump didn't regress non-arrow tests (note any pre-existing main-related failures separately). Fix arrow-attributable issues.

- [ ] **Step 4: CHANGELOG / PR notes** entry (per CONTRIBUTING.md), noting this targets DuckDB main and the EXPRESSION_FILTER coverage.

- [ ] **Step 5: Commit** (`docs(arrow): EXPRESSION_FILTER model docs, bench, lint, full-suite on main`).

---

## Post-implementation
Record the outcome (coverage now matching duckdb-python, benchmark, full-suite status on main) as an addendum to the Daft follow-ups doc, and note the branch pins unreleased DuckDB main (`06eb6b6`) — not upstream-mergeable until re-pinned to a release that has the EXPRESSION_FILTER migration.

## Self-Review notes (for the implementer)
- **The reference file is authoritative** — when this plan and `filter_pushdown_visitor.cpp` differ on node handling, follow the reference (adapt pybind→wire-format/Rust). Mirror its AND-drop / OR-fail / optional-skip / table-filter-function-skip logic exactly.
- **Column-path semantics**: `path[0]` is the projected-batch position (the `TableFilterSet` key, same as the prior fix); `path[1..]` are struct child indices. The multi-column + struct tests guard it.
- **Fail-loud invariant**: any UNHANDLED node / unresolved path / unsupported scalar → `FilterError` → `error_stream`; never a wrong row set. Keep the `unhandled_filter_kind_errors_not_silently_wrong` unit test.
- **Arrow v58 API drift** (`StructArray::new`/`.column`, `Decimal128Array::with_precision_and_scale`, `Scalar`, cmp kernels): adapt against `crates/duckdb/src/vtab/arrow.rs`; do not invent.
- **Slow first build** (main amalgamation). Subsequent edits are faster.

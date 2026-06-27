# duckdb-rs Arrow Filter Pushdown — v1.5.4 Full-Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing v1.5.4 structured-filter walker for `Connection::register_arrow` to reach duckdb-python coverage parity — column paths + `STRUCT_EXTRACT`, scalar breadth (Date / Time / Timestamp / Decimal128 / Blob + NaN), and a full `EXPRESSION_FILTER` expression-tree port.

**Architecture:** Two C++ front-ends in `arrow_zerocopy_shim.cpp` (the existing structured `emit_filter`, extended; and a new `emit_expression` for the `EXPRESSION_FILTER` kind) both serialize into one little-endian wire format whose column references are **paths** (projected root column + struct child indices). One Rust evaluator (`filter.rs`) decodes the buffer, resolves each path to a leaf array (descending `StructArray` children, folding ancestor-null masks), and produces a `BooleanArray` for `filter_record_batch`. Untranslatable required filters fail loud (UNHANDLED → `error_stream`); only OPTIONAL/DYNAMIC/BLOOM are safe-skipped.

**Tech Stack:** Rust (crate `duckdb`), C++17-not-required (released v1.5.4 builds as C++11), arrow-rs (the version vendored by duckdb-rs), DuckDB v1.5.4 bundled amalgamation, `cc` build of the shim.

## Global Constraints

- **Branch:** all work on `arrow-zerocopy-fullfilter`, branched from `arrow-zerocopy`. Never commit to `main`, `arrow-zerocopy`, or `arrow-zerocopy-duckdb-main`.
- **DuckDB bundle stays v1.5.4** (submodule `08e34c44`, the `v1.5.4` tag). **Do NOT** bump the submodule or touch `build_bundled_cc.rs` / `build_bundled_cmake.rs`. No `-std=c++17`, no jemalloc edits — those belong to the abandoned main-bump branch.
- **Build/test feature flags (every command):** `--features "bundled appender-arrow"`. Package is `duckdb` (`-p duckdb`).
- **First build is slow** (compiles the v1.5.4 amalgamation once). Later builds recompile only the shim object + the `duckdb` crate. Do not interpret the first long build as a hang.
- **Commit message trailer (every commit):** end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Correctness invariant:** never an over/under-inclusive row set for a *translated* filter. DuckDB does **not** re-apply pushed arrow-scan filters, so any untranslatable *required* filter (including any AND/OR child or any sub-expression) → emit UNHANDLED (tag 255) → Rust `FilterError` → `error_stream` (clean query error). Only `OPTIONAL_FILTER` / `DYNAMIC_FILTER` / `BLOOM_FILTER` are safe-skipped (the join above re-applies them). This deviates intentionally from duckdb-python's conjunct-dropping, which relies on re-application that our path does not get.
- **Wire-format contract (extended; the single source of truth both files implement):**
  - **Node:** `tag:u8` + payload
    - `0` COMPARE: `path` + `op:u8` + `scalar`
    - `1` IS_NULL: `path`
    - `2` IS_NOT_NULL: `path`
    - `3` AND: `n:u32` + `n × node`
    - `4` OR: `n:u32` + `n × node`
    - `5` IN: `path` + `n:u32` + `n × scalar`
    - `255` UNHANDLED: `raw:u32`
  - **path:** `depth:u32` + `depth × u32` (depth ≥ 1; `index[0]` = projected root column position, `index[1..]` = struct child indices)
  - **op byte:** `0`=EQ `1`=NE `2`=LT `3`=GT `4`=LE `5`=GE (DuckDB ExpressionType 25–30)
  - **scalar:** `stype:u8` + payload — `0` i64 (8 LE), `1` u64 (8 LE), `2` f64 (8 LE IEEE bits; NaN carried as its bits), `3` bool (1), `4` utf8 (`len:u32`+bytes), `5` null (none), `6` date32 (`i32`, 4 LE), `7` time (`unit:u8`+`i64`), `8` timestamp (`unit:u8`+`i64`), `9` decimal128 (`i128`, 16 LE, + `scale:u8`), `10` blob (`len:u32`+bytes), `255` unsupported (none)
  - **time/timestamp unit byte:** `0`=s `1`=ms `2`=µs `3`=ns

---

## File Structure

- `crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp` — C++ trampoline + filter serialization. Gains: `push_i32`/`push_i128`/`push_column_path`; widened `emit_scalar`; `emit_filter` takes a path and handles `STRUCT_EXTRACT`; new `emit_expression` + `resolve_column_path`; `ShimProduce` dispatches `EXPRESSION_FILTER` to `emit_expression`.
- `crates/duckdb/src/arrow_zerocopy/filter.rs` — wire decoder + evaluator. Gains: `ScalarVal` variants (Date32/Time/Timestamp/Decimal128/Blob); `FilterNode` column fields become `path: Vec<usize>`; `read_i32`/`read_i128`/`read_scalar` extensions; `resolve_path`; new `compare` arms; NaN total-order semantics.
- `crates/duckdb/src/arrow_zerocopy/mod.rs` — registration + tests + doc comment. Gains: updated `register_arrow` doc; new oracle/unit tests per task.
- `crates/duckdb/examples/arrow_zerocopy_bench.rs` — benchmark; touched only in Task 7 if a struct/typed column is added to the workload.
- `crates/duckdb/CHANGELOG.md` (or the crate's changelog file if differently named) — Task 7 entry.

Tasks are ordered so each leaves the branch compiling and green. Tasks 1–2 do the path migration + struct support; 3–4 widen scalars; 5–6 add the expression walker; 7 polishes.

---

### Task 1: Branch + wire-format migration (column → path)

Pure refactor: replace the single `col:u32` in COMPARE/IS_NULL/IS_NOT_NULL/IN with a `path` (depth+indices), end to end, with `resolve_path` handling struct descent. No new filter capability yet (struct paths are produced starting in Task 2). All existing tests must stay green after the field rename.

**Files:**
- Modify: `crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp`
- Modify: `crates/duckdb/src/arrow_zerocopy/filter.rs`
- Test: `crates/duckdb/src/arrow_zerocopy/filter.rs` (the `#[cfg(test)]` direct-evaluator tests live in `mod.rs`; update them here in this task)
- Modify (tests): `crates/duckdb/src/arrow_zerocopy/mod.rs`

**Interfaces:**
- Produces (Rust, used by every later task):
  - `pub(crate) enum FilterNode { Compare { path: Vec<usize>, op: CmpOp, val: ScalarVal }, IsNull { path: Vec<usize> }, IsNotNull { path: Vec<usize> }, And(Vec<FilterNode>), Or(Vec<FilterNode>), In { path: Vec<usize>, vals: Vec<ScalarVal> }, Unhandled(u32) }`
  - `fn resolve_path(batch: &RecordBatch, path: &[usize]) -> Result<(ArrayRef, Option<BooleanArray>), FilterError>` — returns the leaf array and an optional ancestor-valid mask (true where every ancestor struct is non-null; `None` = no ancestor had nulls).
- Produces (C++, used by Task 2/5/6): `static void push_column_path(std::vector<uint8_t>&, const std::vector<duckdb::idx_t>&)`; `emit_filter(const duckdb::TableFilter&, const std::vector<duckdb::idx_t>& path, std::vector<uint8_t>&)`.

- [ ] **Step 1: Create the branch**

```bash
cd /Volumes/Work/Code/duckdb-rs
git checkout arrow-zerocopy
git checkout -b arrow-zerocopy-fullfilter
git rev-parse --abbrev-ref HEAD   # expect: arrow-zerocopy-fullfilter
git ls-tree HEAD crates/libduckdb-sys/duckdb-sources   # expect commit 08e34c44... (v1.5.4)
```

- [ ] **Step 2: Update the wire-format doc comment in the shim**

In `crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp`, replace the "Node format" comment block so each node that had `col:u32` now reads `path` and add the path/scalar additions. Replace the existing block:

```cpp
// Node format:  tag:u8 + payload
//   tag 0  (COMPARE):    path + op:u8 + scalar
//   tag 1  (IS_NULL):    path
//   tag 2  (IS_NOT_NULL): path
//   tag 3  (AND):        n:u32  + n*node
//   tag 4  (OR):         n:u32  + n*node
//   tag 5  (IN):         path + n:u32 + n*scalar
//   tag 255 (UNHANDLED): raw_filter_type:u32
//
// path:  depth:u32 + depth*u32  (depth >= 1; index[0] = projected root column,
//        index[1..] = struct child indices)
//
// Scalar format:  stype:u8 + payload
//   stype 0 (i64) 1 (u64) 2 (f64) 3 (bool) 4 (utf8 len:u32+bytes) 5 (null)
//   stype 255 (unsupported)
//   (Date/Time/Timestamp/Decimal128/Blob scalars are added in later tasks.)
//
// op byte (for tag 0):  0=EQ 1=NE 2=LT 3=GT 4=LE 5=GE  (matches ExpressionType 25-30)
```

- [ ] **Step 3: Add `push_column_path` and switch `emit_filter` to a path**

After `push_f64` in the shim, add:

```cpp
static void push_column_path(std::vector<uint8_t> &buf, const std::vector<duckdb::idx_t> &path) {
    push_u32(buf, (uint32_t)path.size());
    for (auto idx : path) { push_u32(buf, (uint32_t)idx); }
}
```

Change `emit_filter`'s signature from `(const duckdb::TableFilter &filter, duckdb::idx_t projected_col, std::vector<uint8_t> &buf)` to `(const duckdb::TableFilter &filter, const std::vector<duckdb::idx_t> &path, std::vector<uint8_t> &buf)`. Inside it:
- COMPARE: replace `push_u32(buf, (uint32_t)projected_col);` with `push_column_path(buf, path);`
- IS_NULL: replace `push_u32(buf, (uint32_t)projected_col);` with `push_column_path(buf, path);`
- IS_NOT_NULL: same replacement.
- IN: replace `push_u32(buf, (uint32_t)projected_col);` with `push_column_path(buf, path);`
- CONJUNCTION_AND / CONJUNCTION_OR: recurse `emit_filter(*child, path, buf);` (pass the path through unchanged).
- OPTIONAL_FILTER: recurse `emit_filter(*opt.child_filter, path, buf);` (unchanged otherwise).

- [ ] **Step 4: Update `ShimProduce` to pass a single-element path**

In `ShimProduce`, change the filter loop body from `emit_filter(tf, projected_col, filter_buf);` to:

```cpp
duckdb::idx_t projected_col = kv.first; // already the projected batch position
const duckdb::TableFilter &tf = *kv.second;
emit_filter(tf, std::vector<duckdb::idx_t>{projected_col}, filter_buf);
```

- [ ] **Step 5: Migrate the Rust `FilterNode` to paths + add `resolve_path`**

In `crates/duckdb/src/arrow_zerocopy/filter.rs`:

Change the `FilterNode` enum's `col: usize` fields to `path: Vec<usize>` (Compare, IsNull, IsNotNull, In). Add `StructArray` to the imports: in the `use arrow::array::{...}` list add `StructArray`.

Add a path reader to `Cursor`:

```rust
fn read_path(&mut self) -> Vec<usize> {
    let depth = self.read_u32() as usize;
    (0..depth).map(|_| self.read_u32() as usize).collect()
}
```

In `read_node_inner`, replace each `let col = self.read_u32() as usize;` (and the `IsNull { col: self.read_u32() as usize }` / `IsNotNull { ... }` shorthands) with `read_path()` and the `path` field:

```rust
0 => {
    let path = self.read_path();
    let op_byte = self.read_u8();
    let op = match op_byte {
        0 => CmpOp::Eq, 1 => CmpOp::Ne, 2 => CmpOp::Lt,
        3 => CmpOp::Gt, 4 => CmpOp::Le, 5 => CmpOp::Ge,
        _ => { let _ = self.read_scalar(); return FilterNode::Unhandled(255); }
    };
    let val = self.read_scalar();
    FilterNode::Compare { path, op, val }
}
1 => FilterNode::IsNull { path: self.read_path() },
2 => FilterNode::IsNotNull { path: self.read_path() },
// 3 (AND) / 4 (OR) unchanged
5 => {
    let path = self.read_path();
    let n = self.read_u32() as usize;
    let vals = (0..n).map(|_| self.read_scalar()).collect();
    FilterNode::In { path, vals }
}
```

Add `resolve_path` near the evaluator (it needs `boolean` from the existing `use arrow::compute::kernels::{boolean, cmp};`):

```rust
/// Resolve a column path to its leaf array plus an optional "ancestor-valid" mask.
/// `path[0]` selects the projected column; each subsequent index descends into a
/// `StructArray` child. The returned mask (when `Some`) is `true` where every ancestor
/// struct is non-null at that row; `None` means no ancestor struct carried nulls.
fn resolve_path(
    batch: &RecordBatch,
    path: &[usize],
) -> Result<(ArrayRef, Option<BooleanArray>), FilterError> {
    let root = *path
        .first()
        .ok_or_else(|| FilterError::Unhandled("empty column path".into()))?;
    if root >= batch.num_columns() {
        return Err(FilterError::Unhandled(format!("column index {root} out of range")));
    }
    let mut arr: ArrayRef = batch.column(root).clone();
    let mut ancestor_valid: Option<BooleanArray> = None;
    for &idx in &path[1..] {
        let sa = arr.as_any().downcast_ref::<StructArray>().ok_or_else(|| {
            FilterError::Unhandled(format!("path descends into non-struct: {:?}", arr.data_type()))
        })?;
        if let Some(nulls) = sa.nulls() {
            let this_valid = BooleanArray::new(nulls.inner().clone(), None); // set bit = valid
            ancestor_valid = Some(match ancestor_valid {
                Some(prev) => boolean::and(&prev, &this_valid).map_err(|e| FilterError::Unhandled(e.to_string()))?,
                None => this_valid,
            });
        }
        let child = sa
            .columns()
            .get(idx)
            .ok_or_else(|| FilterError::Unhandled(format!("struct child index {idx} out of range")))?;
        arr = child.clone();
    }
    Ok((arr, ancestor_valid))
}
```

Rewrite the `evaluate` arms that referenced `col` to use `resolve_path` and fold the ancestor mask:

```rust
FilterNode::IsNull { path } => {
    let (col, valid) = resolve_path(batch, path)?;
    let mut m = is_null(col.as_ref()).map_err(|e| FilterError::Unhandled(e.to_string()))?;
    if let Some(v) = valid {
        let not_v = boolean::not(&v).map_err(|e| FilterError::Unhandled(e.to_string()))?;
        m = boolean::or(&m, &not_v).map_err(|e| FilterError::Unhandled(e.to_string()))?;
    }
    Ok(m)
}
FilterNode::IsNotNull { path } => {
    let (col, valid) = resolve_path(batch, path)?;
    let mut m = is_not_null(col.as_ref()).map_err(|e| FilterError::Unhandled(e.to_string()))?;
    if let Some(v) = valid {
        m = boolean::and(&m, &v).map_err(|e| FilterError::Unhandled(e.to_string()))?;
    }
    Ok(m)
}
FilterNode::In { path, vals } => {
    let (col_arr, valid) = resolve_path(batch, path)?;
    let mut acc = BooleanArray::from(vec![false; batch.num_rows()]);
    for val in vals {
        let eq_mask = compare(&col_arr, &CmpOp::Eq, val)?;
        let n = eq_mask.len();
        let eq_no_null = BooleanArray::from(
            (0..n).map(|i| eq_mask.is_valid(i) && eq_mask.value(i)).collect::<Vec<bool>>(),
        );
        acc = boolean::or(&acc, &eq_no_null).map_err(|e| FilterError::Unhandled(e.to_string()))?;
    }
    if let Some(v) = valid {
        acc = boolean::and(&acc, &v).map_err(|e| FilterError::Unhandled(e.to_string()))?;
    }
    Ok(acc)
}
FilterNode::Compare { path, op, val } => {
    let (col, valid) = resolve_path(batch, path)?;
    let mask = compare(&col, op, val)?;
    match valid {
        Some(v) => boolean::and(&mask, &v).map_err(|e| FilterError::Unhandled(e.to_string())),
        None => Ok(mask),
    }
}
```

- [ ] **Step 6: Update the existing direct-evaluator unit tests to the `path` field**

In `crates/duckdb/src/arrow_zerocopy/mod.rs`, every `FilterNode::Compare { col: N, ... }` / `IsNull { col: N }` / `IsNotNull { col: N }` / `In { col: N, ... }` in the `#[cfg(test)] mod test` block becomes `{ path: vec![N], ... }`. Affected tests (search for `col:`): `evaluate_is_null_and_is_not_null_direct`, `evaluate_or_direct`, `evaluate_in_does_not_match_null_column`, `evaluate_compare_int32_direct`, `evaluate_compare_uint32_direct`, `evaluate_compare_float64_direct`, `evaluate_compare_boolean_direct`, `evaluate_compare_unsupported_combo_errors`, `evaluate_compare_out_of_range_scalar_errors`. Example transform:

```rust
// before
let node = FilterNode::Compare { col: 0, op: CmpOp::Eq, val: ScalarVal::F64(1.0) };
// after
let node = FilterNode::Compare { path: vec![0], op: CmpOp::Eq, val: ScalarVal::F64(1.0) };
```

- [ ] **Step 7: Build and run the full arrow_zerocopy suite**

Run:
```bash
cd /Volumes/Work/Code/duckdb-rs
cargo test -p duckdb --features "bundled appender-arrow" arrow_zerocopy -- --nocapture
```
Expected: all existing arrow_zerocopy tests PASS (round_trip_select_star, projection_subset, self_join_is_replayable, filter_correct_with_pushdown_enabled, constant_comparison_filter_applied_no_global_setting, multi_column_projection_and_filter_matches_appender, drop_unregisters_view, and all `evaluate_*` direct tests). This proves the path migration is behavior-preserving.

- [ ] **Step 8: Commit**

```bash
git add crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp crates/duckdb/src/arrow_zerocopy/filter.rs crates/duckdb/src/arrow_zerocopy/mod.rs
git commit -m "$(cat <<'EOF'
refactor(arrow): migrate filter wire format from column index to path

Column references in COMPARE/IS_NULL/IS_NOT_NULL/IN become a path
(depth + indices); resolve_path descends StructArray children and folds
ancestor-null masks. Behavior-preserving (depth-1 paths); enables
STRUCT_EXTRACT and struct_extract expression chains in later tasks.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `STRUCT_EXTRACT` structured filter kind

Emit a column path for the `STRUCT_EXTRACT` structured `TableFilter` (recurse `StructFilter.child_filter`, appending `child_idx`), so `WHERE s.field <op> const` pushes down. The Rust side already descends paths (Task 1).

**Files:**
- Modify: `crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp`
- Test: `crates/duckdb/src/arrow_zerocopy/mod.rs`

**Interfaces:**
- Consumes: `emit_filter(filter, path, buf)` and `push_column_path` (Task 1); Rust `resolve_path` (Task 1).
- Produces: struct-field pushdown for the structured kind.

- [ ] **Step 1: Write the failing oracle test (struct field filter)**

Add to `mod.rs` `mod test`. The test builds a `StructArray` column and compares the zero-copy view against an Appender oracle on `WHERE s.id > 2`.

```rust
#[test]
fn struct_extract_filter_matches_appender() {
    use arrow::array::{ArrayRef, Int64Array, StructArray};
    use arrow::buffer::NullBuffer;
    use arrow::datatypes::{DataType, Field, Fields};
    // Column `s STRUCT(id BIGINT, score BIGINT)`, with one NULL struct row (row 2).
    let id = Arc::new(Int64Array::from(vec![1i64, 2, 3, 4])) as ArrayRef;
    let score = Arc::new(Int64Array::from(vec![10i64, 20, 30, 40])) as ArrayRef;
    let fields: Fields = vec![
        Field::new("id", DataType::Int64, false),
        Field::new("score", DataType::Int64, false),
    ]
    .into();
    let nulls = NullBuffer::from(vec![true, true, false, true]); // row 2 = NULL struct
    let s = Arc::new(StructArray::new(fields.clone(), vec![id, score], Some(nulls))) as ArrayRef;
    let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Struct(fields), true)]));
    let batch = RecordBatch::try_new(schema, vec![s]).unwrap();

    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch("CREATE TABLE t (s STRUCT(id BIGINT, score BIGINT))").unwrap();
    {
        let mut app = conn.appender("t").unwrap();
        app.append_record_batch(batch.clone()).unwrap();
    }
    let reg = conn.register_arrow("v", vec![batch]).unwrap();

    let q = |sql: String| conn.query_row::<i64, _, _>(&sql, [], |r| r.get(0)).unwrap();
    for where_clause in ["WHERE s.id > 2", "WHERE s.score = 20", "WHERE s.id IS NOT NULL"] {
        assert_eq!(
            q(format!("SELECT count(*) FROM v {where_clause}")),
            q(format!("SELECT count(*) FROM t {where_clause}")),
            "struct-field filter must match the Appender oracle: {where_clause}"
        );
    }
    drop(reg);
}
```

- [ ] **Step 2: Run it; expect FAIL**

Run:
```bash
cargo test -p duckdb --features "bundled appender-arrow" struct_extract_filter_matches_appender -- --nocapture
```
Expected: FAIL — currently `STRUCT_EXTRACT` hits the `else` branch in `emit_filter` → UNHANDLED → the `s.id > 2` query errors (the test panics on `.unwrap()` of an errored query).

- [ ] **Step 3: Include the struct_filter header and add the `STRUCT_EXTRACT` branch**

In `arrow_zerocopy_shim.cpp`, add the include alongside the other filter includes:
```cpp
#include "duckdb/planner/filter/struct_filter.hpp"
```

In `emit_filter`, add a branch (before the final `else`):
```cpp
} else if (filter.filter_type == TableFilterType::STRUCT_EXTRACT) {
    // Descend into a struct child: append child_idx to the path and recurse on the child filter.
    auto &sf = filter.Cast<duckdb::StructFilter>();
    std::vector<duckdb::idx_t> child_path = path;
    child_path.push_back(sf.child_idx);
    emit_filter(*sf.child_filter, child_path, buf);
```

Also add `STRUCT_EXTRACT` to `is_handled_filter_type` so an `OPTIONAL_FILTER` wrapping a struct filter is unwrapped rather than skipped:
```cpp
case TableFilterType::STRUCT_EXTRACT:
    return true;
```

- [ ] **Step 4: Run the struct test; expect PASS**

Run:
```bash
cargo test -p duckdb --features "bundled appender-arrow" struct_extract_filter_matches_appender -- --nocapture
```
Expected: PASS.

- [ ] **Step 5: Run the whole arrow_zerocopy suite (no regressions)**

Run:
```bash
cargo test -p duckdb --features "bundled appender-arrow" arrow_zerocopy -- --nocapture
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp crates/duckdb/src/arrow_zerocopy/mod.rs
git commit -m "$(cat <<'EOF'
feat(arrow): push down STRUCT_EXTRACT filters via column paths

Cast StructFilter, append child_idx to the path, recurse on child_filter.
Struct-field predicates (s.field <op> const, IS [NOT] NULL) now push into
the Rust evaluator; ancestor-null structs handled by resolve_path.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Scalar breadth — Date / Time / Timestamp / Blob

Widen `emit_scalar` and the Rust decoder/`compare` to Date32, Time64(µs), the four Timestamp units (incl. TZ), and Blob. (Decimal128 + NaN are Task 4.) TIME with nanosecond precision is intentionally out of scope (it falls through to `unsupported` → fail-loud, which is correctness-safe).

**Files:**
- Modify: `crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp`
- Modify: `crates/duckdb/src/arrow_zerocopy/filter.rs`
- Test: `crates/duckdb/src/arrow_zerocopy/mod.rs`

**Interfaces:**
- Consumes: wire scalar tags 6/7/8/10 + unit byte (Global Constraints).
- Produces (Rust): `ScalarVal::Date32(i32)`, `ScalarVal::Time { unit: u8, v: i64 }`, `ScalarVal::Timestamp { unit: u8, v: i64 }`, `ScalarVal::Blob(Vec<u8>)`; `compare` arms for `DataType::{Date32, Time64, Timestamp, Binary, LargeBinary}`.

- [ ] **Step 1: Write failing oracle tests (date / timestamp / blob)**

Add to `mod.rs` `mod test`:

```rust
#[test]
fn typed_scalar_filters_match_appender() {
    use arrow::array::{ArrayRef, BinaryArray, Date32Array, TimestampMicrosecondArray};
    use arrow::datatypes::{DataType, Field, TimeUnit};
    // date d (Date32), ts (Timestamp µs, no tz), b (Binary)
    let d = Arc::new(Date32Array::from(vec![19000i32, 19001, 19002, 19003])) as ArrayRef; // days since epoch
    let ts = Arc::new(TimestampMicrosecondArray::from(vec![
        1_000_000i64, 2_000_000, 3_000_000, 4_000_000,
    ])) as ArrayRef;
    let b = Arc::new(BinaryArray::from(vec![
        b"aa".as_ref(), b"bb".as_ref(), b"cc".as_ref(), b"dd".as_ref(),
    ])) as ArrayRef;
    let schema = Arc::new(Schema::new(vec![
        Field::new("d", DataType::Date32, false),
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), false),
        Field::new("b", DataType::Binary, false),
    ]));
    let batch = RecordBatch::try_new(schema, vec![d, ts, b]).unwrap();

    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch("CREATE TABLE t (d DATE, ts TIMESTAMP, b BLOB)").unwrap();
    {
        let mut app = conn.appender("t").unwrap();
        app.append_record_batch(batch.clone()).unwrap();
    }
    let reg = conn.register_arrow("v", vec![batch]).unwrap();

    let q = |sql: String| conn.query_row::<i64, _, _>(&sql, [], |r| r.get(0)).unwrap();
    for where_clause in [
        "WHERE d > DATE '2022-01-02'",
        "WHERE ts >= TIMESTAMP '1970-01-01 00:00:03'",
        "WHERE b = '\\x63\\x63'::BLOB",
    ] {
        assert_eq!(
            q(format!("SELECT count(*) FROM v {where_clause}")),
            q(format!("SELECT count(*) FROM t {where_clause}")),
            "typed-scalar filter must match the Appender oracle: {where_clause}"
        );
    }
    drop(reg);
}
```

- [ ] **Step 2: Run it; expect FAIL**

Run:
```bash
cargo test -p duckdb --features "bundled appender-arrow" typed_scalar_filters_match_appender -- --nocapture
```
Expected: FAIL — `emit_scalar` emits tag 255 (unsupported) for DATE/TIMESTAMP/BLOB → UNHANDLED → query errors.

- [ ] **Step 3: Add C++ encoders (`push_i32`) and widen `emit_scalar`**

In `arrow_zerocopy_shim.cpp`, add `push_i32` after `push_u32`:
```cpp
static void push_i32(std::vector<uint8_t> &buf, int32_t v) {
    uint32_t u = (uint32_t)v;
    for (int i = 0; i < 4; i++) { buf.push_back((uint8_t)(u & 0xff)); u >>= 8; }
}
```

Add the date/time header include near the top includes:
```cpp
#include "duckdb/common/types/date.hpp"
#include "duckdb/common/types/time.hpp"
#include "duckdb/common/types/timestamp.hpp"
```

In `emit_scalar`, add cases before `default:` (the existing int/float/bool/varchar cases stay):
```cpp
        case duckdb::LogicalTypeId::DATE:
            push_u8(buf, 6);
            push_i32(buf, v.GetValue<duckdb::date_t>().days);
            break;
        case duckdb::LogicalTypeId::TIME:
            push_u8(buf, 7);
            push_u8(buf, 2); // unit = µs
            push_i64(buf, v.GetValue<duckdb::dtime_t>().micros);
            break;
        case duckdb::LogicalTypeId::TIMESTAMP:
        case duckdb::LogicalTypeId::TIMESTAMP_TZ:
            push_u8(buf, 8);
            push_u8(buf, 2); // µs
            push_i64(buf, v.GetValue<duckdb::timestamp_t>().value);
            break;
        case duckdb::LogicalTypeId::TIMESTAMP_MS:
            push_u8(buf, 8);
            push_u8(buf, 1); // ms
            push_i64(buf, v.GetValue<duckdb::timestamp_t>().value);
            break;
        case duckdb::LogicalTypeId::TIMESTAMP_NS:
            push_u8(buf, 8);
            push_u8(buf, 3); // ns
            push_i64(buf, v.GetValue<duckdb::timestamp_t>().value);
            break;
        case duckdb::LogicalTypeId::TIMESTAMP_SEC:
            push_u8(buf, 8);
            push_u8(buf, 0); // s
            push_i64(buf, v.GetValue<duckdb::timestamp_t>().value);
            break;
        case duckdb::LogicalTypeId::BLOB:
            push_u8(buf, 10);
            {
                std::string s = v.GetValueUnsafe<duckdb::string_t>().GetString();
                push_u32(buf, (uint32_t)s.size());
                for (char c : s) { buf.push_back((uint8_t)c); }
            }
            break;
```

(If `v.GetValueUnsafe<duckdb::string_t>()` does not compile for BLOB on v1.5.4, use `v.ToString()` is wrong for blobs; instead use the value's internal string via `duckdb::StringValue::Get(v)` which returns the raw bytes as `std::string`. Verify which compiles and keep the one that yields raw bytes.)

- [ ] **Step 4: Add Rust `ScalarVal` variants + decoder + reader**

In `filter.rs`, extend imports: add `BinaryArray, Date32Array, LargeBinaryArray, Time64MicrosecondArray, TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray` to the `use arrow::array::{...}` list, and add `use arrow::datatypes::TimeUnit;`.

Extend `ScalarVal`:
```rust
#[derive(Debug)]
pub(crate) enum ScalarVal {
    I64(i64),
    U64(u64),
    F64(f64),
    Bool(bool),
    Utf8(String),
    Date32(i32),
    Time { unit: u8, v: i64 },
    Timestamp { unit: u8, v: i64 },
    Blob(Vec<u8>),
    Null,
    Unsupported,
}
```

Add `read_i32` to `Cursor`:
```rust
fn read_i32(&mut self) -> i32 {
    self.read_u32() as i32
}
```

Extend `read_scalar` (add arms before the `_ =>`):
```rust
6 => ScalarVal::Date32(self.read_i32()),
7 => { let unit = self.read_u8(); let v = self.read_i64(); ScalarVal::Time { unit, v } }
8 => { let unit = self.read_u8(); let v = self.read_i64(); ScalarVal::Timestamp { unit, v } }
10 => {
    let len = self.read_u32() as usize;
    let bytes = self.buf.get(self.pos..self.pos + len).unwrap_or(&[]).to_vec();
    self.pos += len;
    ScalarVal::Blob(bytes)
}
```

- [ ] **Step 5: Add `compare` arms (date / time / timestamp / blob)**

In `filter.rs` `compare`, add arms before the final `_ =>`. Timestamps convert the wire value to the column's unit losslessly or fail loud:

```rust
        // ── Date / Time ────────────────────────────────────────────────────────
        (DataType::Date32, ScalarVal::Date32(v)) => dispatch(op, col, &Scalar::new(Date32Array::from(vec![*v]))),
        (DataType::Time64(TimeUnit::Microsecond), ScalarVal::Time { unit: 2, v }) => {
            dispatch(op, col, &Scalar::new(Time64MicrosecondArray::from(vec![*v])))
        }

        // ── Blob ────────────────────────────────────────────────────────────────
        (DataType::Binary, ScalarVal::Blob(b)) => dispatch(op, col, &Scalar::new(BinaryArray::from(vec![b.as_slice()]))),
        (DataType::LargeBinary, ScalarVal::Blob(b)) => {
            dispatch(op, col, &Scalar::new(LargeBinaryArray::from(vec![b.as_slice()])))
        }

        // ── Timestamps (convert wire unit → column unit, lossless or fail loud) ──
        (DataType::Timestamp(col_unit, tz), ScalarVal::Timestamp { unit, v }) => {
            let v_col = convert_time_unit(*v, *unit, time_unit_tag(col_unit))?;
            match col_unit {
                TimeUnit::Second => dispatch(op, col, &Scalar::new(TimestampSecondArray::from(vec![v_col]).with_timezone_opt(tz.clone()))),
                TimeUnit::Millisecond => dispatch(op, col, &Scalar::new(TimestampMillisecondArray::from(vec![v_col]).with_timezone_opt(tz.clone()))),
                TimeUnit::Microsecond => dispatch(op, col, &Scalar::new(TimestampMicrosecondArray::from(vec![v_col]).with_timezone_opt(tz.clone()))),
                TimeUnit::Nanosecond => dispatch(op, col, &Scalar::new(TimestampNanosecondArray::from(vec![v_col]).with_timezone_opt(tz.clone()))),
            }
        }
```

Add the unit helpers above `compare`:
```rust
fn time_unit_tag(u: &TimeUnit) -> u8 {
    match u {
        TimeUnit::Second => 0,
        TimeUnit::Millisecond => 1,
        TimeUnit::Microsecond => 2,
        TimeUnit::Nanosecond => 3,
    }
}

/// Convert an integer timestamp/time `v` from `from` unit to `to` unit.
/// Exact (multiply) when going finer-or-equal; fail loud if a coarser target would lose precision.
fn convert_time_unit(v: i64, from: u8, to: u8) -> Result<i64, FilterError> {
    const FACTORS: [i128; 4] = [1, 1_000, 1_000_000, 1_000_000_000]; // s, ms, µs, ns per second
    let (f, t) = (FACTORS[from as usize], FACTORS[to as usize]);
    let vi = v as i128;
    let out = if t >= f {
        vi * (t / f)
    } else {
        let div = f / t;
        if vi % div != 0 {
            return Err(FilterError::Unhandled(format!(
                "timestamp constant {v} (unit {from}) not representable in column unit {to}"
            )));
        }
        vi / div
    };
    i64::try_from(out).map_err(|_| FilterError::Unhandled(format!("timestamp constant overflow converting unit {from}->{to}")))
}
```

- [ ] **Step 6: Run the typed-scalar test; expect PASS**

Run:
```bash
cargo test -p duckdb --features "bundled appender-arrow" typed_scalar_filters_match_appender -- --nocapture
```
Expected: PASS.

- [ ] **Step 7: Run the whole arrow_zerocopy suite**

Run:
```bash
cargo test -p duckdb --features "bundled appender-arrow" arrow_zerocopy -- --nocapture
```
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp crates/duckdb/src/arrow_zerocopy/filter.rs crates/duckdb/src/arrow_zerocopy/mod.rs
git commit -m "$(cat <<'EOF'
feat(arrow): push down Date/Time/Timestamp/Blob filter scalars

Widen emit_scalar + the Rust decoder/compare to Date32, Time64(µs), all
four Timestamp units (incl. TZ via with_timezone_opt) and Blob. Timestamp
constants convert to the column unit losslessly or fail loud. TIME(ns)
stays out of scope (unsupported → fail-loud).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Decimal128 scalars + NaN-compare semantics

Add Decimal128 (i128 + scale) and DuckDB float **total-order** NaN comparison semantics (NaN sorts highest; `NaN == NaN` true) — arrow's default IEEE cmp differs, so NaN constants need explicit handling.

**Files:**
- Modify: `crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp`
- Modify: `crates/duckdb/src/arrow_zerocopy/filter.rs`
- Test: `crates/duckdb/src/arrow_zerocopy/mod.rs`

**Interfaces:**
- Consumes: wire scalar tag 9 (decimal128 = `i128` + `scale:u8`); f64 NaN bits via tag 2.
- Produces (Rust): `ScalarVal::Decimal128 { v: i128, scale: u8 }`; `compare` arms for `DataType::Decimal128`; NaN total-order handling inside the `Float32`/`Float64` arms.

- [ ] **Step 1: Write failing tests (decimal oracle + NaN direct unit tests)**

Add to `mod.rs` `mod test`:

```rust
#[test]
fn decimal128_filter_matches_appender() {
    use arrow::array::{ArrayRef, Decimal128Array};
    use arrow::datatypes::{DataType, Field};
    // DECIMAL(38,2): values 1.00, 2.50, 3.00, 4.25 → unscaled i128 *100
    let dec = Arc::new(
        Decimal128Array::from(vec![100i128, 250, 300, 425])
            .with_precision_and_scale(38, 2)
            .unwrap(),
    ) as ArrayRef;
    let schema = Arc::new(Schema::new(vec![Field::new("p", DataType::Decimal128(38, 2), false)]));
    let batch = RecordBatch::try_new(schema, vec![dec]).unwrap();

    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch("CREATE TABLE t (p DECIMAL(38,2))").unwrap();
    {
        let mut app = conn.appender("t").unwrap();
        app.append_record_batch(batch.clone()).unwrap();
    }
    let reg = conn.register_arrow("v", vec![batch]).unwrap();
    let q = |sql: String| conn.query_row::<i64, _, _>(&sql, [], |r| r.get(0)).unwrap();
    for where_clause in ["WHERE p > 2.50", "WHERE p = 3.00", "WHERE p <= 2.50"] {
        assert_eq!(
            q(format!("SELECT count(*) FROM v {where_clause}")),
            q(format!("SELECT count(*) FROM t {where_clause}")),
            "decimal128 filter must match the Appender oracle: {where_clause}"
        );
    }
    drop(reg);
}

#[test]
fn nan_compare_total_order_direct() {
    use super::filter::{CmpOp, FilterNode, ScalarVal, evaluate};
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field};
    // values: 1.0, NaN, 3.0, NULL
    let schema = Arc::new(Schema::new(vec![Field::new("f", DataType::Float64, true)]));
    let arr = Float64Array::from(vec![Some(1.0), Some(f64::NAN), Some(3.0), None]);
    let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
    let eval = |op| {
        let node = FilterNode::Compare { path: vec![0], op, val: ScalarVal::F64(f64::NAN) };
        let m = evaluate(&node, &batch).unwrap();
        (0..m.len()).map(|i| m.is_valid(i) && m.value(i)).collect::<Vec<bool>>()
    };
    // DuckDB total order: NaN is the largest; NaN == NaN; NULL never matches.
    assert_eq!(eval(CmpOp::Eq), vec![false, true, false, false]);   // = NaN → only NaN
    assert_eq!(eval(CmpOp::Ne), vec![true, false, true, false]);    // <> NaN → non-NaN, non-null
    assert_eq!(eval(CmpOp::Lt), vec![true, false, true, false]);    // < NaN → everything below NaN
    assert_eq!(eval(CmpOp::Le), vec![true, true, true, false]);     // <= NaN → all non-null
    assert_eq!(eval(CmpOp::Gt), vec![false, false, false, false]);  // > NaN → nothing
    assert_eq!(eval(CmpOp::Ge), vec![false, true, false, false]);   // >= NaN → only NaN
}
```

- [ ] **Step 2: Run them; expect FAIL**

Run:
```bash
cargo test -p duckdb --features "bundled appender-arrow" decimal128_filter_matches_appender nan_compare_total_order_direct -- --nocapture
```
Expected: both FAIL — decimal emits unsupported → query errors; NaN uses arrow IEEE cmp (e.g. `= NaN` yields all-false, so the Eq assertion fails).

- [ ] **Step 3: C++ — encode Decimal128 (`push_i128`) + emit_scalar arm**

In `arrow_zerocopy_shim.cpp`, add after `push_i64`:
```cpp
static void push_i128(std::vector<uint8_t> &buf, duckdb::hugeint_t h) {
    // Little-endian i128: low 64 bits then high 64 bits (two's complement).
    push_u64(buf, h.lower);
    push_i64(buf, h.upper);
}
```

Add the decimal include near the includes:
```cpp
#include "duckdb/common/types/decimal.hpp"
```

In `emit_scalar`, add a case before `default:`:
```cpp
        case duckdb::LogicalTypeId::DECIMAL: {
            // Only INT128-physical decimals are pushed by CanPushdown; the arrow column is Decimal128.
            uint8_t width = duckdb::DecimalType::GetWidth(v.type());
            uint8_t scale = duckdb::DecimalType::GetScale(v.type());
            if (width <= 18) { push_u8(buf, 255); break; } // not INT128-physical → unsupported (fail-loud)
            push_u8(buf, 9);
            push_i128(buf, v.GetValueUnsafe<duckdb::hugeint_t>());
            push_u8(buf, scale);
            break;
        }
```

- [ ] **Step 4: Rust — `ScalarVal::Decimal128` + decoder + `read_i128`**

In `filter.rs`, add `Decimal128Array` to the array imports. Extend `ScalarVal` with:
```rust
    Decimal128 { v: i128, scale: u8 },
```

Add `read_i128` to `Cursor`:
```rust
fn read_i128(&mut self) -> i128 {
    let mut bytes = [0u8; 16];
    for b in bytes.iter_mut() {
        *b = self.read_u8();
    }
    i128::from_le_bytes(bytes)
}
```

Extend `read_scalar` (add before `_ =>`):
```rust
9 => { let v = self.read_i128(); let scale = self.read_u8(); ScalarVal::Decimal128 { v, scale } }
```

- [ ] **Step 5: Rust — Decimal128 compare arm + NaN total-order handling**

In `compare`, add the decimal arm before the final `_ =>`:
```rust
        // ── Decimal128 (column precision/scale drive the scalar; scale must match) ──
        (DataType::Decimal128(p, s), ScalarVal::Decimal128 { v, scale }) => {
            if *scale != *s as u8 {
                return Err(FilterError::Unhandled(format!(
                    "decimal scale mismatch: constant scale {scale} vs column scale {s}"
                )));
            }
            let arr = Decimal128Array::from(vec![*v])
                .with_precision_and_scale(*p, *s)
                .map_err(|e| FilterError::Unhandled(e.to_string()))?;
            dispatch(op, col, &Scalar::new(arr))
        }
```

Replace the two float arms with NaN-aware versions:
```rust
        (DataType::Float64, ScalarVal::F64(v)) => {
            if v.is_nan() {
                return nan_compare(col, op);
            }
            dispatch(op, col, &Scalar::new(Float64Array::from(vec![*v])))
        }
        (DataType::Float32, ScalarVal::F64(v)) => {
            if v.is_nan() {
                return nan_compare(col, op);
            }
            dispatch(op, col, &Scalar::new(Float32Array::from(vec![*v as f32])))
        }
```

Add `nan_compare` above `compare` (uses self-inequality to find NaN positions; nulls never match):
```rust
/// Comparison of a float column against a NaN constant, using DuckDB's float total order
/// (NaN sorts highest; NaN == NaN). `col` is Float32 or Float64.
/// Result is non-null: positions that are SQL-null in `col` yield false.
fn nan_compare(col: &ArrayRef, op: &CmpOp) -> Result<BooleanArray, FilterError> {
    // is_nan: x != x is true exactly where x is NaN; null where x is null.
    let self_ne = cmp::neq(col, col).map_err(|e| FilterError::Unhandled(e.to_string()))?;
    let n = col.len();
    let not_null: Vec<bool> = (0..n).map(|i| col.is_valid(i)).collect();
    let is_nan: Vec<bool> = (0..n).map(|i| self_ne.is_valid(i) && self_ne.value(i)).collect();
    let out: Vec<bool> = (0..n)
        .map(|i| {
            if !not_null[i] {
                return false; // NULL never satisfies a comparison
            }
            let nan = is_nan[i];
            match op {
                CmpOp::Eq => nan,        // = NaN  → only NaN
                CmpOp::Ne => !nan,       // <> NaN → everything that is not NaN
                CmpOp::Lt => !nan,       // < NaN  → everything below NaN
                CmpOp::Le => true,       // <= NaN → all (NaN is the max)
                CmpOp::Gt => false,      // > NaN  → nothing exceeds NaN
                CmpOp::Ge => nan,        // >= NaN → only NaN
            }
        })
        .collect();
    Ok(BooleanArray::from(out))
}
```

- [ ] **Step 6: Run the new tests; expect PASS**

Run:
```bash
cargo test -p duckdb --features "bundled appender-arrow" decimal128_filter_matches_appender nan_compare_total_order_direct -- --nocapture
```
Expected: both PASS.

- [ ] **Step 7: Run the whole arrow_zerocopy suite**

Run:
```bash
cargo test -p duckdb --features "bundled appender-arrow" arrow_zerocopy -- --nocapture
```
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp crates/duckdb/src/arrow_zerocopy/filter.rs crates/duckdb/src/arrow_zerocopy/mod.rs
git commit -m "$(cat <<'EOF'
feat(arrow): push down Decimal128 scalars + NaN total-order compares

Decimal128 carried as i128 + scale (INT128-physical only; column p/s drive
the scalar, scale must match). Float NaN constants use DuckDB's total order
(NaN is the max; NaN == NaN) instead of arrow's IEEE default.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Tasks 5 & 6: DROPPED — `EXPRESSION_FILTER` walker is N/A on v1.5.4

> **⛔ DROPPED after execution-time verification (2026-06-26).** Probing 12 predicate shapes
> against a registered arrow view on DuckDB v1.5.4 showed the optimizer **never emits
> `EXPRESSION_FILTER` (=9)** for an arrow scan: it constant-folds expressions
> (`k+1=3` → `CONSTANT_COMPARISON k=2`), wraps OR/IN in `OPTIONAL_FILTER`, uses
> `CONJUNCTION_AND` / `STRUCT_EXTRACT`, and keeps genuinely-complex or cross-column predicates
> above the scan. Tasks 1–4 already handle every kind v1.5.4 pushes, so the `EXPRESSION_FILTER`
> walker would be dead, untestable code with zero coverage gain on v1.5.4. The existing
> `UNHANDLED → error_stream` already covers the (non-occurring) `EXPRESSION_FILTER` case safely.
> Decision (user-approved): skip Tasks 5–6 and proceed to Task 7. The walker is the path
> duckdb-python needs on DuckDB **main** (where structured kinds are LEGACY); revisit if/when the
> bundle is bumped to such a release. The original Task 5/6 text is retained below for the record.

#### (RETAINED FOR RECORD) Task 5: `EXPRESSION_FILTER` walker — comparison / constant / ref / conjunction

Add `emit_expression` for the `EXPRESSION_FILTER` kind: handle a constant comparison whose column side is a bare `BOUND_REF`, and `AND`/`OR` of such. `BoundOperator` (IS NULL / IN) and struct_extract chains come in Task 6. Any untranslatable sub-expression → the whole filter is UNHANDLED (fail-loud).

**Files:**
- Modify: `crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp`
- Test: `crates/duckdb/src/arrow_zerocopy/mod.rs`

**Interfaces:**
- Consumes: `push_column_path`, `emit_scalar` (Tasks 1–4); wire COMPARE/AND/OR nodes.
- Produces (C++): `bool emit_expression(const duckdb::Expression&, duckdb::idx_t root_col, std::vector<uint8_t>& out)` returning `true` iff fully translated (on `true`, `out` holds exactly one node); `bool resolve_column_path(const duckdb::Expression&, duckdb::idx_t root_col, std::vector<duckdb::idx_t>& path_out)` (BOUND_REF only in this task; struct_extract added in Task 6); `static uint8_t expr_type_to_op(int)`; `static int flip_comparison(int)`.

- [ ] **Step 1: Write a failing test that forces an EXPRESSION_FILTER**

DuckDB routes a non-structured predicate through `EXPRESSION_FILTER`. An OR of comparisons on the *same* column is a reliable trigger on v1.5.4 (it cannot be a single structured `CONSTANT_COMPARISON`). Add to `mod.rs` `mod test`:

```rust
#[test]
fn expression_filter_or_same_column_matches_appender() {
    let conn = Connection::open_in_memory().unwrap();
    let batch = sample(); // k Int64 [1,2,3,4], region Utf8 [us,eu,NULL,us]
    conn.execute_batch("CREATE TABLE t (k BIGINT, region VARCHAR)").unwrap();
    {
        let mut app = conn.appender("t").unwrap();
        app.append_record_batch(batch.clone()).unwrap();
    }
    let reg = conn.register_arrow("v", vec![batch]).unwrap();
    let q = |sql: String| conn.query_row::<i64, _, _>(&sql, [], |r| r.get(0)).unwrap();
    // Same-column OR (k=1 OR k=4): not a single structured comparison → arrives as EXPRESSION_FILTER.
    for where_clause in ["WHERE k = 1 OR k = 4", "WHERE (k = 1 OR k = 3) AND region = 'us'"] {
        assert_eq!(
            q(format!("SELECT count(*) FROM v {where_clause}")),
            q(format!("SELECT count(*) FROM t {where_clause}")),
            "EXPRESSION_FILTER predicate must match the Appender oracle: {where_clause}"
        );
    }
    drop(reg);
}
```

- [ ] **Step 2: Run it; expect FAIL**

Run:
```bash
cargo test -p duckdb --features "bundled appender-arrow" expression_filter_or_same_column_matches_appender -- --nocapture
```
Expected: FAIL — `EXPRESSION_FILTER` hits the `else` in `emit_filter` (only reached if it routes there) or, more precisely, `ShimProduce` calls `emit_filter` which emits UNHANDLED for the `EXPRESSION_FILTER` type → query errors. (If, unexpectedly, DuckDB does NOT produce an `EXPRESSION_FILTER` for these and the test already passes, record that finding in the task report; the walker is still added for the kinds that do route here. See the spec's EXPRESSION_FILTER-rarity risk.)

- [ ] **Step 3: Add expression headers + comparison/flip/op helpers**

In `arrow_zerocopy_shim.cpp`, add includes:
```cpp
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
```

Add helpers above `emit_filter`:
```cpp
// Map a comparison ExpressionType (25-30) to the wire op byte; 255 = not a comparison.
static uint8_t expr_type_to_op(int et) {
    switch (et) {
        case 25: return 0; // EQUAL
        case 26: return 1; // NOTEQUAL
        case 27: return 2; // LESSTHAN
        case 28: return 3; // GREATERTHAN
        case 29: return 4; // LESSTHANOREQUALTO
        case 30: return 5; // GREATERTHANOREQUALTO
        default: return 255;
    }
}

// Flip a comparison when the constant is on the left (a < X  ==>  X > a).
static int flip_comparison(int et) {
    switch (et) {
        case 27: return 28; // LT -> GT
        case 28: return 27; // GT -> LT
        case 29: return 30; // LE -> GE
        case 30: return 29; // GE -> LE
        default: return et; // EQ / NE unchanged
    }
}
```

- [ ] **Step 4: Add `resolve_column_path` (BOUND_REF only) and `emit_expression`**

Add above `emit_filter` (after the helpers):
```cpp
// Resolve a column-side expression to a path. This task: BOUND_REF only (the whole
// ExpressionFilter is rooted at `root_col`, so any BOUND_REF maps to that column).
// Struct_extract chains are added in Task 6. Returns false if not a column path.
static bool resolve_column_path(const duckdb::Expression &expr, duckdb::idx_t root_col,
                                std::vector<duckdb::idx_t> &path_out) {
    using duckdb::ExpressionClass;
    if (expr.GetExpressionClass() == ExpressionClass::BOUND_REF) {
        path_out.clear();
        path_out.push_back(root_col);
        return true;
    }
    return false;
}

// Port of duckdb-python's TransformExpression for v1.5.4's classic bound-expression model.
// On success, `out` holds exactly one wire node and the function returns true. On any
// untranslatable sub-expression it returns false (caller emits UNHANDLED — never drops a
// required filter, because DuckDB does not re-apply pushed arrow-scan filters).
static bool emit_expression(const duckdb::Expression &expr, duckdb::idx_t root_col,
                            std::vector<uint8_t> &out) {
    using duckdb::ExpressionClass;
    using duckdb::ExpressionType;
    auto cls = expr.GetExpressionClass();

    if (cls == ExpressionClass::BOUND_COMPARISON) {
        auto &cmp = expr.Cast<duckdb::BoundComparisonExpression>();
        int et = (int)cmp.type;
        const duckdb::Expression *col_side = nullptr;
        const duckdb::BoundConstantExpression *const_side = nullptr;
        if (cmp.right->GetExpressionType() == ExpressionType::VALUE_CONSTANT) {
            col_side = cmp.left.get();
            const_side = &cmp.right->Cast<duckdb::BoundConstantExpression>();
        } else if (cmp.left->GetExpressionType() == ExpressionType::VALUE_CONSTANT) {
            col_side = cmp.right.get();
            const_side = &cmp.left->Cast<duckdb::BoundConstantExpression>();
            et = flip_comparison(et);
        } else {
            return false; // column-to-column comparison: not pushable here
        }
        uint8_t op = expr_type_to_op(et);
        if (op == 255) {
            return false;
        }
        std::vector<duckdb::idx_t> path;
        if (!resolve_column_path(*col_side, root_col, path)) {
            return false;
        }
        push_u8(out, 0); // COMPARE
        push_column_path(out, path);
        push_u8(out, op);
        emit_scalar(const_side->value, out);
        return true;
    }

    if (cls == ExpressionClass::BOUND_CONJUNCTION) {
        auto &conj = expr.Cast<duckdb::BoundConjunctionExpression>();
        bool is_and = expr.GetExpressionType() == ExpressionType::CONJUNCTION_AND;
        std::vector<uint8_t> tmp;
        push_u8(tmp, is_and ? 3 : 4); // AND / OR
        push_u32(tmp, (uint32_t)conj.children.size());
        for (auto &child : conj.children) {
            if (!emit_expression(*child, root_col, tmp)) {
                return false; // required child untranslatable → whole filter fails loud
            }
        }
        out.insert(out.end(), tmp.begin(), tmp.end());
        return true;
    }

    return false; // BOUND_OPERATOR / struct_extract added in Task 6; everything else unsupported
}
```

- [ ] **Step 5: Dispatch `EXPRESSION_FILTER` in `ShimProduce`**

In `ShimProduce`, change the per-filter loop body to special-case `EXPRESSION_FILTER`:
```cpp
duckdb::idx_t projected_col = kv.first;
const duckdb::TableFilter &tf = *kv.second;
if (tf.filter_type == duckdb::TableFilterType::EXPRESSION_FILTER) {
    auto &ef = tf.Cast<duckdb::ExpressionFilter>();
    std::vector<uint8_t> tmp;
    if (emit_expression(*ef.expr, projected_col, tmp)) {
        filter_buf.insert(filter_buf.end(), tmp.begin(), tmp.end());
    } else {
        push_u8(filter_buf, 255); // UNHANDLED
        push_u32(filter_buf, (uint32_t)tf.filter_type);
    }
} else {
    emit_filter(tf, std::vector<duckdb::idx_t>{projected_col}, filter_buf);
}
```

- [ ] **Step 6: Run the expression-filter test; expect PASS**

Run:
```bash
cargo test -p duckdb --features "bundled appender-arrow" expression_filter_or_same_column_matches_appender -- --nocapture
```
Expected: PASS.

- [ ] **Step 7: Run the whole arrow_zerocopy suite**

Run:
```bash
cargo test -p duckdb --features "bundled appender-arrow" arrow_zerocopy -- --nocapture
```
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp crates/duckdb/src/arrow_zerocopy/mod.rs
git commit -m "$(cat <<'EOF'
feat(arrow): EXPRESSION_FILTER walker — comparison/ref/conjunction

Port TransformExpression onto v1.5.4's classic bound-expression model:
constant comparison (BOUND_REF column side, flip when constant is on the
left) and AND/OR conjunctions. Any untranslatable sub-expression makes the
whole filter UNHANDLED (fail-loud); DuckDB does not re-apply pushed filters.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `EXPRESSION_FILTER` walker — IS NULL / IS NOT NULL / IN + struct_extract chains

Extend `emit_expression` with `BoundOperatorExpression` (`OPERATOR_IS_NULL`, `OPERATOR_IS_NOT_NULL`, `COMPARE_IN`) and extend `resolve_column_path` to follow `struct_extract` / `struct_extract_at` function chains, so struct-field predicates that arrive inside an `EXPRESSION_FILTER` also push.

**Files:**
- Modify: `crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp`
- Test: `crates/duckdb/src/arrow_zerocopy/mod.rs`

**Interfaces:**
- Consumes: `emit_expression`/`resolve_column_path` (Task 5); `StructExtractBindData` (v1.5.4 `struct_utils.hpp`).
- Produces: IS NULL/IS NOT NULL/IN + struct_extract-chain coverage inside `EXPRESSION_FILTER`.

- [ ] **Step 1: Write a failing test (IN + struct-in-expression)**

Add to `mod.rs` `mod test`. Reuse the struct batch shape from Task 2's test via a local helper here (repeat it; tasks may be read out of order):

```rust
#[test]
fn expression_filter_in_and_struct_matches_appender() {
    use arrow::array::{ArrayRef, Int64Array, StructArray};
    use arrow::buffer::NullBuffer;
    use arrow::datatypes::{DataType, Field, Fields};
    let id = Arc::new(Int64Array::from(vec![1i64, 2, 3, 4])) as ArrayRef;
    let score = Arc::new(Int64Array::from(vec![10i64, 20, 30, 40])) as ArrayRef;
    let fields: Fields = vec![
        Field::new("id", DataType::Int64, false),
        Field::new("score", DataType::Int64, false),
    ]
    .into();
    let nulls = NullBuffer::from(vec![true, true, false, true]);
    let s = Arc::new(StructArray::new(fields.clone(), vec![id, score], Some(nulls))) as ArrayRef;
    let k = Arc::new(Int64Array::from(vec![1i64, 2, 3, 4])) as ArrayRef;
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("s", DataType::Struct(fields), true),
    ]));
    let batch = RecordBatch::try_new(schema, vec![k, s]).unwrap();

    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch("CREATE TABLE t (k BIGINT, s STRUCT(id BIGINT, score BIGINT))").unwrap();
    {
        let mut app = conn.appender("t").unwrap();
        app.append_record_batch(batch.clone()).unwrap();
    }
    let reg = conn.register_arrow("v", vec![batch]).unwrap();
    let q = |sql: String| conn.query_row::<i64, _, _>(&sql, [], |r| r.get(0)).unwrap();
    for where_clause in [
        "WHERE k IN (1, 3, 4)",
        "WHERE s.id = 1 OR s.id = 4",       // struct_extract inside an EXPRESSION_FILTER (same-col OR)
        "WHERE s.score IS NULL",
    ] {
        assert_eq!(
            q(format!("SELECT count(*) FROM v {where_clause}")),
            q(format!("SELECT count(*) FROM t {where_clause}")),
            "EXPRESSION_FILTER (IN/struct/null) must match the Appender oracle: {where_clause}"
        );
    }
    drop(reg);
}
```

- [ ] **Step 2: Run it; expect FAIL**

Run:
```bash
cargo test -p duckdb --features "bundled appender-arrow" expression_filter_in_and_struct_matches_appender -- --nocapture
```
Expected: FAIL — `BoundOperatorExpression` and struct_extract chains aren't handled yet → UNHANDLED → query errors. (If a given clause routes to a structured kind already handled, that sub-case will pass; the IN-as-expression and struct-as-expression sub-cases drive the failure.)

- [ ] **Step 3: Add operator + struct headers**

In `arrow_zerocopy_shim.cpp`, add includes:
```cpp
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/function/scalar/struct_utils.hpp"
```

- [ ] **Step 4: Extend `resolve_column_path` for struct_extract chains**

Replace `resolve_column_path` (from Task 5) with the chain-following version:
```cpp
static bool resolve_column_path(const duckdb::Expression &expr, duckdb::idx_t root_col,
                                std::vector<duckdb::idx_t> &path_out) {
    using duckdb::ExpressionClass;
    if (expr.GetExpressionClass() == ExpressionClass::BOUND_REF) {
        path_out.clear();
        path_out.push_back(root_col);
        return true;
    }
    if (expr.GetExpressionClass() == ExpressionClass::BOUND_FUNCTION) {
        auto &func = expr.Cast<duckdb::BoundFunctionExpression>();
        const auto &name = func.function.name;
        if ((name == "struct_extract" || name == "struct_extract_at") && func.bind_info &&
            !func.children.empty()) {
            // Resolve the inner column first, then append this child index (root -> leaf order).
            if (!resolve_column_path(*func.children[0], root_col, path_out)) {
                return false;
            }
            auto &bind = func.bind_info->Cast<duckdb::StructExtractBindData>();
            path_out.push_back(bind.index);
            return true;
        }
    }
    return false;
}
```

- [ ] **Step 5: Extend `emit_expression` with `BOUND_OPERATOR`**

In `emit_expression`, add a block before the final `return false;`:
```cpp
    if (cls == ExpressionClass::BOUND_OPERATOR) {
        auto &op_expr = expr.Cast<duckdb::BoundOperatorExpression>();
        auto et = expr.GetExpressionType();
        if (et == ExpressionType::OPERATOR_IS_NULL || et == ExpressionType::OPERATOR_IS_NOT_NULL) {
            if (op_expr.children.empty()) {
                return false;
            }
            std::vector<duckdb::idx_t> path;
            if (!resolve_column_path(*op_expr.children[0], root_col, path)) {
                return false;
            }
            push_u8(out, et == ExpressionType::OPERATOR_IS_NULL ? 1 : 2); // IS_NULL / IS_NOT_NULL
            push_column_path(out, path);
            return true;
        }
        if (et == ExpressionType::COMPARE_IN) {
            if (op_expr.children.empty()) {
                return false;
            }
            std::vector<duckdb::idx_t> path;
            if (!resolve_column_path(*op_expr.children[0], root_col, path)) {
                return false;
            }
            // children[1..] must all be constants.
            for (duckdb::idx_t i = 1; i < op_expr.children.size(); i++) {
                if (op_expr.children[i]->GetExpressionType() != ExpressionType::VALUE_CONSTANT) {
                    return false;
                }
            }
            push_u8(out, 5); // IN
            push_column_path(out, path);
            push_u32(out, (uint32_t)(op_expr.children.size() - 1));
            for (duckdb::idx_t i = 1; i < op_expr.children.size(); i++) {
                emit_scalar(op_expr.children[i]->Cast<duckdb::BoundConstantExpression>().value, out);
            }
            return true;
        }
    }
```

- [ ] **Step 6: Run the test; expect PASS**

Run:
```bash
cargo test -p duckdb --features "bundled appender-arrow" expression_filter_in_and_struct_matches_appender -- --nocapture
```
Expected: PASS.

- [ ] **Step 7: Run the whole arrow_zerocopy suite**

Run:
```bash
cargo test -p duckdb --features "bundled appender-arrow" arrow_zerocopy -- --nocapture
```
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp crates/duckdb/src/arrow_zerocopy/mod.rs
git commit -m "$(cat <<'EOF'
feat(arrow): EXPRESSION_FILTER walker — IS NULL/IN + struct_extract chains

Handle BoundOperatorExpression (IS NULL / IS NOT NULL / COMPARE_IN) and
follow struct_extract / struct_extract_at function chains via
StructExtractBindData, so struct-field predicates inside an
EXPRESSION_FILTER push down with full ancestor-null correctness.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Docs, benchmark, lint, full-suite green + CHANGELOG

Update the `register_arrow` doc comment to reflect the new coverage, refresh the wire-format scalar comment, run clippy and the full `duckdb` test suite (confirm the new coverage didn't regress non-arrow paths), and add a CHANGELOG entry.

**Files:**
- Modify: `crates/duckdb/src/arrow_zerocopy/mod.rs` (doc comment)
- Modify: `crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp` (scalar comment completeness)
- Modify: `crates/duckdb/CHANGELOG.md` (or the crate's actual changelog file)
- Possibly modify: `crates/duckdb/examples/arrow_zerocopy_bench.rs` (only if adding a typed/struct column to the workload)

**Interfaces:** none new.

- [ ] **Step 1: Update the `register_arrow` doc comment**

In `mod.rs`, in the `register_arrow` doc comment, replace the filter-pushdown sentence:
```
    /// **filter pushdown**: constant comparisons, IS [NOT] NULL, IN, AND/OR, and STRUCT_EXTRACT
    /// (struct-field) predicates are evaluated in Rust over scalars of all pushable types (ints,
    /// unsigned, float/double incl. NaN total-order, bool, Utf8, Blob, Date, Time, Timestamp,
    /// Decimal128). OPTIONAL/DYNAMIC/BLOOM filters are safely skipped. On DuckDB v1.5.4 the
    /// optimizer routes every pushable arrow-scan filter to one of these structured kinds (it
    /// constant-folds expressions and keeps genuinely-complex predicates above the scan), so the
    /// catch-all EXPRESSION_FILTER kind is not emitted here; should one ever appear it produces a
    /// clean error stream (fail-loud, never silently wrong).
```

- [ ] **Step 2: Complete the scalar wire-format comment in the shim**

In `arrow_zerocopy_shim.cpp`, update the "Scalar format" comment to list the added stypes (6 date32, 7 time, 8 timestamp, 9 decimal128, 10 blob) and the unit byte, matching the Global Constraints contract.

- [ ] **Step 3: Run clippy (arrow features) — expect clean**

Run:
```bash
cargo clippy -p duckdb --features "bundled appender-arrow" --all-targets -- -D warnings
```
Expected: no warnings. Fix any clippy findings in the changed files.

- [ ] **Step 4: Run the FULL duckdb test suite — expect green**

Run:
```bash
cargo test -p duckdb --features "bundled appender-arrow" -- --nocapture
```
Expected: the entire `duckdb` crate suite passes (arrow_zerocopy + all pre-existing tests). This confirms staying on v1.5.4 with these changes didn't regress non-arrow paths.

- [ ] **Step 5: Add a CHANGELOG entry**

Find the changelog file:
```bash
ls crates/duckdb/CHANGELOG.md 2>/dev/null || git -C /Volumes/Work/Code/duckdb-rs ls-files | grep -i changelog
```
Add an "Unreleased" entry under the duckdb crate changelog:
```markdown
- `Connection::register_arrow` filter pushdown now covers every filter kind DuckDB v1.5.4
  pushes to an arrow scan: STRUCT_EXTRACT (struct-field) predicates via column paths, plus
  scalar breadth (Date/Time/Timestamp/Decimal128/Blob + float NaN total-order) on top of the
  existing constant-comparison / IS [NOT] NULL / AND·OR / IN coverage. OPTIONAL/DYNAMIC/BLOOM
  are safely skipped; any unhandled required filter fails loud.
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
docs(arrow): document full filter coverage; changelog; lint + full suite green

Update register_arrow docs + wire-format scalar comment for the new
coverage. clippy clean and the full duckdb suite green on v1.5.4.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**1. Spec coverage**

- "STRUCT_EXTRACT (column paths)" → Task 1 (path plumbing + `resolve_path`) + Task 2 (structured kind). ✓
- "Scalar breadth: Date/Time/Timestamp/Decimal128/Blob + NaN" → Task 3 (Date/Time/Timestamp/Blob) + Task 4 (Decimal128 + NaN). ✓ (TIME nanosecond explicitly out of scope → fail-loud, noted in Task 3 and consistent with the spec's correctness-safe fallback.)
- "EXPRESSION_FILTER full expression-tree port (Approach A: dual front-ends, one wire format, one evaluator)" → Tasks 5–6; structured front-end (`emit_filter`) and expression front-end (`emit_expression`) both emit the same wire format decoded by one evaluator. ✓
- "Branch arrow-zerocopy-fullfilter off arrow-zerocopy; no main bump" → Task 1 Step 1 + Global Constraints. ✓
- Correctness policy (fail-loud; OPTIONAL/DYNAMIC/BLOOM skip) → Global Constraints + carried in Tasks 5–6. Deviation from duckdb-python's conjunct-dropping is documented with rationale (DuckDB does not re-apply pushed arrow-scan filters). ✓
- Testing (oracle + struct both paths + scalar types + arbitrary-expression predicate + multi-column regression + self-join + negative) → struct via Task 2 (structured) and Task 6 (expression); scalars Tasks 3–4; arbitrary-expression Task 5; the pre-existing `multi_column_projection_and_filter_matches_appender` and `self_join_is_replayable` stay green every task; negative/fail-loud is covered by the pre-existing `unhandled_filter_kind_errors_not_silently_wrong` and `evaluate_compare_unsupported_combo_errors`/`evaluate_compare_out_of_range_scalar_errors`. ✓
- Full `duckdb` suite green → Task 7 Step 4. ✓
- Docs/bench/lint/CHANGELOG → Task 7. ✓

No spec requirement is left without a task.

**2. Placeholder scan**

No "TBD/implement later". Two conditional verifications are explicit, not placeholders: the BLOB byte-accessor fallback in Task 3 Step 3 (pick the call that yields raw bytes) and the EXPRESSION_FILTER-rarity note in Task 5 Step 2 (record if a clause already passes). Both name the exact decision and the safe default (fail-loud).

**3. Type consistency**

- `FilterNode` column fields are `path: Vec<usize>` everywhere from Task 1 on; the existing tests are migrated in Task 1 Step 6; later tasks construct `FilterNode::Compare { path: vec![..], .. }`. ✓
- `resolve_path` returns `(ArrayRef, Option<BooleanArray>)` and is used identically in every evaluate arm. ✓
- C++ `emit_filter(filter, const std::vector<idx_t>& path, buf)` signature is set in Task 1 and used by Tasks 2/5. `emit_expression(expr, root_col, out) -> bool` and `resolve_column_path(expr, root_col, path_out) -> bool` are introduced in Task 5 and extended (not re-signatured) in Task 6. ✓
- Wire scalar tags (6 date32, 7 time, 8 timestamp, 9 decimal128, 10 blob) and unit byte are defined once in Global Constraints and used consistently by the C++ `emit_scalar` and the Rust `read_scalar`. ✓
- `ScalarVal` variant names (`Date32`, `Time{unit,v}`, `Timestamp{unit,v}`, `Decimal128{v,scale}`, `Blob`) match between the enum definition (Tasks 3–4) and the `compare` arms. ✓

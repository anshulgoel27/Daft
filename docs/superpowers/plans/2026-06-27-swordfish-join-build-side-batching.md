# swordfish: batch the hash-join build-side output assembly (OP-2) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the row-at-a-time build-side gather in the inner hash join (the ~71%-of-CPU `probe_inner` hotspot) by hoisting per-call lookups out of the shared `ArrowGrowable::extend` and restructuring `probe_inner` to pre-size + bulk-`take` like the probe side already does.

**Architecture:** Two independent changes. (1) `ArrowGrowable` precomputes each source's offset/value-buffers/validity once in `new()` (owned `Arc`-backed clones — no self-referential borrow) so `extend()` stops re-deriving them per call. (2) `probe_inner` pre-sizes its index `Vec`s to `input_table.len()` and, when the build side is a single `RecordBatch`, gathers it with one vectorized `RecordBatch::take` instead of the growable; a multi-table build falls back to a pre-sized `GrowableRecordBatch`.

**Tech Stack:** Rust, arrow-rs (`Buffer`, `NullBuffer`, `MutableBuffer`), Daft `daft-core` growable + `daft-local-execution` hash join.

## Global Constraints

- Branch: work on `duckdb-poc`. Never commit to `main`.
- Output must be **byte-for-byte identical** — these are pure perf changes (hoist + bulk gather), no semantic change.
- `ArrowGrowable` is used by every growable consumer (joins, concat, explode, …); preserve fixed-width / boolean / var-len + validity + offset semantics exactly.
- Inner join only (Component 1 touches `probe_inner`); do not change outer/left/semi/anti paths.
- Build/test: `cargo test -p daft-core -p daft-recordbatch -p daft-local-execution`. Existing growable + join tests must stay green.
- No `Co-Authored-By` trailer in commits.

---

## File Structure

- `src/daft-core/src/array/growable/arrow_growable.rs` — Task 1: per-source cache + `extend` hoist.
- `src/daft-local-execution/src/join/inner_join.rs` — Task 2: `probe_inner` pre-size + single-table bulk-take.

---

### Task 1: Hoist per-call lookups in `ArrowGrowable::extend`

Replace the `source_data: Vec<ArrayData>` field with a precomputed `sources: Vec<SourceCache>` (offset + value `Buffer`(s) + `NullBuffer`, all owned `Arc` clones), computed once in `new()`. `extend` then indexes the cache instead of calling `source.buffers()`/`offset()`/`nulls()` every call. Semantics unchanged.

**Files:**
- Modify: `src/daft-core/src/array/growable/arrow_growable.rs`

**Interfaces:**
- Consumes: arrow `Buffer`, `NullBuffer`, `ArrayData` (already imported / via `arrays[i].to_data()`).
- Produces: unchanged public API — `ArrowGrowable::new(name, dtype, arrays, use_validity, capacity)` and the `Growable` impl (`extend`/`add_nulls`/`build`/`len`). Behavior identical; only internals change.

- [ ] **Step 1: Write the failing growable tests (multi-source + nulls + sliced offset; var-len)**

Add to the `#[cfg(test)] mod tests` block in `arrow_growable.rs` (keep the existing `test_extend_from_nonzero_start`):

```rust
    #[test]
    fn extend_multi_source_with_nulls_and_offset() {
        use crate::array::ops::as_arrow::AsArrow;
        let field = Field::new("v", DataType::Int32);
        // src0 (no nulls), then slice it to offset=1 -> logical [20,30,40,50]
        let src0_full =
            Int32Array::from_iter(field.clone(), vec![Some(10), Some(20), Some(30), Some(40), Some(50)]);
        let src0 = src0_full.slice(1, 5); // offset=1, values [20,30,40,50]
        // src1 has a null
        let src1 = Int32Array::from_iter(field.clone(), vec![Some(1), None, Some(3)]);

        let mut g = ArrowGrowable::<Int32Type>::new(
            "v", &DataType::Int32, vec![&src0, &src1], false, 0,
        );
        // from src0 (offset=1): take logical idx 1..3 -> [30,40]
        g.extend(0, 1, 2);
        // from src1: take 0..3 -> [1, null, 3]
        g.extend(1, 0, 3);
        // from src0: take logical idx 3 -> [50]
        g.extend(0, 3, 1);

        let out = g.build().unwrap();
        let arr = out.i32().unwrap();
        let got: Vec<Option<i32>> = (0..arr.len()).map(|i| arr.get(i)).collect();
        assert_eq!(
            got,
            vec![Some(30), Some(40), Some(1), None, Some(3), Some(50)]
        );
    }

    #[test]
    fn extend_varlen_utf8_multi_source() {
        let field = Field::new("s", DataType::Utf8);
        let a = Utf8Array::from_iter("s", vec![Some("a"), Some("bb"), Some("ccc")].into_iter());
        let b = Utf8Array::from_iter("s", vec![Some("dddd"), Some("e")].into_iter());
        let mut g = ArrowGrowable::<Utf8Type>::new("s", &DataType::Utf8, vec![&a, &b], false, 0);
        g.extend(0, 1, 2); // ["bb","ccc"]
        g.extend(1, 0, 1); // ["dddd"]
        g.extend(0, 0, 1); // ["a"]
        let out = g.build().unwrap();
        let arr = out.utf8().unwrap();
        let got: Vec<Option<&str>> = (0..arr.len()).map(|i| arr.get(i)).collect();
        assert_eq!(got, vec![Some("bb"), Some("ccc"), Some("dddd"), Some("a")]);
    }
```

Note: the exact constructors (`Int32Array::from_iter`, `Utf8Array::from_iter`, `.slice`, `.i32()/.utf8()/.get()`) follow the existing test + `daft-core` array API; if a helper name differs, match the existing `test_extend_from_nonzero_start` style and the `daft_core::array::prelude` in scope. The assertions (values/nulls/order) are the contract — do not weaken them.

- [ ] **Step 2: Run the tests to verify they fail/compile-pass against current code**

Run:
```bash
cargo test -p daft-core arrow_growable -- --nocapture
```
Expected: the two new tests **pass against the current implementation too** (the hoist is behavior-preserving, so these are characterization tests). If they don't pass on current code, the test is wrong — fix the test before refactoring. (This is the safety net for Step 3.)

- [ ] **Step 3: Add the per-source cache + rewrite `new`/`extend`**

Add the cache type (module scope) and replace the `source_data` field + `new` + `extend` in `arrow_growable.rs`:

```rust
/// One source array's buffers/offset/validity, precomputed once so `extend` does no per-call
/// `ArrayData::buffers()/offset()/nulls()` lookups. All fields are owned Arc-backed clones,
/// so there is no borrow into the growable itself.
struct SourceCache {
    offset: usize,
    nulls: Option<arrow::buffer::NullBuffer>,
    buf0: arrow::buffer::Buffer,         // fixed-width/boolean: values; var-len: offsets
    buf1: Option<arrow::buffer::Buffer>, // var-len: values
}
```

In the struct, replace `source_data: Vec<ArrayData>,` with `sources: Vec<SourceCache>,` (keep `arrow_dtype`, `grower`, `validity`, `len`, `name`, `dtype`, `_phantom`).

Replace `new`'s body that built `source_data`/`arrow_dtype`/`needs_validity` with:

```rust
    pub fn new(
        name: &str,
        dtype: &DataType,
        arrays: Vec<&'a DataArray<T>>,
        use_validity: bool,
        capacity: usize,
    ) -> Self {
        let mut sources = Vec::with_capacity(arrays.len());
        let mut needs_validity = use_validity;
        let mut arrow_dtype: Option<arrow::datatypes::DataType> = None;
        for a in &arrays {
            let data = a.to_data();
            if arrow_dtype.is_none() {
                arrow_dtype = Some(data.data_type().clone());
            }
            let bufs = data.buffers();
            let nulls = data.nulls().cloned();
            if nulls.is_some() {
                needs_validity = true;
            }
            sources.push(SourceCache {
                offset: data.offset(),
                nulls,
                buf0: bufs[0].clone(),
                buf1: bufs.get(1).cloned(),
            });
        }
        let arrow_dtype = arrow_dtype
            .unwrap_or_else(|| dtype.to_arrow().unwrap_or(arrow::datatypes::DataType::Null));

        let grower = grower_from_dtype(dtype, capacity);
        let validity = if needs_validity {
            Some(NullBufferBuilder::new(capacity))
        } else {
            None
        };

        Self {
            name: name.to_string(),
            dtype: dtype.clone(),
            arrow_dtype,
            sources,
            grower,
            validity,
            len: 0,
            _phantom: PhantomData,
        }
    }
```

Replace `extend` to read from the cache:

```rust
    #[inline]
    fn extend(&mut self, index: usize, start: usize, len: usize) {
        let src = &self.sources[index];

        if let Some(ref mut validity) = self.validity {
            if let Some(nulls) = &src.nulls {
                validity.append_buffer(&nulls.slice(start, len));
            } else {
                validity.append_n_non_nulls(len);
            }
        }

        let offset = src.offset;
        match &mut self.grower {
            ValueGrower::FixedWidth { buffer, byte_width } => {
                let bw = *byte_width;
                let s = src.buf0.as_slice();
                let byte_start = (offset + start) * bw;
                buffer.extend_from_slice(&s[byte_start..byte_start + len * bw]);
            }
            ValueGrower::Boolean { builder } => {
                let s = src.buf0.as_slice();
                let base = offset + start;
                for i in 0..len {
                    let bit_idx = base + i;
                    let is_set = (s[bit_idx / 8] >> (bit_idx % 8)) & 1 == 1;
                    builder.append(is_set);
                }
            }
            ValueGrower::VarLen { offsets, values } => {
                let src_offsets: &[i64] = src.buf0.typed_data();
                let src_values = src.buf1.as_ref().expect("var-len source has values buffer").as_slice();
                let base = offset + start;
                for i in 0..len {
                    let idx = base + i;
                    let val_start = src_offsets[idx] as usize;
                    let val_end = src_offsets[idx + 1] as usize;
                    values.extend_from_slice(&src_values[val_start..val_end]);
                    offsets.push(offsets.last().unwrap() + (val_end - val_start) as i64);
                }
            }
        }

        self.len += len;
    }
```

`add_nulls` and `build` are unchanged (they don't touch `source_data`). Remove the now-unused `ArrayData` import only if nothing else uses it (it's used via `a.to_data()` return + `ArrayData::builder` in `build` — keep the import).

- [ ] **Step 4: Run the growable tests (incl. existing) — expect pass**

Run:
```bash
cargo test -p daft-core arrow_growable -- --nocapture
```
Expected: `test_extend_from_nonzero_start`, `extend_multi_source_with_nulls_and_offset`, `extend_varlen_utf8_multi_source` all PASS.

- [ ] **Step 5: Run the broader growable + recordbatch suites (no regressions)**

Run:
```bash
cargo test -p daft-core growable
cargo test -p daft-recordbatch
```
Expected: green (growable is used by `GrowableRecordBatch` + many ops).

- [ ] **Step 6: Commit**

```bash
git add src/daft-core/src/array/growable/arrow_growable.rs
git commit -m "perf(core): hoist per-call source lookups out of ArrowGrowable::extend

Precompute each source's offset/value-buffers/validity once in new() (owned
Arc clones) so extend() no longer re-derives buffers()/offset()/nulls() per
call. Behavior-preserving hoist on the growable hot path used by joins/concat/etc."
```

---

### Task 2: `probe_inner` — pre-size + single-table bulk `take`

Pre-reserve the index `Vec`s, and when the build side is one `RecordBatch`, gather it with a vectorized `take` (mirroring the probe side) instead of the row-at-a-time growable. Multi-table build uses a pre-sized growable.

**Files:**
- Modify: `src/daft-local-execution/src/join/inner_join.rs`

**Interfaces:**
- Consumes: `RecordBatch::take(&UInt64Array)` (exists), `GrowableRecordBatch::new(tables, use_validity, capacity)` (now realloc-friendly after Task 1, but Task 2 is independent of Task 1), `probe_state.probe_indices`, `MicroPartition::record_batches`.
- Produces: unchanged `probe_inner(input, probe_state, params) -> DaftResult<MicroPartition>`; identical output.

- [ ] **Step 1: Write the failing test (single- vs multi-table build give identical, correct results)**

Add a `#[cfg(test)] mod tests` block to `inner_join.rs`. This builds a minimal Int64-key inner join and runs `probe_inner` with the build side provided as one `RecordBatch` and as two, asserting both equal the expected 2-row inner join.

```rust
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use common_error::DaftResult;
    use daft_core::prelude::*;
    use daft_dsl::expr::bound_expr::BoundExpr;
    use daft_dsl::resolved_col;
    use daft_micropartition::MicroPartition;
    use daft_recordbatch::{ProbeState, RecordBatch};
    use daft_recordbatch::probeable::make_probeable_builder;
    use indexmap::IndexSet;
    use daft_core::join::JoinType;

    use super::*;
    use crate::join::hash_join::HashJoinParams;

    fn i64_batch(name1: &str, v1: Vec<i64>, name2: &str, v2: Vec<i64>) -> RecordBatch {
        let a = Int64Array::from_vec(name1, v1).into_series();
        let b = Int64Array::from_vec(name2, v2).into_series();
        RecordBatch::from_nonempty_columns(vec![a, b]).unwrap()
    }

    fn run(build_tables: Vec<RecordBatch>) -> DaftResult<usize> {
        // build side: columns (k, v); probe side: columns (k, w); inner join on k.
        let key_schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64)]));
        let left_schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64),
            Field::new("v", DataType::Int64),
        ]));
        let right_schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64),
            Field::new("w", DataType::Int64),
        ]));
        let output_schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64),
            Field::new("v", DataType::Int64),
            Field::new("w", DataType::Int64),
        ]));
        let build_on = vec![BoundExpr::try_new(resolved_col("k"), &left_schema)?];
        let probe_on = vec![BoundExpr::try_new(resolved_col("k"), &right_schema)?];

        // Probe table is built over the build-side KEY columns.
        let mut builder = make_probeable_builder(key_schema.clone(), None, true)?;
        for t in &build_tables {
            let keys = t.eval_expression_list(&build_on)?;
            builder.add_table(&keys)?;
        }
        let probeable = builder.build();
        let probe_state = ProbeState::new(probeable, build_tables);

        let mut common = IndexSet::new();
        common.insert("k".to_string());
        let params = HashJoinParams {
            key_schema,
            build_on,
            probe_on,
            nulls_equal_aware: None,
            track_indices: true,
            join_type: JoinType::Inner,
            build_on_left: true,
            left_schema,
            right_schema: right_schema.clone(),
            common_join_cols: common,
            output_schema,
            spill_config: None,
        };

        let probe_input = MicroPartition::new_loaded(
            right_schema,
            Arc::new(vec![i64_batch("k", vec![2, 3, 4], "w", vec![200, 300, 400])]),
            None,
        );
        let out = probe_inner(&probe_input, &probe_state, &params)?;
        Ok(out.len())
    }

    #[test]
    fn inner_join_single_and_multi_table_build_match() {
        // build (k,v): k in {1,2,3}; probe k in {2,3,4} -> inner matches k=2,3 -> 2 rows.
        let single = vec![i64_batch("k", vec![1, 2, 3], "v", vec![10, 20, 30])];
        let multi = vec![
            i64_batch("k", vec![1, 2], "v", vec![10, 20]),
            i64_batch("k", vec![3], "v", vec![30]),
        ];
        assert_eq!(run(single).unwrap(), 2, "single-table build");
        assert_eq!(run(multi).unwrap(), 2, "multi-table build");
    }
}
```

Note: verify `make_probeable_builder` is exported at `daft_recordbatch::probeable::make_probeable_builder` (it is `pub` in `probeable/mod.rs`); if the re-export path differs, adjust the `use`. `HashJoinParams` fields are listed verbatim above from the current struct. The existing Daft join suite is the broader correctness backstop — do not weaken the asserted row counts.

- [ ] **Step 2: Run the test to verify it passes on current code (characterization)**

Run:
```bash
cargo test -p daft-local-execution inner_join_single_and_multi_table_build_match -- --nocapture
```
Expected: PASS against the current implementation (both build shapes already produce 2 rows). This locks correctness before the refactor. If it fails to compile, fix the construction (signatures) before Step 3.

- [ ] **Step 3: Restructure `probe_inner`**

Replace the per-`input_table` closure body (the `DEFAULT_GROWABLE_SIZE` setup through `build_side_table`) with the pre-sized + single-table-take version. Full replacement of lines from `let mut build_side_growable = …` through the `let build_side_table = …;`:

```rust
            let est = input_table.len();
            let join_keys = input_table.eval_expression_list(&params.probe_on)?;
            let idx_iter = probe_state.probe_indices(join_keys)?;

            let mut probe_side_idxs: Vec<u64> = Vec::with_capacity(est);
            let build_side_table = if build_side_tables.len() == 1 {
                // Single build table: collect matched build row indices and gather with one
                // vectorized take (mirrors the probe side) — no row-at-a-time growable.
                let single = build_side_tables[0];
                let mut build_row_idxs: Vec<u64> = Vec::with_capacity(est);
                for (probe_row_idx, inner_iter) in idx_iter.enumerate() {
                    if let Some(inner_iter) = inner_iter {
                        for (build_rb_idx, build_row_idx) in inner_iter {
                            debug_assert_eq!(build_rb_idx, 0, "single build table => rb idx 0");
                            build_row_idxs.push(build_row_idx);
                            probe_side_idxs.push(probe_row_idx as u64);
                        }
                    }
                }
                single.take(&UInt64Array::from_vec("", build_row_idxs))?
            } else {
                // Multiple build tables: pre-sized growable gather.
                let mut build_side_growable =
                    GrowableRecordBatch::new(&build_side_tables, false, est)?;
                for (probe_row_idx, inner_iter) in idx_iter.enumerate() {
                    if let Some(inner_iter) = inner_iter {
                        for (build_rb_idx, build_row_idx) in inner_iter {
                            build_side_growable.extend(
                                build_rb_idx as usize,
                                build_row_idx as usize,
                                1,
                            );
                            probe_side_idxs.push(probe_row_idx as u64);
                        }
                    }
                }
                build_side_growable.build()?
            };

            let probe_side_table = {
                let indices_arr = UInt64Array::from_vec("", probe_side_idxs);
                input_table.take(&indices_arr)?
            };
```

Remove the now-unused `const DEFAULT_GROWABLE_SIZE: usize = 20;` and the old `let join_keys = …; let idx_iter = …;` lines that were below (they are moved up into the new block — ensure they appear exactly once). The downstream code (`build_on_left` swap, `common_join_keys`, column unions, `final_table`) is unchanged.

- [ ] **Step 4: Run the probe_inner test — expect pass**

Run:
```bash
cargo test -p daft-local-execution inner_join_single_and_multi_table_build_match -- --nocapture
```
Expected: PASS (both single- and multi-table build → 2 rows).

- [ ] **Step 5: Run the full join + execution suite (no regressions)**

Run:
```bash
cargo test -p daft-local-execution
```
Expected: green — all existing inner/outer/semi/anti join tests pass (only `probe_inner`'s internals changed; output identical).

- [ ] **Step 6: Commit**

```bash
git add src/daft-local-execution/src/join/inner_join.rs
git commit -m "perf(swordfish): bulk-gather inner-join build side (pre-size + single-table take)

probe_inner pre-reserves its index Vecs to input_table.len() and, when the
build side is a single RecordBatch, gathers it with one vectorized take instead
of the row-at-a-time growable.extend(..,1) (was capacity 20 -> realloc churn).
Multi-table build uses a pre-sized growable. Output identical."
```

---

## Verification (post-implementation; not a code task)

```bash
make build-release
# CPU-time profile (samply + threadCPUDelta re-weight) on the 10M join, and the bench:
.venv/bin/python -m tests.duckdb_poc.bench_duckdb_vs_swordfish
```
Expect `probe_inner` self-CPU to drop sharply and swordfish 10M (~184 ms) to fall toward / below DuckDB's ~107 ms. Report before/after.

---

## Self-Review

**1. Spec coverage:** Component 2 (ArrowGrowable hoist) → Task 1 (cache struct + `new`/`extend` rewrite, Step 3) with multi-source/nulls/offset/var-len test (Step 1). Component 1 (`probe_inner` pre-size + single-table take + multi-table pre-sized growable fallback) → Task 2 Step 3 with single/multi-table-build test (Step 1). Identical-output requirement → characterization tests (run on current code first, Task 1 Step 2 / Task 2 Step 2). Test commands + existing-suite-green → Steps 5. Verification bench → Verification section. All covered.

**2. Placeholder scan:** No TBD/"handle errors". Every code step has complete code; run steps have exact commands + expected results. The two "verify the constructor signature" notes name the exact symbol to check and the backstop — engineering guidance, not placeholders.

**3. Type consistency:** `SourceCache { offset: usize, nulls: Option<NullBuffer>, buf0: Buffer, buf1: Option<Buffer> }` defined once and used in `new`/`extend`. `ArrowGrowable` field rename `source_data` → `sources` applied in struct + `new` + `extend` (build/add_nulls untouched). `probe_inner` keeps its signature; `est: usize`, `build_row_idxs: Vec<u64>`, `probe_side_idxs: Vec<u64>` consistent; `RecordBatch::take(&UInt64Array)` and `GrowableRecordBatch::new(&[&RecordBatch], bool, usize)` match existing signatures. `HashJoinParams` fields in the test match the struct definition.

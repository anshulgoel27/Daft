# swordfish: batch the hash-join build-side output assembly (OP-2) — Design

**Date:** 2026-06-27
**Status:** Design approved; ready for implementation plan
**Repo:** Daft (`/Volumes/Work/Code/Daft`); files `src/daft-local-execution/src/join/inner_join.rs`, `src/daft-core/src/array/growable/arrow_growable.rs`
**Branch:** `duckdb-poc` (where this perf work is happening)

## Goal

Eliminate the row-at-a-time build-side gather in the inner hash join, which a CPU-time profile (samply, re-weighted by `threadCPUDelta`) showed as the dominant swordfish cost on the join→filter→groupby-agg workload: `inner_join::probe_inner` is ~71% of CPU, and the build-side `GrowableRecordBatch::extend(.., 1)` loop (per-element `Map`/`ArrowGrowable` work + reallocation) is ~33% self / ~45% across the output-assembly path. The probe side already uses a single bulk `take`; only the build side is row-at-a-time.

## Background (current hot path)

`src/daft-local-execution/src/join/inner_join.rs::probe_inner`:
```rust
let mut build_side_growable = GrowableRecordBatch::new(&build_side_tables, false, DEFAULT_GROWABLE_SIZE)?; // = 20
let mut probe_side_idxs = Vec::new();                                  // unreserved
for (probe_row_idx, inner_iter) in idx_iter.enumerate() {
    if let Some(inner_iter) = inner_iter {
        for (build_rb_idx, build_row_idx) in inner_iter {
            build_side_growable.extend(build_rb_idx as usize, build_row_idx as usize, 1);  // row-at-a-time
            probe_side_idxs.push(probe_row_idx as u64);
        }
    }
}
let build_side_table = build_side_growable.build()?;
let probe_side_table = input_table.take(&UInt64Array::from_vec("", probe_side_idxs))?;     // bulk (good)
```
Two cost sources: (1) capacity starts at 20 → the per-column `MutableBuffer`s reallocate ~19× growing to millions of rows; (2) `ArrowGrowable::extend` re-derives `source.buffers()[0].as_slice()`, `source.offset()`, `source.nulls()` on every call (×rows×columns).

The build sink concats pending build partitions into one MicroPartition (`join/build.rs:147`), so `build_side_tables` is **often a single `RecordBatch`** (e.g. a 1M-row `from_pydict` build), but may be several.

## Component 1 — `probe_inner` restructure (`inner_join.rs`)

- Compute `est = input_table.len()` — a tight capacity upper bound for unique-build-key joins (output ≈ matching probe rows ≤ input rows); under fan-out the Vecs grow with a few reallocs, still far better than 20.
- `probe_side_idxs = Vec::with_capacity(est)`.
- **Single build table** (`build_side_tables.len() == 1`, common): also collect `build_row_idxs: Vec<u64>` with `Vec::with_capacity(est)`; in the loop push `build_row_idx as u64` (assert/rely on `build_rb_idx == 0`) and `probe_row_idx as u64`; then
  `build_side_table = build_side_tables[0].take(&UInt64Array::from_vec("", build_row_idxs))?` — a vectorized gather, no growable.
- **Multi build table** (fallback): `GrowableRecordBatch::new(&build_side_tables, false, est)` (pre-sized) + `extend(build_rb_idx, build_row_idx, 1)` as today; `build_side_table = growable.build()?`.
- `probe_side_table = input_table.take(&UInt64Array::from_vec("", probe_side_idxs))?` (unchanged; now reserved).
- Everything downstream (schema assembly, union of join keys + non-join columns) is unchanged.

## Component 2 — `ArrowGrowable::extend` hoist (`arrow_growable.rs`)

In `ArrowGrowable::new`, precompute once per source array a cached record holding: `offset: usize`, the value buffer(s) needed by the grower variant (the fixed-width/boolean value `Buffer`; for var-len the offsets `Buffer` + values `Buffer`), and `nulls: Option<NullBuffer>` — all owned `Arc`-backed clones (no self-referential borrow into `source_data`). Store these in a `Vec` parallel to `source_data`, indexed by `index`.

`extend(index, start, len)` then indexes the cache (`let src = &self.sources[index];`) and does the same slice copy / validity append / var-len offset work as today, but without re-calling `source.buffers()`, `source.offset()`, `source.nulls()` per call. Semantics are byte-for-byte identical — this is a pure hoist of loop-invariant lookups. The `FixedWidth` / `Boolean` / `VarLen` branches, validity handling, and `add_nulls`/`build` are otherwise unchanged.

Benefit applies to every growable consumer (joins, concat, explode, take-like ops), not just this join.

## Correctness

The output rows and their order are identical:
- Single-table `take(build_row_idxs)` gathers exactly the rows the growable would append, in the same probe-driven order; `probe_side_idxs` is unchanged.
- The growable cache is a pure hoist (same buffers, same offset/validity math).
Capacity changes never affect contents, only allocation.

## Testing

- **`ArrowGrowable`**: a test extending from multiple source arrays with non-zero offsets, with nulls/validity present, and a var-len (`Utf8`/`LargeUtf8`) column — asserting value + validity equal the expected gathered result. Keep `test_extend_from_nonzero_start`. (Guards the Component-2 hoist.)
- **`probe_inner`**: a join test asserting correct results when the build side is a **single** RecordBatch and when it is **multiple** RecordBatches (exercising both Component-1 paths) — e.g. build a join where the build input arrives as one vs several partitions/batches; assert identical, correct output. Plus the existing inner-join tests.
- `cargo test -p daft-core -p daft-recordbatch -p daft-local-execution` green.

## Verification (post-implementation; not a code task)

Re-run the CPU-time profile (samply + `threadCPUDelta` re-weighting) and the swordfish 10M bench (`bench_duckdb_vs_swordfish`); expect `probe_inner` self-CPU to drop sharply and the swordfish 10M time (~184 ms) to fall toward or below DuckDB's ~107 ms.

## Out of scope

- Outer / left / semi / anti join probe functions (separate code paths) — inner join only.
- ProbeState single-table caching (Option C) and morsel sizing (OP-3) — separate, deprioritized by the profile (build is not CPU-bound).
- Run-coalescing of contiguous matches (the unique-build-key workload has 1 match/row, so no contiguous runs to coalesce).

## Risks

- `ArrowGrowable` is widely used; the per-source cache must preserve validity, offset, and var-len semantics exactly (the added growable test + the broad suite guard this).
- `est = input_table.len()` under-counts on heavy build-key fan-out → a few Vec/buffer reallocs (acceptable; still O(1) amortized and far better than capacity 20).
- The single-table fast path assumes `build_rb_idx == 0` when `build_side_tables.len() == 1` — guaranteed by construction; a `debug_assert` documents it.

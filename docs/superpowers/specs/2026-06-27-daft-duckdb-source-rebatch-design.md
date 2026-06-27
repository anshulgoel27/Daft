# daft-duckdb: auto-batch sources for parallel arrow scans — Design

**Date:** 2026-06-27
**Status:** Design approved; ready for implementation plan
**Repo:** Daft (`/Volumes/Work/Code/Daft`), crate `src/daft-duckdb`, file `src/daft-duckdb/src/executor.rs`
**Builds on:** the zero-copy `register_arrow` integration (merged on `duckdb-poc`).

## Goal

Split each registered source's Arrow data into multiple record batches before `register_arrow`, so DuckDB's native `arrow_scan` — which parallelizes at **record-batch granularity** — can use all cores. Today the executor hands `register_arrow` whatever batch count `source_to_arrow` produces; a single-partition Daft source becomes **one** Arrow batch, which DuckDB scans single-threaded.

## Motivation (measured)

DuckDB's `arrow_scan` parallelism is bounded by the number of record batches. Measured on the standard benchmark query (10M-row probe ⨝ 1M build, filter, groupby-agg), 10 cores, through the real Rust `register_arrow` path:

| source batches | swordfish | duck cold | duck / native |
|---|---|---|---|
| 1 per source | 180 ms | 244 ms | 1.35× slower |
| 8 per source | 181 ms | 112 ms | ~1.6× faster |
| 16–32 per source | 186 ms | ~107 ms | ~1.75× faster |

Registration stays ~0.2 ms (zero-copy) at every batch count, cold ≈ warm, and `duck(split=16) == swordfish` (correctness). The win saturates around ~8 batches/source (≈ core count). So splitting a large single-batch source into N batches flips the 10M case from a loss to a ~1.7× win with no downside.

## Decision: target rows-per-batch

Split any batch whose row count exceeds a fixed constant `TARGET_ROWS_PER_BATCH = 122_880` into `ceil(rows / target)` equal slices; batches at or under the target pass through unchanged.

- `122_880` = DuckDB's row-group/morsel size (60 × 2048), so each Arrow batch ≈ one scan morsel.
- **Data-adaptive:** 10K → 1 batch (no overhead), 1M → ~8, 10M → ~82. More batches than cores is fine — DuckDB load-balances them.
- **No core detection** (a fixed target gives ≥ cores batches for large data on any machine) and **no config knob** (YAGNI; a `DuckDbConfig` field can be added later if a workload needs it).

Rejected alternatives: core-count-based splitting (needs a min-batch-size floor to avoid splitting tiny sources into many tiny batches); a configurable knob (adds config surface for a value a good default handles).

## Architecture / data flow

```
inputs[source_id] (Vec<MicroPartition>) ──source_to_arrow──► Vec<RecordBatch>
                                          ──rebatch(target)──► Vec<RecordBatch>  (large batches sliced)
                                          ──register_arrow──►  zero-copy view (DuckDB scans batches in parallel)
```

Only `register_sources` changes; everything downstream (`register_arrow`, `execute`, result path) is unchanged.

## Component (single file: `src/daft-duckdb/src/executor.rs`)

A pure helper:
```rust
/// Split any batch larger than `target_rows` into ~equal zero-copy slices so DuckDB's
/// per-batch-parallel arrow scan can use all cores. Batches with <= target_rows pass through
/// unchanged. Slicing uses RecordBatch::slice (an offset view) — no row copy, so the zero-copy
/// property is preserved.
fn rebatch(batches: Vec<arrow_array::RecordBatch>, target_rows: usize) -> Vec<arrow_array::RecordBatch>
```
- For each input batch: if `num_rows() <= target_rows` (or `target_rows == 0`), keep as-is; else emit `ceil(num_rows / target_rows)` slices via `batch.slice(offset, len)`, the last slice taking the remainder.
- `register_sources` inserts one call: `let arrow_batches = rebatch(source_to_arrow(mps)?, TARGET_ROWS_PER_BATCH);` before `register_arrow`. The empty-source guard (no batches → `DaftError::ValueError`) is unchanged and still applies (rebatch of an empty Vec is empty).

`TARGET_ROWS_PER_BATCH` is a module constant.

## Correctness

- `RecordBatch::slice` is a zero-copy offset view; concatenating the slices reproduces the original batch exactly. Registration stays O(1).
- Filter/projection pushdown over offset-sliced batches is handled by arrow-compute and DuckDB's arrow scan — the result row-set is identical regardless of how the source is batched (verified end-to-end: `duck(split=16) == swordfish`).

## Testing

- **`rebatch` unit tests** (pure function):
  - a batch of M rows with `target = t` (`t < M`) → `ceil(M/t)` batches; summed rows == M; concatenating the output equals the input (value-level), proving the slices are correct and lossless.
  - a batch with `rows <= target` → returned unchanged (same count, same data).
  - `target_rows == 0` → no split (guard against divide-by-zero / pathological).
  - empty input Vec → empty output.
- **Executor integration test:** a single ~300K-row Int64 source (> target → 3 batches) with a filter (e.g. `value > X`), run through `DuckDbExecutor::run`, asserting the correct row count — proves sliced multi-batch sources push down correctly through the real `register_arrow` path. Fast (<1s).
- **Existing executor tests stay green** (small data → no split → behavior unchanged).
- `cargo test -p daft-duckdb` green.

## Verification (post-implementation, not a code change)

Re-run `python -m tests.duckdb_poc.bench_duckdb_vs_swordfish` (after `make build-release`) and confirm the 10M `duck` numbers improve automatically through the normal path now that sources are auto-batched. (The bench's single-partition inputs will be split by `rebatch` to ~82 batches.)

## Out of scope

- Making the target configurable or core-count-based.
- Any change outside `executor.rs`.
- Daft-side partitioning of sources (orthogonal; `rebatch` handles the single-batch case regardless).

## Risks

- **Over-splitting tiny data** — avoided: batches ≤ target are untouched, so small sources stay one batch.
- **Offset-slice handling** in `register_arrow` / filter eval — already exercised (the end-to-end split bench passed); the integration test guards it.
- **Target tuning** — `122_880` is a principled default (DuckDB morsel size); if a workload shows it's wrong, it becomes a `DuckDbConfig` field (future).

# pbldg-full-pipeline slowness: root cause and fix

**Job:** `pbldg-full-pipeline-ba7188e846cf` (Ray job, `daft-cluster` in `ray-system`, k8s cluster `geo-addressing`)
**Stage:** `bronze.preprocess`, entity `building`, source `pbldg`, vintage `202608`
**Date investigated:** 2026-09-09
**Status:** Diagnosed. Fix designed, not yet implemented.

## TL;DR

The job is not stuck or hung — it makes steady forward progress. It's slow because
[`MultiPartDataProcessor.run()`](../../../../li-dis-data-geo-fabric/src/geo_fabric/bronze/common/multi_part_data_processor.py)
processes US counties **strictly one at a time** in a plain Python `for` loop, and each
county's Daft query is tiny (8 partitions, sub-second to a few seconds). The 20-node /
200-CPU Ray cluster sits almost completely idle throughout the run because no single
county's query ever has enough work to spread across it. Wall-clock is bounded by
`(number of parts) × (per-part latency)` — about 3,200 counties × ~5.2s ≈ 4.5 hours —
regardless of cluster size.

**Recommended fix:** batch multiple counties into fewer, bigger writes, so each Daft
query actually has enough scan/join work to use the cluster. Writes stay fully
serialized (no concurrent Delta commits), so there's no new correctness risk.

## Evidence

- Job status: `RUNNING`, started `2026-09-09 09:12:41 UTC`. Not `FAILED`, not `PENDING`.
- Driver (`python query.py ...`) and the `RemoteFlotillaRunner` actor log files
  (`/tmp/ray/session_*/logs/worker-*-4f080000-296049.err` on `daft-cluster-head-4sbcq`)
  had `mtime` within ~1 second of "now" every time they were checked — the job is
  actively writing output, not hung.
- Log lines show steady progress through US counties in FIPS/alphabetical order
  (`[building] Processing part: ak.02013` → ... → `tx.48233`), 2,725 of an estimated
  ~3,200 total parts done (~85%) after 3h55m — consistent with the observed pace.
- `ray status`, sampled 4 times over ~25 seconds while the job was actively running:

  ```
  Total Usage:
   0.0/200.0 CPU              (one sample briefly showed 2.0/200.0)
   0.0/20.0 daft_data_workers  (every sample)
   0B/4.72TiB memory
  ```

  The cluster's dedicated `daft_data_workers` resource (20 slots) was never used in any
  sample. This is the direct signature of "no per-part query is ever big enough to need
  more than one worker."
- Root cause in code:
  [`multi_part_data_processor.py:129`](../../../../li-dis-data-geo-fabric/src/geo_fabric/bronze/common/multi_part_data_processor.py#L129):

  ```python
  for part in parts:
      print(f"[{self.label}] Processing part: {part.name}")
      df = self.process_fn(part)
      collected.append(df)
      if self.incremental_write:
          write_mode = self.cfg.write_mode if is_first_write else "append"
          write_table_with_directives(df, self.target_table, ..., write_mode=write_mode)
          is_first_write = False
  ```

  One part is fully read, joined, and written before the next one starts. There is no
  fan-out across parts anywhere in the call chain — `_preprocess_part` in
  `bronze/pbldg/preprocessors/building.py` is a pure per-county function with no shared
  state, but nothing ever calls it for more than one county concurrently.
- Secondary (minor) finding: `collected.append(df)` runs unconditionally even when
  `incremental_write=True`, so every part's lazy `DataFrame` is retained for the whole
  run, and `align_frames(collected)` is still called on all ~3,200 of them at the end —
  pure waste in incremental mode. Driver RSS had grown to ~3.4GB after 2,725 parts
  (not dangerous yet, but avoidable).

## Why "just run counties concurrently" doesn't work

The obvious alternative — fire off many counties' read+join+write pipelines at once
(threads or `ray.remote`), each ending in its own `write_deltalake(mode="append")` to
the same target table — was considered and **rejected**: it's unsafe.

Checked `DataFrame.write_deltalake()` in `daft/dataframe/dataframe.py`:
- The default `mode="append"` path reads the table version once, writes files, then
  does a single one-shot `create_write_transaction(...)` commit — no retry, no
  conflict handling.
- If a concurrent writer commits first, the losing writer's commit raises
  `CommitFailedError` and the whole `write_deltalake()` call fails outright.
- Daft's only retry path (`checkpoint=IdempotentCommit`, up to 2 retries) is
  explicitly documented as *not* designed for concurrent-writer safety — it's
  defensive against transient failures, not real contention.
- The underlying `deltalake` (delta-rs) binding exposes no retry/backoff parameter
  either; conflict handling would have to be implemented by the caller.

Verdict: **unsafe** for 10-20 processes concurrently appending to the same Delta table.
Any fix must keep writes serialized.

## Recommended fix

Change `MultiPartDataProcessor.run()`'s `incremental_write=True` path (shared by
pbldg's `building` preprocessor today, and available to `ttmnr` preprocessors) to
process parts in **batches** instead of one at a time:

- Add a `batch_size: int = 25` constructor parameter.
- Group `parts` into chunks of `batch_size`. For each chunk: call `process_fn` for
  every part (cheap — builds a lazy `DataFrame`, no execution yet), `align_frames()`
  them into one combined `DataFrame`, then issue **one** `write_table_with_directives`
  call for the whole chunk (first chunk uses `cfg.write_mode`, later chunks use
  `"append"` — same semantics as today, just per-chunk instead of per-part).
- Leave the `incremental_write=False` path (`ttmnr` address pipeline, `folder_merge.py`)
  untouched — it already does a single write at the end.
- Drop the unconditional `collected.append(df)` / final `align_frames(collected)` in
  the incremental path — no caller uses `.run()`'s return value (verified: every call
  site discards it), and dropping it removes the unbounded memory retention noted
  above.

**Why this works:** each chunk's Daft query now scans+joins `batch_size` counties at
once, giving the query planner enough partitions to actually spread across the 20-node
cluster, instead of ~3,200 queries of 8 partitions each that never gave the cluster
anything to parallelize. Number of Delta commits drops from ~3,200 to ~3,200/25 ≈ 128.
Writes remain fully serialized — zero concurrent-write risk. Peak memory stays bounded
to one batch's worth of data at a time (the same rationale that motivated
`incremental_write=True` in the first place, just at batch instead of per-part
granularity).

**Scope:** fix the shared `MultiPartDataProcessor` class rather than special-casing
`pbldg`, since the bottleneck lives in the shared orchestrator and every current/future
caller benefits from one change.

**Not in scope / explicitly rejected:** true concurrent per-part writes relying on
Delta's optimistic concurrency — confirmed unsafe above.

## Notes

- This fix does **not** affect the currently-running job (`pbldg-full-pipeline-ba7188e846cf`)
  — it's an in-flight Ray job running already-submitted code. It will finish on its
  current (slow) trajectory; the fix applies to future runs after the code change ships.
- Estimated completion for the current run at the time of investigation: ~35-40 more
  minutes from 2026-09-09 13:07 UTC.

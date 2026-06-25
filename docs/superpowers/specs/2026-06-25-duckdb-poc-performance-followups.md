# DuckDB Local-Engine POC — Performance Follow-ups

**Date:** 2026-06-25
**Status:** Backlog (POC complete; these are next-step optimizations, not yet planned)
**Related:** [design](2026-06-25-duckdb-local-engine-poc-design.md) · [plan](../plans/2026-06-25-duckdb-local-engine-poc.md)

## Context

The POC measured DuckDB executing a hash-join+filter/agg `LocalPhysicalPlan` vs the native swordfish engine, single-node at the plan seam:

| size | duckdb / swordfish |
|---|---|
| 10K | 1.48 (swordfish faster) |
| 1M | 0.70 (DuckDB ~30% faster) |
| 10M | 0.66 (DuckDB ~34% faster) |

**The measured ratio is conservative** — it includes per-task overheads (an input copy, connection setup, SQL re-bind) that a real backend would shed. The optimizations below would (a) make the win larger and (b) lower the ~100K crossover where swordfish currently wins, widening the range where DuckDB pays off. They also gate whether a per-task swap is safe at distributed scale.

## Prioritized optimizations

### 1. Zero-copy Arrow ingestion (highest value)
**Now:** `register_arrow_table` in `src/daft-duckdb/src/executor.rs` does `CREATE TABLE` + `Appender::append_record_batch` — a full copy of every input row into DuckDB's native storage, inside the timed region. duckdb-rs 1.10501 exposes no Arrow-view API, so this was the fallback.
**Change:** scan the Arrow buffers in place via DuckDB's `arrow_scan` / replacement scan over the C Data Interface (call the C API `duckdb_arrow_scan` via FFI, or move to a duckdb-rs that exposes Arrow registration).
**Impact:** removes the input copy entirely → DuckDB time drops (ratios improve past 0.70/0.66) and the small-size loss (10K = 1.48) shrinks, since a large fraction of that is the copy + DDL fixed cost. Flagged by the final whole-branch review.
**Effort:** real engineering (FFI to `duckdb_arrow_scan`, or a duckdb-rs bump). Highest payoff.

### 2. Connection reuse + DuckDB thread/memory control (distributed-critical)
**Now:** `Connection::open_in_memory()` is created fresh per `run()`; no `threads`/`memory_limit` pragmas are set, so DuckDB defaults to **all cores per query**.
**Change:**
- Pool one connection per worker thread (thread-local); register/unregister inputs per task instead of re-opening.
- Set `SET threads=<worker slot>` and `SET memory_limit=<worker allocation>` per connection.
**Impact:** connection reuse removes repeated init cost across many partitions (invisible single-query, real at scale). **Thread control is essential before any per-task swap:** N concurrent Daft tasks each running DuckDB with all-cores-each is massive core oversubscription that would make a real deployment *slower*. This won't show in the single-node single-query benchmark, so it must be addressed deliberately, not measured into existence.
**Effort:** thread/memory pragmas are cheap (a couple of `SET`s); pooling is medium.

### 3. Skip re-translation / re-bind per partition
**Now:** the distributed planner sends the same `LocalPhysicalPlan` *shape* to every partition, but each task re-runs `plan_to_sql` + SQL parse/bind.
**Change:** cache the translated SQL by plan fingerprint and use a **prepared statement**; or switch to DuckDB's relational API (Approach B) to skip SQL parsing entirely.
**Impact:** amortizes per-task fixed cost; helps most at small partition sizes where fixed costs dominate (further lowering the crossover).
**Effort:** medium.

### 4. Benchmark-measurement cleanups (trust the number, not engine speed)
**Now:** `tests/duckdb_poc/_harness.py` `run_native`/`run_duckdb` both apply `sort_by(all columns)` (needed for the correctness assert, not for timing). It cancels in the ratio but is a meaningful fraction of total time at 10K, adding noise.
**Change:**
- Add bench-local execute+materialize variants *without* the sort.
- Report **two DuckDB numbers**: cold (incl. registration — the honest per-task swap cost) and warm (register once, run K times — pure compute). These answer two different decisions: "is DuckDB's engine faster" vs "is the swap cost worth it."
**Effort:** cheap.

## Architectural note (fuller-backend scope, not incremental)
The biggest lever beyond the above is pushing **larger plan subtrees** into a single DuckDB query per worker, so per-task fixed costs amortize over more compute and intermediate Daft↔Arrow↔DuckDB conversions are avoided. This belongs to a "fuller backend" effort (with the cost-model engine-selection threshold around ~100K rows the POC indicated), not to these incremental tweaks.

## Suggested sequencing
1. **Cheap wins first** to sharpen the measurement: #2 thread cap + #4 warm/cold split. (Low effort; clarifies what #1 will buy.)
2. **#1 zero-copy ingestion** — the headline optimization; brainstorm → plan it.
3. **#2 pooling + #3 SQL/prepared-statement cache** — scale-facing, do alongside any move toward an actual per-task scheduler swap.

## Phase 1 results (DONE — #4 + #2)

Delivered: `DuckDbConfig { threads, memory_limit }` + `DuckDbExecutor::run_with_config` (the #2 per-task thread/memory cap, applied as connection PRAGMAs); `DuckDbExecutor::bench_runs` (register once, time K executes → `(registration_s, [run_s])`); pyo3 bindings for both; benchmark rewritten to drop the correctness sort and report the warm/cold/registration split. 25 crate tests pass; correctness gate still green.

Benchmark (native runner, 16 GiB box, ITERS=5, no sort, DuckDB default threads):

| size | sword ms | duck cold ms | duck warm ms | reg ms | warm/native | cold/native |
|---|---|---|---|---|---|---|
| 10K | 16.4 | 5.8 | 2.4 | 3.4 | 0.15 | 0.35 |
| 1M | 158.3 | 94.1 | 23.8 | 70.3 | 0.15 | 0.59 |
| 10M | 1128.7 | 780.9 | 127.2 | 653.7 | 0.11 | 0.69 |

**Findings that set up Phase 2:**
- **DuckDB's pure compute (warm) is ~7–9× faster than swordfish** (warm/native 0.11–0.15) across sizes.
- **The Arrow→DuckDB registration copy dominates the cold cost** — at 10M, `reg`=654 ms vs `warm`=127 ms, i.e. **~84% of DuckDB's per-task time is the `CREATE TABLE` + `Appender` copy**, not query execution.
- So **#1 (zero-copy ingestion) is quantified as the headline win**: removing the copy should pull cold toward warm (≈0.11–0.15× native instead of 0.35–0.69×). The thread/memory cap (#2) is a production-concurrency knob, intentionally left at DuckDB defaults in this single-query bench.

## Phase 2 finding (#1 BLOCKED by duckdb-rs 1.10501)

Attempted Phase 2a (bulk-materialize via the `arrow()` table function). **It does not work here, and true zero-copy is not available in this dependency version:**

- duckdb-rs's `ArrowVTab::func` writes a *whole* RecordBatch into a *single* DuckDB `DataChunk` (`vtab/arrow.rs:108`, no chunking) and a DataChunk caps at `STANDARD_VECTOR_SIZE` = 2048 rows. So `SELECT * FROM arrow(?, ?)` **panics (`data.len() <= self.capacity()`, non-unwinding abort)** on any batch > 2048 rows — fine for the tiny correctness test, fatal on real data. Reverted.
- More fundamentally, the ArrowVTab is **not zero-copy**: `record_batch_to_duckdb_data_chunk` → `FlatVector::copy` is a `memcpy` into DuckDB's vector memory. The `Appender` path does the *same* copy but **chunks correctly** (`appender/arrow.rs`: slices into `vector_size` chunks). So in duckdb-rs 1.10501, **both ingestion paths copy Arrow into DuckDB's format** — there is no in-place scan. The Appender is already the better of the two (correct chunking, columnar) and is what the POC uses.
- Therefore the ~84% registration cost is the **inherent Arrow→DuckDB materialization copy**, not a fixable "switch to a zero-copy scan." #1 as scoped is blocked without one of: a newer DuckDB/duckdb-rs exposing a true zero-copy arrow scan (uncertain it exists — DuckDB generally copies Arrow into its native vectors), or the C `duckdb_arrow_scan` API (not surfaced by these Rust bindings). A custom chunking VTab would fix the 2048 crash but **still copies**, so it buys no perf over the Appender — not worth building.

**Decision:** keep the Appender (optimal available); mark #1 BLOCKED/deferred; the achievable remaining wins are Phase 3 (#2 connection pooling + #3 SQL/prepared-statement cache), which cut per-task *fixed* overhead rather than the copy.

## Phase 3 results (DONE — #2 connection reuse + #3a translation cache)

Delivered: `DuckDbSession` (`executor.rs`) — one connection (config applied once) reused across runs, with the translated SQL cached per plan identity; each run drops the prior partition's `daft_src_*` tables and registers the new ones. Unit test (`session_reuses_connection_across_runs`) + a Rust benchmark example (`examples/session_bench.rs`). 26 crate tests pass.

Benchmark (debug build, 500 runs of a small 1000-row partition):

| path | total | per run |
|---|---|---|
| fresh connection per run (`DuckDbExecutor::run`) | 9277 ms | 18.55 ms |
| reused `DuckDbSession` | 690 ms | 1.38 ms |
| **speedup** | | **13.4×** |

**Finding:** the per-task fixed overhead (DuckDB `open_in_memory` + plan translation) is **~17 ms/run**, dominated by connection open. Reusing the connection (a per-worker session) eliminates it — a **13.4× win for small/many partitions** (the fixed-overhead-dominated regime). For large partitions this barely matters (the inherent ingestion copy from #1 dominates), so #2 and #1 are complementary: pooling fixes the small-partition tax, while the large-partition copy stays blocked.

**#3b not done (intentional):** prepared-statement reuse across partitions doesn't fit the table-based registration model — each partition re-creates its `daft_src_*` tables, so the data isn't a bindable parameter and a cached prepared statement would re-resolve the catalog anyway. The achievable part of #3 — caching the *translation* (`plan_to_sql`) per plan — is in `DuckDbSession`.

## Overall conclusion

Phase 1 quantified it and the phases bear it out: **DuckDB's compute is 7–9× faster than swordfish**, but the per-task swap has two costs — the **Arrow→DuckDB ingestion copy** (dominates large partitions; **blocked** by duckdb-rs having no in-place arrow scan) and **connection/translation fixed overhead** (dominates small partitions; **fixed** by `DuckDbSession`, 13.4×). A production per-task swap would use a per-worker `DuckDbSession` with `threads`/`memory_limit` caps; it wins big on small partitions and on compute-heavy queries, and is a wash on large-partition ingestion until a zero-copy arrow scan becomes available.

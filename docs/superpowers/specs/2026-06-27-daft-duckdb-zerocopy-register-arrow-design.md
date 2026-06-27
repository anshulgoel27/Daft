# Daft DuckDB Zero-Copy Execution via `register_arrow` — Design

**Date:** 2026-06-27
**Status:** Design approved; ready for implementation plan
**Repo:** Daft (`/Volumes/Work/Code/Daft`), crate `src/daft-duckdb`
**Depends on:** duckdb-rs branch `arrow-zerocopy` (`anshulgoel27/duckdb-rs`, on arrow-rs 59) which adds `Connection::register_arrow(name, Vec<RecordBatch>) -> ArrowView<'conn>` — zero-copy Arrow registration over the native `arrow_scan` table function with projection + filter pushdown.

## Goal

Replace `daft-duckdb`'s Appender-based source registration — which copies every row into DuckDB native storage — with `register_arrow`, so DuckDB scans Daft's Arrow buffers **in place** with **filter + projection pushdown**. Because the branch is on arrow-rs **59** (matching Daft's workspace), this also unifies the Arrow version across the dependency graph and lets us **delete** the v58↔v59 `FFI_ArrowArrayStream` bridges and the `CREATE TABLE` DDL. Net effect: zero-copy ingest + pushdown, and a substantial code reduction.

## Background (current state)

`src/daft-duckdb/src/executor.rs` today:
- `plan_to_sql(plan)` translates a Daft `LocalPhysicalPlan` to SQL (`plan_sql.rs`, unchanged by this work).
- `register_sources` → `register_arrow_table`: builds a `CREATE TABLE` DDL from the arrow schema, then loads rows with `Appender::append_record_batch`. **This copies all rows into DuckDB storage** (the code comments call the copy "inherent").
- `duckdb = "1.1"` resolves to a duckdb-rs on arrow-rs **58**, while Daft's workspace is arrow-rs **59**. So ingest and result paths bridge between the two via the Arrow C Stream Interface (`arrow_array_to_duck_rb`, `v59_reader_to_duck_batches`, `duck_rb_to_arrow_array`).
- `execute` runs the SQL and collects results with `query_arrow`, bridging v58→v59, then `arrow_to_daft_batches`.
- `DuckDbSession` reuses one connection across partitions, dropping/recreating tables between runs.

Daft↔arrow-rs-59 conversion already exists (`arrow_bridge.rs`: `source_to_arrow`, `daft_batches_to_arrow`, `arrow_to_daft_batches`, via Daft's `RecordBatch` `TryFrom` impls).

## Decision: Approach A (full replace)

Chosen over (B) keeping the Appender path as a fallback and (C) zero-copy ingest with filter pushdown disabled. Rationale: the v1.5.4 filter coverage is comprehensive for every kind DuckDB actually pushes to an arrow scan (constant comparison, IS [NOT] NULL, IN, conjunction, struct-extract; optional/dynamic/bloom skipped; `EXPRESSION_FILTER` not emitted — all verified during the duckdb-rs work). A delivers the full benefit (zero-copy + pushdown) and is mostly deletion. The accepted tradeoff: an unhandled pushed filter/type fails loud (error stream) rather than silently copying — if a real query ever trips it, that's a shim bug to fix, not a silent wrong result.

## Architecture / data flow

```
Daft MicroPartitions ──source_to_arrow──► arrow_array v59 RecordBatch(es)        (existing; unchanged)
        │
        └─ conn.register_arrow("daft_src_<id>", batches) ──► zero-copy DuckDB view
                                                              (returns ArrowView, held for the query)
   plan_to_sql(plan) ──► SQL ── execute ──► DuckDB pushes WHERE/projection INTO the arrow scan
                                            (Rust evaluator filters the Arrow batches in place)
                          query_arrow() ──► arrow_array v59 RecordBatch(es) ──arrow_to_daft_batches──► Daft RecordBatches
```

With the dependency on the arrow-59 branch, the graph resolves a **single** `arrow`/`arrow-array` 59 shared by Daft and duckdb, so `duckdb::arrow::*` **is** `arrow_array::*` — no version bridging anywhere.

## Components

### 1. `src/daft-duckdb/Cargo.toml`
- Replace `duckdb = {version = "1.1", features = ["bundled", "appender-arrow"]}` with:
  ```toml
  duckdb = { git = "https://github.com/anshulgoel27/duckdb-rs", branch = "arrow-zerocopy", features = ["bundled"] }
  ```
- Drop the `appender-arrow` feature **iff** the Appender is fully removed (Approach A removes it). If anything still needs it, keep it; the implementer confirms by compiling.

### 2. `src/daft-duckdb/src/executor.rs`
- `register_sources`: for each binding, call `conn.register_arrow(&format!("daft_src_{id}"), batches)` and **collect the returned `ArrowView`s into a `Vec` that outlives `execute()`**. `register_arrow` takes ownership of the `Vec<RecordBatch>` (moves it into the factory the `ArrowView` holds), so holding the views alone keeps the referenced Arrow buffers alive for the query — the executor does not separately retain the batches.
- `execute`: `query_arrow([])` yields `arrow_array` v59 batches directly → `arrow_to_daft_batches` (no `duck_rb_to_arrow_array`).
- **Delete** (now dead under unified arrow + Approach A): `register_arrow_table`, `build_create_table_ddl`, `arrow_dtype_to_duckdb_type`, `arrow_array_to_duck_rb`, `v59_reader_to_duck_batches`, `duck_rb_to_arrow_array`, and the `ffi_bridge_propagates_reader_error_cleanly` test.
- `DuckDbExecutor::run` / `run_with_config` / `bench_runs`: keep public signatures; bodies register (holding views), execute, then drop views. `bench_runs` registers once (holding views for all repeats) and times each execute.
- `DuckDbSession`: keep one connection; between runs, **drop the prior run's held `ArrowView`s** (Drop auto-unregisters the view) and register the new sources — replacing the current `DROP TABLE IF EXISTS` loop. Store the live views on the session for the duration of a `run`.

### 3. `src/daft-duckdb/src/arrow_bridge.rs`
Unchanged: `source_to_arrow`, `daft_batches_to_arrow`, `arrow_to_daft_batches` (Daft↔arrow_array) are still required. Its round-trip test stays.

### 4. `src/daft-duckdb/src/python.rs`, `plan_sql.rs`, `expr_sql.rs`
Unchanged. The Python binding calls the same executor API; SQL/expr translation is independent of the registration mechanism.

## Correctness

- **View / buffer lifetime:** `ArrowView<'conn>` borrows the connection and owns the factory holding the Arrow batches (referenced in place by the scan). The views must live until `execute()` finishes; holding them keeps the buffers alive. After they drop, the view auto-unregisters (`DROP VIEW IF EXISTS`) so a late scan errors cleanly rather than dereferencing freed memory.
- **Arrow unification (must verify, gates the deletions):** the build must resolve exactly one `arrow-array` 59 shared by Daft and duckdb. The implementer confirms with `cargo tree -i arrow-array` (one version) before deleting the bridges. If unification fails, the bridges are still required and the design must be revisited.
- **Fail-loud (accepted):** an unhandled pushed filter or unsupported column type → duckdb-rs error stream → `DaftError::External`. Never a silently wrong row set. Comprehensive for what v1.5.4 emits; any real trip is a duckdb-rs shim fix.

## Error handling

Registration/query failures surface as `DaftError::External` (as today). The empty-source guard (a source with no record batches) keeps an explicit `DaftError::ValueError`.

## Testing

- Existing executor tests stay green and now exercise the zero-copy path:
  - `filter_over_scan_runs_in_duckdb` — filter is pushed into the zero-copy scan; result still `[150, 250]`.
  - `run_with_config_applies_thread_and_memory_caps` — caps applied, same result.
  - `session_reuses_connection_across_runs` — now via view drop + re-register across two runs.
  - `bench_runs_registers_once_and_times_each_run`.
- `arrow_bridge` round-trip test stays.
- **Remove** `ffi_bridge_propagates_reader_error_cleanly` (the bridge is deleted).
- **Add** one executor test that a multi-type / struct-field predicate round-trips correctly through the zero-copy path (proves pushdown coverage end-to-end from a Daft plan): e.g. a source with an int + a struct column, a plan filtering on the struct field, asserting the row count matches the expected.
- `cargo build -p daft-duckdb` and `cargo test -p daft-duckdb` green (the `bundled` build compiles the DuckDB amalgamation from the branch — slow first build). Confirm a single arrow version with `cargo tree -i arrow-array`.

## Out of scope

- Whether `source_to_arrow` (Daft `RecordBatch` → arrow_array) is itself zero-copy (a Daft-internal arrow2↔arrow-rs concern). This change removes the **DuckDB-storage** copy regardless.
- Any change to `plan_to_sql` / SQL translation, the Python API, or the benchmark harness behavior.
- Merging/releasing the duckdb-rs branch upstream (tracked separately; the dep points at the fork branch until then).

## Risks

- **Cargo arrow unification** doesn't hold → bridges still needed (mitigated: verify before deleting).
- **First build cost** — the `bundled` feature compiles the DuckDB v1.5.4 amalgamation from the branch (minutes); expected, not a hang.
- **Fail-loud regression** for an exotic pushed filter/type not covered by the shim → query errors where the Appender path would have copied-and-filtered. Accepted per Approach A; comprehensive coverage makes it unlikely; fix in duckdb-rs if hit.
- **Branch drift** — the git-branch dep follows `arrow-zerocopy`; if that branch is force-updated, a rebuild picks up changes. (Pin to a rev later if reproducibility demands it.)

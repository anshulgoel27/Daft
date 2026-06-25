# DuckDB Local-Engine POC — Design

**Status:** Approved design, pending implementation plan
**Date:** 2026-06-25
**Author:** Anshul Goel (with Claude)

## Goal

Validate, with a faithful prototype, whether **DuckDB executing a task's `LocalPhysicalPlan`** can beat Daft's native "swordfish" engine on a representative analytical query — producing a trustworthy, apples-to-apples wall-clock ratio with verified-identical results, enough to decide go/no-go on a fuller DuckDB local backend.

This is a **measurement spike**, not a product feature. Its deliverable is a number (the speedup ratio) plus a correctness proof, not a shippable alternative engine.

## Motivation

Daft's distributed planner breaks a query into `SwordfishTask`s, each carrying a `LocalPhysicalPlanRef` + input partitions, executed on each Ray worker by the native engine (`NativeExecutor` in `daft-local-execution`). The hypothesis: DuckDB's vectorized engine may execute the per-partition relational work faster than swordfish. Because the win has to overcome plan-translation and Arrow-registration cost — and swordfish is already columnar/vectorized — the premise must be measured before any backend is built.

## Decisions (locked during brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Primary goal | **Performance** | Beat swordfish on per-partition compute. |
| Coverage | **Prototype one query shape** | Validate the premise cheaply before a backend. |
| Query shape | **Hash join + filter/agg** | Where DuckDB's reputation is strongest → a fair fight. |
| Integration point | **Real Rust `LocalPhysicalPlan` seam** | Measures the actual swap, not just "DuckDB on this data." |
| Translation mechanism | **Plan → DuckDB SQL string (Approach A)** | Least code for a fixed shape; lets DuckDB's optimizer plan it (the thing we're measuring); trivially debuggable. |

The spatial/nested-loop join is **excluded** as a test shape: its local plan is the R-tree `NestedLoopJoin`, for which DuckDB's core engine has no equivalent — an unfair, unrepresentative comparison.

## Architecture

A new **feature-gated crate `daft-duckdb`** isolates the DuckDB dependency from default builds. It consumes the same plan + inputs the native engine does and is exercised by a benchmark harness that runs **both** engines and compares.

```
SwordfishTask's LocalPhysicalPlan  ─┬─▶  NativeExecutor (swordfish)  ─▶ Arrow result   [baseline]
   (+ input partitions as Arrow)    └─▶  DuckDbExecutor (new)        ─▶ Arrow result   [candidate]
                                              ↑ benchmark harness runs BOTH, compares + times
```

**Boundary decision:** the POC runs **single-node, directly at the plan seam** — it is *not* wired into the distributed scheduler. The harness constructs the plan + inputs and calls both executors itself. This is the cleanest measurement, and because each Ray worker executes this exact `LocalPhysicalPlan`, a single-node result transfers directly to "swap per task" later.

### Seam reference (existing code)

- `NativeExecutor::run(local_physical_plan, exec_cfg, subscribers, additional_context, inputs: HashMap<SourceId, Input>, input_id, maintain_order)` — `src/daft-local-execution/src/run.rs`.
- Relevant `LocalPhysicalPlan` variants — `src/daft-local-plan/src/plan.rs`: `InMemoryScan`, `PhysicalScan`, `Filter`, `HashJoin`, `HashAggregate`, `UnGroupedAggregate`.
- Inputs arrive as `MicroPartition`s exposing `record_batches() -> &[RecordBatch]` — `src/daft-micropartition/src/micropartition.rs`.

## Components

1. **`DuckDbExecutor`** — entry point, mirrors the relevant slice of the `NativeExecutor::run` seam, simplified for the POC:
   `run(plan: &LocalPhysicalPlanRef, inputs: HashMap<SourceId, Vec<RecordBatch>>) -> DaftResult<Vec<RecordBatch>>`.
   Runs the whole local plan as **one** DuckDB query and returns the materialized Arrow result. No streaming/subscribers/`maintain_order` — those are swordfish-internal concerns the POC does not need.

2. **`plan_to_sql`** — walks the `LocalPhysicalPlan` DAG → `(sql_string, input_table_bindings)`. Handles the five node types below. Any other node → explicit `DaftError`.

3. **`expr_to_sql`** — Daft `ExprRef` → DuckDB SQL fragment for a fixed subset: column refs, literals, comparison/logical/arithmetic ops, equi-join key equality, and the agg functions SUM/COUNT/MIN/MAX/AVG. Unsupported expr → explicit `DaftError`.

4. **`arrow_bridge`** — registers each input partition as a DuckDB Arrow view via the **Arrow C Data Interface** (version-agnostic FFI, decoupling Daft's arrow-rs from duckdb-rs's), and collects the result back into Daft `RecordBatch`es.

5. **`bench` harness** — builds the real planned `LocalPhysicalPlan` for the query (via Daft's planner, not hand-rolled), runs it through both executors on identical in-memory Arrow, checks equivalence, reports wall-clock.

## Data flow (in the harness)

1. **Preload inputs as Arrow.** Load both tables into memory and build the query on *in-memory* sources via Daft's builder (`from_arrow → join → filter → groupby/agg`), then optimize to a `LocalPhysicalPlanRef` whose leaves are `InMemoryScan`s. In-memory sources make the comparison **compute-only, I/O excluded**, and ensure both engines see the identical Arrow buffers.
2. **Baseline:** `NativeExecutor::run(plan, inputs)` → `R_native`. Timed.
3. **Candidate:** `DuckDbExecutor::run(plan, inputs)`, timed end-to-end:
   - register each `InMemoryScan`'s batches as a DuckDB Arrow view `daft_src_<source_id>` (C Data Interface),
   - `plan_to_sql(plan)` → SQL over those view names,
   - `con.query_arrow(sql)` → Arrow → `R_duck`.
4. **Compare** `R_native` vs `R_duck`; record both timings.

The DuckDB timing **deliberately includes** Arrow registration + SQL bind + execution — that is the real per-task swap cost, so the number is honest.

## Plan → SQL translation scope

Each node emits `SELECT <explicit output columns from the node's schema> FROM (<child sql>) …` — explicit projection at every level (no `SELECT *`), so column names stay deterministic and match Daft's expected output schema even through the join.

| Local plan node | SQL |
|---|---|
| `InMemoryScan(src)` | view name `daft_src_<src>` |
| `Filter(pred, child)` | `SELECT … FROM (child) WHERE expr_to_sql(pred)` |
| `HashJoin(l, r, l_on, r_on, type)` | `SELECT … FROM (l) JOIN (r) ON l_on = r_on` — **inner only** |
| `HashAggregate(keys, aggs, child)` | `SELECT keys, agg_fns FROM (child) GROUP BY keys` |
| `UnGroupedAggregate(aggs, child)` | same, no `GROUP BY` |

Anything outside this subset (other node types, window functions, non-inner joins, unsupported exprs) → **explicit error, no fallback**.

## Correctness validation

The gate that makes the perf number trustworthy:

- Cast `R_duck` to the plan's expected output schema first (DuckDB returns its own Arrow types).
- Compare as a **row multiset**: sort both results by all output columns, assert row-for-row equality. Both engines' join/group-by output order is nondeterministic, so sorting is required.
- Exact equality for integer/key columns; a small epsilon for float aggregates (SUM/AVG) if needed.
- Test data explicitly includes **NULLs** in the join key and the filter column so null semantics are covered.

## Benchmark

- Synthetic data: two tables with a controllable-cardinality join key, a numeric measure, and a filter column. A small correctness size (~10K) and two perf sizes (~1M and ~10M build rows), configurable.
- Method: one warm-up, then median wall-clock over N iterations (~5) per engine; print `(engine, size, median ms)` and the **DuckDB/native ratio**. Simple timed loop — no criterion dependency. A recorded measurement, **not** a CI pass/fail gate.

## Error handling

- Unsupported node/expr → `DaftError` naming the offender.
- DuckDB bind/exec errors propagate as `DaftError`.
- Schema mismatch surfaces via the correctness check.
- No silent fallback anywhere — a measurement tool must fail loudly.

## Non-goals (explicitly deferred — each a follow-up only if the POC wins)

- Streaming / morsel execution (the POC materializes).
- Distributed scheduler wiring (per-task swap).
- Spilling / memory accounting.
- General operator & expression coverage; non-inner joins.
- Substrait / DuckDB relational API translation paths (Approaches B/C).
- Any engine-selection cost model or config toggle.

## Tests shipped

- One correctness test: DuckDB == swordfish on sorted rows, including NULLs.
- `expr_to_sql` unit tests for the supported subset.
- The benchmark (prints a measurement, does not assert).

## Success criterion

A trustworthy, apples-to-apples wall-clock ratio of DuckDB vs swordfish on the hash-join+filter/agg local plan, with verified-identical results — enough to decide go/no-go on a fuller backend.

## Open implementation details to pin (during planning, not blockers)

- The exact `duckdb-rs` API for registering an Arrow stream / collecting Arrow results (the **capability** is known; the precise call/feature flags need confirming against the pinned crate version).
- `daft-duckdb` crate wiring: feature flag name, workspace `Cargo.toml` entry, and how the harness/test target pulls it in without affecting default builds.
- How the harness obtains a `LocalPhysicalPlan` with `InMemoryScan` leaves from the Python/Rust builder for the chosen query.
- Mapping of Daft `SourceId` → stable DuckDB view name across the plan walk.

## Results

**Benchmark run:** 2026-06-25, macOS Darwin 25.5.0, Apple Silicon (M-series), single-node native runner.
**Sizes:** 10 K, 1 M, 10 M probe rows (build table is 1/10 of probe). **Iterations:** 5 (1 warm-up + 5 timed). **Sorting included:** yes — both `run_native` and `run_duckdb` include a final `sort_by(all columns)` for correctness; the sort cost is present in both timings and roughly cancels in the ratio.

```
        size   swordfish ms    duckdb ms  duckdb/native
      10,000           16.4         24.2           1.48
   1,000,000          152.4        107.3           0.70
  10,000,000         1185.4        787.1           0.66
```

**Go/no-go:** GO — DuckDB is ~30-34% faster than swordfish at production-relevant sizes (1 M+ rows), despite including Arrow-registration overhead in the DuckDB timing; at small sizes (10 K) swordfish is faster (ratio 1.48), so a cost-model threshold around ~100 K rows is warranted before enabling DuckDB on real tasks.

from __future__ import annotations

import daft
from daft.daft import DuckDbExecutor, LocalPhysicalPlan
from daft.execution.native_executor import NativeExecutor
from daft.recordbatch import MicroPartition
from daft.runners import get_or_create_runner


def build_plan_and_inputs(df: daft.DataFrame):
    """Produce the (LocalPhysicalPlan, inputs) the native runner would execute for `df`.

    Optimizes the DataFrame; mirrors native_runner.py:148-152.
    """
    from daft.context import get_context

    ctx = get_context()
    builder = df._builder.optimize(ctx.daft_execution_config)
    runner = get_or_create_runner()
    psets = {
        k: [v.micropartition()._micropartition for v in parts.values()]
        for k, parts in runner._part_set_cache.get_all_partition_sets().items()
    }
    plan, inputs = LocalPhysicalPlan.from_logical_plan_builder(builder._builder, psets)
    return plan, inputs


def run_native(plan, inputs):
    """Run using the native (swordfish) executor. Returns a sorted Arrow table."""
    from daft.context import get_context

    parts = list(NativeExecutor().run(plan, inputs, get_context(), None))
    mps = [p.partition() for p in parts]
    return _to_sorted_arrow(mps)


def run_duckdb(plan, inputs):
    """Run using the DuckDB executor. Returns a sorted Arrow table."""
    py_mps = DuckDbExecutor().run(plan, inputs)
    mps = [MicroPartition._from_pymicropartition(p) for p in py_mps]
    return _to_sorted_arrow(mps)


def _to_sorted_arrow(mps):
    import pyarrow as pa

    if not mps:
        return None
    table = pa.concat_tables([m.to_arrow() for m in mps])
    return _sort_arrow(table)


def _sort_arrow(table):
    if table is None:
        return None
    # Sort by all columns for a deterministic row-multiset comparison.
    # Use stable sort; NULLs sort last in ascending order.
    return table.sort_by([(c, "ascending") for c in table.column_names])


# ── Phase 2b: zero-copy Arrow ingestion via the duckdb Python package ─────────────
# duckdb-rs 1.10501 has no in-place Arrow scan — both its ingestion paths copy every row into
# DuckDB's native storage (see the follow-ups doc, Phase 2 finding). The duckdb *Python* package
# does have one: `con.register(name, arrow_table)` creates an Arrow view that DuckDB scans in place
# (zero-copy for fixed-width primitives; strings/nested still convert). These helpers reuse the
# Rust SQL translation (`plan_to_sql`) and feed it zero-copy-registered Arrow, to measure how far
# the per-task "cold" cost collapses toward "warm" once the ingestion copy is removed.


def _zerocopy_sql_and_tables(plan, inputs):
    """Translate `plan` to SQL and materialize each bound source as a pyarrow Table.

    Returns ``(sql, {source_id: pyarrow.Table})``. The `to_arrow()` conversion (Daft -> pyarrow)
    is done here, OUTSIDE any timed region, so the benchmark can attribute it separately from the
    DuckDB-side register + scan it is actually measuring.
    """
    import pyarrow as pa

    sql, source_ids = DuckDbExecutor.plan_to_sql(plan)
    tables = {}
    for sid in source_ids:
        if sid in tables:  # a source can be bound twice (e.g. self-join); build it once
            continue
        mps = [MicroPartition._from_pymicropartition(p) for p in inputs[sid]]
        tables[sid] = pa.concat_tables([m.to_arrow() for m in mps])
    return sql, tables


def run_duckdb_zerocopy(plan, inputs):
    """Run `plan` via the duckdb Python package using zero-copy Arrow registration.

    Returns a sorted Arrow table (row-multiset comparable with `run_native`).
    """
    import duckdb

    sql, tables = _zerocopy_sql_and_tables(plan, inputs)
    con = duckdb.connect()
    try:
        for sid, table in tables.items():
            con.register(f"daft_src_{sid}", table)
        result = con.sql(sql).to_arrow_table()
    finally:
        con.close()
    return _sort_arrow(result)


def duckdb_bench_zerocopy(plan, inputs, repeat, threads=None, memory_limit=None):
    """Zero-copy counterpart to `duckdb_bench`, via the duckdb Python package.

    Registers each input as a pyarrow Table view (no row copy — DuckDB scans the Arrow buffers in
    place) instead of the Rust `CREATE TABLE` + `Appender` copy, then times `repeat` executions.

    Returns ``(to_arrow_seconds, register_seconds, [run_seconds, ...])``:
    - ``to_arrow_seconds``  -- Daft MicroPartition -> pyarrow conversion (the analog of Daft
      producing Arrow; reported separately, not part of the DuckDB ingestion being measured).
    - ``register_seconds``  -- `con.register` of every source. With zero-copy this is ~0: register
      only creates the Arrow view; the in-place scan is deferred to query time.
    - ``[run_seconds]``     -- each execution, fully materialized to a pyarrow Table (`.arrow()`
      alone returns a lazy reader and would NOT time the scan). Every run re-scans the Arrow view
      in place, so this already includes the (zero-copy) scan — there is no cheaper "warm" state in
      the register-view model, which is the point.

    ``threads`` / ``memory_limit`` cap DuckDB per task (None = DuckDB defaults).
    """
    import time

    import duckdb

    t_arrow = time.perf_counter()
    sql, tables = _zerocopy_sql_and_tables(plan, inputs)
    to_arrow_s = time.perf_counter() - t_arrow

    con = duckdb.connect()
    try:
        if threads is not None:
            con.execute(f"SET threads={threads}")
        if memory_limit is not None:
            con.execute(f"SET memory_limit='{memory_limit}'")
        t_reg = time.perf_counter()
        for sid, table in tables.items():
            con.register(f"daft_src_{sid}", table)
        register_s = time.perf_counter() - t_reg
        run_s = []
        for _ in range(repeat):
            t_run = time.perf_counter()
            con.sql(sql).to_arrow_table()  # materialize the result (forces the scan + compute)
            run_s.append(time.perf_counter() - t_run)
        return to_arrow_s, register_s, run_s
    finally:
        con.close()


# ── Benchmark helpers (no correctness sort — timing only) ────────────────────────
def run_native_unsorted(plan, inputs):
    """Run the native (swordfish) executor and materialize the result partitions (no sort)."""
    from daft.context import get_context

    parts = list(NativeExecutor().run(plan, inputs, get_context(), None))
    return [p.partition() for p in parts]


def duckdb_bench(plan, inputs, repeat, threads=None, memory_limit=None):
    """Register `inputs` into DuckDB once, then time `repeat` executions of the query.

    Returns ``(registration_seconds, [run_seconds, ...])`` so the caller can separate the per-task
    registration/copy cost ("cold") from steady-state query compute ("warm"). ``threads`` /
    ``memory_limit`` cap DuckDB per task (None = DuckDB defaults).
    """
    return DuckDbExecutor().bench_runs(plan, inputs, repeat, threads, memory_limit)

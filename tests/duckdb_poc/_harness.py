from __future__ import annotations

import daft
from daft.daft import DuckDbExecutor, LocalPhysicalPlan
from daft.execution.native_executor import NativeExecutor
from daft.recordbatch import MicroPartition
from daft.runners import get_or_create_runner


def build_plan_and_inputs(df: daft.DataFrame):
    """Optimize the DataFrame and produce the (LocalPhysicalPlan, inputs) the native
    runner would execute. Mirrors native_runner.py:148-152."""
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
    # Sort by all columns for a deterministic row-multiset comparison.
    # Use stable sort; NULLs sort last in ascending order.
    return table.sort_by([(c, "ascending") for c in table.column_names])


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

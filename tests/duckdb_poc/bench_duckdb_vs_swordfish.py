from __future__ import annotations

import statistics
import time

import daft
import numpy as np

from tests.duckdb_poc._harness import build_plan_and_inputs, duckdb_bench, run_native_unsorted

SIZES = [10_000, 1_000_000, 10_000_000]
ITERS = 5

# DuckDB per-task caps. Left at DuckDB defaults (None) for this SINGLE-query benchmark — a thread
# cap would only slow a lone query. In production (many concurrent tasks per worker) you'd set
# THREADS to the per-task slot and MEMORY_LIMIT to its share to avoid core/RAM oversubscription.
THREADS: int | None = None
MEMORY_LIMIT: str | None = None  # e.g. "8GB"


def make_query(n: int) -> daft.DataFrame:
    rng = np.random.default_rng(0)
    build = daft.from_pydict({
        "k": np.arange(n // 10, dtype="int64"),
        "region": rng.integers(0, 8, n // 10).astype("int64"),
    })
    probe = daft.from_pydict({
        "k": rng.integers(0, n // 10, n, dtype="int64"),
        "amount": rng.integers(0, 1000, n, dtype="int64"),
    })
    joined = probe.join(build, on="k")
    filtered = joined.where(daft.col("amount") > 100)
    return filtered.groupby("region").agg(daft.col("amount").sum().alias("total"))


def median_ms_native(plan, inputs) -> float:
    run_native_unsorted(plan, inputs)  # warm-up
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        run_native_unsorted(plan, inputs)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main() -> None:
    daft.set_runner_native()
    # cold  = registration (Arrow->DuckDB copy) + one query  -> the real per-task swap cost
    # warm  = query only, inputs already registered          -> pure DuckDB compute
    # reg   = registration alone                             -> the copy overhead the swap pays
    hdr = (
        f"{'size':>12} {'sword ms':>10} {'duck cold ms':>13} {'duck warm ms':>13} "
        f"{'reg ms':>9} {'warm/native':>12} {'cold/native':>12}"
    )
    print(hdr)
    for n in SIZES:
        plan, inputs = build_plan_and_inputs(make_query(n))
        sword = median_ms_native(plan, inputs)
        reg_s, run_s = duckdb_bench(plan, inputs, ITERS, THREADS, MEMORY_LIMIT)
        warm = statistics.median(run_s) * 1000.0
        reg = reg_s * 1000.0
        cold = reg + warm
        print(
            f"{n:>12,} {sword:>10.1f} {cold:>13.1f} {warm:>13.1f} "
            f"{reg:>9.1f} {warm / sword:>12.2f} {cold / sword:>12.2f}"
        )


if __name__ == "__main__":
    main()

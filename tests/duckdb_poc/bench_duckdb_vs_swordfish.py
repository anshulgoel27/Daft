from __future__ import annotations

import statistics
import time

import numpy as np

import daft
from tests.duckdb_poc._harness import (
    build_plan_and_inputs,
    duckdb_bench,
    duckdb_bench_zerocopy,
    run_native_unsorted,
)

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

    # Two DuckDB ingestion paths, both vs swordfish:
    #   APPENDER  (duckdb-rs)        -- CREATE TABLE + Appender: copies every row into DuckDB
    #                                   native storage up front (reg), then queries cheaply (warm).
    #   ZERO-COPY (duckdb Python)    -- con.register Arrow view: no up-front copy (reg ~= 0), each
    #                                   query scans the Arrow buffers in place (so the scan is in
    #                                   every warm run). `arr` = Daft -> pyarrow conversion.
    #   cold = the honest per-task swap cost = get-data-in + one query.
    #     appender cold = reg + warm
    #     zero-copy cold = arr + reg + warm   (includes Daft->Arrow, the analog of appender's reg)
    hdr = (
        f"{'size':>12} {'sword ms':>9} "
        f"{'app cold':>9} {'app warm':>9} {'app reg':>8} "
        f"{'zc cold':>9} {'zc warm':>9} {'zc reg':>7} {'zc arr':>7} "
        f"{'zc/native':>10} {'app/native':>11} {'zc/app cold':>12}"
    )
    print(hdr)
    # Warm up the duckdb Python package's Arrow-scan machinery once: the first register in a
    # process pays a one-time init that would otherwise be misattributed to the first size's reg.
    _wp, _wi = build_plan_and_inputs(make_query(SIZES[0]))
    duckdb_bench_zerocopy(_wp, _wi, 1, THREADS, MEMORY_LIMIT)
    for n in SIZES:
        plan, inputs = build_plan_and_inputs(make_query(n))
        sword = median_ms_native(plan, inputs)

        app_reg_s, app_run_s = duckdb_bench(plan, inputs, ITERS, THREADS, MEMORY_LIMIT)
        app_warm = statistics.median(app_run_s) * 1000.0
        app_reg = app_reg_s * 1000.0
        app_cold = app_reg + app_warm

        arr_s, zc_reg_s, zc_run_s = duckdb_bench_zerocopy(plan, inputs, ITERS, THREADS, MEMORY_LIMIT)
        zc_warm = statistics.median(zc_run_s) * 1000.0
        zc_reg = zc_reg_s * 1000.0
        zc_arr = arr_s * 1000.0
        zc_cold = zc_arr + zc_reg + zc_warm

        print(
            f"{n:>12,} {sword:>9.1f} "
            f"{app_cold:>9.1f} {app_warm:>9.1f} {app_reg:>8.1f} "
            f"{zc_cold:>9.1f} {zc_warm:>9.1f} {zc_reg:>7.1f} {zc_arr:>7.1f} "
            f"{zc_cold / sword:>10.2f} {app_cold / sword:>11.2f} {zc_cold / app_cold:>12.2f}"
        )


if __name__ == "__main__":
    main()

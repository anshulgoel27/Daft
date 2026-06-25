from __future__ import annotations

import daft
import pytest

from tests.duckdb_poc._harness import build_plan_and_inputs, run_duckdb, run_native


@pytest.fixture(autouse=True)
def _native_runner():
    daft.set_runner_native()


def _query() -> daft.DataFrame:
    build = daft.from_pydict({
        "k": [1, 2, 3, None, 2],
        "region": ["us", "eu", "us", "eu", None],
    })
    probe = daft.from_pydict({
        "k": [1, 2, 2, 3, 5],
        "amount": [10, 20, 30, 40, 999],
    })
    joined = probe.join(build, on="k")  # inner
    filtered = joined.where(daft.col("amount") > 15)
    return filtered.groupby("region").agg(
        daft.col("amount").sum().alias("total"),
        daft.col("amount").count().alias("n"),
    )


def test_duckdb_matches_swordfish():
    plan, inputs = build_plan_and_inputs(_query())
    native = run_native(plan, inputs)
    duck = run_duckdb(plan, inputs)
    assert native is not None, "Native executor returned no partitions"
    assert duck is not None, "DuckDB executor returned no partitions"
    assert native.schema.names == duck.schema.names, (
        f"Column name mismatch: native={native.schema.names}, duck={duck.schema.names}"
    )
    assert native.equals(duck, check_metadata=False), (
        f"Results differ!\nNative:\n{native.to_pydict()}\nDuckDB:\n{duck.to_pydict()}"
    )

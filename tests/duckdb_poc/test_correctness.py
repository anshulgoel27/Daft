from __future__ import annotations

import pytest

import daft
from tests.duckdb_poc._harness import (
    build_plan_and_inputs,
    run_duckdb,
    run_duckdb_zerocopy,
    run_native,
)


@pytest.fixture(scope="module", autouse=True)
def _native_runner():
    # The native runner can only be set once per process; set it once for the module.
    try:
        daft.set_runner_native()
    except daft.exceptions.DaftCoreException:
        pass  # already set (e.g. DAFT_RUNNER=native or a prior module)


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


def _normalize_string_width(table):
    """Normalize Arrow string/binary offset width for value-only comparison.

    swordfish emits string; DuckDB's native Arrow output emits large_string. Cast
    large_string/large_binary -> string/binary so the two compare on value, not representation.
    """
    import pyarrow as pa

    fields = []
    for f in table.schema:
        t = f.type
        if pa.types.is_large_string(t):
            t = pa.string()
        elif pa.types.is_large_binary(t):
            t = pa.binary()
        fields.append(pa.field(f.name, t))
    return table.cast(pa.schema(fields))


def test_duckdb_zerocopy_matches_swordfish():
    """Phase 2b: zero-copy Arrow registration via the duckdb Python package matches swordfish.

    Validates the SQL translation + zero-copy register path end-to-end. The duckdb Python package
    returns its result directly as Arrow (no Daft round-trip), so VARCHAR surfaces as large_string
    where swordfish uses string — normalize the offset width before comparing values.
    """
    plan, inputs = build_plan_and_inputs(_query())
    native = run_native(plan, inputs)
    duck = run_duckdb_zerocopy(plan, inputs)
    assert native is not None, "Native executor returned no partitions"
    assert duck is not None, "DuckDB zero-copy executor returned no result"
    assert native.schema.names == duck.schema.names, (
        f"Column name mismatch: native={native.schema.names}, duck={duck.schema.names}"
    )
    native_n = _normalize_string_width(native)
    duck_n = _normalize_string_width(duck)
    assert native_n.equals(duck_n, check_metadata=False), (
        f"Results differ!\nNative:\n{native.to_pydict()}\nDuckDB(zero-copy):\n{duck.to_pydict()}"
    )

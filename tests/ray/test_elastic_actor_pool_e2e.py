"""End to end: with ``actor_udf_min_ready`` set, a partial actor pool grows when capacity appears mid-query.

A query that starts on a partial actor pool grows into the rest of it when the cluster gains capacity.
Scenarios live in _elastic_actor_pool_scenario.py and run in their own interpreter (Daft's runner can be set
only once per process).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("ray")

SCENARIO = os.path.join(os.path.dirname(__file__), "_elastic_actor_pool_scenario.py")


def _run(mode: str) -> list[str]:
    proc = subprocess.run(
        [sys.executable, SCENARIO, mode],
        capture_output=True,
        text=True,
        timeout=420,
        check=False,
        env={**os.environ, "DAFT_RUNNER": "ray"},
    )
    assert proc.returncode == 0, f"scenario failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
    line = next(line for line in reversed(proc.stdout.splitlines()) if line.startswith("RESULT "))
    return json.loads(line[len("RESULT ") :])


def test_pool_grows_into_capacity_that_appears_mid_query():
    actors_used = _run("elastic")

    # Only 2 slots existed when the query started; actors that joined later must have been used too.
    assert len(actors_used) > 2, f"pool never grew past the initial actors: {actors_used}"
    assert len(actors_used) <= 4


def test_fixed_pool_does_not_grow_by_default():
    # Control: without actor_udf_min_ready the pool is fixed to the actors ready at start.
    actors_used = _run("fixed")

    assert len(actors_used) == 2, f"fixed pool should stay at the 2 ready actors: {actors_used}"

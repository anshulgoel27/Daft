"""Elastic UDF actor start: returns once enough actors are ready, and keeps the rest so they can join later."""

from __future__ import annotations

import asyncio
import time

import pytest

ray = pytest.importorskip("ray")
from ray.cluster_utils import Cluster

import daft
from daft.execution import ray_actor_pool_udf as pool_udf

SLOT = {"slot": 1}  # every test actor needs one custom "slot"; nodes advertise how many they have


@pytest.fixture
def cluster():
    """A local Ray cluster whose head has 2 slots; tests add nodes to free up more."""
    if ray.is_initialized():
        pytest.skip("Ray is already initialized in this process")
    c = Cluster(initialize_head=True, head_node_args={"num_cpus": 1, "resources": {"slot": 2}})
    ray.init(address=c.address, logging_level="ERROR", include_dashboard=False)
    yield c
    ray.shutdown()
    c.shutdown()


def _start(num_actors: int, *, min_ready: int, timeout: int = 30, resources: dict | None = None):
    return asyncio.run(
        pool_udf.start_udf_actors_elastic(
            daft.col("x")._expr,
            num_actors,
            0.0,  # gpus
            0.0,  # cpus: only the custom slot gates scheduling
            0,  # memory
            {"resources": resources or SLOT},
            timeout,
            None,
            min_ready,
        )
    )


def _up_ids(handles) -> set[str]:
    """Actors a worker would use right now: what get_ready_actors_by_location keeps."""
    local, remote = pool_udf.get_ready_actors_by_location(handles)
    return {h.actor_id() for h in local + remote}


def _wait_until_up(handles, n: int, within: float = 60.0) -> set[str]:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        up = _up_ids(handles)
        if len(up) >= n:
            return up
        time.sleep(0.2)
    raise AssertionError(f"never reached {n} usable actors (has {len(_up_ids(handles))})")


def _teardown(handles) -> None:
    for h in handles:
        h.teardown()


def test_returns_every_actor_but_only_the_ready_ones_are_usable(cluster):
    handles = _start(4, min_ready=2)
    try:
        assert len(handles) == 4  # pending actors are kept, not dropped
        assert len(_up_ids(handles)) == 2  # only the 2 that fit are usable
    finally:
        _teardown(handles)


def test_pending_actors_become_usable_when_capacity_appears(cluster):
    handles = _start(4, min_ready=2)
    try:
        first = _up_ids(handles)
        assert len(first) == 2

        cluster.add_node(num_cpus=1, resources={"slot": 2})

        grown = _wait_until_up(handles, 4)
        assert first < grown  # the original actors are still there; two more joined
    finally:
        _teardown(handles)


def test_proceeds_with_partial_pool_after_timeout(cluster):
    # 2 slots exist, 3 wanted: after the timeout it proceeds with the 2 that are ready.
    start = time.monotonic()
    handles = _start(4, min_ready=3, timeout=3)
    try:
        assert time.monotonic() - start >= 3
        assert len(_up_ids(handles)) == 2
    finally:
        _teardown(handles)


def test_raises_when_no_actor_becomes_ready(cluster):
    with pytest.raises(RuntimeError, match="failed to start within"):
        _start(2, min_ready=1, timeout=2, resources={"never_available": 1})


def test_min_ready_is_clamped_to_the_pool_size(cluster):
    handles = _start(2, min_ready=100)  # asking for more than exist must not hang
    try:
        assert len(_up_ids(handles)) == 2
    finally:
        _teardown(handles)


def test_when_no_actor_is_up_yet_all_are_kept(cluster):
    # get_ready_actors_by_location must never return an empty pool for a non-empty one: the operator
    # indexes into it. Here every actor is pending (no slot resource matches), so none are "up".
    actors = [pool_udf.UDFActor.options(resources={"never_available": 1}, num_cpus=0).remote(None) for _ in range(2)]
    handles = [pool_udf.UDFActorHandle(a) for a in actors]
    try:
        local, remote = pool_udf.get_ready_actors_by_location(handles)
        assert len(local) + len(remote) == 2
    finally:
        _teardown(handles)


def test_teardown_releases_ready_and_pending_actors(cluster):
    handles = _start(4, min_ready=2)
    ids = {h.actor_id() for h in handles}
    _teardown(handles)

    # Killed actors release their slots, so a fresh pool can use all of them again.
    second = _start(2, min_ready=2)
    try:
        assert len(_up_ids(second)) == 2
        assert not ids & {h.actor_id() for h in second}
    finally:
        _teardown(second)

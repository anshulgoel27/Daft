from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from daft.expressions.expressions import Expression, ExpressionsProjection
from daft.recordbatch.micropartition import MicroPartition
from daft.runners.ray_compat import validate_and_normalize_ray_options

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ray.actor import ActorHandle as RayActorHandle

    from daft.daft import PyExpr, PyMicroPartition

try:
    import ray
except ImportError:
    raise

# Ray actor states before the actor's __init__ has finished (see ray.util.state.ActorState).
_ACTOR_STATES_NOT_YET_UP = ("DEPENDENCIES_UNREADY", "PENDING_CREATION")

MAX_UDFACTOR_ACTOR_RESTARTS = 4
MAX_UDFACTOR_ACTOR_TASK_RETRIES = 4


@ray.remote(
    max_restarts=MAX_UDFACTOR_ACTOR_RESTARTS,
    max_task_retries=MAX_UDFACTOR_ACTOR_TASK_RETRIES,
)
class UDFActor:
    def __init__(self, projection: ExpressionsProjection) -> None:
        self.projection = projection

    def get_node_id(self) -> str:
        return ray.get_runtime_context().get_node_id()

    def eval_input(self, input: PyMicroPartition) -> PyMicroPartition:
        mp = MicroPartition._from_pymicropartition(input)
        res = mp.eval_expression_list(self.projection)
        return res._micropartition


class UDFActorHandle:
    def __init__(self, actor_ref: RayActorHandle) -> None:
        self.actor = actor_ref

    def actor_id(self) -> str:
        return self.actor._actor_id.hex()

    async def eval_input(self, input: PyMicroPartition) -> PyMicroPartition:
        return await self.actor.eval_input.remote(input)

    def teardown(self) -> None:
        ray.kill(self.actor)


def get_ready_actors_by_location(
    actor_handles: list[UDFActorHandle],
) -> tuple[list[UDFActorHandle], list[UDFActorHandle], int]:
    """Split the actors that can take work into those on this node and those on other nodes.

    Also returns how many actors are not up yet. Only an elastic pool (see ``start_udf_actors_elastic``) has
    such actors; they join the pool when a later call finds them up. Dead actors are left out. If no actor is
    up, the pending ones (or, if there are none, all of them) are returned as remote, so calls queue on them
    instead of the caller getting an empty pool.
    """
    from ray._private.state import actors

    current_node_id = ray.get_runtime_context().get_node_id()

    local_actors = []
    remote_actors = []
    pending_actors = []
    for actor_handle in actor_handles:
        actor_state = actors(actor_handle.actor_id()) or {}
        state = actor_state.get("State")
        if state in _ACTOR_STATES_NOT_YET_UP:
            pending_actors.append(actor_handle)
        elif state is None or state == "DEAD":
            continue
        elif actor_state["Address"]["NodeID"] == current_node_id:
            local_actors.append(actor_handle)
        else:
            remote_actors.append(actor_handle)

    if not local_actors and not remote_actors:
        remote_actors = pending_actors or list(actor_handles)

    return local_actors, remote_actors, len(pending_actors)


def _create_udf_actors(
    projection: PyExpr,
    num_actors: int,
    num_gpus_per_actor: float,
    num_cpus_per_actor: float,
    memory_per_actor: float,
    ray_options: dict[str, Any] | None,
    timeout: int,
    actor_name: str | None = None,
) -> list[RayActorHandle]:
    """Validate the resource request and submit ``num_actors`` UDF actors to Ray.

    Returns immediately: the actors are created asynchronously by Ray and may not be ready (or even
    scheduled, if the cluster has no free resources for them yet).
    """
    expr_projection = ExpressionsProjection([Expression._from_pyexpr(projection)])

    # If resources are already in ray_options, we should prioritize them and avoid passing them twice
    # to avoid "keyword argument repeated" errors.
    ray_options = (ray_options or {}).copy()

    udf_options = validate_and_normalize_ray_options(
        {
            "scheduling_strategy": "SPREAD",
            "num_gpus": num_gpus_per_actor,
            "num_cpus": num_cpus_per_actor,
            "memory": memory_per_actor,
            **ray_options,
        }
    )

    # Read from udf_options (not the params) because ray_options can override num_cpus / num_gpus via the spread above.
    cpus_per_actor = udf_options.get("num_cpus", 1.0)
    gpus_per_actor = udf_options.get("num_gpus", 0.0)
    alive_nodes = [n for n in ray.nodes() if n["Alive"]]
    can_schedule = any(
        n["Resources"].get("CPU", 0) >= cpus_per_actor and n["Resources"].get("GPU", 0) >= gpus_per_actor
        for n in alive_nodes
    )
    if not can_schedule:
        raise RuntimeError(
            f"Actor UDF requires {cpus_per_actor} CPUs and {gpus_per_actor} GPUs per actor, "
            f"but no single node can satisfy this. "
            f"No single actor can be scheduled."
        )

    if cpus_per_actor > 0 or gpus_per_actor > 0:
        cluster_cpus = ray.cluster_resources().get("CPU", 0)
        cluster_gpus = ray.cluster_resources().get("GPU", 0)
        limits = []
        if cpus_per_actor > 0:
            limits.append(cluster_cpus / cpus_per_actor)
        if gpus_per_actor > 0:
            limits.append(cluster_gpus / gpus_per_actor)
        max_schedulable = int(min(limits))
        if num_actors > max_schedulable:
            logger.warning(
                "with_concurrency(%d) requested but only %d actors can be scheduled "
                "(%g CPUs, %g GPUs available). Will proceed with partial actor pool "
                "after actor_udf_ready_timeout (%ds).",
                num_actors,
                max_schedulable,
                cluster_cpus,
                cluster_gpus,
                timeout,
            )

    actors: list[RayActorHandle] = [
        UDFActor.options(  # type: ignore
            name=None if actor_name is None else f"{actor_name}:{str(uuid.uuid4())[:8]}-{rank}",
            **udf_options,
        ).remote(expr_projection)
        for rank in range(num_actors)
    ]
    return actors


async def start_udf_actors_elastic(
    projection: PyExpr,
    num_actors: int,
    num_gpus_per_actor: float,
    num_cpus_per_actor: float,
    memory_per_actor: float,
    ray_options: dict[str, Any] | None,
    timeout: int,
    actor_name: str | None = None,
    min_ready: int = 1,
) -> list[UDFActorHandle]:
    """Start ``num_actors`` UDF actors but return as soon as ``min_ready`` of them are ready.

    Unlike ``start_udf_actors``, actors that are not ready yet (for example because the cluster has no free
    resources for them) are returned too, not dropped. Ray keeps scheduling them, and every task skips the
    ones that are not alive yet when it resolves its actors (see ``get_ready_actors_by_location``), so
    later tasks use a larger pool as the pending actors come up.

    If fewer than ``min_ready`` are ready after ``timeout`` seconds, it proceeds with the ones that are;
    if none are ready, it kills them all and raises. Actors that fail to start are killed and left out.
    """
    needed = min(max(min_ready, 1), num_actors)
    actors = _create_udf_actors(
        projection,
        num_actors,
        num_gpus_per_actor,
        num_cpus_per_actor,
        memory_per_actor,
        ray_options,
        timeout,
        actor_name,
    )
    pending = {actor.__ray_ready__.remote(): i for i, actor in enumerate(actors)}
    failed: set[int] = set()
    num_ready = 0

    def _teardown() -> None:
        for actor in actors:
            try:
                ray.kill(actor)
            except (ValueError, ray.exceptions.RayError) as e:
                logger.warning("Error tearing down UDF actor: %s", e)

    deadline = time.monotonic() + timeout
    while True:
        if pending:
            done, _ = ray.wait(list(pending), num_returns=len(pending), timeout=0)
            for ref in done:
                i = pending.pop(ref)
                try:
                    ray.get(ref)
                    num_ready += 1
                except ray.exceptions.RayError as e:
                    # Drop it: a dead actor left in the pool would fail every batch sent to it.
                    logger.warning("UDF actor failed to start: %s", e)
                    failed.add(i)
                    try:
                        ray.kill(actors[i])
                    except (ValueError, ray.exceptions.RayError):
                        pass
        if num_ready >= needed:
            break
        if not pending:
            # Every actor either became ready or failed, and fewer than `needed` are usable.
            if num_ready:
                break
            _teardown()
            raise RuntimeError(f"All {num_actors} UDF actors failed to start; see the warnings above for the cause.")
        if time.monotonic() >= deadline:
            if num_ready:
                logger.warning(
                    "Only %d of the %d UDF actors wanted are ready after %ds; starting with them. "
                    "The remaining actors join the pool when Ray schedules them.",
                    num_ready,
                    needed,
                    timeout,
                )
                break
            _teardown()
            raise RuntimeError(
                f"UDF actors failed to start within {timeout} seconds, please increase the actor_udf_ready_timeout config "
                f"via daft.set_execution_config(actor_udf_ready_timeout=timeout)"
            )
        await asyncio.sleep(0.2)

    return [UDFActorHandle(actor) for i, actor in enumerate(actors) if i not in failed]


async def start_udf_actors(
    projection: PyExpr,
    num_actors: int,
    num_gpus_per_actor: float,
    num_cpus_per_actor: float,
    memory_per_actor: float,
    ray_options: dict[str, Any] | None,
    timeout: int,
    actor_name: str | None = None,
) -> list[UDFActorHandle]:
    actors = _create_udf_actors(
        projection,
        num_actors,
        num_gpus_per_actor,
        num_cpus_per_actor,
        memory_per_actor,
        ray_options,
        timeout,
        actor_name,
    )

    # Wait for actors to be ready
    ready_futures = [asyncio.wrap_future(actor.__ray_ready__.remote().future()) for actor in actors]
    ready_refs, _ = await asyncio.wait(ready_futures, return_when=asyncio.ALL_COMPLETED, timeout=timeout)

    # Verify that the __ray_ready__ calls were successful
    await asyncio.gather(*ready_refs)

    if not ready_refs:
        raise RuntimeError(
            f"UDF actors failed to start within {timeout} seconds, please increase the actor_udf_ready_timeout config "
            f"via daft.set_execution_config(actor_udf_ready_timeout=timeout)"
        )

    # Return the ready actors
    ready_indices = [ready_futures.index(ref) for ref in ready_refs]
    ready_actors = [UDFActorHandle(actors[i]) for i in ready_indices]
    return ready_actors

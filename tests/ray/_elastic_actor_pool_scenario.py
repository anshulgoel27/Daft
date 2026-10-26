"""Subprocess helper for test_elastic_actor_pool_e2e.py.

Daft's runner can be set only once per process, so each scenario runs in its own interpreter:

    python _elastic_actor_pool_scenario.py elastic|fixed

It prints one line, ``RESULT <json list of the distinct Ray actor ids that processed the data>``.

The cluster is shaped like production: the head runs the tasks, and the UDF actors live on separate
nodes that advertise no CPU. One such node (2 `slot`s) exists up front and a second one is added
mid-query. The actors must not share a node with the tasks: workers prefer actors on their own node,
which would hide actors that joined the pool elsewhere.
"""

from __future__ import annotations

import json
import sys
import threading
import time

import daft
from daft import DataType

NUM_ACTORS = 4
PARTITIONS = 60
ADD_NODE_AFTER_SECONDS = 5.0


@daft.cls(cpus=0, max_concurrency=NUM_ACTORS, ray_options={"resources": {"slot": 1}})
class WhoAmI:
    """Returns the id of the Ray actor that handled each row (after a pause, so the query stays running)."""

    @daft.method.batch(return_dtype=DataType.string())
    def run(self, xs):
        import time

        import ray

        time.sleep(1.0)
        actor_id = ray.get_runtime_context().get_actor_id()
        return [actor_id] * len(xs)


def main(mode: str) -> None:
    from ray.cluster_utils import Cluster

    cluster = Cluster(initialize_head=True, head_node_args={"num_cpus": 4})
    cluster.add_node(num_cpus=0, resources={"slot": 2})
    daft.set_runner_ray(address=cluster.address)

    if mode == "elastic":
        daft.set_execution_config(actor_udf_min_ready=2, actor_udf_ready_timeout=60)
    else:
        daft.set_execution_config(actor_udf_ready_timeout=5)

    result: dict[str, list[str]] = {}
    errors: list[BaseException] = []

    def query() -> None:
        try:
            df = daft.range(start=0, end=PARTITIONS, partitions=PARTITIONS)
            df = df.with_column("actor", WhoAmI().run(df["id"]))
            result["actors"] = sorted(set(df.to_pydict()["actor"]))
        except BaseException as e:  # noqa: BLE001  (any failure is re-raised in the main thread below)
            errors.append(e)

    # `Cluster.add_node` installs signal handlers, which Python allows only on the main thread, so the
    # query is the one that runs in the background.
    t = threading.Thread(target=query, daemon=True)
    t.start()
    time.sleep(ADD_NODE_AFTER_SECONDS)
    cluster.add_node(num_cpus=0, resources={"slot": 2})
    t.join(timeout=300)
    if t.is_alive():
        raise SystemExit("query did not finish")
    if errors:
        raise errors[0]
    print("RESULT " + json.dumps(result["actors"]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

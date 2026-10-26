# Elastic UDF actor pools

## Summary

Add an opt-in mode, `actor_udf_min_ready`, in which a stateful UDF (`@daft.cls` / `with_concurrency`) starts
running as soon as a minimum number of its actors are ready, and **uses the remaining actors as they come
up** instead of discarding them. With the option unset (the default) behaviour is unchanged.

```python
daft.set_execution_config(actor_udf_min_ready=8, actor_udf_ready_timeout=300)
df.with_column("out", MyUDF().run(df["x"])).collect()   # MyUDF has with_concurrency(64) / max_concurrency=64
```

Also settable via `DAFT_ACTOR_UDF_MIN_READY`.

## Problem

A stateful UDF with concurrency `N` is backed by a pool of `N` Ray actors. Today the pool is sized once,
when the first task reaches the UDF:

1. `start_udf_actors` submits `N` actors and waits up to `actor_udf_ready_timeout`.
2. Whatever is ready by then is the pool. The rest are dropped (their handles go out of scope and Ray
   kills them).
3. That pool is used, unchanged, for the whole query.

This is a poor fit when capacity is not all available at the moment the query starts, which is common:

- the cluster autoscales, so actors beyond the current nodes are `PENDING` for minutes;
- other workloads hold resources that are released while the query is running;
- the query is long (hours), so capacity that frees up later is never used.

The only options are to wait for the full pool before doing any work (idle time, and a timeout that has to be
sized for the worst case), or to start with a partial pool that stays small for the rest of the run. Re-running
the query to pick up more capacity throws away the work done so far.

## Behaviour with `actor_udf_min_ready > 0`

- All `N` actors are submitted, as before.
- Execution starts once `min(actor_udf_min_ready, N)` are ready. If fewer are ready when
  `actor_udf_ready_timeout` expires, it proceeds with the ones that are, as long as at least one is.
  If none are ready it raises, as today.
- Actors that are not up yet are **kept**. Ray keeps scheduling them, and work that is dispatched after they
  come up is spread over them too, so the pool grows during the query up to `N`.
- The pool only grows. Actors are torn down at the end of the query, including ones that never came up.

With the option unset, the code path is the existing one.

## Implementation

### Python: `daft/execution/ray_actor_pool_udf.py`

- `_create_udf_actors`: the resource validation and actor submission from `start_udf_actors`, extracted so
  both start functions share it.
- `start_udf_actors_elastic`: submits `N` actors, polls readiness without blocking, returns when enough are
  ready (or the timeout passes with at least one), and returns **all** handles, ready or not. If none become
  ready it kills every actor before raising.
- `get_ready_actors_by_location`: runs on the worker when an operator is built and when it refreshes. It now
  skips actors whose Ray state is `DEPENDENCIES_UNREADY` / `PENDING_CREATION`. If none are up it keeps all of
  them, so the caller never receives an empty pool.

### Rust: `daft-distributed/src/pipeline_node/actor_udf.rs`

`ActorUDF` reads `actor_udf_min_ready` from the execution config and calls `start_udf_actors_elastic` instead of
`start_udf_actors` when it is non-zero. Nothing else about how the actor list is attached to tasks changes.

### Rust: `daft-local-execution/.../distributed_actor_pool_project.rs`

This is the part that makes growth actually take effect, and it is why filtering at task creation was not
enough:

- Tasks that share a plan fingerprint and run on the same worker **share one pipeline**, so the actor
  operator is built **once per worker**, not once per task. Whatever actor list it resolves at that moment is
  frozen for the rest of the query.
- Each of the operator's concurrency slots was also bound to a single actor for the lifetime of the pipeline.

So, when the operator is built and some actors in the list are not up, it creates a `DynamicActorPool`:

- it keeps the full list and the currently usable subset (same-node actors if any, else the rest);
- it re-resolves the usable subset every 5 seconds until every actor is up, then stops checking;
- it picks the actor **per batch** (round robin over the usable subset) instead of per concurrency slot;
- `max_concurrency` counts all actors, including pending ones. Concurrency is fixed when the pipeline
  starts, and batches queue on the actors that are up until the rest join.

When every actor is up at build time (every pool today, and any pool with the option off) `dynamic` is `None`
and the operator behaves exactly as before: per-slot actor binding and `max_concurrency` from the filtered
list.

### Config plumbing

`actor_udf_min_ready` (`usize`, default 0) is added next to `actor_udf_ready_timeout` in `common-daft-config`
(struct, `Default`, `DAFT_ACTOR_UDF_MIN_READY` env parsing, Python bindings), `daft.set_execution_config`, and
the `.pyi` stub.

## Why not

- **Re-read the pool at task creation.** The first attempt did this. Tasks are created in bulk at the start
  of the query (all of them are queued within the first second), long before capacity changes, and tasks on
  the same worker share a pipeline anyway. The growth never showed up in an end-to-end test, which is what
  led to the per-batch design above.
- **Throttle task creation so each task is built late.** This would couple the actor node to downstream task
  completion. Tasks feeding a blocking sink only complete after the sink flushes, so waiting on completion can
  deadlock. It also still would not help, because the actor list is resolved once per worker pipeline.
- **A manager actor that tracks the pool.** More moving parts (another long-lived actor, another failure
  mode) for what the worker can already learn from Ray's actor state.
- **Resize on request (`add_actors(n)`).** Needs a new API and a way to ask Daft's running plan to change; the
  pending-actors approach reuses Ray's own queue and needs no new control path.

## Compatibility

- Default off: no change to plans, config, or behaviour for existing users.
- No change to the serialized plan; the actor list attached to tasks is the same shape.
- The new config field is additive. Rust and Python bindings are updated together.

## Limitations and risks

- **No shrink.** Idle actors stay until the query ends.
- **Pending actors hold a position in Ray's queue.** On a shared cluster, asking for a large `N` makes this
  query compete for resources as they free up, and makes an autoscaler provision for the full `N`. Pick `N`
  accordingly.
- **Refresh latency.** A newly-up actor can wait up to 5 seconds (plus the next batch) before receiving work.
- **Same-node preference.** Workers prefer actors on their own node. Pending actors that come up elsewhere
  are only used when no same-node actors exist, as before.
- **Ray runner only.** Actor pools on the native runner are unaffected.

## Testing

- `tests/actor_pool/test_elastic_actor_pool.py`: the Python start function and the worker-side filter on a
  real local multi-node Ray cluster (a custom resource gates scheduling): returns every actor while only the
  ready ones are usable; pending actors become usable when a node joins; partial pool after the timeout;
  raises when none become ready; clamps `min_ready` to the pool size; keeps all actors when none are up yet;
  teardown releases ready and pending actors.
- `tests/ray/test_elastic_actor_pool_e2e.py` (+ `_elastic_actor_pool_scenario.py`, run in a subprocess
  because the runner can only be set once per process): a real query on a local cluster whose actors live on
  CPU-less nodes, with a second node added mid-query. With the option on, more actors than were ready at the
  start process data; with it off (control), the pool stays at its initial size.
- `common-daft-config` unit tests cover the new environment variable.

## Rollout

Because the option is off by default, it can ship dark and be enabled per query or per deployment with
`set_execution_config` or the environment variable.

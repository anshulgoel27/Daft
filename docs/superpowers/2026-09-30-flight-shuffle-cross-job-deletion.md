# Flight Shuffle — Cross-Job Shuffle File Deletion

Why `fix/flight-shuffle-per-query-dirs` exists: concurrent Daft jobs on one Ray cluster could
delete or overwrite each other's live flight-shuffle files, failing queries with
`IoError No such file or directory (os error 2)`.

**Status:** fixed on `fix/flight-shuffle-per-query-dirs` (`82f6ffed3` plus cleanup follow-ups;
not yet merged or released). Regression tests reproduce the bug before the fix and pass after it.

---

## Symptom

A long-running Ray job (`geo_fabric` `gold.resolve-links`, ~40–100 min) was run with
`shuffle_algorithm="flight_shuffle"`, with every driver on the cluster pointing at one shared
`flight_shuffle_dirs` (`DAFT_FLIGHT_SHUFFLE_DIR=/mnt/nvme/shuffle`). It failed repeatedly
partway through a hash join:

```
ray::((ShuffleRead, ShuffleRead)->HashJoin, InMemoryScan)->HashJoin->Project->Project()
daft.exceptions.DaftCoreException: Tonic error: code: 'Internal error',
  message: "flight stream: DaftError::IoError No such file or directory (os error 2)"
```

The job had moved to `flight_shuffle` because `pre_shuffle_merge` was OOM-killing the head
node: Daft estimated ~930 GiB of head-node shuffle metadata for an 18,030-partition shuffle,
against a 96 GB head.

| Run | Shuffle | Other jobs running at the same time | Outcome |
|---|---|---|---|
| `…1790693164` | flight | **0** | ran 99 min, no errors (stopped manually) |
| `…1790674827` | flight | 130 | failed, `No such file` (44 min) |
| `…1790679030` | flight | 11 | failed, `No such file` (97 min) |
| `…1790759348` | flight | 93 | failed, `No such file` (39 min) |
| `…1790699459` | auto | 37 | succeeded (40 min) — object-store shuffle, no shared dir |

The only flight-shuffle run that stayed clean was the one with the cluster to itself.

---

## Hypotheses ruled out

| Hypothesis | Why it was ruled out |
|---|---|
| **Data skew** causing worker OOM | A temporary skew diagnostic found no hot keys: the largest fingerprint group had 324 rows, the target join table was unique on its key, and set/group keys were similarly flat. |
| **Worker OOM** destroying shuffle files | No OOM was logged before the first `No such file` error in any failed run. The one worker OOM (`…1790679030`) happened *after* the errors began, so it wasn't the trigger. |
| **Autoscaler idle-reaping** nodes that hold shuffle data | On Ray 2.58, a live actor that holds no resources still keeps its node non-idle (verified locally); every worker node runs a `RaySwordfishActor`. The error also comes from `do_get` in a **live** flight server, so the server was up and the files were gone. A reaped node would surface as a connection error instead. |

A prototype keeping shuffle-holding nodes busy (pinned 1-byte keep-alive actors) was built and
tested, then discarded for the reason in the last row.

---

## Root cause

Two independent bugs let one plan delete another plan's live shuffle files.

### 1. Shuffle directories collided across drivers

Files lived at `{dir}/daft_shuffle/{shuffle_id}` with

```rust
// pipeline_node/shuffles/backends/mod.rs
((context.query_idx as u64) << 32) | (context.node_id as u64)
```

where `query_idx` comes from `QUERY_IDX_COUNTER` in `plan/mod.rs`, a **per-process** counter
starting at 0. Two drivers running the same pipeline therefore produce the same `shuffle_id`s,
so on a shared directory:

- the one-shot writer's `File::create(".../map_{input_id}.arrow")` **truncates** the other
  job's map files, and
- success cleanup in `PlanRunner::run_plan_impl` deletes `{dir}/daft_shuffle/{shuffle_id}` on
  **every node**, removing the other job's files.

### 2. Failure cleanup deleted the whole shuffle root

`RemoteFlotillaRunner.cleanup_plan_shuffle` (runs when any plan fails or is cancelled) deleted
`{dir}/daft_shuffle` — **every job's** shuffle data — on every node. It built that path from
`plan.flight_shuffle_dirs()`, which returned the configured dirs for **every** plan, including
plans that never used flight shuffle.

### Evidence on the cluster

Ray's task history shows `_clear_flight_shuffle_dirs` wipes across all 20 nodes, issued by two
separate interactive Ray Client sessions (10:01:39 and 10:07:41 UTC on 2026-09-30), not by the
failing job. Ray keeps only recent task history, so no wipe could be shown inside a failed run's
own window; the link to the failures is strong but circumstantial.

---

## The fix

| Change | Where |
|---|---|
| Each `DistributedPhysicalPlan` gets a random 64-bit `shuffle_namespace` at creation | `src/daft-distributed/src/plan/mod.rs` |
| All of a plan's shuffle files live under `{dir}/daft_shuffle/{namespace:016x}` | `PlanConfig::flight_shuffle_dirs()` in `src/daft-distributed/src/plan/runner.rs` |
| Writers, success cleanup, and failure cleanup all derive paths from that one function | `translate_shuffle.rs`, `backends/flight.rs`, `oneshot_writer.rs`, `shuffle_cache.rs`, `daft/runners/flotilla.py` |
| `flight_shuffle_dirs()` is empty unless the plan uses flight shuffle, so other plans never trigger cleanup | `PlanConfig::flight_shuffle_dirs()` |
| Registered cleanup dirs are deduplicated, so the plan root is removed once | `PlanExecutionContext::register_shuffle_dirs` |
| A plan's namespace is deleted however it ends: success, failure, cancellation, or being abandoned early (`.show()`, breaking out of `iter_partitions()`) | `PlanRunner::run_plan_impl` (now also on error), `FlotillaRunner.stream_plan` (now also on `GeneratorExit` / `KeyboardInterrupt`) |

A random namespace was chosen over `query_id` because `query_id` is a cosmetic display name
(`eager-phoenix-a4f821`, ~36 bits of entropy) and shouldn't decide what `rm -rf` touches.

Because a namespace is never reused, a missed cleanup is never repaired by a later run. Before
this fix, a re-run with the same `shuffle_id` (or the destructive root wipe) incidentally removed
leftovers. So every way a plan can end now cleans up its own namespace.

User-facing configuration is unchanged: `flight_shuffle_dirs` still takes base directories.

---

## Verification

New tests in `tests/dataframe/test_shuffles.py`:

| Test | Before fix | After fix |
|---|---|---|
| `test_flight_shuffle_dirs_are_unique_per_plan` | failed (identical dirs) | pass |
| `test_flight_shuffle_dirs_empty_without_flight_shuffle` | failed | pass |
| `test_flight_shuffle_cleanup_spares_other_plans_files[True]` (failing query) | **failed — deleted another plan's live file** | pass |
| `test_flight_shuffle_cleanup_spares_other_plans_files[False]` (succeeding query) | pass | pass |
| `test_flight_shuffle_cleanup_after_early_stop` (query abandoned after one partition) | failed — namespace dir leaked | pass |

The cleanup tests also assert that nothing is written outside `{dir}/daft_shuffle`.

The shuffle, sort, join, and flotilla test files pass on the Ray runner (3,930 passed, 0 failed).

---

## Not addressed

- **Flight shuffle still has no recovery for a lost worker.** Unlike the Ray object-store shuffle,
  disk spill has no lineage, so a real worker loss (OOM, node failure) still fails the query.
- **Daft's own downscaler** (`downscale_enabled=True`) can still retire a worker that holds
  shuffle output.
- **Some shuffle directories can still be left behind for good.** If the driver is killed
  (`SIGKILL`, OOM, node loss), no cleanup runs. On failure or early stop, a map task already
  running on a worker can also recreate its directory after cleanup has deleted it. A namespace
  is never reused, so these leftovers stay until removed by hand. Removing
  `{dir}/daft_shuffle/<namespace>` directories older than the longest job is safe.

## Until this ships

Build and deploy a Daft wheel with this fix to the cluster image **and to every driver that
uses the shared directory, including interactive Ray Client sessions**. The cluster wipes in the
incident came from interactive sessions. A driver still on an old wheel deletes the whole
`{dir}/daft_shuffle` root when any of its plans fails, and that root now holds every patched
job's namespaces too. Until every driver is upgraded, either:

- use `shuffle_algorithm="auto"` (object-store shuffle; `…1790699459` succeeded with it), or
- give each job its own directory, e.g. `DAFT_FLIGHT_SHUFFLE_DIR=/mnt/nvme/shuffle/<job_id>`,
  which other sessions' cleanup of `/mnt/nvme/shuffle/daft_shuffle` does not reach.

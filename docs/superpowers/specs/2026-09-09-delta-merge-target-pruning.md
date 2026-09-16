# Spec: Target pruning for `distributed_merge_deltalake`

**Date:** 2026-09-09 · **Verified against:** `eff608c99`

## Problem

A 24-row delete+append into an unpartitioned Delta table takes **18m22s**. Three
compounding costs, all confirmed in code:

1. **The target is the probe side.** On Ray with a source ≤ 512 MiB,
   `_should_decompose_join` (`daft/io/delta_lake/_deltalake.py:1895`) rewrites the full
   outer join into broadcast `LEFT` + `ANTI` (`_decomposed_outer_join`, `:1152`).
   `translate_join.rs:291` sets `is_swapped = true` for `JoinType::Left`, so the small
   source becomes the build side and the whole target is probed.
2. **That join executes twice.** `annotated = _annotate(joined)` (`:1685`) stays lazy;
   pass 1 (`.agg`, `:1693`) runs it and pass 2 (the write, `:1772+`) re-runs it.
   `materialize_join` defaults to `False` (`:991`).
3. **Pruning is write-pass-only and partition-column-only.** Pass 1 (`:1543`) takes no
   filter. Pass 2 prunes only under `if partition_cols and affected is not None` (`:1781`).

### Non-goal: re-siding the join

`is_swapped = true` for `Left` is **semantically required**, not a defect.
`broadcast_join.rs:170-181` gives the node `ClusteringStrategy::Passthrough { child: &receiver }`
(receiver partitioned, broadcaster replicated to every partition) and `:248` ties
`build_on_left = !is_swapped` (build is always the broadcast side). Each task therefore
needs a *complete* copy of the right side to decide "no match" for its slice of left rows.
Broadcasting the left instead would let a left row unmatched in one partition's right-slice
emit a spurious NULL while matching in another slice. **The only fix is scanning less
target.** Do not attempt to re-side the join.

## Enabling facts

- **`MicroPartition::filter` skips files on stats without reading them.**
  `daft-micropartition/src/ops/filter.rs:22-27` evaluates the predicate against
  `self.statistics` and returns `Self::empty(...)` on `TruthValue::False` *before*
  `self.record_batches()` triggers the load.
- **Delta scan already supplies per-file min/max.** `delta_lake_scan.py:378-412` populates
  each ScanTask's `stats` from the add actions' `min`/`max`. The `TODO(Clark)` at
  `delta_lake_scan.py:292` is only about pruning the *action fetch* — not a blocker.
- **`daft-stats` supports only a narrow expression set.** `table_stats.rs:126-160`:
  `Alias`, `Column`, `Literal`, `Not`, `Cast`, and `BinaryOp` over
  `Lt/LtEq/Eq/NotEq/GtEq/Gt/Plus/Minus/And/Or`. Everything else falls to
  `ColumnRangeStatistics::Missing` → `TruthValue::Maybe` (`column_stats/mod.rs:110`).
  **`is_in` prunes nothing.** Range predicates (`>=` / `<=` joined by `&`) are required.
- **The join-level stats short-circuit is `Inner`-only.** `ops/join.rs:41` guards the whole
  range-pruning block with `if how == JoinType::Inner`, carrying
  `TODO(Kevin): short circuits are also possible for other join types`. The merge uses
  `left` + `anti`; neither benefits.
- **Nulls are not tracked in range stats.** `column_stats/mod.rs:166` computes `_null_count`
  and discards it. `hash_join` (`ops/join.rs:80-98`) defaults `null_equals_nulls` to
  `vec![false; n]` at `:88` but **does not pass it** to the private `join` at `:98`.

## Design

### Piece 1 — prune pass 1's target (Python, fork-local)

Build a conjunction of per-key-column range containments from the materialized source:

```
(col(k) >= lit(min_k)) & (col(k) <= lit(max_k))   for each k in self._on
```

**Correctness invariant:** the predicate is derived *solely from source key values*, and
any target row matching a source row on all keys necessarily satisfies every per-column
range containment. So pruning the target by it **cannot drop a row that would match**.
This is a strictly different property from the pass-2 partition prune, whose
`anti_join_target` warning at `_deltalake.py:1163-1166` exists precisely because
`affected` is *not* source-derived. The pass-1 prune may therefore also serve the ANTI
side without producing false inserts.

Unmatched target rows *can* be dropped — that is the point, and it is why metrics need
correcting.

**Metric accounting.** From `_deltalake.py:1603-1614`, with no by-source clauses, an
unmatched target row yields exactly `deleted=false`, `updated=false`, `inserted=false`,
`dropped_source=false`, `emitted=true`, `copied=true`. So each skipped row owes exactly
`+1 copied` and `+1 emitted`, and nothing else.

- **Partitioned target:** no correction needed. `num_copied`/`num_output` are already
  scoped to `affected` groups (`:1748-1759`), and a pruned partition contains no modified
  rows, so it was never in `affected`. Pruning is a pure win.
- **Unpartitioned target:** `num_copied += skipped` and `num_output += skipped`, where
  `skipped = total_target_rows - kept_target_rows`, `total_target_rows` is the exact
  `sum(num_records)` from `get_add_actions()` (the pattern already exists at `:1835`), and
  `kept_target_rows` is counted in pass 1 as `sum(_TGT_MARKER.not_null())`.

**Gates — skip the prune entirely when any hold:**
- any `by_source_update` / `by_source_delete` clause is present (an unmatched target row's
  own values decide whether it is deleted or updated, so it must be read);
- `self._validate_unique_keys` is `False` (duplicate source keys fan out the LEFT join and
  break the `kept_target_rows` count);
- any key column's min or max is absent/NULL, or the source is empty;
- any key column's dtype is not an ordered scalar (restrict to numeric / string / bool /
  date / timestamp).

**Ceiling — state this plainly.** For an **unpartitioned** target, pass 2 does a *full
overwrite* (`:1826+`), so it must still see the complete target; `annotated` cannot be
built on the pruned frame. Pass 1 becomes cheap, pass 2 stays full. That is roughly
**2x**, not 100x — the same order as `materialize_join=True`, but composable and without
the memory cost. The 100x case for unpartitioned tables needs **Piece 3 (out of scope):
file-level replace in the write pass** — remove only the add-actions containing matched
keys and add their rewrites, instead of overwriting the table. For a **partitioned** target
whose keys are partition-correlated, Piece 1 plus the existing pass-2 prune means both
passes are pruned, which is the large win available today.

### Piece 2 — extend the join stats short-circuit to Left/Anti/Semi (Rust, upstreamable)

Hoist the `tv` computation out of `if how == JoinType::Inner` and act per join type on
`TruthValue::False` ("no pair of key ranges can overlap"):

| `how`   | Output on `False`                                    |
|---------|------------------------------------------------------|
| `Inner` | empty (existing)                                     |
| `Semi`  | empty — no left row has a match                      |
| `Anti`  | **all of left, unchanged** — `infer_join_schema` returns `left_schema.clone()` for `Anti`/`Semi` (`daft-dsl/src/join.rs:23`), so the schema already matches |
| `Left`  | **all left rows, right-only columns null-extended** — never empty |
| `Right` | mirror of `Left`                                     |
| `Outer` | out of scope (common columns take a supertype, `join.rs:39-43`) |

`Left` and `Anti` are exactly the merge's two joins, and they replace the per-row no-match
path in `join/left_right_join.rs:113-135` (per-row `add_nulls(1)` + `probe_side_idxs.push`
then a full `take`) with a single bulk construction.

**Null-safety gate (also fixes a latent `Inner` bug).** Range stats carry no null
information, so disjoint non-null ranges do *not* prove "no match" when
`null_equals_nulls` is set and both sides contain NULL keys — the NULL/NULL pair matches.
The private `join` cannot currently see the flag (`:98`). Thread
`null_equals_nulls: Option<&[bool]>` through and apply the short-circuit only when every
entry is `false`. This is a pre-existing hazard for `Inner`, so Piece 2's first task is a
failing test that proves it before anything is extended.

## Independence

Piece 1 is Python in `_deltalake.py`; Piece 2 is Rust in `daft-micropartition`. Neither
depends on the other and either can ship alone. Piece 1 helps only this merge; Piece 2
helps every `left`/`anti`/`semi` join in Daft and is upstreamable against
[#4047](https://github.com/Eventual-Inc/Daft/issues/4047), which is `help wanted` and
unassigned — the maintainer declined native merge for bandwidth reasons and pointed at
`DataSink` instead, so an engine-level join improvement is the more welcome contribution.

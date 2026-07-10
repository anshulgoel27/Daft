# Spatial Correctness Branch — Deferred Findings

Issues raised by task-scoped reviews on `fix/spatial-correctness` that were **not** fixed
in the task that surfaced them, plus pre-existing bugs discovered during execution.

**Status:** covers Tasks 1, 2, and 4 (reviewed & approved). Tasks 3 and 5 were still
implementing when this was written; Tasks 6–8 not yet started. Append their findings as
their reviews land.

**Severity** is the reviewer's, per the SDD rubric: *Important* = would block a merge;
*Minor* = polish. "Plan-mandated" means the plan's own text prescribed the thing the
reviewer flagged — those need a human decision, because the plan doesn't get to grade
its own work.

**Two items need a decision from you before merge:** [A1](#a1) and [C1](#c1).

---

## A. Design gaps in shipped code

### <a id="a1"></a>A1. No cap on the geohash covering-set size — a large query polygon stalls plan optimization
**Severity:** Important, plan-mandated · **From:** Task 4 review · **Decision needed**

**Where:** `src/daft-geo/src/st_geohash.rs` (the new `collect_covering_cells` flood fill),
consumed at `src/daft-logical-plan/src/optimization/rules/geohash_pruning.rs:151-153`.

**What:** The flood fill is bounded only by the query bbox's cell block at the rule's fixed
`DEFAULT_GEOHASH_PRECISION = 5` (`geohash_pruning.rs:90`). A query
`WHERE st_intersects(col, <continent-sized polygon literal>)` against a table carrying a
`{col}_geohash` column — which is the **default** optimizer pipeline — enumerates roughly
1M cells for a 60°×30° bbox, and up to ~33.5M (`32^5`) for a near-global one. Each is a
`String` in two `HashSet`s built at plan time, then `join("\n")`-ed into the plan text
(`geohash_pruning.rs:159`), then linearly rescanned per row by `geohash_in_covering_cells`.

**Why it matters:** Reachable from ordinary user input; produces a multi-minute, multi-GB
stall during *planning*, before any data is read.

**Not a regression.** The pre-fix code flooded an O(D²) Chebyshev disk *around* the start
cell, which was strictly worse. Task 4 improved the worst case from "unbounded slow" to
"bounded but huge." The plan simply never specified a cap.

**Fix shape (does not weaken correctness):** bail out of the flood once `cells.len()`
exceeds a few thousand and return an empty `Vec`. The consumer already handles this:
`geohash_pruning.rs:155-157` does `if geom.is_empty() { return None; }`, i.e. injects no
predicate and prunes nothing. Pruning is an optimization, so prune-nothing is a correct
result, not a silent wrong answer.

**Why it wasn't fixed:** the plan mandated the uncapped design, and changing the contract of
`geohash_covers_geometry` (it would now return empty for very large geometries) is a design
call, not a bug fix.

---

## B. Test coverage gaps

### B1. Two of Task 1's three new tests cannot fail on pre-fix code
**Severity:** Important, plan-mandated · **From:** Task 1 review

**Where:** `tests/expressions/test_spatial_join.py` —
`test_spatial_join_null_key_never_matches` and
`test_spatial_join_literal_none_string_key_does_not_match_null`.

**What:** Both pass identically before and after Task 1's fix, so neither pins the change.

**Root cause, and a correction to the original review's claim:** the pre-existing optimizer
rule `FilterNullJoinKey`
(`src/daft-logical-plan/src/optimization/rules/filter_null_join_key.rs`, registered in
`optimizer.rs`) inserts `Filter(key.is_null().not())` on *both* join children for any
standard, non-null-safe `=` equi-key, before `daft-local-plan`'s `translate.rs` ever runs.
NULL-keyed rows therefore never reach the buggy candidate-generation path via a plain `==`
join. **The `str_value` NULL-collision bug is only reachable through null-safe `<=>`
equality** — materially narrower than the incoming review asserted. The rule explicitly does
not apply to `EqNullSafe`.

**Fix shape:** add a test using `Expression.eq_null_safe` with the literal *string* `"None"`
on one side and a real NULL on the other, both spatially matching. Pre-fix, `str_value`
renders both as `"None"`, they collide into one partition group, and the spatial-only filter
emits a spurious joined row. Post-fix, the complete filter evaluates `key <=> rkey`, which is
false, and no row is emitted. Alias one side's column (e.g. `rkey`) to dodge the pre-existing
"Ambiguous column reference in join predicate" error — `test_spatial_join_partitioned_by_key`
in the same file shows the precedent.

**Why it wasn't fixed:** a fix pass was dispatched and you cancelled it. Nothing was left
behind — `src/` was verified byte-identical to the task commit afterward.

### B2. Task 1's dtype guard has no test that can fail without it
**Severity:** Important · **From:** Task 1 review

**Where:** `src/daft-local-plan/src/translate.rs`, the `partition_key` closure.

**What:** No test constructs a genuine cross-dtype equi-key. The one test that touches dtypes
(`test_spatial_join_literal_none_string_key_does_not_match_null`) explicitly `.cast()`s both
sides to the *same* dtype, so it can never exercise the guard.

**Fix shape:** join two spatially-matching rows whose equi-key columns hold equal values at
different dtypes (Int64 `1` vs Float64 `1.0`, forced with `.cast()`), so their `str_value`
renderings differ (`"1"` vs `"1.0"`). Pre-guard the rows land in different partition groups
and the row is lost; post-guard `partition_key = None` and the row joins. **If the planner
supertype-casts both sides before `translate.rs` runs, the guard may be unreachable** — in
that case record that finding rather than shipping a vacuous test.

### B3. Task 2's dtype-guard fall-through path has no test
**Severity:** Minor · **From:** Task 2 review

**Where:** `src/daft-distributed/src/pipeline_node/translate.rs:253-266`.

**What:** Nothing exercises mismatched equi-key dtypes under Ray to prove the
`TreeNodeRecursion::Continue` early-return yields correct results via the normal
key-normalizing join path. The reviewer verified by code reading that the early-return fires
before any state mutation and is byte-identical to `f_down`'s no-match fall-through, so the
join is translated rather than skipped — but no test locks that in.

### B4. Task 1's Filter-over-Join call site has no repo-tracked test
**Severity:** Minor · **From:** Task 1 review

**Where:** `src/daft-local-plan/src/translate.rs`, the `LogicalPlan::Filter` spatial-rewrite
branch (the `.and(strip_join_side_cols(on_expr))` fold). Verified only by an ad hoc script
during implementation.

### B5. An assumption shared by Tasks 1 and 2, untested on both paths
**Severity:** ⚠️ unverified · **From:** Task 2 review

Neither task tests that `ResolvedColumn::JoinSide(field, _)` always carries the
**post-deduplication** field name that `rebind_predicate` needs when both sides have a column
of the same name. Both fixes assume it. Worth one test through each path.

### B6. Task 4 edge cases untested
**Severity:** Minor, plan-mandated · **From:** Task 4 review

Single-cell bbox, degenerate (zero-area) point bbox, and the no-bounding-rect centroid
fallback at `src/daft-geo/src/st_geohash.rs:83-91`. The plan mandated the exact test bodies.

---

## C. Maintainability

### <a id="c1"></a>C1. `strip_join_side_cols` is verbatim-duplicated across crates
**Severity:** Important, plan-mandated · **From:** Task 2 review · **Decision needed**

**Where:** `src/daft-distributed/src/pipeline_node/translate.rs:110-124` is a byte-for-byte
logic copy of the private helper at `src/daft-local-plan/src/translate.rs:75-84`.

**Why it matters:** both call sites enforce the same correctness invariant (stripping
`JoinSide` markers so a join predicate can be rebound against the join output schema). If
`ResolvedColumn` semantics ever change and only one copy is updated, the result is a silent
correctness bug, not a style problem.

**Recommendation (reviewer's, which I endorse):** hoist it into `daft-dsl`, beside
`unresolved_col`. The function depends only on `daft-dsl` types, so that is its natural home
— better than exporting it from `daft-local-plan` (which `daft-distributed` already depends
on, so either would work structurally).

**Why it wasn't fixed:** the plan explicitly mandated "write a local copy," because the
original is private.

### C2. Task 4's `visited` set doubles peak string memory
**Severity:** Minor, plan-mandated · **From:** Task 4 review

**Where:** `src/daft-geo/src/st_geohash.rs`, `collect_covering_cells`.

`visited` ends up holding exactly the same `String`s as `cells` (every enqueued cell is
eventually dequeued and inserted). Inserting into `cells` at enqueue time and using *it* as
the visited check would halve peak storage. Compounds with [A1](#a1).

### C3. Cosmetic inconsistencies
**Severity:** Minor

- Task 2's copy of `strip_join_side_cols` uses a function-local `use` plus a fully-qualified
  path, where its sibling copy uses top-level imports. Semantics identical.
- Two defensive guards in Task 4's flood fill cannot fire with the current `geohash` crate:
  `neighbor.len() != precision` (the crate's `neighbor` always re-encodes at input length)
  and the `let Ok(neighbors) = ... else` (inputs here are always valid hashes). Harmless.
- Task 2's reported Ray run shows an unidentified `1 warning`. Almost certainly pre-existing
  amid 11 panicking tests, but never named. Test output should be pristine.
- `geohash_covers_geometry` returns a `Vec<String>` in `HashSet` iteration order, which is
  `join("\n")`-ed into the plan string at `geohash_pruning.rs:159` — so identical queries can
  produce different plan text. Pre-existing; unchanged by Task 4.
- The workspace build carries ~12 pre-existing warnings in `common/error`, `daft-core`, and
  `daft-dsl`. None attributable to this branch.

---

## D. Pre-existing bugs found during execution — NOT part of this branch

### D1. Distributed spatial joins panic in their most natural syntax
**Severity:** Important (a crash, not silent corruption) · **Predates the branch**

**Where:** `src/daft-distributed/src/pipeline_node/join/translate_join.rs:396` —
`todo!("FLOTILLA_MS?: Implement non-equality joins")`.

**What:** Under `DAFT_RUNNER=ray`, `df.join(other, on=(a == b) & st_intersects(...))` panics.
The distributed spatial rewrite in `f_down`
(`src/daft-distributed/src/pipeline_node/translate.rs`) only matches the shape
`Filter(spatial) → Join(equi)`; it never matches a bare `Join` carrying a residual predicate.
Such a join falls through to the generic distributed translator, which hits the `todo!()` as
soon as `split_eq_preds` leaves a non-empty residual.

**Verified present at `main` (94909e69b)**, so it predates this work entirely. It is why 11 of
12 tests in `tests/expressions/test_spatial_join.py` currently fail under Ray, and why Task 2's
regression test had to be written with the `.where(spatial)` shape instead.

**Impact:** distributed spatial joins are effectively unusable as normally written. Needs its
own task — folding it into this branch silently would be wrong.

**Fix shape:** teach `f_down` to also match a bare `Join` whose `on` splits into equi-keys plus
a spatial residual, routing it to `SpatialHashJoinNode` the same way the `Filter`-wrapped shape
is. Note this interacts with Task 2: the rewrite must keep carrying the **full** ON predicate
into the local NLJ filter.

---

## Corrections to the original incoming review

The document this branch was planned from (`~/Downloads/2026-07-10-spatial-review-fixes.md`)
overstated two findings. Recorded here so they aren't re-litigated:

1. **NULL-key collision** is reachable only via null-safe `<=>` equality, not plain `=`,
   because `FilterNullJoinKey` strips NULL-keyed rows first. See [B1](#b1).
2. **Collocated-join `continue` on a partition-value miss** is *correct* for the Inner-only
   rewrite — an absent partition value genuinely contributes zero rows. Only the *systematic*
   all-miss case (e.g. dtype-divergent `PartitionSpec`s that can never compare equal) is a bug,
   and only when a left unkeyed group lets it bypass the `len == 1` guard. Task 5 fixes exactly
   that, and no more.

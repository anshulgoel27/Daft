# Spatial Correctness Branch — Findings Ledger

Complete record of review findings across all of `fix/spatial-correctness`: what was
**fixed on the branch**, what is **deferred for a decision or a follow-up**, and the
**pre-existing bugs discovered during execution** that are out of this branch's scope.

**Status:** all 8 planned tasks + 5 follow-up fixes implemented, committed, individually
reviewed (Tasks 3–8 and every follow-up reviewed by Fable 5), and integrated-verified
(full `make build` clean; native `test_spatial*.py` + `test_geoparquet.py` = 96 passed /
1 skipped / 1 xfailed; `cargo test` across the six touched crates green). A final
whole-branch integration review is the last gate.

**Severity** is the reviewer's, per the SDD rubric: *Important* = would block a merge on
its own; *Minor* = polish. "Plan-mandated" = the plan's own text prescribed the flagged
behavior, so it needed a human decision rather than an implementer fix.

---

## Branch commits (main = 94909e69b)

| SHA | What |
|-----|------|
| `7080a579d` | Task 1 — local NLJ filter carries the full ON predicate |
| `a1c58f97e` | Task 2 — distributed spatial hash join enforces equi-keys in the filter |
| `26300e7fd` | Task 4 — geohash covering set complete + tight (flood fill) |
| `0b4423f86` | Task 3 — refuse R-tree acceleration for st_disjoint / negated / OR predicates |
| `8d08f7747` | Task 5 — collocated-join symmetric unkeyed handling + systematic-mismatch abort |
| `733715e07` | Task 3 fixup — xfail the negated-spatial-predicate join test (unrelated pre-existing bug) |
| `e9b526a2a` | Task 6 — reject casting non-Binary sources to Geometry |
| `3ba1d814a` | Task 7 — warn on non-default GeoParquet CRS |
| `e462b5788` | Task 3b (review-driven) — derive st_dwithin R-tree pad from the accelerated node |
| `5b5696b59` | Task 8 — read the real `_spatial_index.idx` format, per directory |
| `ae01f2192` | Fix A1 — cap the geohash covering set, decline pruning for oversized geometries |
| `d805f9117` | Fix B1/B2 — pin null-safe key collision + cross-dtype equi-key fallback |
| `1e9420349` | Fix Task-8 — keep scan tasks when their H3 index cells fail to parse |
| `c4093bb27` | Fix C1 — hoist `strip_join_side_cols` into `daft-dsl`, remove the cross-crate duplicate |

Plus two docs commits (`ee910f4b0` plan, `15259956e` an earlier snapshot of this ledger).

---

## FIXED on this branch (was flagged, now resolved)

### ✅ Task-3b — `st_dwithin` pad divergence (was a NEW hole this branch introduced)
Task 3 made geom-extraction `And`-only, but the separate `extract_dwithin_distance` walk
still recursed through `Not`/any-`BinaryOp`. For `(st_dwithin(a,b,5) | cond) & st_dwithin(c,d,100)`
the join accelerated on the within-100 node but padded the R-tree query box by 5, silently
dropping pairs 5–100 apart. **Fixed** in `e462b5788`: one unified walk returns
`(build_col, probe_col, pad)`, so geom columns and pad come from the same node by
construction; a non-literal / negative / non-finite distance now refuses acceleration
(falls back to the complete-filter scan) instead of padding by 0.0. Reviewer ran the tests
(9 passed), confirmed the old function is deleted with zero remaining references.

### ✅ A1 — geohash covering set had no size cap
A `st_intersects` query with a continent-sized polygon enumerated ~1M cells (up to ~33.5M
near-global) at plan time, as Strings spliced into the plan text — stalling *planning*.
**Fixed** in `ae01f2192`: capped at 4096 cells; on overflow the covering set is returned
EMPTY (all-or-nothing — never partial, which would be unsound pruning), and the consumer
already treats empty as "don't prune." Verified the capped path discards the partial set.

### ✅ C1 — `strip_join_side_cols` duplicated across crates
Was byte-for-byte duplicated in `daft-local-plan` and `daft-distributed`; drift would have
been a silent correctness bug since both enforce the same invariant. **Fixed** in
`c4093bb27`: hoisted to `daft_dsl::join::strip_join_side_cols`, both copies deleted, single
definition, no unused-import warnings.

### ✅ B1/B2 — two Task-1 tests couldn't fail on pre-fix code; dtype guard untested
**Fixed** in `d805f9117`. Two new tests, both genuine RED pre-fix:
- `test_spatial_join_null_safe_key_collision` — uses `eq_null_safe` (which the pre-existing
  `FilterNullJoinKey` rule does NOT strip, unlike plain `=`), literal string `"None"` vs a
  real NULL. Pre-fix the `str_value` collision emitted a spurious row; post-fix the complete
  filter's `key <=> rkey` rejects it.
- `test_spatial_join_cross_dtype_equi_key` — the dtype guard. **Useful negative result:**
  Int64/Float64 does NOT reach the guard (Rust Display renders `1.0f64` as `"1"`, identical
  to `1i64`); Date-vs-Timestamp does. So the guard is reachable from Python, just not via
  the numeric pair the plan first suggested.

### ✅ Task-8 conservative-keep gap — corrupt H3 cells could prune
`parse_h3_cells` silently drops unparseable cell strings, so a corrupt sidecar shrank a
file's coverage and could cause it to be *pruned* — silent row loss on corrupt data.
**Fixed** in `1e9420349` (design A, skip-file): when a file's parsed cell count ≠ its
recorded count, that file's entry is omitted, so the lookup misses and the existing
"basename absent → keep" path fires. Corrupt cells now KEEP the task. Needs a malformed
sidecar to trigger (the shipped writer only emits valid hex).

### ✅ Task-8 third bug — `file://` path prefix (found during implementation)
Scan-task paths are canonicalized to `file:///…` URIs, which `std::fs` can't resolve — so
even after the format + per-directory fixes the pruning rule would have stayed dead for
real local-FS usage. Fixed in `5b5696b59` with a `local_fs_path` helper mirroring the
codebase's existing `daft_io::strip_file_uri_to_path`.

---

## DEFERRED — needs your decision or a follow-up (NOT fixed on this branch)

### ⚠️ Task-7 — CRS warning false-positives on PROJJSON-object CRS (you chose to defer)
**Severity:** Important, plan-mandated · `src/daft-schema/src/geo_metadata.rs:113-118`
GeoParquet's spec form for `crs` is a PROJJSON **object**, and GeoPandas embeds the full
object even for plain WGS84. Our `non_default_crs_columns` stringifies any non-string `crs`
and compares to `"OGC:CRS84"`, so it flags EVERY GeoPandas-written default-CRS file, and
`other.to_string()` dumps 1–2 KB of JSON into each log line. Two sub-parts:
1. Inspect the PROJJSON `id` member (`authority == "OGC" && code == "CRS84"`, arguably also
   `CRS84h` / `EPSG:4326`) and treat it as default before flagging; log the extracted
   `authority:code`, not the raw blob.
2. The function also skips `detect_geo_columns`'s WKB-encoding and schema-presence filters,
   so it warns for native-GeoArrow-encoded columns Daft never reads as Geometry, and for
   metadata columns absent from the file schema. Add `.filter(encoding == WKB)` inside, or
   intersect with the already-computed `geo_cols` at each call site.
**You elected to leave this for yourself.** The axis-order question (EPSG:4326 lat/lon vs
CRS84 lon/lat) is a genuine open design question worth a separate look either way.

### Minor / polish — safe to defer, none block merge
- **Per-file CRS warning volume** (Task 7): one `log::warn!` per offending column per file;
  a 10k-file projected dataset emits ~10k lines. Confirmed file-level, not per-batch.
  Consider hoisting to once-per-scan — compounds with the wall-of-JSON above if unfixed.
- **`optimizer.rs:248-249` stale comment** (near Task 8's edit): still says "R-tree index"
  / "per-file MBR" — wrong for the H3 design. Outside the grep mandate, pre-existing.
- **Geohash `visited` doubles peak String memory** (Task 4): inserting into `cells` at
  enqueue time and using it as the visited check would halve it.
- **Example leans on private `idx._load_full()`** (Task 8): `SpatialIndex` wants a public
  eager-load accessor; user-facing example code shouldn't touch a private method.
- **dwithin boundary tests** (Task 3b): no test pins the `0.0`-distance-accelerated boundary
  or NaN/±inf refusal (both are code-covered, not test-covered).
- **Collocated-join `l_keyed_tasks` order is HashMap-nondeterministic** (Task 5): affects
  sub-join/Concat order, not the row set; benign today (no ordering guarantee, no snapshot
  tests). Sort deterministically if plan-snapshot tests are ever added.
- **Task-5 `l_keyed_tasks` materialization untested** (Task 5): the 7 unit tests pin the
  pure `plan_sub_joins`, but a regression substituting `l_tasks` for `l_keyed_tasks` in the
  materialization would pass them. A mixed-hive-layout integration test would close it.
- **Empty `task.sources` → prune** (Task 8): harmless (zero-source task carries no rows) but
  an explicit `if task.sources.is_empty() { return true; }` would make the invariant airtight.
- **Cosmetic** (Task 6): `cast.rs` Geometry error uses `{:?}` while the sibling
  `GeometryArray::cast` error uses `{}`; minor inconsistency.

---

## PRE-EXISTING bugs found during execution — NOT this branch's scope, need their own tasks

### D1. Negating any spatial function is broken in Daft (a crash / hang)
**Severity:** Important · predates the branch (reproduced on `main`).
`~st_intersects(a, b)` raises `DaftCoreException: Mismatch of expected expression name and
name from computed series (g vs st_intersects)` from `daft-recordbatch` — even OUTSIDE a
join and even over materialized, non-aliased columns — while plain `st_intersects(a,b)` and
`~bool_col` both work. Inside a nested-loop join the same defect *deadlocks* instead of
erroring. This is why `test_python_join_not_st_intersects` is `xfail`ed (`733715e07`). It
also means Task 3's negation-refusal guards a path no user can currently execute — correct
to have, but not independently reachable until this is fixed. **Fix shape:** the
`Not`-over-`ScalarFn` name-resolution mismatch in expression evaluation; remove the xfail
once fixed (the assertions are already correct).

### D2. Distributed spatial joins panic in their most natural syntax
**Severity:** Important (a crash, not silent corruption) · predates the branch (on `main`).
Under Ray, `df.join(other, on=(a == b) & st_intersects(...))` hits
`todo!("FLOTILLA_MS?: Implement non-equality joins")` at
`src/daft-distributed/src/pipeline_node/join/translate_join.rs:396`, because the distributed
spatial rewrite only matches `Filter(spatial) → Join(equi)`, never a bare `Join` carrying a
residual predicate. This is why 11/12 tests in `tests/expressions/test_spatial_join.py` fail
under Ray today, and why Task 2's regression test uses the `.where(spatial)` shape.
**Fix shape:** teach the distributed `f_down` to also match a bare `Join` whose `on` splits
into equi-keys + a spatial residual, routing it to `SpatialHashJoinNode` — while keeping the
full ON predicate in the local NLJ filter (the Task 1/2 invariant).

---

## Corrections to the original incoming review

The document this branch was planned from (`~/Downloads/2026-07-10-spatial-review-fixes.md`)
overstated two findings; recorded so they aren't re-litigated:
1. **NULL-key collision** is reachable only via null-safe `<=>`, not plain `=` — the
   pre-existing `FilterNullJoinKey` rule strips NULL keys first for `=`. (See B1 above.)
2. **Collocated-join `continue` on a partition-value miss** is *correct* for the Inner-only
   rewrite; only the systematic all-miss case (dtype-divergent PartitionSpecs) combined with
   a left unkeyed group is a bug. Task 5 fixes exactly that and no more.

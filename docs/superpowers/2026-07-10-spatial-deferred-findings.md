# Spatial Correctness Branch — Findings Ledger

Complete record of the `fix/spatial-correctness` branch: the bugs fixed, the two
pre-existing crash bugs also fixed here at the user's request, and the residual
follow-up items (all Minor) that remain.

**Status:** all 8 planned tasks + follow-up fixes committed and individually reviewed
(Tasks 3–8 and every follow-up reviewed by Fable 5; a whole-branch integration review
also ran). Two originally out-of-scope pre-existing bugs (D1, D2) and the deferred Task 7
CRS item were subsequently fixed on this branch, along with all nine Minor polish items.
26 commits total (24 code/test + 2 docs).

**Severity** is the reviewer's, per the SDD rubric: *Important* = would block a merge on
its own; *Minor* = polish.

---

## What this branch fixes

### Core correctness (the 10 original findings → 8 tasks)
| Area | Commit |
|------|--------|
| Local NLJ carries the full ON predicate (fixes dropped equi-keys) | `7080a579d` |
| Distributed spatial hash join enforces equi-keys in the filter | `a1c58f97e` |
| Geohash covering set complete + tight (flood fill) | `26300e7fd` |
| Refuse R-tree acceleration for st_disjoint / negated / OR predicates | `0b4423f86` |
| Collocated-join symmetric unkeyed handling + systematic-mismatch abort | `8d08f7747` |
| Reject casting non-Binary sources to Geometry | `e9b526a2a` |
| Warn on non-default GeoParquet CRS | `3ba1d814a` |
| Read the real `_spatial_index.idx` format, per directory | `5b5696b59` |

### Review-driven correctness fixes (found while executing the plan)
| What | Commit |
|------|--------|
| Derive the st_dwithin R-tree pad from the accelerated node (closes a pad/geom-node divergence Task 3 widened) | `e462b5788` |
| Cap the geohash covering set; decline pruning for oversized geometries | `ae01f2192` |
| Keep scan tasks when their H3 index cells fail to parse (conservative-keep) | `1e9420349` |
| Task 8's third bug: `file://` path prefix (rule was dead for local FS) | (in `5b5696b59`) |

### Test-coverage and maintainability follow-ups
| What | Commit |
|------|--------|
| Pin null-safe key collision + cross-dtype equi-key fallback | `d805f9117` |
| Hoist `strip_join_side_cols` into `daft-dsl` (remove cross-crate duplicate) | `c4093bb27` |
| Pin st_dwithin zero-distance acceleration + non-finite refusal | `dbc180bc5` |
| Geometry-cast error uses Display not Debug | `a50a176c6` |

### Two pre-existing bugs fixed at user request (were "out of scope", section D below)
| What | Commit |
|------|--------|
| **D1** — negating any spatial ScalarFn crashed/deadlocked | `327c08a3d` |
| **D2** — distributed `on=(a==b) & st_intersects(...)` panicked | `8b2302580` |

### Task 7 CRS follow-up (was the one deferred item)
| What | Commit |
|------|--------|
| Recognize PROJJSON-object default CRS; only warn for WKB geometry columns actually read; log compact `authority:code` not the raw blob | `1f13de799` |

### The nine Minor polish items (all fixed)
`ab4a00290` geohash `visited` memory · `606075d69` + `236e6aa7c` collocated determinism +
materialization test + honest tie-break comment · `e39fcaed4` optimizer stale comment +
empty-sources keep guard + public `SpatialIndex.load_full()` accessor · `dbc180bc5` dwithin
boundary tests · `a50a176c6` cast Display · (Task-5 materialization test and Task-4 memory
also covered above).

---

## D1 — negating any spatial function (FIXED, `327c08a3d`)

Was: `~st_intersects(a,b)` raised `ComputeError: Mismatch of expected expression name...
(g vs st_intersects)` from `daft-recordbatch`, even outside a join; deadlocked inside one.
Root cause: `Not`/`IsNull`/`NotNull` `to_field` derived their output NAME from `expr.name()`,
which for a builtin ScalarFn returns the FIRST ARGUMENT's name, while evaluation names the
series after the function's return field. Fix: derive the wrapper's output name from the
child's actual evaluated field name (`expr.to_field(schema)?.name`). `Expr::name()`'s
first-arg semantics (intentional across the fluent API, and `#[deprecated]`) were left
untouched. Reviewer confirmed the blast radius collapses to previously-broken inputs: the
recordbatch eval check enforces `to_field name == computed name` at every node, so any input
whose column name changes under the fix is one that previously errored/hung. 28,202
`tests/expressions/` tests pass; the `test_python_join_not_st_intersects` xfail was removed
and it now passes.

**Integration fallout (found + fixed, `09b5317c8`):** making `IsNull`/`NotNull::to_field`
validate their child — which `Not::to_field` already did on `main` — surfaced ONE internal
test, `push_down_filter::filter_with_udf_not_pushed_down_into_scan`, that built a
`url_download` UDF with an invalid signature (Int64 input where a string is required, plus
bogus literals for optional args). Dormant only because `is_null().to_field()` never
recursed before. Fixed the test to use a well-formed `url_download(col_b)`; no production
change. A full-crate Rust sweep (daft-dsl, -logical-plan, -local-plan, -sql, -recordbatch,
-distributed, -scan, -core, -schema, -geo, -local-execution) found no other fallout. The
eager validation is correct fail-loud behavior: an expression whose `to_field` fails cannot
be evaluated, so `is_null` of it is equally ill-formed.

## D2 — distributed spatial join panic on the natural syntax (FIXED, `8b2302580`)

Was: under Ray, `df.join(other, on=(a==b) & st_intersects(...))` panicked at
`translate_join.rs:396` (`todo!("Implement non-equality joins")`), because the distributed
rewrite only matched `Filter(spatial) → Join(equi)`, never a bare `Join` carrying an equi +
spatial-residual `on`. Fix: `f_down` now also routes a bare `Join(equi + spatial residual,
Inner)` to `SpatialHashJoinNode`, carrying the FULL `on` (equi AND spatial) into the local
NLJ filter — the equi-key is enforced there, not by hash-partitioning. Ray
`test_spatial_join.py` went 3→9 passing; native stayed green; the new Ray test pins key
enforcement (200 distinct keys, spatial universally true, asserts pid==qid and exactly N rows).

**Residual limitation (NOT a regression — a genuinely unimplemented feature):** the remaining
11 Ray failures all hit the same `todo!()`, for two out-of-scope reasons the fix deliberately
does not cover:
1. **Distributed spatial-only joins** (no equi-key at all — pure `on=st_intersects(...)` or
   OR-composed): with no key to hash-partition by, these need a distributed
   broadcast/cross-join spatial strategy — a real new feature. The failure stays LOUD (panic),
   not silently wrong. Needs its own task.
2. **Distributed cross-dtype equi-key**: hits the pre-existing dtype-mismatch guard, which
   correctly refuses to hash-partition mismatched-dtype keys and falls through to the same
   `todo!()`. Separate unimplemented feature; also loud.

---

## Remaining follow-ups — all Minor, none block merge

**Task 7 CRS (new sub-items found reviewing the fix):**
- Legacy GeoParquet 0.x used WKT2 *strings* for `crs`; a string-form non-default value is not
  length-capped, so a ~1 KB WKT2 string would go whole into a log line. One-line fix
  (`truncate_for_log`). Object-form blobs ARE already capped.
- PROJJSON permits an `ids` (plural) array; only singular `id` is recognized. pyproj/PROJ emit
  singular `id`, so exposure is near zero; a plural-only CRS84 would be flagged with a truncated
  blob (safe direction).
- `CRS84h` (3D) is recognized but untested; authority/code compare is case-sensitive (errs
  toward flagging = safe).
- Per-file warning volume: one warn per offending column per file (a 10k-file projected dataset
  emits ~10k lines). Deduped per-column within a file; a once-per-scan hoist is left as a
  documented `TODO`.

**D2 (from the review):**
- Mixed residuals like `(a.x < b.y) & st_intersects(...)` now route to the SHJ and execute
  correctly (the NLJ only accelerates AND-sound sub-predicates, else full-scan) — a free scope
  widening beyond the fix's stated scope. Correct; worth a documenting test.
- The new Ray test's "collisions guaranteed" claim relies on the default partition count; pinning
  a small `default_num_partitions` would make it unconditional. Same convention as the sibling
  Task-2 test.
- `strip_join_side_cols` + rebind-by-name against `join.output_schema` could mis-bind if both
  sides carry identically-named columns. PRE-EXISTING and shared with the Filter-shape path AND
  the native runner's helper — out of scope here, worth a tracked follow-up across all three.

**Task 8 (from the review):**
- The empty-sources unit-test fixture is a struct literal, so it breaks compilation if `ScanTask`
  gains a field (acknowledged trade-off).
- An empty-index `.idx` sidecar (zero rows) leaves the lazy `file_h3_cells` truthiness-guard
  falsy, so the property re-reads the Parquet on every access. Pre-existing guard; a `_loaded:
  bool` flag fixes it. Empty sidecars are a corner case.

**D1 (from the review):** `IsNull`/`NotNull` `to_field` arms are verbatim duplicates (could be a
combined match arm); the Rust unit test pins `to_field` alignment while the eval-side pinning
lives in the Python select test (structural — Not/IsNull eval lives in `daft-recordbatch`).

**Collocated (already fixed):** the tie-break "unique across groups" comment overstated a
type-level guarantee; softened in `236e6aa7c` to note a true tie falls back to stable-sort =
HashMap order (unreachable for real hive layouts).

**Geohash cap (pre-existing design note):** the cap is `MAX_COVERING_CELLS = 4096`, all-or-empty;
this is the tightening that fixed the plan's original uncapped design.

---

## Corrections to the original incoming review

The document this branch was planned from (`~/Downloads/2026-07-10-spatial-review-fixes.md`)
overstated two findings:
1. **NULL-key collision** is reachable only via null-safe `<=>`, not plain `=` — the pre-existing
   `FilterNullJoinKey` rule strips NULL keys first for `=`. Pinned by the null-safe test in
   `d805f9117`.
2. **Collocated-join `continue` on a partition-value miss** is *correct* for the Inner-only
   rewrite; only the systematic all-miss case (dtype-divergent PartitionSpecs) combined with a
   left unkeyed group is a bug. `8d08f7747` fixes exactly that and no more.

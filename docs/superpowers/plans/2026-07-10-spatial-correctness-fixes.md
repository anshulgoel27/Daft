# Spatial Correctness Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the verified spatial correctness bugs on `main` (join engine dropping/fabricating rows, geohash pruning dropping rows, collocated-join row loss, unchecked Geometry casts, silent CRS misinterpretation, dead spatial-index pruning rule) so every spatial query either returns the correct result or fails loudly.

**Architecture:** This plan supersedes the earlier draft in `~/Downloads/2026-07-10-spatial-review-fixes.md` after adversarial verification of its findings. The central design change: instead of generalizing the nested-loop join's partition key to composite/NULL-aware keys across three crates (old Tasks 7–9, breaking signatures), we make the NLJ **filter complete** — it always carries the full join ON predicate (equality AND spatial), so every acceleration structure (R-tree, per-key grouping, hash shuffle) becomes a candidate-generation optimization that only has to be *superset-sound*, never semantics-bearing. This fixes the NULL-key collision, dropped composite equi-keys, and the distributed hash-collision bug in two small tasks with zero signature changes, and also closes a hole the old plan missed (equi-keys dropped when the spatial arg is an expression). The remaining tasks tighten unsound accelerations (st_disjoint/NOT/OR, geohash BFS), fix the collocated-join pairing logic (corrected to avoid the old plan's double-counting of unkeyed×unkeyed pairs), restrict the Geometry cast, warn on non-default CRS, and make the spatial-index pruning rule read the real `.idx` format per-directory.

**Tech Stack:** Rust (daft-geo, daft-core, daft-schema, daft-parquet, daft-local-plan, daft-local-execution, daft-logical-plan, daft-distributed), Python integration tests (pytest, `DAFT_RUNNER=native` / `ray`).

## Global Constraints

- All work happens on a new branch: `git checkout -b fix/spatial-correctness main` (first action, before Task 1).
- Rebuild after every Rust change that a Python test exercises: `make build`.
- Python tests: `DAFT_RUNNER=native make test EXTRA_ARGS="-v <path>"` (`DAFT_RUNNER=ray` only where a task says so).
- Rust unit tests: `cargo test -p <crate> <filter>`.
- Never weaken a correctness fix into a warning-only fallback: each fix either produces the correct result or fails loudly. The one deliberate exception is Task 7 (CRS), which is warn-only by design because a CRS-aware Geometry type is out of scope.
- Where a step notes "adjust per compiler feedback", the *shape* of the fix is the requirement; exact method/constructor names may drift — fix names, never semantics.
- PR titles use Conventional Commits (repo convention).

---

## Task 1: Make the local nested-loop-join filter carry the complete join predicate

Today `build_spatial_nested_loop_join` receives only the **spatial residual** as the NLJ filter ([translate.rs:651-665](../../src/daft-local-plan/src/translate.rs)); the equality conjuncts of `join.on` survive *only* as a single-column `partition_key` that silently becomes `None` for composite keys, expression keys, or unresolvable names ([translate.rs:140-161](../../src/daft-local-plan/src/translate.rs)) — the equality is then enforced nowhere. Even when the key resolves, grouping uses `str_value`, which renders NULL as the literal string `"None"`, so NULL keys on both sides collide and spurious NULL=NULL rows are emitted. Fix: pass the **full ON predicate** (equality AND spatial) as the NLJ filter. Grouping then becomes a pure candidate-pruning optimization: NULL collisions and unresolved keys generate at most *extra candidates*, which the equality in the filter rejects. Also add a dtype-equality guard on the partition key (grouping compares raw per-side `str_value`s; equal values of *different* dtypes can render differently and would split matching keys into different groups, losing rows).

**Files:**
- Modify: `src/daft-local-plan/src/translate.rs` (two call sites of `build_spatial_nested_loop_join` + the `partition_key` closure)
- Test: `tests/expressions/test_spatial_join.py`

**Interfaces:**
- Consumes: `ExprRef::and(self, other) -> ExprRef` (`src/daft-dsl/src/expr/mod.rs:1531`), `join.on.inner() -> Option<&ExprRef>`, existing `strip_join_side_cols` / `rebind_predicate` helpers in the same file.
- Produces: no signature changes. The `BoundExpr` filter stored on `LocalPhysicalPlan::NestedLoopJoin` now contains the full ON predicate; Task 3's operator changes and Task 2's distributed changes rely on this invariant ("the NLJ filter is complete").

- [ ] **Step 1: Write the failing tests**

Add to `tests/expressions/test_spatial_join.py`:

```python
def test_spatial_join_composite_equi_key():
    """Two equality columns + a spatial predicate must both be enforced, not just the spatial one."""
    pts = daft.from_pydict(
        {
            "region": ["a", "a", "b"],
            "zone": [1, 2, 1],
            "pid": [1, 2, 3],
            "x": [1.0, 1.0, 1.0],
            "y": [1.0, 1.0, 1.0],
        }
    ).select("region", "zone", "pid", st_point(daft.col("x"), daft.col("y")).alias("pg"))
    polys = daft.from_pydict(
        {
            "region": ["a", "b"],
            "zone": [1, 1],
            "qid": [10, 20],
            "wkt": ["POLYGON((0 0,2 0,2 2,0 2,0 0))", "POLYGON((0 0,2 0,2 2,0 2,0 0))"],
        }
    ).select(
        daft.col("region").alias("pregion"),
        daft.col("zone").alias("pzone"),
        "qid",
        st_geomfromtext(daft.col("wkt")).alias("qg"),
    )
    got = (
        pts.join(
            polys,
            on=(pts["region"] == polys["pregion"])
            & (pts["zone"] == polys["pzone"])
            & st_intersects(polys["qg"], pts["pg"]),
        )
        .select("pid", "qid")
        .sort(["pid", "qid"])
        .to_pydict()
    )
    # pid=2 (region=a, zone=2) has no polygon with zone=2 and must NOT match qid=10,
    # even though it is spatially inside qid=10's polygon.
    assert list(zip(got["pid"], got["qid"])) == [(1, 10), (3, 20)]


def test_spatial_join_null_key_never_matches():
    """A NULL equi-join key must never match another NULL key under standard `=` semantics."""
    left = daft.from_pydict(
        {"lid": [1, 2], "key": [None, "k"], "x": [1.0, 1.0], "y": [1.0, 1.0]}
    ).select("lid", "key", st_point(daft.col("x"), daft.col("y")).alias("lg"))
    right = daft.from_pydict(
        {"rid": [10, 11], "key": [None, "k"], "wkt": ["POLYGON((0 0,2 0,2 2,0 2,0 0))"] * 2}
    ).select("rid", "key", st_geomfromtext(daft.col("wkt")).alias("rg"))
    got = (
        left.join(right, on=(left["key"] == right["key"]) & st_intersects(right["rg"], left["lg"]))
        .select("lid", "rid")
        .sort(["lid", "rid"])
        .to_pydict()
    )
    assert list(zip(got["lid"], got["rid"])) == [(2, 11)]


def test_spatial_join_literal_none_string_key_does_not_match_null():
    """A key column holding the literal string 'None' must not join against NULL keys
    (today both str_value-render to "None" and collide into one group)."""
    left = daft.from_pydict(
        {"lid": [1], "key": ["None"], "x": [1.0], "y": [1.0]}
    ).select("lid", "key", st_point(daft.col("x"), daft.col("y")).alias("lg"))
    right = daft.from_pydict(
        {"rid": [10], "key": [None], "wkt": ["POLYGON((0 0,2 0,2 2,0 2,0 0))"]}
    ).select("rid", "key", st_geomfromtext(daft.col("wkt")).alias("rg"))
    right = right.with_column("key", right["key"].cast(daft.DataType.string()))
    got = (
        left.join(right, on=(left["key"] == right["key"]) & st_intersects(right["rg"], left["lg"]))
        .select("lid", "rid")
        .to_pydict()
    )
    assert got["lid"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `DAFT_RUNNER=native make test EXTRA_ARGS="-v tests/expressions/test_spatial_join.py::test_spatial_join_composite_equi_key tests/expressions/test_spatial_join.py::test_spatial_join_null_key_never_matches tests/expressions/test_spatial_join.py::test_spatial_join_literal_none_string_key_does_not_match_null"`
Expected: FAIL — composite returns an extra `(2, 10)` row (second equi-key silently dropped); NULL test returns an extra `(1, 10)` row (NULL keys collide via `str_value`); literal-"None" test returns an extra row.

- [ ] **Step 3: Implement the fix**

In `src/daft-local-plan/src/translate.rs`, three changes:

**(a) Join call site (~line 653)** — pass the full ON predicate instead of only the residual:

```rust
            if !remaining_on.is_empty() {
                let resid = remaining_on.inner().expect("non-empty residual has an expr").clone();
                if join.join_type == JoinType::Inner && is_spatial_predicate(&resid) {
                    // The NLJ filter is the ONLY place join semantics are enforced —
                    // candidate generation (R-tree, per-key grouping) is a superset
                    // optimization. Pass the FULL on-predicate (equality AND spatial),
                    // never just the residual: partition grouping may fail to resolve
                    // (composite keys, expression keys) and must not carry semantics.
                    let full_on = join.on.inner().expect("non-empty on has an expr").clone();
                    let predicate = strip_join_side_cols(full_on)?;
                    return build_spatial_nested_loop_join(
                        join,
                        predicate,
                        left_plan,
                        right_plan,
                        left_inputs,
                        join.stats_state.clone(),
                    );
                }
                return Err(DaftError::not_implemented("Execution of non-equality join"));
            }
```

**(b) Filter-over-Join call site (~line 286)** — conjoin the WHERE predicate with the join's ON (if any):

```rust
                if let Some(join) = maybe_join {
                    if join.join_type == JoinType::Inner {
                        let (left_plan, mut left_inputs) =
                            translate_helper(&join.left, source_counter, psets)?;
                        let (right_plan, right_inputs) =
                            translate_helper(&join.right, source_counter, psets)?;
                        left_inputs.extend(right_inputs);
                        // Fold the join's own ON condition (if any) into the NLJ filter
                        // so its equality conjuncts are actually evaluated — they are
                        // not enforced by candidate generation.
                        let mut predicate = filter.predicate.clone();
                        if let Some(on_expr) = join.on.inner() {
                            predicate = predicate.and(strip_join_side_cols(on_expr.clone())?);
                        }
                        return build_spatial_nested_loop_join(
                            join,
                            predicate,
                            left_plan,
                            right_plan,
                            left_inputs,
                            filter.stats_state.clone(),
                        );
                    }
                }
```

**(c) `partition_key` closure (~line 140)** — add the dtype guard after resolving indices:

```rust
        let probe_idx = phys_left.schema().get_index(&probe_col_name).ok()?;
        let build_idx = phys_right.schema().get_index(&build_col_name).ok()?;
        // Grouping compares raw per-side values via `str_value`. Equal values of
        // DIFFERENT dtypes can render differently (e.g. Date vs Timestamp), which
        // would place matching keys in different groups and silently drop rows —
        // now that the filter is complete, grouping must be match-preserving.
        if phys_left.schema().fields()[probe_idx].dtype
            != phys_right.schema().fields()[build_idx].dtype
        {
            return None;
        }
        Some([build_idx, probe_idx])
```

(If `Schema::fields()` is not the accessor, use whatever `geo_metadata.rs:39` uses — `schema.fields().iter()` exists there. Adjust per compiler feedback.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `make build`
Run: `DAFT_RUNNER=native make test EXTRA_ARGS="-v tests/expressions/test_spatial_join.py"`
Expected: PASS (all tests in the file, including the three new ones).

- [ ] **Step 5: Commit**

```bash
git add src/daft-local-plan/src/translate.rs tests/expressions/test_spatial_join.py
git commit -m "fix: carry the full join ON predicate in the spatial nested-loop-join filter"
```

---

## Task 2: Complete the filter in the distributed spatial hash join

`SpatialHashJoinNode::produce_tasks` builds the local NLJ with only the spatial predicate and `partition_key: None`, with a comment asserting hash partitions "already contain only rows with compatible partition keys" ([spatial_hash_join.rs:165-175](../../src/daft-distributed/src/pipeline_node/join/spatial_hash_join.rs)). Hash partitioning only co-locates *equal* keys; different keys routinely share a partition, and nothing re-checks the equality — false-positive joined rows. Same remedy as Task 1: fold the equi-conjuncts into the filter at rewrite time. Also guard the rewrite on matching key dtypes (mismatched dtypes could hash equal values to different partitions; the regular join path normalizes keys, this rewrite does not).

**Files:**
- Modify: `src/daft-distributed/src/pipeline_node/translate.rs` (the spatial rewrite branch of `f_down`, ~lines 206-355)
- Modify: `src/daft-distributed/src/pipeline_node/join/spatial_hash_join.rs` (comment only, ~line 170)
- Test: `tests/expressions/test_spatial_join.py` (Ray runner)

**Interfaces:**
- Consumes: Task 1's invariant is independent — this task establishes the same invariant on the distributed path. Uses `ExprRef::and`, `join.on.inner()`, and a local copy of `strip_join_side_cols` (the daft-local-plan one is private).
- Produces: `SpatialHashJoinNode.spatial_filter` now contains the full ON predicate conjoined with the WHERE spatial predicate. No signature changes.

- [ ] **Step 1: Write the failing test**

Add to `tests/expressions/test_spatial_join.py`:

```python
import os


@pytest.mark.skipif(
    os.environ.get("DAFT_RUNNER") != "ray",
    reason="SpatialHashJoinNode only exists in the distributed (Ray) pipeline",
)
def test_distributed_spatial_hash_join_does_not_hash_collide_different_keys():
    """Many distinct keys, few partitions: hash collisions are guaranteed. The equi-key
    must still be enforced, not just the spatial predicate."""
    n = 200
    pts = daft.from_pydict(
        {
            "key": [str(i) for i in range(n)],
            "pid": list(range(n)),
            "x": [1.0] * n,
            "y": [1.0] * n,
        }
    ).select("key", "pid", st_point(daft.col("x"), daft.col("y")).alias("pg"))
    # Every polygon covers the SAME area as every point, so the spatial predicate
    # alone matches everything — only the equi-key restricts matches.
    polys = daft.from_pydict(
        {
            "pkey": [str(i) for i in range(n)],
            "qid": list(range(n)),
            "wkt": ["POLYGON((0 0,2 0,2 2,0 2,0 0))"] * n,
        }
    ).select("pkey", "qid", st_geomfromtext(daft.col("wkt")).alias("qg"))

    got = (
        pts.join(polys, on=(pts["key"] == polys["pkey"]) & st_intersects(polys["qg"], pts["pg"]))
        .select("pid", "qid")
        .to_pydict()
    )
    assert sorted(got["pid"]) == list(range(n))
    assert all(pid == qid for pid, qid in zip(got["pid"], got["qid"]))
```

Note: the distributed rewrite fires on the pattern `Filter(spatial) → Join(equi)`, which the optimizer produces from this `on=` expression by splitting equality and residual conjuncts. If the test does not exercise `SpatialHashJoinNode` (check via `df.explain(True)` showing `SpatialHashJoin`), rewrite the query as an equi-join plus `.where(st_intersects(...))` — the semantics assertion stays identical.

- [ ] **Step 2: Run test to verify it fails**

Run: `DAFT_RUNNER=ray make test EXTRA_ARGS="-v tests/expressions/test_spatial_join.py::test_distributed_spatial_hash_join_does_not_hash_collide_different_keys"`
Expected: FAIL — far more than `n` rows returned; `pid == qid` violated (every point matches every polygon that hash-collided into its partition).

- [ ] **Step 3: Implement the fix**

**(a)** In `src/daft-distributed/src/pipeline_node/translate.rs`, add a private helper next to the existing `rebind_predicate` (copy of `daft-local-plan/src/translate.rs:75-84`; adjust imports — `daft_dsl::expr::{Column, ResolvedColumn}`, `common_treenode::{Transformed, TreeNode}`, `daft_dsl::unresolved_col`):

```rust
/// Convert `ResolvedColumn::JoinSide(field, _)` markers in a join predicate into
/// plain unresolved column references (by post-deduplication field name), so the
/// predicate can be re-bound against the join output schema.
fn strip_join_side_cols(expr: daft_dsl::ExprRef) -> DaftResult<daft_dsl::ExprRef> {
    Ok(expr
        .transform(|e| match e.as_ref() {
            Expr::Column(Column::Resolved(ResolvedColumn::JoinSide(field, _))) => {
                Ok(Transformed::yes(unresolved_col(field.name.clone())))
            }
            _ => Ok(Transformed::no(e)),
        })?
        .data)
}
```

**(b)** In the same file's `f_down` spatial branch, two edits. First, a dtype guard right after `split_eq_preds` (~line 233):

```rust
                    if remaining.is_empty() && !left_eq_keys.is_empty() {
                        // The rewrite hash-partitions each side by its own raw key.
                        // Mismatched key dtypes could hash equal values differently
                        // and the local filter can't recover cross-partition rows —
                        // fall through to the normal (normalizing) join translation.
                        let dtypes_match = left_eq_keys.iter().zip(right_eq_keys.iter()).all(
                            |(l, r)| {
                                match (
                                    l.to_field(&join.left.schema()),
                                    r.to_field(&join.right.schema()),
                                ) {
                                    (Ok(lf), Ok(rf)) => lf.dtype == rf.dtype,
                                    _ => false,
                                }
                            },
                        );
                        if !dtypes_match {
                            return Ok(TreeNodeRecursion::Continue);
                        }
```

Second, replace the `spatial_filter` construction (~line 247):

```rust
                        // The local NLJ evaluates ONLY this filter, and it is the only
                        // place join semantics are enforced: hash-partitioning co-locates
                        // equal keys but does NOT prevent different keys from sharing a
                        // partition. Carry the full ON predicate (equality conjuncts)
                        // alongside the WHERE spatial predicate.
                        let join_schema = join.output_schema.clone();
                        let full_on = join
                            .on
                            .inner()
                            .expect("non-empty equi join has an on expr")
                            .clone();
                        let combined =
                            filter.predicate.clone().and(strip_join_side_cols(full_on)?);
                        let spatial_filter = rebind_predicate(combined, &join_schema)?;
```

**(c)** In `src/daft-distributed/src/pipeline_node/join/spatial_hash_join.rs` (~line 170), correct the misleading comment:

```rust
                                None, // no per-key R-tree grouping in the distributed path;
                                      // the equi-keys are enforced by `spatial_filter`, which
                                      // carries the full ON predicate — hash-partitioning
                                      // alone does NOT prevent different key values from
                                      // sharing a partition.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `make build`
Run: `DAFT_RUNNER=ray make test EXTRA_ARGS="-v tests/expressions/test_spatial_join.py"`
Run: `DAFT_RUNNER=native make test EXTRA_ARGS="-v tests/expressions/test_spatial_join.py"`
Expected: PASS under both runners.

- [ ] **Step 5: Commit**

```bash
git add src/daft-distributed/src/pipeline_node/translate.rs src/daft-distributed/src/pipeline_node/join/spatial_hash_join.rs tests/expressions/test_spatial_join.py
git commit -m "fix: enforce equi-join keys in the distributed spatial hash join filter"
```

---

## Task 3: Stop R-tree-accelerating `st_disjoint`, negated, and OR-composed spatial predicates

R-tree candidate generation emits only bbox-**intersecting** pairs, and the filter runs only on candidates. That is a sound superset filter only for a predicate that (a) implies bbox intersection and (b) sits in an AND-conjunction. Three violations exist in `extract_from_expr` ([nested_loop_join.rs:77-142](../../src/daft-local-execution/src/join/nested_loop_join.rs)): `st_disjoint` is in `SPATIAL_FNS` (true pairs mostly have *non*-intersecting bboxes), `Expr::Not` recurses into the wrapped predicate (negation flips truth), and `Expr::BinaryOp` recurses through **any** operator including `Or` (a pair failing the spatial conjunct can still satisfy the other OR branch). All three silently drop true rows. Fix: remove `st_disjoint`, refuse `Not`, and recurse only through `And`. Non-accelerated predicates fall back to the plain nested-loop scan, which is correct (and, after Task 1, evaluates the complete filter).

**Files:**
- Modify: `src/daft-local-execution/src/join/nested_loop_join.rs` (`SPATIAL_FNS` const + `extract_from_expr` match arms)
- Test: `tests/expressions/test_spatial_join.py` and inline `#[cfg(test)]` in the same Rust file

**Interfaces:**
- Consumes: nothing new (`daft_core::prelude::Operator` for the `And` match — add to the existing `daft_core` import).
- Produces: `extract_geom_col_indices` (unchanged signature) now returns `None` for `st_disjoint`, any `Expr::Not(..)`, and any non-`And` binary composition — routing those joins to the plain nested-loop fallback.

- [ ] **Step 1: Write the failing Python tests**

Add to `tests/expressions/test_spatial_join.py`:

```python
def test_python_join_st_disjoint():
    """st_disjoint must not be R-tree-accelerated: its true matches typically have
    NON-intersecting bboxes, which bbox candidate generation silently drops."""
    from daft.functions import st_disjoint

    left = daft.from_pydict({"lid": [1, 2], "lx": [0.0, 100.0], "ly": [0.0, 100.0]}).select(
        daft.col("lid"), st_point(daft.col("lx"), daft.col("ly")).alias("lg")
    )
    right = daft.from_pydict({"rid": [10], "rx": [0.0], "ry": [0.0]}).select(
        daft.col("rid"), st_point(daft.col("rx"), daft.col("ry")).alias("rg")
    )
    result = (
        left.join(right, on=st_disjoint(left["lg"], right["rg"]))
        .select("lid", "rid")
        .sort("lid")
        .to_pydict()
    )
    # lid=1 coincides with rid=10 (not disjoint); lid=2 is far away (disjoint).
    assert result["lid"] == [2]
    assert result["rid"] == [10]


def test_python_join_not_st_intersects():
    """A negated spatial predicate must skip bbox-based R-tree acceleration."""
    left = daft.from_pydict({"lid": [1, 2], "lx": [1.0, 100.0], "ly": [1.0, 100.0]}).select(
        daft.col("lid"), st_point(daft.col("lx"), daft.col("ly")).alias("lg")
    )
    right = daft.from_pydict(
        {"rid": [10], "wkt": ["POLYGON((0 0,2 0,2 2,0 2,0 0))"]}
    ).select(daft.col("rid"), st_geomfromtext(daft.col("wkt")).alias("rg"))
    result = (
        left.join(right, on=~st_intersects(left["lg"], right["rg"]))
        .select("lid", "rid")
        .sort("lid")
        .to_pydict()
    )
    # lid=1 is inside the polygon -> excluded by NOT; lid=2 is far outside -> included.
    assert result["lid"] == [2]
    assert result["rid"] == [10]


def test_python_join_or_composed_spatial_predicate():
    """An OR-composed predicate must skip acceleration: a pair failing the spatial
    branch (and its bbox test) can still satisfy the other OR branch."""
    left = daft.from_pydict({"lid": [1, 2], "lx": [1.0, 100.0], "ly": [1.0, 100.0]}).select(
        daft.col("lid"), st_point(daft.col("lx"), daft.col("ly")).alias("lg")
    )
    right = daft.from_pydict(
        {"rid": [10], "wkt": ["POLYGON((0 0,2 0,2 2,0 2,0 0))"]}
    ).select(daft.col("rid"), st_geomfromtext(daft.col("wkt")).alias("rg"))
    result = (
        left.join(
            right,
            on=st_intersects(left["lg"], right["rg"]) | ((left["lid"] + 8) == right["rid"]),
        )
        .select("lid", "rid")
        .sort("lid")
        .to_pydict()
    )
    # lid=1 intersects; lid=2 does not intersect but satisfies lid+8 == rid (2+8=10).
    assert result["lid"] == [1, 2]
    assert result["rid"] == [10, 10]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `DAFT_RUNNER=native make test EXTRA_ARGS="-v tests/expressions/test_spatial_join.py::test_python_join_st_disjoint tests/expressions/test_spatial_join.py::test_python_join_not_st_intersects tests/expressions/test_spatial_join.py::test_python_join_or_composed_spatial_predicate"`
Expected: FAIL — each returns fewer rows than expected (true matches with non-intersecting bboxes never became candidates).

- [ ] **Step 3: Implement the fix**

In `src/daft-local-execution/src/join/nested_loop_join.rs`:

Remove `"st_disjoint"` from `SPATIAL_FNS`:

```rust
const SPATIAL_FNS: &[&str] = &[
    "st_intersects", "st_contains", "st_within", "st_covers",
    "st_covered_by", "st_touches", "st_overlaps",
    "st_crosses", "st_equals", "st_dwithin",
];
```

Replace the `BinaryOp` and `Not` arms of `extract_from_expr` (add `Operator` to the `daft_core::prelude` import):

```rust
        // Only an AND-conjunction preserves superset-soundness: every true pair
        // must satisfy the spatial conjunct, hence bbox-intersect. Under OR the
        // other branch can be true for pairs whose bboxes do NOT intersect, and
        // under NOT the truth of the wrapped predicate is inverted — in both
        // cases bbox candidate generation would silently drop true rows.
        Expr::BinaryOp { op: Operator::And, left, right } => {
            extract_from_expr(left, build_side, output_schema_len, build_n)
                .or_else(|| extract_from_expr(right, build_side, output_schema_len, build_n))
        }
        _ => None,
```

(This also subsumes the previous `Expr::Not(inner) => ...` arm — delete it. Leave `extract_dwithin_distance` unchanged: its padding only *enlarges* the query box, which is superset-sound.)

- [ ] **Step 4: Add Rust unit tests pinning the contract**

Add at the bottom of `src/daft-local-execution/src/join/nested_loop_join.rs`:

```rust
#[cfg(test)]
mod acceleration_tests {
    use daft_core::prelude::{DataType, Field, Schema};
    use daft_dsl::{expr::bound_expr::BoundExpr, unresolved_col};

    use super::*;

    fn geom_schema() -> Schema {
        Schema::new(vec![
            Field::new("a", DataType::Geometry),
            Field::new("b", DataType::Geometry),
        ])
    }

    fn extract(expr: daft_dsl::ExprRef) -> Option<(usize, usize)> {
        let bound = BoundExpr::try_new(expr, &geom_schema()).unwrap();
        extract_geom_col_indices(&bound, JoinSide::Left, 2, 1)
    }

    #[test]
    fn st_intersects_is_accelerated() {
        let e = daft_geo::st_intersects::st_intersects(unresolved_col("a"), unresolved_col("b"));
        assert!(extract(e).is_some());
    }

    #[test]
    fn st_disjoint_is_never_accelerated() {
        let e = daft_geo::st_disjoint::st_disjoint(unresolved_col("a"), unresolved_col("b"));
        assert!(extract(e).is_none());
    }

    #[test]
    fn negated_intersects_is_never_accelerated() {
        let e = daft_geo::st_intersects::st_intersects(unresolved_col("a"), unresolved_col("b"))
            .not();
        assert!(extract(e).is_none());
    }

    #[test]
    fn or_composed_intersects_is_never_accelerated() {
        let spatial =
            daft_geo::st_intersects::st_intersects(unresolved_col("a"), unresolved_col("b"));
        let e = spatial.or(unresolved_col("a").is_null());
        assert!(extract(e).is_none());
    }

    #[test]
    fn and_composed_intersects_is_still_accelerated() {
        let spatial =
            daft_geo::st_intersects::st_intersects(unresolved_col("a"), unresolved_col("b"));
        let e = spatial.and(unresolved_col("a").not_null());
        assert!(extract(e).is_some());
    }
}
```

(Free-function names `daft_geo::st_intersects::st_intersects` / `st_disjoint::st_disjoint` follow the crate's per-module pattern seen in `st_geohash.rs:71`; `.not()`/`.and()`/`.or()`/`.is_null()`/`.not_null()` are `ExprRef` combinators — adjust names per compiler feedback, not semantics.)

Run: `cargo test -p daft-local-execution acceleration_tests`
Expected: PASS (5 tests)

- [ ] **Step 5: Run the Python tests to verify they pass**

Run: `make build`
Run: `DAFT_RUNNER=native make test EXTRA_ARGS="-v tests/expressions/test_spatial_join.py"`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/daft-local-execution/src/join/nested_loop_join.rs tests/expressions/test_spatial_join.py
git commit -m "fix: refuse bbox R-tree acceleration for st_disjoint, negated, and OR-composed spatial join predicates"
```

---

## Task 4: Make the geohash covering set geometrically complete (and tight)

`collect_covering_cells` ([st_geohash.rs:114-150](../../src/daft-geo/src/st_geohash.rs)) BFS-walks from the bbox's min-corner cell and `break`s when it *dequeues* the max-corner cell — but cells are inserted into the result at dequeue time, so same-layer cells still in the queue are silently lost. Verified empirically: even a 2×2-cell bbox misses its SE cell (queue push order `[n, ne, e, se, ...]` dequeues NE before SE). The `GeohashPruning` rule (default optimizer pipeline) then drops rows whose centroid geohash falls in a missing cell. Bonus defect: neighbors are enqueued with no bbox filter, so the BFS floods an O(D²) disk around the start (a ~23×1-cell bbox yields 1,915 cells). Fix: replace the corner-hit BFS with a flood fill that only enqueues neighbors whose own decoded bbox intersects the query bbox — complete (bbox cells form a connected region) and tight (nothing outside the bbox).

**Files:**
- Modify: `src/daft-geo/src/st_geohash.rs:79-150`
- Test: inline `#[cfg(test)]` in the same file

**Interfaces:**
- Consumes: `geohash::decode_bbox` (returns the cell's `geo::Rect<f64>`; the `geohash` crate is already a dependency), `geo::Intersects`.
- Produces: `geohash_covers_geometry(g: &Geometry, precision: usize) -> Vec<String>` — same signature, now returns exactly the cells whose bbox intersects the geometry's bbox.

- [ ] **Step 1: Write the failing tests**

Add a `#[cfg(test)]` module at the bottom of `src/daft-geo/src/st_geohash.rs`:

```rust
#[cfg(test)]
mod covering_tests {
    use std::collections::HashSet;

    use geo::{Geometry, Intersects, Rect, coord};

    use super::*;

    #[test]
    fn covering_includes_all_cells_of_a_2x2_bbox() {
        // Straddles the precision-5 cell corner at 0.0439453125°, spanning 2x2 cells.
        // The old corner-hit BFS dequeues the NE (end) cell before the SE cell and
        // breaks with SE still in the queue — SE is silently missing.
        let rect = Rect::new(coord! { x: 0.02, y: 0.02 }, coord! { x: 0.06, y: 0.06 });
        let cells = geohash_covers_geometry(&Geometry::Rect(rect), 5);
        let got: HashSet<String> = cells.into_iter().collect();
        for (x, y) in [(0.02, 0.02), (0.06, 0.02), (0.02, 0.06), (0.06, 0.06)] {
            let h = geohash::encode(GeohashCoord { x, y }, 5).unwrap();
            assert!(got.contains(&h), "missing cell {h} containing bbox corner ({x},{y})");
        }
    }

    #[test]
    fn covering_includes_all_cells_of_a_wide_bbox() {
        // Multi-cell in both axes at precision 3: every cell a swept point hashes to
        // must be in the covering set.
        let rect = Rect::new(coord! { x: -10.0, y: -10.0 }, coord! { x: 10.0, y: 10.0 });
        let cells = geohash_covers_geometry(&Geometry::Rect(rect), 3);
        let got: HashSet<String> = cells.into_iter().collect();
        let mut x = -10.0;
        while x <= 10.0 {
            let mut y = -10.0;
            while y <= 10.0 {
                let h = geohash::encode(GeohashCoord { x, y }, 3).unwrap();
                assert!(got.contains(&h), "missing cell {h} for point ({x},{y})");
                y += 0.5;
            }
            x += 0.5;
        }
    }

    #[test]
    fn covering_cells_all_intersect_the_bbox() {
        // Tightness: the old BFS flooded an O(D²) disk far outside the bbox.
        let rect = Rect::new(coord! { x: -1.0, y: -1.0 }, coord! { x: 1.0, y: 1.0 });
        let cells = geohash_covers_geometry(&Geometry::Rect(rect), 4);
        assert!(!cells.is_empty());
        for cell in &cells {
            let cell_bbox = geohash::decode_bbox(cell).unwrap();
            assert!(
                cell_bbox.intersects(&rect),
                "cell {cell} does not intersect the query bbox"
            );
        }
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p daft-geo covering_`
Expected: `covering_includes_all_cells_of_a_2x2_bbox` FAILS (SE cell missing) and `covering_cells_all_intersect_the_bbox` FAILS (flooded cells outside the bbox). The wide-bbox test may pass or fail depending on queue order — either is acceptable at this step.

- [ ] **Step 3: Implement the fix**

Replace `geohash_covers_geometry` and `collect_covering_cells` in `src/daft-geo/src/st_geohash.rs` (add `use geo::Intersects;` to the imports):

```rust
/// Compute all geohash cells at `precision` that overlap with the given geometry's bounding box.
/// Used for automatic geohash-based partition pruning.
pub fn geohash_covers_geometry(g: &Geometry, precision: usize) -> Vec<String> {
    let bbox = match g.bounding_rect() {
        Some(b) => b,
        None => {
            if let Some(c) = g.centroid() {
                let coord = GeohashCoord { x: c.x(), y: c.y() };
                return encode(coord, precision).ok().into_iter().collect();
            }
            return vec![];
        }
    };

    let Ok(start_hash) = geohash::encode(
        GeohashCoord { x: bbox.min().x, y: bbox.min().y },
        precision,
    ) else {
        return vec![];
    };

    let mut cells = std::collections::HashSet::new();
    collect_covering_cells(&mut cells, &start_hash, &bbox, precision);
    cells.into_iter().collect()
}

/// Flood-fill outward from `start` (a cell known to overlap `bbox`), continuing
/// only into neighbor cells whose own decoded bounding rectangle intersects
/// `bbox`. Complete: the cells overlapping an axis-aligned bbox form a single
/// 8-connected region, all reachable from any member. Tight and terminating:
/// the flood never leaves the bbox's finite cell set. (The previous
/// implementation broke on dequeuing the max-corner cell, losing same-layer
/// cells still in the queue, and enqueued neighbors unfiltered, flooding an
/// O(D²) disk around the start.)
fn collect_covering_cells(
    cells: &mut std::collections::HashSet<String>,
    start: &str,
    bbox: &geo::Rect<f64>,
    precision: usize,
) {
    if start.is_empty() {
        return;
    }
    let mut queue = std::collections::VecDeque::new();
    let mut visited = std::collections::HashSet::new();
    queue.push_back(start.to_string());
    visited.insert(start.to_string());

    while let Some(current) = queue.pop_front() {
        cells.insert(current.clone());

        let Ok(neighbors) = geohash::neighbors(&current) else {
            continue;
        };
        for neighbor in [
            neighbors.n, neighbors.ne, neighbors.e, neighbors.se,
            neighbors.s, neighbors.sw, neighbors.w, neighbors.nw,
        ] {
            if neighbor.len() != precision || visited.contains(&neighbor) {
                continue;
            }
            let Ok(neighbor_bbox) = geohash::decode_bbox(&neighbor) else {
                continue;
            };
            if neighbor_bbox.intersects(bbox) {
                visited.insert(neighbor.clone());
                queue.push_back(neighbor);
            }
        }
    }
}
```

(The `end`/`max_hash` computation in the old `geohash_covers_geometry` is deleted — termination is bbox-driven now.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p daft-geo covering_`
Expected: PASS (3 tests).

Regression check on the consumers:

Run: `make build`
Run: `DAFT_RUNNER=native make test EXTRA_ARGS="-v tests/expressions/test_spatial_geohash_pruning.py"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/daft-geo/src/st_geohash.rs
git commit -m "fix: make the geohash bbox covering set complete and tight via bbox-bounded flood fill"
```

---

## Task 5: Fix collocated-join group pairing (symmetric unkeyed handling, systematic-mismatch abort)

Verified in [collocated_join.rs](../../src/daft-logical-plan/src/optimization/rules/collocated_join.rs): (a) left-side tasks with no resolvable partition value are joined against all right tasks (lines 196-208), but right-side unkeyed tasks appear in **no** sub-join — their rows silently vanish (reachable via mixed hive layouts, where per-file spec parsing yields keyed+unkeyed tasks in one source, or via differently-named partition columns, where *every* right task groups under `None`). (b) When both sides have keyed groups but **zero** values pair up (systematic mismatch — e.g. the sides inferred different partition-value dtypes, so no `PartitionSpec` can ever compare equal), keyed groups are silently skipped; combined with a left unkeyed group this bypasses the `len == 1` guard and drops all keyed left tasks. Note the correction versus the earlier draft plan: `continue` on a *partial* miss is **correct** for this Inner-only rewrite (an absent partition value contributes zero rows), so we do not abort on any single miss — only on the systematic all-miss case. Also note: the earlier draft's symmetric fix joined right-unkeyed against ALL left tasks, double-counting left-unkeyed × right-unkeyed pairs (both sub-joins would emit them) — here right-unkeyed pairs only with *keyed* left tasks.

**Files:**
- Modify: `src/daft-logical-plan/src/optimization/rules/collocated_join.rs:187-237`
- Test: inline `#[cfg(test)]` in the same file

**Interfaces:**
- Consumes: nothing new.
- Produces: a pure, generically-testable planner `fn plan_sub_joins<K: Eq + Hash + Clone>(l_keys: &[Option<K>], r_keys: &[Option<K>]) -> Option<Vec<SubJoin<K>>>` and `enum SubJoin<K> { LeftUnkeyedVsAllRight, RightUnkeyedVsKeyedLeft, KeyedPair(K) }`; `try_optimize_node` materializes its output.

- [ ] **Step 1: Write the failing tests**

Add a `#[cfg(test)]` module at the bottom of `src/daft-logical-plan/src/optimization/rules/collocated_join.rs` (this fails to compile until Step 3 introduces `plan_sub_joins`/`SubJoin`):

```rust
#[cfg(test)]
mod pairing_tests {
    use super::*;

    // K = i32 stands in for PartitionSpec: the planner is generic over the key.

    #[test]
    fn keyed_pairs_and_partial_overlap() {
        // Left has {1,2,3}, right has {2,3,4}: pairs {2,3}; 1 and 4 are genuine
        // inner-join misses and are correctly skipped (no abort).
        let l = vec![Some(1), Some(2), Some(3)];
        let r = vec![Some(2), Some(3), Some(4)];
        let plan = plan_sub_joins(&l, &r).expect("partial overlap must not abort");
        assert!(plan.contains(&SubJoin::KeyedPair(2)));
        assert!(plan.contains(&SubJoin::KeyedPair(3)));
        assert_eq!(plan.len(), 2);
    }

    #[test]
    fn right_unkeyed_tasks_are_joined_against_keyed_left() {
        let l = vec![Some(1), Some(2)];
        let r = vec![Some(1), None];
        let plan = plan_sub_joins(&l, &r).unwrap();
        assert!(plan.contains(&SubJoin::RightUnkeyedVsKeyedLeft));
        assert!(plan.contains(&SubJoin::KeyedPair(1)));
    }

    #[test]
    fn unkeyed_on_both_sides_is_not_double_counted() {
        // left-unkeyed × right-unkeyed must be covered by LeftUnkeyedVsAllRight
        // ONLY (RightUnkeyedVsKeyedLeft pairs with keyed left tasks exclusively).
        let l = vec![Some(1), None];
        let r = vec![Some(1), None];
        let plan = plan_sub_joins(&l, &r).unwrap();
        assert!(plan.contains(&SubJoin::LeftUnkeyedVsAllRight));
        assert!(plan.contains(&SubJoin::RightUnkeyedVsKeyedLeft));
        assert!(plan.contains(&SubJoin::KeyedPair(1)));
        assert_eq!(plan.len(), 3);
    }

    #[test]
    fn systematic_mismatch_aborts() {
        // Both sides keyed, zero pairs: e.g. dtype-divergent PartitionSpecs that can
        // never compare equal. Skipping would silently drop every row — abort instead.
        let l = vec![Some(1), Some(2)];
        let r = vec![Some(10), Some(20)];
        assert!(plan_sub_joins(&l, &r).is_none());
    }

    #[test]
    fn systematic_mismatch_with_left_unkeyed_still_aborts() {
        // The dangerous variant: a left unkeyed group used to bypass the guards and
        // fire a rewrite that drops all keyed left tasks.
        let l = vec![Some(1), Some(2), None];
        let r = vec![Some(10)];
        assert!(plan_sub_joins(&l, &r).is_none());
    }

    #[test]
    fn all_right_unkeyed_pairs_with_keyed_left() {
        // e.g. differently-named partition column on the right: every right task
        // groups under None; keyed left must still see them.
        let l = vec![Some(1), Some(2)];
        let r = vec![None];
        let plan = plan_sub_joins(&l, &r).unwrap();
        assert_eq!(plan, vec![SubJoin::RightUnkeyedVsKeyedLeft]);
    }

    #[test]
    fn find_shared_partition_key_requires_both_sides() {
        use daft_dsl::resolved_col;
        let l_pkeys = vec!["region".to_string()];
        let r_pkeys: Vec<String> = vec![];
        let l_eq = vec![resolved_col("region")];
        let r_eq = vec![resolved_col("region")];
        assert_eq!(find_shared_partition_key(&l_pkeys, &r_pkeys, &l_eq, &r_eq), None);
        let r_pkeys = vec!["region".to_string()];
        assert_eq!(
            find_shared_partition_key(&l_pkeys, &r_pkeys, &l_eq, &r_eq),
            Some("region")
        );
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p daft-logical-plan pairing_tests`
Expected: FAIL to compile — `plan_sub_joins` / `SubJoin` do not exist yet.

- [ ] **Step 3: Implement the fix**

In `src/daft-logical-plan/src/optimization/rules/collocated_join.rs`, add above the `impl CollocatedJoin` block:

```rust
/// One sub-join to materialize during the collocated rewrite.
#[derive(Debug, Clone, PartialEq, Eq)]
enum SubJoin<K> {
    /// Left tasks with no resolvable partition value × ALL right tasks.
    LeftUnkeyedVsAllRight,
    /// Right tasks with no resolvable partition value × left KEYED tasks only
    /// (left-unkeyed × right-unkeyed is already covered by the arm above —
    /// pairing against ALL left tasks would emit those pairs twice).
    RightUnkeyedVsKeyedLeft,
    /// One partition value present on both sides.
    KeyedPair(K),
}

/// Decide the sub-joins for the rewrite, given each side's group keys
/// (`None` = tasks with no resolvable partition value for the shared column).
///
/// Returns `None` to abort the rewrite: when both sides have keyed groups but
/// not a single value pairs up, the mismatch is systematic (e.g. the two sides
/// inferred different partition-value dtypes, so no `PartitionSpec` can ever
/// compare equal), not a genuine absence — only the original, un-split join is
/// safe. A *partial* pairing, by contrast, is genuine: this rewrite only fires
/// for Inner joins, where a left value absent on the right correctly
/// contributes zero rows, so unpaired keyed groups are simply skipped.
fn plan_sub_joins<K: Eq + std::hash::Hash + Clone>(
    l_keys: &[Option<K>],
    r_keys: &[Option<K>],
) -> Option<Vec<SubJoin<K>>> {
    let l_keyed: Vec<&K> = l_keys.iter().flatten().collect();
    let r_keyed: std::collections::HashSet<&K> = r_keys.iter().flatten().collect();
    let l_has_unkeyed = l_keys.iter().any(Option::is_none);
    let r_has_unkeyed = r_keys.iter().any(Option::is_none);

    let pairs: Vec<&K> = l_keyed
        .iter()
        .copied()
        .filter(|k| r_keyed.contains(*k))
        .collect();
    if !l_keyed.is_empty() && !r_keyed.is_empty() && pairs.is_empty() {
        return None;
    }

    let mut out = Vec::new();
    if l_has_unkeyed {
        out.push(SubJoin::LeftUnkeyedVsAllRight);
    }
    if r_has_unkeyed && !l_keyed.is_empty() {
        out.push(SubJoin::RightUnkeyedVsKeyedLeft);
    }
    out.extend(pairs.into_iter().cloned().map(SubJoin::KeyedPair));
    Some(out)
}
```

Then replace the sub-join construction in `try_optimize_node` (everything from `// Build one sub-join per partition value...` at line 191 through the `len() == 1` guard at line 237) with:

```rust
        let l_keys: Vec<Option<PartitionSpec>> = l_groups.keys().cloned().collect();
        let r_keys: Vec<Option<PartitionSpec>> = r_groups.keys().cloned().collect();
        let Some(planned) = plan_sub_joins(&l_keys, &r_keys) else {
            // Systematic partition-value mismatch: fall back to the original join.
            return Ok(Transformed::no(plan));
        };
        // A single sub-join can't reduce peak memory; keep the original plan.
        if planned.len() <= 1 {
            return Ok(Transformed::no(plan));
        }

        let make_sub_join = |l_tasks_sub: Vec<ScanTaskRef>,
                             r_tasks_sub: Vec<ScanTaskRef>|
         -> DaftResult<Arc<LogicalPlan>> {
            let l_src = source_with_tasks(&join.left, l_tasks_sub);
            let r_src = source_with_tasks(&join.right, r_tasks_sub);
            let sub = Join::try_new(
                l_src,
                r_src,
                join.on.clone(),
                join.join_type,
                join.join_strategy,
            )?;
            Ok(Arc::new(LogicalPlan::Join(sub)))
        };

        let l_keyed_tasks: Vec<ScanTaskRef> = l_groups
            .iter()
            .filter(|(k, _)| k.is_some())
            .flat_map(|(_, ts)| ts.iter().cloned())
            .collect();

        let mut sub_plans: Vec<Arc<LogicalPlan>> = Vec::with_capacity(planned.len());
        for sub in planned {
            match sub {
                SubJoin::LeftUnkeyedVsAllRight => {
                    let l_unkeyed = l_groups.get(&None).expect("planned implies present");
                    sub_plans.push(make_sub_join(
                        l_unkeyed.clone(),
                        r_tasks.iter().cloned().collect(),
                    )?);
                }
                SubJoin::RightUnkeyedVsKeyedLeft => {
                    let r_unkeyed = r_groups.get(&None).expect("planned implies present");
                    sub_plans.push(make_sub_join(l_keyed_tasks.clone(), r_unkeyed.clone())?);
                }
                SubJoin::KeyedPair(ps) => {
                    let l_sub = l_groups.get(&Some(ps.clone())).expect("planned implies present");
                    let r_sub = r_groups.get(&Some(ps)).expect("planned implies present");
                    sub_plans.push(make_sub_join(l_sub.clone(), r_sub.clone())?);
                }
            }
        }
```

The trailing `Concat`-folding reduce (lines 239-249) is unchanged; delete the now-dead `sub_plans.is_empty()` guard (the `len() <= 1` check above subsumes it).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p daft-logical-plan pairing_tests`
Expected: PASS (7 tests).

Regression check (the rewrite feeds ordinary joins, so the join suites cover materialization):

Run: `make build`
Run: `DAFT_RUNNER=native make test EXTRA_ARGS="-v tests/expressions/test_spatial_join.py"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/daft-logical-plan/src/optimization/rules/collocated_join.rs
git commit -m "fix: symmetric unkeyed-task handling and systematic-mismatch abort in collocated join"
```

---

## Task 6: Restrict the `Geometry` cast to `Binary` sources

The `DataType::Geometry` arm of the generic `DataArray<T>::cast` ([cast.rs:82-88](../../src/daft-core/src/array/ops/cast.rs)) fires for **any** arrow-backed source. Int64→Binary succeeds by design (string-representation semantics: `1i64` → `b"1"`), so `cast(geometry())` on an Int64/Utf8/Boolean column silently yields a Geometry column of non-WKB bytes — verified end-to-end; downstream ST_* functions then produce silent all-null output (current `wkb` 0.9.2 pin; the prior pin panicked). Fix: only `Binary` sources may cast to Geometry.

**Files:**
- Modify: `src/daft-core/src/array/ops/cast.rs:82-88`
- Test: `tests/expressions/test_spatial.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `DataArray<T>::cast(&DataType::Geometry)` returns `Err(DaftError::TypeError(..))` for any non-`Binary` source; Binary→Geometry (the GeoParquet/Delta retype path) is unchanged.

- [ ] **Step 1: Write the failing tests**

Add to `tests/expressions/test_spatial.py`:

```python
def test_cast_to_geometry_rejects_int_source():
    df = daft.from_pydict({"x": [1, 2, 3]})
    with pytest.raises(Exception, match="Geometry"):
        df.select(daft.col("x").cast(daft.DataType.geometry())).collect()


def test_cast_to_geometry_rejects_utf8_source():
    df = daft.from_pydict({"x": ["POINT(1 2)", "not wkb"]})
    with pytest.raises(Exception, match="Geometry"):
        df.select(daft.col("x").cast(daft.DataType.geometry())).collect()


def test_cast_to_geometry_from_binary_still_works():
    from daft.functions import st_point

    df = daft.from_pydict({"x": [1.0], "y": [2.0]}).select(
        st_point(daft.col("x"), daft.col("y")).alias("g")
    )
    wkb = df.select(daft.col("g").cast(daft.DataType.binary()).alias("wkb"))
    back = wkb.select(daft.col("wkb").cast(daft.DataType.geometry()).alias("g2"))
    assert back.to_pydict()["g2"] == df.to_pydict()["g"]
```

(If `test_spatial.py` lacks module-level `import pytest` / `import daft`, add them.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `DAFT_RUNNER=native make test EXTRA_ARGS="-v tests/expressions/test_spatial.py::test_cast_to_geometry_rejects_int_source tests/expressions/test_spatial.py::test_cast_to_geometry_rejects_utf8_source tests/expressions/test_spatial.py::test_cast_to_geometry_from_binary_still_works"`
Expected: the first two FAIL (`pytest.raises` doesn't fire — the cast silently succeeds); the third passes.

- [ ] **Step 3: Implement the fix**

In `src/daft-core/src/array/ops/cast.rs`, replace the `DataType::Geometry` arm:

```rust
            DataType::Geometry => {
                // Only Binary carries a WKB representation. Numeric/Utf8/Boolean
                // sources cast to Binary via their STRING representation (see the
                // arrow2-compat routing below), so allowing them here would mint a
                // Geometry column of non-WKB bytes that every ST_* function
                // null-ifies — silent garbage far from the actual mistake.
                if !matches!(self.data_type(), DataType::Binary) {
                    return Err(DaftError::TypeError(format!(
                        "Cannot cast {:?} to Geometry: only Binary (WKB-encoded) data can \
                         be cast to Geometry",
                        self.data_type()
                    )));
                }
                let binary_series = self.cast(&DataType::Binary)?;
                let physical = binary_series.binary().unwrap().clone();
                let field = Field::new(self.name(), DataType::Geometry);
                Ok(GeometryArray::new(field, physical).into_series())
            }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `make build`
Run: `DAFT_RUNNER=native make test EXTRA_ARGS="-v tests/expressions/test_spatial.py"`
Run: `DAFT_RUNNER=native make test EXTRA_ARGS="-v tests/io/test_geoparquet.py"`
Expected: PASS (the second command confirms the real Binary retype path is unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/daft-core/src/array/ops/cast.rs tests/expressions/test_spatial.py
git commit -m "fix: reject casting non-Binary sources to Geometry"
```

---

## Task 7: Warn when a GeoParquet file declares a non-default CRS

`GeoColumn.crs` is parsed then discarded by `detect_geo_columns` ([geo_metadata.rs:83](../../src/daft-schema/src/geo_metadata.rs)); `DataType::Geometry` carries no CRS; no read-path code surfaces it. A file declaring e.g. `EPSG:3857` is read with zero indication — and Daft's geodesic functions (`use_spheroid` variants, `great_circle_distance`, `st_geohash`, H3 helpers) silently misinterpret projected coordinates as lon/lat. (Planar defaults like `st_distance` are CRS-agnostic, so this is a warn-only fix, deliberately — a CRS-aware Geometry type is out of scope.)

**Files:**
- Modify: `src/daft-schema/src/geo_metadata.rs` (new pure function; `detect_geo_columns` unchanged)
- Modify: `src/daft-parquet/src/lib.rs` (`infer_schema_from_daft_metadata`, the `options.geometry` block)
- Modify: `src/daft-parquet/src/reader/mod.rs` (`stream_parquet`, the geo-cols closure ~line 611-637)
- Test: inline `#[cfg(test)]` in `geo_metadata.rs`

**Interfaces:**
- Consumes: nothing new (`log` is already used in `daft-parquet`; verify and add `log = {workspace = true}` if missing).
- Produces: `pub fn non_default_crs_columns(geo_json: &str) -> Vec<(String, String)>` in `daft_schema::geo_metadata` — `(column_name, crs_string)` for every declared geometry column whose `crs` is present and not `"OGC:CRS84"`.

- [ ] **Step 1: Write the failing tests**

Add to the `#[cfg(test)]` module in `src/daft-schema/src/geo_metadata.rs`:

```rust
    #[test]
    fn non_default_crs_columns_flags_non_wgs84() {
        let json = r#"{
            "version": "1.1.0",
            "primary_column": "geom",
            "columns": {
                "geom": {"encoding": "WKB", "crs": "EPSG:3857"},
                "geom_wgs84": {"encoding": "WKB", "crs": "OGC:CRS84"},
                "geom_default": {"encoding": "WKB"}
            }
        }"#;
        let flagged = non_default_crs_columns(json);
        assert_eq!(flagged, vec![("geom".to_string(), "EPSG:3857".to_string())]);
    }

    #[test]
    fn non_default_crs_columns_empty_for_malformed_json() {
        assert!(non_default_crs_columns("{not json").is_empty());
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p daft-schema non_default_crs_columns`
Expected: FAIL to compile — `non_default_crs_columns` does not exist.

- [ ] **Step 3: Implement the fix**

Add to `src/daft-schema/src/geo_metadata.rs`, after `detect_geo_columns`:

```rust
/// The GeoParquet spec's default CRS when the `crs` key is absent: lon/lat WGS84.
const DEFAULT_CRS: &str = "OGC:CRS84";

/// Returns `(column_name, crs)` for every geometry column in `geo_json` whose `crs`
/// is present and is not the GeoParquet default (`OGC:CRS84`). Lenient: returns
/// empty on any parse failure, matching `detect_geo_columns`.
///
/// Daft's `Geometry` type has no CRS field. Planar ST_* defaults are CRS-agnostic
/// (results are in coordinate units), but the geodesic family (`use_spheroid`
/// variants, `great_circle_distance`, `st_geohash`, H3 helpers) assumes lon/lat
/// WGS84 — callers use this to warn instead of silently misinterpreting projected
/// coordinates.
pub fn non_default_crs_columns(geo_json: &str) -> Vec<(String, String)> {
    let Ok(meta) = serde_json::from_str::<GeoMetadata>(geo_json) else {
        return vec![];
    };
    meta.columns
        .into_iter()
        .filter_map(|(name, c)| {
            let crs = c.crs?;
            let crs_str = match &crs {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            (crs_str != DEFAULT_CRS).then_some((name, crs_str))
        })
        .collect()
}
```

Wire into `src/daft-parquet/src/lib.rs` (`infer_schema_from_daft_metadata`):

```rust
    if options.geometry {
        if let Some(geo_json) = metadata.geo_metadata() {
            let geo_cols = geo_metadata::detect_geo_columns(&geo_json, &schema);
            if !geo_cols.is_empty() {
                schema = retype_geo_schema(&schema, &geo_cols);
            }
            for (col, crs) in geo_metadata::non_default_crs_columns(&geo_json) {
                log::warn!(
                    "GeoParquet column '{col}' declares CRS '{crs}' (not the default \
                     OGC:CRS84 / WGS84 lon-lat). Daft's Geometry type has no CRS concept: \
                     geodesic functions (use_spheroid distance/dwithin, \
                     great_circle_distance, st_geohash, H3 helpers) assume lon/lat WGS84 \
                     and will silently produce wrong results for projected coordinates."
                );
            }
        }
    }
```

Wire into `src/daft-parquet/src/reader/mod.rs` (`stream_parquet`), inside the existing `.map(|geo_json| { ... })` closure, before the `arrow_schema_for_detect ... detect_geo_columns` return expression:

```rust
                for (col, crs) in daft_schema::geo_metadata::non_default_crs_columns(&geo_json) {
                    log::warn!(
                        "GeoParquet column '{col}' in {path} declares CRS '{crs}' (not the \
                         default OGC:CRS84 / WGS84 lon-lat); Daft's Geometry type has no \
                         CRS concept — geodesic functions assume lon/lat WGS84."
                    );
                }
```

(`path` is in scope from `let path = cs_builder.path().clone();` a few lines above; if its type doesn't implement `Display`, use the accessor other log/error sites in this file use for it. If `daft-parquet` lacks a `log` dependency, add `log = {workspace = true}` to its `Cargo.toml`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p daft-schema non_default_crs_columns`
Run: `cargo build -p daft-parquet`
Expected: tests PASS; build clean.

- [ ] **Step 5: Commit**

```bash
git add src/daft-schema/src/geo_metadata.rs src/daft-parquet/src/lib.rs src/daft-parquet/src/reader/mod.rs
git commit -m "fix: warn when a GeoParquet column declares a non-default CRS"
```

---

## Task 8: Make `SpatialPartitionPruning` read the real `.idx` index, per directory

Two verified defects fixed together (the earlier draft's Tasks 3 and 10 — merged so the JSON-format test scaffolding is never built): (a) the rule reads `_spatial_index.json`, but the only shipped writer — `daft.functions.spatial_index.build_spatial_index()` — writes `_spatial_index.idx`, an inverted Parquet file (two dictionary-encoded string columns `h3_cell`/`filename` + schema metadata `geom_col`/`h3_resolution`); the rule is dead code (git: commit `5deb7d0bd` migrated the Python writer and left the Rust reader behind). (b) The rule loads ONE index from the first task's directory and matches every task by bare basename — in multi-directory layouts, wrong index consulted (wrong pruning on basename collisions, lost pruning otherwise). Fix: read the `.idx` format, cache one index per directory, look each task up in its own directory's index. Also fix the stale docstrings/example, and replace the file's vacuous tests (`assert!(1 <= 1)`) with real ones.

**Files:**
- Modify: `src/daft-logical-plan/src/optimization/rules/spatial_partition_pruning.rs` (doc-comment, `INDEX_FILENAME`, `load_index_for_dir`, `try_optimize_node`, `task_passes_h3`; delete `first_file_dir`; delete the vacuous `tests` module)
- Modify: `src/daft-logical-plan/Cargo.toml` (add `parquet = {workspace = true}` to `[dependencies]`; add `arrow = {workspace = true}` and `tempfile = {workspace = true}` to `[dev-dependencies]` — all three are already workspace deps at the root)
- Modify: `daft/functions/spatial_index.py`, `examples/spatial_geohash_pruning/spatial_partition_index.py`, `examples/spatial_geohash_pruning/generate_data.py` (stale-docstring fixes)
- Test: inline `#[cfg(test)]` in the Rust file

**Interfaces:**
- Consumes: `parquet::file::reader::{FileReader, SerializedFileReader}`, `parquet::record::RowAccessor` (parquet 59.0.0 — adjust exact method names per compiler feedback; the shape "open file → read schema KV metadata → iterate rows → HashMap<filename, Vec<cell>>" is the requirement). Test-side: `arrow` array builders, `tempfile`, `daft_scan::test_utils::DummyScanOperator` (exists: `src/daft-scan/src/test_utils.rs:15`), `ScanTask::new` (signature: `(sources, source_config, schema, storage_config, pushdowns, generated_fields)` — `src/daft-scan/src/lib.rs:400`).
- Produces: `load_index_for_dir(dir: &str) -> Option<LoadedIndex>` — same signature and `LoadedIndex` shape, new format; `INDEX_FILENAME = "_spatial_index.idx"`; `task_passes_h3` takes an `IndexCache` and resolves each source file against its own directory.

- [ ] **Step 1: Write the failing tests**

Add the Cargo.toml dependencies first (build fails otherwise):

```toml
# [dependencies] — append:
parquet = {workspace = true}

# [dev-dependencies] — append (create the section if absent):
arrow = {workspace = true}
tempfile = {workspace = true}
```

Replace the entire vacuous `#[cfg(test)] mod tests` block at the bottom of `spatial_partition_pruning.rs` with:

```rust
#[cfg(test)]
mod tests {
    use std::sync::Arc as StdArc;

    use super::*;

    /// Write a `.idx` file with the exact schema `_build_h3_index` in
    /// `daft/functions/spatial_index.py` produces: two dictionary-encoded string
    /// columns (`h3_cell`, `filename`) plus `geom_col`/`h3_resolution` metadata.
    fn write_test_idx_file(
        path: &std::path::Path,
        geom_col: &str,
        h3_resolution: u8,
        rows: &[(&str, &str)],
    ) {
        use arrow::array::{Int32Type, StringDictionaryBuilder};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;

        let mut h3_builder: StringDictionaryBuilder<Int32Type> = StringDictionaryBuilder::new();
        let mut fname_builder: StringDictionaryBuilder<Int32Type> = StringDictionaryBuilder::new();
        for (cell, fname) in rows {
            h3_builder.append_value(*cell);
            fname_builder.append_value(*fname);
        }
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("geom_col".to_string(), geom_col.to_string());
        metadata.insert("h3_resolution".to_string(), h3_resolution.to_string());
        let schema = StdArc::new(
            Schema::new(vec![
                Field::new(
                    "h3_cell",
                    DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                    false,
                ),
                Field::new(
                    "filename",
                    DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                    false,
                ),
            ])
            .with_metadata(metadata),
        );
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                StdArc::new(h3_builder.finish()),
                StdArc::new(fname_builder.finish()),
            ],
        )
        .unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    #[test]
    fn load_index_for_dir_parses_real_idx_parquet_format() {
        let dir = tempfile::tempdir().unwrap();
        write_test_idx_file(
            &dir.path().join(INDEX_FILENAME),
            "geom",
            5,
            &[
                ("851fb467fffffff", "part-0.parquet"),
                ("851fb467fffffff", "part-1.parquet"),
            ],
        );
        let index = load_index_for_dir(&dir.path().to_string_lossy()).expect("index must parse");
        assert_eq!(index.geom_col, "geom");
        assert_eq!(index.h3_resolution, 5);
        assert!(index.file_cells.contains_key("part-0.parquet"));
        assert!(index.file_cells.contains_key("part-1.parquet"));
    }

    #[test]
    fn index_cache_isolates_directories() {
        let dir_a = tempfile::tempdir().unwrap();
        let dir_b = tempfile::tempdir().unwrap();
        write_test_idx_file(
            &dir_a.path().join(INDEX_FILENAME),
            "geom", 5,
            &[("85be0e37fffffff", "part-0.parquet")],
        );
        write_test_idx_file(
            &dir_b.path().join(INDEX_FILENAME),
            "geom", 5,
            &[("851fb467fffffff", "part-0.parquet")],
        );
        let mut cache = IndexCache::new();
        let a = cache.get(&dir_a.path().to_string_lossy()).unwrap();
        let b = cache.get(&dir_b.path().to_string_lossy()).unwrap();
        // Same basename, different directories, different cells: per-directory
        // isolation, never a shared global basename map.
        assert_ne!(
            a.file_cells.get("part-0.parquet"),
            b.file_cells.get("part-0.parquet")
        );
        // Missing directory -> None (conservative keep at the call site).
        let dir_c = tempfile::tempdir().unwrap();
        assert!(cache.get(&dir_c.path().to_string_lossy()).is_none());
    }

    #[test]
    fn end_to_end_prunes_task_whose_own_directory_index_does_not_cover_query() {
        use daft_scan::{
            FileFormatConfig, ParquetSourceConfig, PhysicalScanInfo, Pushdowns, ScanSource,
            ScanSourceKind, ScanTask, SourceConfig, storage_config::StorageConfig,
            test_utils::DummyScanOperator,
        };
        use daft_schema::{dtype::DataType as DaftDataType, field::Field, schema::Schema, time_unit::TimeUnit};

        fn make_task(path: &str, schema: daft_schema::schema::SchemaRef) -> ScanTaskRef {
            StdArc::new(ScanTask::new(
                vec![ScanSource {
                    size_bytes: None,
                    metadata: None,
                    statistics: None,
                    partition_spec: None,
                    kind: ScanSourceKind::File {
                        path: path.to_string(),
                        chunk_spec: None,
                        iceberg_delete_files: None,
                        parquet_metadata: None,
                    },
                }],
                StdArc::new(SourceConfig::File(FileFormatConfig::Parquet(
                    ParquetSourceConfig {
                        coerce_int96_timestamp_unit: TimeUnit::Seconds,
                        field_id_mapping: None,
                        row_groups: None,
                        chunk_size: None,
                        ignore_corrupt_files: false,
                        geometry: true,
                    },
                ))),
                schema,
                StdArc::new(StorageConfig::new_internal(false, None)),
                Pushdowns::default(),
                None,
            ))
        }

        let dir_a = tempfile::tempdir().unwrap();
        let dir_b = tempfile::tempdir().unwrap();
        // dir_a covers a Sydney cell; dir_b covers a Paris cell. Query point: Paris.
        write_test_idx_file(&dir_a.path().join(INDEX_FILENAME), "geom", 5, &[("85be0e37fffffff", "part-0.parquet")]);
        write_test_idx_file(&dir_b.path().join(INDEX_FILENAME), "geom", 5, &[("851fb467fffffff", "part-0.parquet")]);

        let schema = StdArc::new(Schema::new(vec![Field::new("geom", DaftDataType::Binary)]));
        let task_a = make_task(&dir_a.path().join("part-0.parquet").to_string_lossy(), schema.clone());
        let task_b = make_task(&dir_b.path().join("part-0.parquet").to_string_lossy(), schema.clone());

        let scan_op = StdArc::new(DummyScanOperator { schema: schema.clone(), ..Default::default() });
        let mut psi = PhysicalScanInfo::new(scan_op.into(), schema.clone(), vec![], Pushdowns::default(), None);
        psi.scan_state = ScanState::Tasks(StdArc::new(vec![task_a, task_b]));
        let source = crate::ops::Source::new(schema.clone(), StdArc::new(SourceInfo::Physical(psi)));

        let query_wkb = daft_geo::geom_to_wkb(&geo::Geometry::Point(geo::Point::new(2.35, 48.85))).unwrap();
        let predicate = daft_geo::st_intersects::st_intersects(
            daft_dsl::resolved_col("geom"),
            daft_dsl::lit(query_wkb).into(),
        );
        let filter = crate::ops::Filter::try_new(
            StdArc::new(LogicalPlan::Source(source)),
            predicate,
        )
        .unwrap();
        let plan = StdArc::new(LogicalPlan::Filter(filter));

        let result = SpatialPartitionPruning.try_optimize(plan).unwrap();
        assert!(result.transformed, "rule must prune the non-covering task");
        let LogicalPlan::Filter(f) = result.data.as_ref() else { panic!("expected Filter") };
        let LogicalPlan::Source(s) = f.input.as_ref() else { panic!("expected Source") };
        let SourceInfo::Physical(p) = s.source_info.as_ref() else { panic!() };
        let ScanState::Tasks(remaining) = &p.scan_state else { panic!() };
        assert_eq!(remaining.len(), 1);
        assert!(remaining[0].sources[0]
            .get_path()
            .starts_with(&*dir_b.path().to_string_lossy()));
    }
}
```

Notes for the implementer, all shape-preserving: (a) `extract_spatial_pred` requires arg1 to be `Expr::Literal(Literal::Binary(..))` — `daft_dsl::lit(Vec<u8>)` should produce that; if the H3 cells chosen don't match the H3 encoding of (2.35, 48.85) at resolution 5, compute the true cell in the test itself via `daft_geo::h3_index::wkb_to_h3_cells(&query_wkb, 5)` and write *that* into dir_b's index (and any other cell into dir_a's) — the assertion logic stays identical. (b) Constructor field names for `ScanSource`/`PhysicalScanInfo` may drift; fix names per compiler feedback. (c) If `daft-logical-plan` does not already depend on `daft-geo` for tests, add `daft-geo = {path = "../daft-geo"}` under `[dev-dependencies]`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p daft-logical-plan spatial_partition_pruning`
Expected: FAIL — first at compile (until Step 3's `IndexCache` exists), then at runtime: `load_index_for_dir` still expects JSON named `_spatial_index.json` and cannot parse the Parquet bytes.

- [ ] **Step 3: Implement the fix**

In `src/daft-logical-plan/src/optimization/rules/spatial_partition_pruning.rs`:

**(a)** Replace the module doc-comment (lines 1-27) and constant:

```rust
/// Partition-level spatial pruning using a per-directory H3 spatial index.
///
/// When a spatial predicate (`st_intersects`, `st_contains`, `st_within`) filters a
/// geometry column and a `_spatial_index.idx` sidecar exists in a source file's own
/// directory, this rule skips every scan task whose H3 coverage does not overlap the
/// query geometry.
///
/// # Index format — version 1 (H3 inverted Parquet index)
///
/// An inverted Parquet file with two dictionary-encoded string columns:
///
/// | h3_cell         | filename       |
/// |-----------------|----------------|
/// | 8a283473fffffff | part-0.parquet |
/// | 8a2834b3fffffff | part-0.parquet |
/// | 8a283477fffffff | part-1.parquet |
///
/// plus schema-level key-value metadata: `geom_col` (geometry column name) and
/// `h3_resolution` (decimal string). This matches what
/// `daft.functions.spatial_index.build_spatial_index()` writes. The rule loads each
/// directory's whole (small) sidecar into memory once, cached per directory.
///
/// Build the index from Python:
///
/// ```python
/// from daft.functions.spatial_index import build_spatial_index
/// build_spatial_index("output/", geom_col="geom")   # requires h3
/// ```
```

```rust
const INDEX_FILENAME: &str = "_spatial_index.idx";
```

**(b)** Replace `load_index_for_dir` (the `LoadedIndex` struct is unchanged):

```rust
fn load_index_for_dir(dir: &str) -> Option<LoadedIndex> {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use parquet::record::RowAccessor;

    let index_path = Path::new(dir).join(INDEX_FILENAME);
    let file = std::fs::File::open(index_path).ok()?;
    let reader = SerializedFileReader::new(file).ok()?;

    let kv_meta = reader.metadata().file_metadata().key_value_metadata()?;
    let geom_col = kv_meta
        .iter()
        .find(|kv| kv.key == "geom_col")
        .and_then(|kv| kv.value.clone())?;
    let h3_resolution: u8 = kv_meta
        .iter()
        .find(|kv| kv.key == "h3_resolution")
        .and_then(|kv| kv.value.clone())?
        .parse()
        .ok()?;

    // Column 0 = h3_cell, column 1 = filename (the schema `_build_h3_index` in
    // daft/functions/spatial_index.py writes). Dictionary encoding is a physical
    // storage detail — the row-based reader yields plain logical strings.
    let mut raw: HashMap<String, Vec<String>> = HashMap::new();
    let row_iter = reader.get_row_iter(None).ok()?;
    for row in row_iter {
        let row = row.ok()?;
        let h3_cell = row.get_string(0).ok()?.clone();
        let filename = row.get_string(1).ok()?.clone();
        raw.entry(filename).or_default().push(h3_cell);
    }

    let file_cells = raw
        .into_iter()
        .map(|(fname, cell_strs)| (fname, parse_h3_cells(&cell_strs)))
        .collect();
    Some(LoadedIndex { geom_col, h3_resolution, file_cells })
}
```

(parquet 59.0.0 API drift: if `get_row_iter`/`get_string`/`KeyValue` field names differ, adjust names — the shape is the requirement.)

**(c)** Add the per-directory cache, and rewrite the pruning body of `try_optimize_node` (everything from `let dir = match first_file_dir...` at line 123 through the `pruned` collection at line 144); delete `first_file_dir`:

```rust
/// One loaded index per directory, so a scan spanning many partition directories
/// reads each sidecar once, not once per task.
struct IndexCache {
    by_dir: HashMap<String, Option<Arc<LoadedIndex>>>,
}

impl IndexCache {
    fn new() -> Self {
        Self { by_dir: HashMap::new() }
    }

    fn get(&mut self, dir: &str) -> Option<Arc<LoadedIndex>> {
        self.by_dir
            .entry(dir.to_string())
            .or_insert_with(|| load_index_for_dir(dir).map(Arc::new))
            .clone()
    }
}
```

```rust
        let mut cache = IndexCache::new();
        // Different directories may use different H3 resolutions; resolve the
        // query's covering cells once per resolution.
        let mut query_cells_by_res: HashMap<u8, Option<Vec<u64>>> = HashMap::new();

        let pruned: Vec<ScanTaskRef> = tasks
            .iter()
            .filter(|t| task_passes_h3(t, &geom_col, &query_wkb, &mut cache, &mut query_cells_by_res))
            .cloned()
            .collect();
```

**(d)** Replace `task_passes_h3` — each source file is looked up in **its own directory's** index:

```rust
/// Keep the task when any of its source files' H3 cells (from that file's OWN
/// directory's index) intersect the query cells. Every unresolvable situation
/// (no parent dir, no index, wrong geom_col, no query cells, file not indexed)
/// keeps the task conservatively — pruning must never be based on a different
/// directory's index.
fn task_passes_h3(
    task: &ScanTask,
    geom_col: &str,
    query_wkb: &[u8],
    cache: &mut IndexCache,
    query_cells_by_res: &mut HashMap<u8, Option<Vec<u64>>>,
) -> bool {
    for source in &task.sources {
        let path = source.get_path();
        let Some(dir) = Path::new(path).parent().map(|p| p.to_string_lossy().into_owned())
        else {
            return true;
        };
        let Some(index) = cache.get(&dir) else {
            return true;
        };
        if index.geom_col != geom_col {
            return true;
        }
        let query_cells = query_cells_by_res
            .entry(index.h3_resolution)
            .or_insert_with(|| {
                wkb_to_h3_cells(query_wkb, index.h3_resolution)
                    .map(|c| c.into_iter().collect())
            });
        let Some(query_cells) = query_cells.as_ref() else {
            return true;
        };
        let fname = Path::new(path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned());
        match fname.and_then(|f| index.file_cells.get(&f)) {
            Some(cells) => {
                if h3_cells_intersect(cells, query_cells) {
                    return true;
                }
            }
            None => return true,
        }
    }
    false
}
```

Adjust `try_optimize_node`'s earlier index-dependent code accordingly: the old `index.geom_col != geom_col` check and the single `wkb_to_h3_cells` call at lines 132-139 move into `task_passes_h3` as shown; the surrounding match/rebuild logic (lines 146-160) is unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p daft-logical-plan spatial_partition_pruning`
Expected: PASS (3 tests).

- [ ] **Step 5: Fix the stale Python docstrings and example**

All references found by: `grep -rn "_spatial_index.json\|file_mbrs" daft/ examples/`

- `daft/functions/spatial_index.py`: module docstring opening paragraph — replace "a small JSON sidecar file (``_spatial_index.json``)" with "a small inverted-Parquet sidecar file (``_spatial_index.idx``)"; `build_spatial_index` docstring — replace "write ``_spatial_index.json``" and "Defaults to ``{directory}/_spatial_index.json``" with the `.idx` name; `spatial_join` docstring — same substitution at both mentions.
- `examples/spatial_geohash_pruning/spatial_partition_index.py`: docstring line ~9 — replace "Build a ``_spatial_index.json`` sidecar that records the MBR of each file" with "Build a ``_spatial_index.idx`` sidecar that records the H3 cells of each file"; the existence check (~line 273) `os.path.exists(os.path.join(pdir, "_spatial_index.json"))` → `"_spatial_index.idx"`; the broken attribute access (~line 173) `idx.file_mbrs` → `idx.file_h3_cells` (the attribute `SpatialIndex` actually has).
- `examples/spatial_geohash_pruning/generate_data.py`: docstring line ~5 — same `.json` → `.idx` substitution.

Run: `grep -rn "_spatial_index.json\|file_mbrs" daft/ examples/ src/`
Expected: no matches.

- [ ] **Step 6: Commit**

```bash
git add src/daft-logical-plan/Cargo.toml src/daft-logical-plan/src/optimization/rules/spatial_partition_pruning.rs Cargo.lock daft/functions/spatial_index.py examples/spatial_geohash_pruning/
git commit -m "fix: read the real _spatial_index.idx format per directory in SpatialPartitionPruning"
```

---

## Final verification

After all tasks:

Run: `DAFT_RUNNER=native make test EXTRA_ARGS="-v tests/expressions/test_spatial.py tests/expressions/test_spatial_join.py tests/expressions/test_spatial_geohash_pruning.py tests/io/test_geoparquet.py"`
Run: `DAFT_RUNNER=ray make test EXTRA_ARGS="-v tests/expressions/test_spatial_join.py"`
Run: `cargo test -p daft-geo -p daft-core -p daft-schema -p daft-local-execution -p daft-local-plan -p daft-logical-plan -p daft-parquet -p daft-distributed`

Expected: all PASS.

## Explicitly out of scope (deferred, not forgotten)

- **Composite/NULL-aware partition-key grouping in the NLJ** (the old draft's Tasks 7–9 machinery): after Tasks 1–2 it is purely a performance optimization (composite-key joins fall back to a single global R-tree with the complete filter — correct, possibly slower on huge multi-tenant joins). Follow-up if profiling shows a need.
- **CRS-aware `Geometry` type**: Task 7 warns instead; a typed CRS ripples through ~15 call sites across 3 crates.
- **Left/Right/Outer collocated joins**: the rule stays Inner-only (its doc already defers this).

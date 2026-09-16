# Delta Merge Pass-1 Target Prune — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop pass 1 of `distributed_merge_deltalake` from scanning the entire target table by filtering it to the source's key ranges, so Delta files that cannot contain a match are never opened.

**Architecture:** Derive a per-key-column `min`/`max` range predicate from the (already materialized, ≤512 MiB) source and apply it with `.where()` to the target frame used by pass 1 only. `MicroPartition::filter` short-circuits on ScanTask stats before loading, and the Delta scan already supplies per-file min/max, so no engine change is needed. Pass 2 keeps the unpruned frame because the unpartitioned write path is a full overwrite. Metrics are corrected arithmetically from the Delta log's exact `num_records`.

**Tech Stack:** Python, Daft DataFrame API, `deltalake` (`DeltaTable.get_add_actions()`), pytest.

**Spec:** `docs/superpowers/specs/2026-09-09-delta-merge-target-pruning.md`

## Global Constraints

- Target file: `daft/io/delta_lake/_deltalake.py` (fork-local; no upstream equivalent).
- `is_in` must never be used in the prune predicate — `daft-stats` (`table_stats.rs:126-160`) cannot evaluate it and it would prune nothing. Range predicates only: `>=` / `<=` combined with `&`.
- The prune predicate must be derived **only** from source key values, preserving the invariant that no matching target row can be dropped.
- Never build pass 2's `annotated` frame on the pruned target — the unpartitioned write path (`_deltalake.py:1826+`) is a full overwrite and would delete every skipped row.
- Rebuild is not required: this task touches no Rust. Run tests with `DAFT_RUNNER=native` unless a step says `ray`.
- Existing behavior must be bit-identical when the prune is skipped by a gate.

---

### Task 1: Source key-range extraction and predicate construction

**Files:**
- Modify: `daft/io/delta_lake/_deltalake.py` (add module-level helpers next to `_should_decompose_join`, ~line 1892)
- Test: `tests/io/delta_lake/test_delta_merge_pass1_prune.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `_source_key_ranges(source: DataFrame, keys: list[str]) -> dict[str, tuple[Any, Any]] | None`
  - `_key_range_filter(ranges: dict[str, tuple[Any, Any]]) -> Expression`

- [ ] **Step 1: Write the failing test**

Create `tests/io/delta_lake/test_delta_merge_pass1_prune.py`:

```python
from __future__ import annotations

import daft
from daft.io.delta_lake._deltalake import _key_range_filter, _source_key_ranges


def test_source_key_ranges_single_key():
    src = daft.from_pydict({"id": [5, 1, 9], "v": ["a", "b", "c"]})
    assert _source_key_ranges(src, ["id"]) == {"id": (1, 9)}


def test_source_key_ranges_multi_key():
    src = daft.from_pydict({"a": [2, 4], "b": ["x", "z"]})
    assert _source_key_ranges(src, ["a", "b"]) == {"a": (2, 4), "b": ("x", "z")}


def test_source_key_ranges_ignores_nulls_in_keys():
    # NULL keys never match under Daft's default null-unequal join semantics,
    # so min/max over the non-null values is still a valid covering range.
    src = daft.from_pydict({"id": [3, None, 7]})
    assert _source_key_ranges(src, ["id"]) == {"id": (3, 7)}


def test_source_key_ranges_all_null_returns_none():
    src = daft.from_pydict({"id": [None, None]}).select(daft.col("id").cast(daft.DataType.int64()))
    assert _source_key_ranges(src, ["id"]) is None


def test_source_key_ranges_empty_source_returns_none():
    src = daft.from_pydict({"id": [1]}).where(daft.col("id") < 0)
    assert _source_key_ranges(src, ["id"]) is None


def test_key_range_filter_prunes_disjoint_and_keeps_overlapping():
    ranges = {"id": (10, 20)}
    pred = _key_range_filter(ranges)

    disjoint = daft.from_pydict({"id": [1, 2, 3]})
    assert disjoint.where(pred).count_rows() == 0

    overlapping = daft.from_pydict({"id": [9, 15, 21]})
    assert overlapping.where(pred).to_pydict()["id"] == [15]


def test_key_range_filter_multi_key_is_conjunctive():
    pred = _key_range_filter({"a": (1, 2), "b": (5, 6)})
    df = daft.from_pydict({"a": [1, 1, 3], "b": [5, 9, 5]})
    assert df.where(pred).to_pydict() == {"a": [1], "b": [5]}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test EXTRA_ARGS="-v tests/io/delta_lake/test_delta_merge_pass1_prune.py"`
Expected: FAIL with `ImportError: cannot import name '_key_range_filter'`

- [ ] **Step 3: Write minimal implementation**

Add to `daft/io/delta_lake/_deltalake.py` immediately after `_should_decompose_join` (after line 1910):

```python
def _source_key_ranges(source: DataFrame, keys: "list[str]") -> "dict[str, tuple[Any, Any]] | None":
    """Per-key-column (min, max) of the source, or None if unusable for pruning.

    Returns None when the source is empty or any key column is entirely NULL —
    in both cases no covering range exists. NULL key values are ignored, which
    is safe: Daft's default join semantics treat NULL as unequal to everything,
    so a NULL source key cannot match any target row and needs no coverage.
    """
    from daft import col

    lo_alias = {k: f"__daft_dm_min_{i}__" for i, k in enumerate(keys)}
    hi_alias = {k: f"__daft_dm_max_{i}__" for i, k in enumerate(keys)}
    agg_exprs = []
    for k in keys:
        agg_exprs.append(col(k).min().alias(lo_alias[k]))
        agg_exprs.append(col(k).max().alias(hi_alias[k]))

    row = source.agg(*agg_exprs).to_pydict()

    ranges: dict[str, tuple[Any, Any]] = {}
    for k in keys:
        lo_vals = row[lo_alias[k]]
        hi_vals = row[hi_alias[k]]
        if not lo_vals or not hi_vals:
            return None
        lo, hi = lo_vals[0], hi_vals[0]
        if lo is None or hi is None:
            return None
        ranges[k] = (lo, hi)
    return ranges


def _key_range_filter(ranges: "dict[str, tuple[Any, Any]]") -> "Expression":
    """Conjunction of per-column range containments covering every possible match.

    Deliberately uses only ``>=``/``<=``/``&``: ``daft-stats``
    (``table_stats.rs:126-160``) can fold exactly these into per-file min/max
    statistics, so whole Delta files whose ranges are disjoint are skipped
    without being opened. ``is_in`` would fall through to
    ``ColumnRangeStatistics::Missing`` and prune nothing.
    """
    from daft import col, lit

    pred = None
    for k, (lo, hi) in ranges.items():
        clause = (col(k) >= lit(lo)) & (col(k) <= lit(hi))
        pred = clause if pred is None else (pred & clause)
    assert pred is not None, "_key_range_filter requires at least one key range"
    return pred
```

- [ ] **Step 4: Run test to verify it passes**

Run: `make test EXTRA_ARGS="-v tests/io/delta_lake/test_delta_merge_pass1_prune.py"`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/io/delta_lake/test_delta_merge_pass1_prune.py daft/io/delta_lake/_deltalake.py
git commit -m "perf(delta): add source key-range predicate helpers for merge pruning"
```

---

### Task 2: Eligibility gate

**Files:**
- Modify: `daft/io/delta_lake/_deltalake.py` (add after `_key_range_filter`)
- Test: `tests/io/delta_lake/test_delta_merge_pass1_prune.py` (extend)

**Interfaces:**
- Consumes: nothing (pure predicate over clause list and schema).
- Produces: `_prune_keys_eligible(clauses: list[tuple], validate_unique_keys: bool, keys: list[str], target_schema: Schema) -> bool`

- [ ] **Step 1: Write the failing test**

Append to `tests/io/delta_lake/test_delta_merge_pass1_prune.py`:

```python
from daft.io.delta_lake._deltalake import _prune_keys_eligible


def _schema(**cols):
    return daft.from_pydict({k: v for k, v in cols.items()}).schema()


def test_eligible_for_plain_matched_and_insert_clauses():
    clauses = [("matched_update", {"v": "source.v"}, None, None), ("insert", None, None, None)]
    assert _prune_keys_eligible(clauses, True, ["id"], _schema(id=[1], v=["a"])) is True


def test_ineligible_when_by_source_delete_present():
    # An unmatched target row's own values decide whether it is deleted, so it
    # must be read; pruning it away would silently skip the delete.
    clauses = [("by_source_delete", None, None, None)]
    assert _prune_keys_eligible(clauses, True, ["id"], _schema(id=[1])) is False


def test_ineligible_when_by_source_update_present():
    clauses = [("by_source_update", {"v": "1"}, None, None)]
    assert _prune_keys_eligible(clauses, True, ["id"], _schema(id=[1], v=["a"])) is False


def test_ineligible_without_unique_key_validation():
    # Duplicate source keys fan out the LEFT join, breaking the kept-row count
    # that the metric correction depends on.
    assert _prune_keys_eligible([("insert", None, None, None)], False, ["id"], _schema(id=[1])) is False


def test_ineligible_for_unordered_key_dtype():
    schema = daft.from_pydict({"id": [[1, 2]]}).schema()
    assert _prune_keys_eligible([("insert", None, None, None)], True, ["id"], schema) is False


def test_eligible_for_string_and_temporal_keys():
    import datetime

    schema = daft.from_pydict({"s": ["a"], "d": [datetime.date(2026, 1, 1)]}).schema()
    assert _prune_keys_eligible([("insert", None, None, None)], True, ["s", "d"], schema) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test EXTRA_ARGS="-v tests/io/delta_lake/test_delta_merge_pass1_prune.py -k eligible"`
Expected: FAIL with `ImportError: cannot import name '_prune_keys_eligible'`

- [ ] **Step 3: Write minimal implementation**

Add to `daft/io/delta_lake/_deltalake.py` after `_key_range_filter`:

```python
def _is_ordered_scalar_dtype(dtype: "DataType") -> bool:
    """True for dtypes whose Delta min/max statistics support range comparison."""
    return bool(
        dtype.is_numeric()
        or dtype.is_string()
        or dtype.is_boolean()
        or dtype.is_date()
        or dtype.is_timestamp()
    )


def _prune_keys_eligible(
    clauses: "list[tuple]",
    validate_unique_keys: bool,
    keys: "list[str]",
    target_schema: "Schema",
) -> bool:
    """Whether pass 1's target scan may be pruned to the source key ranges.

    Every gate here protects a correctness property, not a heuristic:

    - A ``by_source_*`` clause acts on target rows with NO source match. Those
      rows are exactly the ones a source-derived prune discards, and their own
      column values decide whether the clause fires. They must be read.
    - Without unique-key validation, duplicate source keys fan out the LEFT
      join so a target row appears more than once, invalidating the
      ``kept_target_rows`` count the metric correction relies on.
    - Non-scalar key dtypes have no meaningful Delta min/max range.
    """
    if not keys:
        return False
    if not validate_unique_keys:
        return False
    for kind, _updates, _pred, _exc in clauses:
        if kind in ("by_source_update", "by_source_delete"):
            return False
    for k in keys:
        if k not in [f.name for f in target_schema]:
            return False
        if not _is_ordered_scalar_dtype(target_schema[k].dtype):
            return False
    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `make test EXTRA_ARGS="-v tests/io/delta_lake/test_delta_merge_pass1_prune.py"`
Expected: PASS (13 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/io/delta_lake/test_delta_merge_pass1_prune.py daft/io/delta_lake/_deltalake.py
git commit -m "perf(delta): add eligibility gate for merge pass-1 pruning"
```

---

### Task 3: Wire the prune into pass 1 with metric correction

**Files:**
- Modify: `daft/io/delta_lake/_deltalake.py:1536-1546` (join construction), `:1685-1691` (annotated frames), `:1693-1710` (agg exprs), `:1761-1768` (unpartitioned metric branch)
- Test: `tests/io/delta_lake/test_delta_merge_pass1_prune.py` (extend)

**Interfaces:**
- Consumes: `_source_key_ranges`, `_key_range_filter`, `_prune_keys_eligible` from Tasks 1-2.
- Produces: a `_TGT_KEPT` counter column name constant and a `stats["kept_target"]` aggregate consumed only within this method.

- [ ] **Step 1: Write the failing test**

Append to `tests/io/delta_lake/test_delta_merge_pass1_prune.py`:

```python
import pytest

pytest.importorskip("deltalake")


def _write_target(tmp_path, ids, vals):
    import deltalake

    deltalake.write_deltalake(str(tmp_path), daft.from_pydict({"id": ids, "v": vals}).to_arrow())
    return str(tmp_path)


def test_prune_preserves_metrics_and_rows_unpartitioned(tmp_path):
    """A tiny merge into a wide key range must report the SAME metrics and leave
    the SAME table as the unpruned path — the prune only changes what is read."""
    path = _write_target(tmp_path / "t", list(range(0, 1000)), [f"v{i}" for i in range(1000)])

    result = (
        daft.distributed_merge_deltalake(
            path, daft.from_pydict({"id": [500], "v": ["NEW"]}), on=["id"]
        )
        .when_matched_update({"v": "source.v"})
        .when_not_matched_insert({"id": "source.id", "v": "source.v"})
        .execute()
    )

    assert result["num_updated_rows"] == 1
    assert result["num_inserted_rows"] == 0
    assert result["num_deleted_rows"] == 0
    # The 999 rows the prune never read are still copied and still emitted.
    assert result["num_copied_rows"] == 999
    assert result["num_output_rows"] == 1000

    out = daft.read_deltalake(path).sort("id").to_pydict()
    assert len(out["id"]) == 1000
    assert out["v"][500] == "NEW"
    assert out["v"][499] == "v499"


def test_prune_skipped_when_by_source_delete_present(tmp_path):
    """by_source_delete must still see unmatched rows, so the prune is off and
    the delete actually fires across the whole table."""
    path = _write_target(tmp_path / "t2", [1, 2, 3], ["a", "b", "c"])

    result = (
        daft.distributed_merge_deltalake(
            path, daft.from_pydict({"id": [2], "v": ["B"]}), on=["id"]
        )
        .when_matched_update({"v": "source.v"})
        .when_not_matched_by_source_delete()
        .execute()
    )

    assert result["num_deleted_rows"] == 2
    assert daft.read_deltalake(path).to_pydict()["id"] == [2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test EXTRA_ARGS="-v tests/io/delta_lake/test_delta_merge_pass1_prune.py -k 'metrics or by_source_delete'"`
Expected: both tests PASS against the current unpruned code — they are the behavioral
baseline. Record the reported `num_copied_rows` / `num_output_rows`. After Step 3 they must
be *unchanged* (999 / 1000) while the scan is pruned; any drift means the metric correction
is wrong.

- [ ] **Step 3: Write minimal implementation**

3a. Replace the join construction at `_deltalake.py:1536-1546` with:

```python
        decompose = _should_decompose_join(
            self._broadcast_join,
            runners.get_or_create_runner().name,
            source_size_bytes,
        )

        # Pass-1 prune: restrict the target scan to the source's key ranges.
        # The predicate is derived only from source key values, so every target
        # row that could match is retained (see the spec's correctness
        # invariant); unmatched target rows may be dropped, which is why the
        # unpartitioned metric branch adds them back below.
        prune_filter = None
        if _prune_keys_eligible(self._clauses, self._validate_unique_keys, on, target_schema):
            ranges = _source_key_ranges(source, on)
            if ranges is not None:
                prune_filter = _key_range_filter(ranges)

        def _build_joined(target_frame: DataFrame) -> DataFrame:
            if decompose:
                # anti_join_target may be the pruned frame: the prune is
                # source-derived, so a source row matched anywhere in the
                # target is still present after pruning.
                return self._decomposed_outer_join(target_frame, source_tagged)
            return target_frame.join(source_tagged, on=on, how="outer", suffix=".__src__")

        joined = _build_joined(target_tagged)
        joined_pass1 = (
            _build_joined(target_tagged.where(prune_filter)) if prune_filter is not None else joined
        )
```

3b. At `_deltalake.py:1685-1691`, build both annotated frames:

```python
        annotated = _annotate(joined)
        # Pass 2 must NEVER use the pruned frame: the unpartitioned write path is
        # a full overwrite and would delete every row the prune skipped.
        annotated_pass1 = _annotate(joined_pass1) if prune_filter is not None else annotated
        if self._materialize_join:
            # One join execution shared by both passes: pinned in cluster
            # memory (spillable on the Ray runner) instead of recomputed
            # for the write pass.
            annotated = annotated.collect()
            annotated_pass1 = annotated
```

3c. In the pass-1 agg list (`:1693-1710`), add a kept-target counter and switch the
source frame. Add this aggregate alongside the existing ones:

```python
            col(_TGT_MARKER).not_null().cast(_i64).sum().alias("kept_target"),
```

and change every `annotated.agg(...)` / `annotated.groupby(...)` call in pass 1 to read
from `annotated_pass1` instead.

3d. In the unpartitioned metric branch (`:1761-1768`), add the correction after the
existing assignments:

```python
            if prune_filter is not None:
                # Rows in Delta files the prune skipped were never read. With no
                # by-source clause (gated above), each is exactly one copied and
                # one emitted row — see the spec's metric accounting.
                total_target_rows = sum(
                    self._resolved_table.get_add_actions().column("num_records").to_pylist()
                )
                kept_target_rows = stats["kept_target"][0] or 0
                skipped = total_target_rows - kept_target_rows
                if skipped < 0:
                    raise RuntimeError(
                        "distributed merge pass-1 prune saw more target rows "
                        f"({kept_target_rows}) than the snapshot holds ({total_target_rows}); "
                        "refusing to report corrupt metrics"
                    )
                num_copied += skipped
                num_output += skipped
```

- [ ] **Step 4: Run test to verify it passes**

Run: `make test EXTRA_ARGS="-v tests/io/delta_lake/test_delta_merge_pass1_prune.py"`
Expected: PASS (15 passed)

Then confirm nothing regressed in the existing merge suites:

Run: `make test EXTRA_ARGS="-v tests/io/delta_lake/"`
Expected: PASS, no new failures

- [ ] **Step 5: Commit**

```bash
git add tests/io/delta_lake/test_delta_merge_pass1_prune.py daft/io/delta_lake/_deltalake.py
git commit -m "perf(delta): prune distributed merge pass-1 scan to source key ranges"
```

---

### Task 4: Prove files are actually skipped, and equivalence on Ray

**Files:**
- Test: `tests/integration/delta_lake/test_distributed_merge_pass1_prune.py` (create)

**Interfaces:**
- Consumes: the wired prune from Task 3. Produces nothing.

- [ ] **Step 1: Write the failing test**

Create `tests/integration/delta_lake/test_distributed_merge_pass1_prune.py`:

```python
from __future__ import annotations

import pytest

import daft

pytestmark = pytest.mark.integration()


def _write_many_files(path, n_files, rows_per_file):
    """One add-action per write so min/max statistics differ per file."""
    import deltalake

    for f in range(n_files):
        lo = f * rows_per_file
        ids = list(range(lo, lo + rows_per_file))
        deltalake.write_deltalake(
            str(path),
            daft.from_pydict({"id": ids, "v": [f"v{i}" for i in ids]}).to_arrow(),
            mode="append" if f else "overwrite",
        )
    return str(path)


def test_prune_reads_only_matching_files(tmp_path):
    """A single-key merge must touch one file's worth of rows, not the table."""
    path = _write_many_files(tmp_path / "t", n_files=20, rows_per_file=100)

    plan = (
        daft.read_deltalake(path)
        .where((daft.col("id") >= 250) & (daft.col("id") <= 250))
        .explain(show_all=True)
    )
    # Sanity: the range predicate is the shape daft-stats can fold.
    assert "id" in str(plan)

    scanned = (
        daft.read_deltalake(path)
        .where((daft.col("id") >= 250) & (daft.col("id") <= 250))
        .count_rows()
    )
    assert scanned == 1

    result = (
        daft.distributed_merge_deltalake(
            path, daft.from_pydict({"id": [250], "v": ["NEW"]}), on=["id"]
        )
        .when_matched_update({"v": "source.v"})
        .when_not_matched_insert({"id": "source.id", "v": "source.v"})
        .execute()
    )
    assert result["num_updated_rows"] == 1
    assert result["num_copied_rows"] == 1999
    assert result["num_output_rows"] == 2000


def test_pruned_and_unpruned_agree(tmp_path):
    """Same merge, prune forced off via a by-source clause vs. on: identical table."""
    src = daft.from_pydict({"id": [3, 4001], "v": ["X", "Y"]})

    pruned_path = _write_many_files(tmp_path / "a", n_files=10, rows_per_file=100)
    (
        daft.distributed_merge_deltalake(pruned_path, src, on=["id"])
        .when_matched_update({"v": "source.v"})
        .when_not_matched_insert({"id": "source.id", "v": "source.v"})
        .execute()
    )

    unpruned_path = _write_many_files(tmp_path / "b", n_files=10, rows_per_file=100)
    (
        daft.distributed_merge_deltalake(
            unpruned_path, src, on=["id"], validate_unique_keys=False
        )
        .when_matched_update({"v": "source.v"})
        .when_not_matched_insert({"id": "source.id", "v": "source.v"})
        .execute()
    )

    a = daft.read_deltalake(pruned_path).sort("id").to_pydict()
    b = daft.read_deltalake(unpruned_path).sort("id").to_pydict()
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test EXTRA_ARGS="-v --integration tests/integration/delta_lake/test_distributed_merge_pass1_prune.py"`
Expected: FAIL if Task 3 is incomplete (wrong `num_copied_rows`); otherwise PASS.

- [ ] **Step 3: Write minimal implementation**

No production change. If `test_prune_reads_only_matching_files` shows `scanned != 1`, the
per-file statistics are not reaching the ScanTask — inspect
`delta_lake_scan.py:378-412` and confirm `add_actions["min"]`/`["max"]` are populated for
the `id` column (they are absent when the table was written without statistics, in which
case regenerate the fixture with a writer that emits them).

- [ ] **Step 4: Run test to verify it passes**

Run with both runners:

```bash
DAFT_RUNNER=native make test EXTRA_ARGS="-v --integration tests/integration/delta_lake/test_distributed_merge_pass1_prune.py"
DAFT_RUNNER=ray make test EXTRA_ARGS="-v --integration tests/integration/delta_lake/test_distributed_merge_pass1_prune.py"
```

Expected: PASS on both. The `ray` run is the one that exercises the decomposed
broadcast `LEFT` + `ANTI` path.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/delta_lake/test_distributed_merge_pass1_prune.py
git commit -m "test(delta): assert merge pass-1 prune skips non-matching Delta files"
```

---

## Known ceiling

For an **unpartitioned** target this plan removes one of two full target passes — roughly
**2x**, the same order as `materialize_join=True` but without the memory cost and
composable with it. It does not approach the 18m→seconds improvement on its own, because
pass 2 still overwrites the whole table. Closing that gap requires **file-level replace in
the write pass** (remove only the add-actions holding matched keys, add their rewrites),
which is deliberately out of scope here and should get its own spec.

For a **partitioned** target whose merge keys are partition-correlated, this plan plus the
existing pass-2 prune (`_deltalake.py:1781`, commit `7798d140a`) means both passes are
pruned — that is the large win available today.

# Join Stats Short-Circuit for Left/Right/Semi/Anti — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `MicroPartition`'s existing range-statistics join short-circuit from `Inner`-only to `Semi`, `Anti`, `Left`, and `Right`, and close the null-equality correctness gap it already has.

**Architecture:** `ops/join.rs:41` proves "no key ranges can overlap" from per-side min/max statistics but only acts on it for `Inner`. The verdict is equally decisive for the other types — it just implies a different output than "empty". Hoist the verdict, then dispatch per join type. Because range statistics discard null counts, the whole short-circuit must first be gated on `null_equals_nulls` being all-false, which requires threading that flag into the private `join`.

**Tech Stack:** Rust, `daft-micropartition`, `daft-stats`, `daft-recordbatch`, cargo test.

**Spec:** `docs/superpowers/specs/2026-09-09-delta-merge-target-pruning.md`

## Global Constraints

- Target file: `src/daft-micropartition/src/ops/join.rs`. `join.rs` currently has **no** `mod tests`; add one modeled on `src/daft-micropartition/src/ops/filter.rs:46-88`.
- `JoinType::Outer` is **out of scope** — its common columns take a supertype (`daft-dsl/src/join.rs:39-43`), so null-extension needs casts. Leave it falling through to the normal path.
- `Left` and `Right` short-circuits must **never** return empty. A `False` verdict means "no matches", which for an outer-preserving join means *null-extended rows*, not *no rows*.
- Use `map_or(true, ...)` rather than `is_none_or` to avoid a Rust version floor.
- After every Rust change run `make build` before running Python tests.
- Do not touch `translate_join.rs` or `broadcast_join.rs`. The `is_swapped = true` for `Left` is semantically required (see spec, "Non-goal").

---

### Task 1: Prove and fix the null-equality gap in the existing `Inner` short-circuit

**Files:**
- Modify: `src/daft-micropartition/src/ops/join.rs:12-19` (private `join` signature), `:41-61` (verdict block), `:98` and `:117` and `:126` (call sites)
- Test: `src/daft-micropartition/src/ops/join.rs` (add `mod tests`)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: private `fn join<F>(&self, right: &Self, left_on: &[BoundExpr], right_on: &[BoundExpr], null_equals_nulls: Option<&[bool]>, how: JoinType, table_join: F) -> DaftResult<Self>` — note the new fifth parameter, used by every later task.

- [ ] **Step 1: Write the failing test**

Append to `src/daft-micropartition/src/ops/join.rs`:

```rust
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use daft_core::{
        datatypes::{DataType, Field, Int64Array},
        join::JoinType,
        prelude::Schema,
        series::IntoSeries,
    };
    use daft_dsl::{expr::bound_expr::BoundExpr, resolved_col};
    use daft_recordbatch::RecordBatch;
    use daft_stats::TableStatistics;

    use crate::MicroPartition;

    /// Builds a one-column i64 MicroPartition whose declared stats range is
    /// [stat_min, stat_max], independent of the rows it actually holds.
    fn mp(name: &str, rows: Vec<Option<i64>>, stat_min: i64, stat_max: i64) -> MicroPartition {
        let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Int64)]));
        let data = RecordBatch::from_nonempty_columns(vec![
            Int64Array::from_iter(Field::new(name, DataType::Int64), rows).into_series(),
        ])
        .unwrap();
        let stats_table = RecordBatch::from_nonempty_columns(vec![
            Int64Array::from_slice(name, &[stat_min, stat_max]).into_series(),
        ])
        .unwrap();
        let stats = TableStatistics::from_stats_table(&stats_table).unwrap();
        MicroPartition::new_loaded(schema, Arc::new(vec![data]), Some(stats))
    }

    fn on(mp: &MicroPartition, name: &str) -> Vec<BoundExpr> {
        vec![BoundExpr::try_new(resolved_col(name), &mp.schema).unwrap()]
    }

    #[test]
    fn test_inner_short_circuit_respects_null_equals_nulls() {
        // Non-null ranges are disjoint ([10,20] vs [30,40]) so the range
        // verdict is False — but both sides hold NULL keys, and with
        // null_equals_nulls the NULL/NULL pair is a real match. Range stats
        // discard null counts (`column_stats/mod.rs:166`), so the
        // short-circuit MUST NOT fire here.
        let left = mp("a", vec![Some(15), None], 10, 20);
        let right = mp("a", vec![Some(35), None], 30, 40);

        let result = left
            .hash_join(
                &right,
                &on(&left, "a"),
                &on(&right, "a"),
                Some(vec![true]),
                JoinType::Inner,
            )
            .unwrap();

        assert_eq!(result.len(), 1, "NULL == NULL match was wrongly pruned");
    }

    #[test]
    fn test_inner_short_circuit_still_fires_when_nulls_unequal() {
        let left = mp("a", vec![Some(15), None], 10, 20);
        let right = mp("a", vec![Some(35), None], 30, 40);

        let result = left
            .hash_join(
                &right,
                &on(&left, "a"),
                &on(&right, "a"),
                None,
                JoinType::Inner,
            )
            .unwrap();

        assert_eq!(result.len(), 0);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p daft-micropartition null_equals_nulls`
Expected: `test_inner_short_circuit_respects_null_equals_nulls` FAILS with `assertion failed: left == right` (`0 != 1`) — the short-circuit fired and dropped a real match.

- [ ] **Step 3: Write minimal implementation**

3a. Change the private signature at `join.rs:12-19` to accept the flag:

```rust
    fn join<F>(
        &self,
        right: &Self,
        left_on: &[BoundExpr],
        right_on: &[BoundExpr],
        null_equals_nulls: Option<&[bool]>,
        how: JoinType,
        table_join: F,
    ) -> DaftResult<Self>
```

3b. Replace the verdict block at `join.rs:40-61` with a gated version:

```rust
        // Range statistics carry no null information: `column_stats/mod.rs:166`
        // computes `_null_count` and throws it away. So disjoint NON-NULL ranges
        // do not prove "no match" when NULL keys are allowed to compare equal —
        // the NULL/NULL pair would still match. Only short-circuit when nulls
        // are unequal.
        let nulls_never_equal = null_equals_nulls.map_or(true, |n| n.iter().all(|&x| !x));

        // TODO(Kevin): short circuits are also possible for other join types
        if how == JoinType::Inner && nulls_never_equal {
            let tv = match (&self.statistics, &right.statistics) {
                (_, None) => TruthValue::Maybe,
                (None, _) => TruthValue::Maybe,
                (Some(l), Some(r)) => {
                    let l_eval_stats = l.eval_expression_list(left_on)?;
                    let r_eval_stats = r.eval_expression_list(right_on)?;
                    let mut curr_tv = TruthValue::Maybe;
                    for (lc, rc) in l_eval_stats.into_iter().zip(&r_eval_stats) {
                        if lc.equal(rc)?.to_truth_value() == TruthValue::False {
                            curr_tv = TruthValue::False;
                            break;
                        }
                    }
                    curr_tv
                }
            };
            if tv == TruthValue::False {
                return Ok(Self::empty(Some(join_schema)));
            }
        }
```

3c. Update the three call sites. At `join.rs:98` (inside `hash_join`, where
`null_equals_nulls` is already a local `Vec<bool>` from `:88`):

```rust
        self.join(
            right,
            left_on,
            right_on,
            Some(null_equals_nulls.as_slice()),
            how,
            table_join,
        )
```

At `join.rs:117` (`sort_merge_join`) and `join.rs:126` (the cross join), pass `None`:

```rust
        self.join(right, left_on, right_on, None, how, table_join)
```

```rust
        self.join(right, &[], &[], None, JoinType::Inner, table_join)
```

Note: `hash_join` shadows `null_equals_nulls` at `:88` with the defaulted `Vec<bool>`;
borrow that shadowed value, not the original `Option<Vec<bool>>` parameter.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p daft-micropartition null_equals_nulls`
Expected: PASS (2 passed)

Then confirm no join regressions:

Run: `cargo test -p daft-micropartition && cargo test -p daft-recordbatch`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/daft-micropartition/src/ops/join.rs
git commit -m "fix(join): don't prune joins on range stats when nulls compare equal"
```

---

### Task 2: Short-circuit `Semi` and `Anti`

**Files:**
- Modify: `src/daft-micropartition/src/ops/join.rs` (verdict block from Task 1)
- Test: `src/daft-micropartition/src/ops/join.rs` (`mod tests`)

**Interfaces:**
- Consumes: the `null_equals_nulls`-gated verdict block and the `mp`/`on` test helpers from Task 1.
- Produces: verdict block restructured so `tv` is computed for every join type and dispatched via `match how`.

- [ ] **Step 1: Write the failing test**

Append inside `mod tests`:

```rust
    #[test]
    fn test_semi_join_short_circuits_to_empty() {
        // No left row can have a match, so a semi join keeps nothing.
        let left = mp("a", vec![Some(15), Some(16)], 10, 20);
        let right = mp("a", vec![Some(35)], 30, 40);

        let result = left
            .hash_join(&right, &on(&left, "a"), &on(&right, "a"), None, JoinType::Semi)
            .unwrap();

        assert_eq!(result.len(), 0);
        assert_eq!(result.schema, left.schema);
    }

    #[test]
    fn test_anti_join_short_circuits_to_all_of_left() {
        // No left row can have a match, so an anti join keeps ALL of left —
        // emphatically not empty.
        let left = mp("a", vec![Some(15), Some(16)], 10, 20);
        let right = mp("a", vec![Some(35)], 30, 40);

        let result = left
            .hash_join(&right, &on(&left, "a"), &on(&right, "a"), None, JoinType::Anti)
            .unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result.schema, left.schema);
        let col = result.concat_or_get().unwrap().unwrap();
        let vals = col.get_column(0).i64().unwrap();
        assert_eq!((vals.get(0), vals.get(1)), (Some(15), Some(16)));
    }

    #[test]
    fn test_anti_join_overlapping_ranges_still_probes() {
        // Ranges overlap → Maybe → normal path, which must actually exclude 15.
        let left = mp("a", vec![Some(15), Some(16)], 10, 20);
        let right = mp("a", vec![Some(15)], 10, 20);

        let result = left
            .hash_join(&right, &on(&left, "a"), &on(&right, "a"), None, JoinType::Anti)
            .unwrap();

        assert_eq!(result.len(), 1);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p daft-micropartition -- semi_join anti_join`
Expected: all three PASS on the existing probe path — they encode the semantics the
short-circuit must preserve, so they are a behavioral baseline rather than a red test. The
short-circuit is a pure performance change here; Step 4 re-runs them against it. If
`test_anti_join_short_circuits_to_all_of_left` fails now, the expectation is wrong — fix it
against `infer_join_schema` before touching production code.

- [ ] **Step 3: Write minimal implementation**

Replace the `if how == JoinType::Inner && nulls_never_equal { ... }` block from Task 1 with:

```rust
        if nulls_never_equal {
            let tv = match (&self.statistics, &right.statistics) {
                (_, None) => TruthValue::Maybe,
                (None, _) => TruthValue::Maybe,
                (Some(l), Some(r)) => {
                    let l_eval_stats = l.eval_expression_list(left_on)?;
                    let r_eval_stats = r.eval_expression_list(right_on)?;
                    let mut curr_tv = TruthValue::Maybe;
                    for (lc, rc) in l_eval_stats.into_iter().zip(&r_eval_stats) {
                        if lc.equal(rc)?.to_truth_value() == TruthValue::False {
                            curr_tv = TruthValue::False;
                            break;
                        }
                    }
                    curr_tv
                }
            };

            if tv == TruthValue::False {
                // No pair of key ranges can overlap, so no row can match. What
                // that implies depends on which side the join preserves.
                match how {
                    // Nothing survives.
                    JoinType::Inner | JoinType::Semi => {
                        return Ok(Self::empty(Some(join_schema)));
                    }
                    // Every left row survives untouched. `infer_join_schema`
                    // returns `left_schema.clone()` for Anti/Semi
                    // (`daft-dsl/src/join.rs:23`), so `self` already carries the
                    // output schema — and cloning keeps it unloaded, skipping
                    // the read entirely.
                    JoinType::Anti => return Ok(self.clone()),
                    // Outer joins are handled in Task 3; Outer stays on the
                    // normal path because its common columns take a supertype.
                    JoinType::Left | JoinType::Right | JoinType::Outer => {}
                }
            }
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p daft-micropartition -- semi_join anti_join`
Expected: PASS (3 passed)

Run: `cargo test -p daft-micropartition`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/daft-micropartition/src/ops/join.rs
git commit -m "perf(join): short-circuit semi and anti joins on disjoint key ranges"
```

---

### Task 3: Short-circuit `Left` and `Right` via bulk null-extension

**Files:**
- Modify: `src/daft-micropartition/src/ops/join.rs` (add `null_extend` helper; extend the `match how`)
- Test: `src/daft-micropartition/src/ops/join.rs` (`mod tests`)

**Interfaces:**
- Consumes: the `match how` dispatch from Task 2.
- Produces: `fn null_extend(base: &MicroPartition, join_schema: &SchemaRef) -> DaftResult<MicroPartition>` (module-private).

- [ ] **Step 1: Write the failing test**

Append inside `mod tests`:

```rust
    /// Two-column builder so the join has a genuinely right-only column to null-fill.
    fn mp2(
        key: &str,
        other: &str,
        keys: Vec<Option<i64>>,
        others: Vec<Option<i64>>,
        stat_min: i64,
        stat_max: i64,
    ) -> MicroPartition {
        let schema = Arc::new(Schema::new(vec![
            Field::new(key, DataType::Int64),
            Field::new(other, DataType::Int64),
        ]));
        let data = RecordBatch::from_nonempty_columns(vec![
            Int64Array::from_iter(Field::new(key, DataType::Int64), keys).into_series(),
            Int64Array::from_iter(Field::new(other, DataType::Int64), others).into_series(),
        ])
        .unwrap();
        let stats_table = RecordBatch::from_nonempty_columns(vec![
            Int64Array::from_slice(key, &[stat_min, stat_max]).into_series(),
            Int64Array::from_slice(other, &[stat_min, stat_max]).into_series(),
        ])
        .unwrap();
        let stats = TableStatistics::from_stats_table(&stats_table).unwrap();
        MicroPartition::new_loaded(schema, Arc::new(vec![data]), Some(stats))
    }

    #[test]
    fn test_left_join_short_circuits_to_null_extended_left() {
        // Disjoint ranges: every left row survives with the right-only column NULL.
        // This must NOT be empty.
        let left = mp2("a", "lv", vec![Some(15), Some(16)], vec![Some(1), Some(2)], 10, 20);
        let right = mp2("a", "rv", vec![Some(35)], vec![Some(9)], 30, 40);

        let result = left
            .hash_join(
                &right,
                &on(&left, "a"),
                &on(&right, "a"),
                None,
                JoinType::Left,
            )
            .unwrap();

        assert_eq!(result.len(), 2, "left join must preserve all left rows");

        let batch = result.concat_or_get().unwrap().unwrap();
        let names: Vec<&str> = result.schema.into_iter().map(|f| f.name.as_ref()).collect();
        assert_eq!(names, vec!["a", "lv", "rv"]);

        let rv_idx = names.iter().position(|n| *n == "rv").unwrap();
        let rv = batch.get_column(rv_idx);
        assert_eq!(rv.len(), 2);
        assert_eq!(rv.i64().unwrap().get(0), None);
        assert_eq!(rv.i64().unwrap().get(1), None);

        let lv_idx = names.iter().position(|n| *n == "lv").unwrap();
        let lv = batch.get_column(lv_idx).i64().unwrap();
        assert_eq!((lv.get(0), lv.get(1)), (Some(1), Some(2)));
    }

    #[test]
    fn test_right_join_short_circuits_to_null_extended_right() {
        let left = mp2("a", "lv", vec![Some(15)], vec![Some(1)], 10, 20);
        let right = mp2("a", "rv", vec![Some(35), Some(36)], vec![Some(9), Some(8)], 30, 40);

        let result = left
            .hash_join(
                &right,
                &on(&left, "a"),
                &on(&right, "a"),
                None,
                JoinType::Right,
            )
            .unwrap();

        assert_eq!(result.len(), 2, "right join must preserve all right rows");

        let batch = result.concat_or_get().unwrap().unwrap();
        let names: Vec<&str> = result.schema.into_iter().map(|f| f.name.as_ref()).collect();
        let lv_idx = names.iter().position(|n| *n == "lv").unwrap();
        assert_eq!(batch.get_column(lv_idx).i64().unwrap().get(0), None);

        let a_idx = names.iter().position(|n| *n == "a").unwrap();
        let a = batch.get_column(a_idx).i64().unwrap();
        assert_eq!((a.get(0), a.get(1)), (Some(35), Some(36)));
    }

    #[test]
    fn test_left_join_overlapping_ranges_still_probes() {
        let left = mp2("a", "lv", vec![Some(15)], vec![Some(1)], 10, 20);
        let right = mp2("a", "rv", vec![Some(15)], vec![Some(9)], 10, 20);

        let result = left
            .hash_join(&right, &on(&left, "a"), &on(&right, "a"), None, JoinType::Left)
            .unwrap();

        assert_eq!(result.len(), 1);
        let batch = result.concat_or_get().unwrap().unwrap();
        let names: Vec<&str> = result.schema.into_iter().map(|f| f.name.as_ref()).collect();
        let rv_idx = names.iter().position(|n| *n == "rv").unwrap();
        assert_eq!(batch.get_column(rv_idx).i64().unwrap().get(0), Some(9));
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p daft-micropartition -- left_join right_join`
Expected: all three PASS on the existing probe path. Like Task 2 these are a semantics
baseline, not a red test — the short-circuit must reproduce these exact values and column
order. The column-order assertions (`["a", "lv", "rv"]`) are the load-bearing part: if they
fail now, `infer_join_schema` orders differently than assumed and both the test and
`null_extend`'s contract need correcting before proceeding.

- [ ] **Step 3: Write minimal implementation**

3a. Add these imports at the top of `join.rs`:

```rust
use std::collections::HashMap;

use daft_core::series::Series;
use daft_schema::schema::SchemaRef;
```

3b. Add the helper above `impl MicroPartition`:

```rust
/// All rows of `base`, with every output column absent from `base` filled with
/// typed NULLs, laid out to match `join_schema`.
///
/// `infer_join_schema` puts common columns first — taking the *preserved*
/// side's field (left's for `Left`, right's for `Right`) — then unique left,
/// then unique right (`daft-dsl/src/join.rs:26-60`). So resolving each output
/// field by name against `base` and null-filling the misses yields the correct
/// layout for both directions with one implementation.
fn null_extend(base: &MicroPartition, join_schema: &SchemaRef) -> DaftResult<MicroPartition> {
    let Some(batch) = base.concat_or_get()? else {
        return Ok(MicroPartition::empty(Some(join_schema.clone())));
    };
    let num_rows = batch.len();

    let base_idx: HashMap<&str, usize> = base
        .schema
        .into_iter()
        .enumerate()
        .map(|(i, f)| (f.name.as_ref(), i))
        .collect();

    let mut columns = Vec::with_capacity(join_schema.len());
    for field in join_schema.into_iter() {
        match base_idx.get(field.name.as_ref()) {
            Some(&i) => columns.push(batch.get_column(i).clone()),
            None => columns.push(Series::full_null(
                field.name.as_ref(),
                &field.dtype,
                num_rows,
            )),
        }
    }

    let extended = RecordBatch::new_with_size(join_schema.clone(), columns, num_rows)?;
    Ok(MicroPartition::new_loaded(
        join_schema.clone(),
        Arc::new(vec![extended]),
        None,
    ))
}
```

3c. Replace the `JoinType::Left | JoinType::Right | JoinType::Outer => {}` arm from
Task 2 with:

```rust
                    // No match exists, so the preserved side's rows all come
                    // through null-extended. This replaces the per-row no-match
                    // path in `join/left_right_join.rs:113-135` (a per-row
                    // `add_nulls(1)` plus `probe_side_idxs.push`, then a full
                    // `take`) with one bulk construction.
                    JoinType::Left => return null_extend(self, &join_schema),
                    JoinType::Right => return null_extend(right, &join_schema),
                    // Outer's common columns take a supertype, so null-extension
                    // would need casts. Left on the normal path deliberately.
                    JoinType::Outer => {}
```

Ensure `use std::sync::Arc;` is present at the top of `join.rs` (needed by
`Arc::new(vec![extended])`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p daft-micropartition -- left_join right_join`
Expected: PASS (3 passed)

Run: `cargo test -p daft-micropartition`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/daft-micropartition/src/ops/join.rs
git commit -m "perf(join): short-circuit left and right joins on disjoint key ranges"
```

---

### Task 4: Verify end-to-end and guard against regression

**Files:**
- Test: `tests/table/test_joins.py` (extend — verify the file exists; otherwise `tests/dataframe/test_joins.py`)

**Interfaces:**
- Consumes: Tasks 1-3. Produces nothing.

- [ ] **Step 1: Write the failing test**

Locate the join test module first:

```bash
ls tests/table/test_joins.py tests/dataframe/test_joins.py 2>/dev/null
```

Append to whichever exists (prefer `tests/dataframe/test_joins.py`):

```python
import daft


def test_left_join_disjoint_ranges_preserves_all_left_rows():
    """Regression guard: the range-stats short-circuit must null-extend, not empty."""
    left = daft.from_pydict({"a": [1, 2, 3], "lv": ["x", "y", "z"]})
    right = daft.from_pydict({"a": [100, 200], "rv": ["p", "q"]})

    out = left.join(right, on="a", how="left").sort("a").to_pydict()
    assert out["a"] == [1, 2, 3]
    assert out["lv"] == ["x", "y", "z"]
    assert out["rv"] == [None, None, None]


def test_anti_join_disjoint_ranges_keeps_all_left_rows():
    left = daft.from_pydict({"a": [1, 2, 3]})
    right = daft.from_pydict({"a": [100, 200]})
    assert left.join(right, on="a", how="anti").sort("a").to_pydict()["a"] == [1, 2, 3]


def test_semi_join_disjoint_ranges_is_empty():
    left = daft.from_pydict({"a": [1, 2, 3]})
    right = daft.from_pydict({"a": [100, 200]})
    assert left.join(right, on="a", how="semi").to_pydict()["a"] == []

```

- [ ] **Step 2: Run test to verify it fails**

```bash
make build
make test EXTRA_ARGS="-v tests/dataframe/test_joins.py -k disjoint"
```

Expected: PASS if Tasks 1-3 are complete. A failure on
`test_left_join_disjoint_ranges_preserves_all_left_rows` with an empty result means the
`Left` arm returned `Self::empty` instead of null-extending — re-check Task 3 step 3c.

- [ ] **Step 3: Write minimal implementation**

No production change. Note that `DataFrame.join` exposes no `null_equals_nulls` parameter
(`daft/dataframe/dataframe.py:4381-4391`), so the null-equality fix from Task 1 is covered
only by the Rust test `test_inner_short_circuit_respects_null_equals_nulls`. Do not add a
Python test for it.

- [ ] **Step 4: Run test to verify it passes**

```bash
DAFT_RUNNER=native make test EXTRA_ARGS="-v tests/dataframe/test_joins.py"
DAFT_RUNNER=ray make test EXTRA_ARGS="-v tests/dataframe/test_joins.py"
```

Expected: PASS on both.

- [ ] **Step 5: Commit**

```bash
git add tests/dataframe/test_joins.py
git commit -m "test(join): cover range-stats short-circuit for left/anti/semi joins"
```

---

## Upstreaming

Tasks 1-3 touch only `src/daft-micropartition/src/ops/join.rs` and are independent of the
fork's Delta merge, so they cherry-pick cleanly onto `upstream/main`. Task 1 is a genuine
bug fix and worth its own PR. Tasks 2-3 close the standing
`TODO(Kevin): short circuits are also possible for other join types` at `join.rs:41`.
Reference [#4047](https://github.com/Eventual-Inc/Daft/issues/4047) for motivation only —
the merge itself is not being proposed upstream here.

**CORRECTION (post-implementation review, 2026-09-16):** "closes the standing TODO" is
accurate, but do not overstate what that buys — this does **not** "help every
`left`/`anti`/`semi` join in Daft". `MicroPartition::hash_join` is not on
`DataFrame.join()`'s execution path for either the native or Ray runner (both go through
`LocalPhysicalPlan::hash_join` → `HashJoinOperator` in `daft-local-execution`, a separate
code path that never calls `MicroPartition::hash_join`). What this plan actually helps is
narrower: `MicroPartition::hash_join`'s only callers — the PyO3 binding — and
`MicroPartition::sort_merge_join`'s `SortMergeJoinOperator`, reached only when a caller
explicitly requests `strategy="sort_merge"`. It is still a correct, worthwhile, and
independently upstreamable fix on its own terms; it just doesn't move the needle on the
primary `DataFrame.join()` query-execution path, and should not be pitched to upstream
maintainers as if it did.

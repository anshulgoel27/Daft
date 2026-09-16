use std::{collections::HashMap, sync::Arc};

use common_error::DaftResult;
use daft_core::{
    array::ops::DaftCompare,
    join::{JoinSide, JoinType},
    prelude::SchemaRef,
    series::Series,
};
use daft_dsl::{expr::bound_expr::BoundExpr, join::infer_join_schema};
use daft_recordbatch::RecordBatch;
use daft_stats::TruthValue;

use crate::micropartition::MicroPartition;

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

impl MicroPartition {
    fn join<F>(
        &self,
        right: &Self,
        left_on: &[BoundExpr],
        right_on: &[BoundExpr],
        null_equals_nulls: Option<&[bool]>,
        how: JoinType,
        table_join: F,
    ) -> DaftResult<Self>
    where
        F: FnOnce(
            &RecordBatch,
            &RecordBatch,
            &[BoundExpr],
            &[BoundExpr],
            JoinType,
        ) -> DaftResult<RecordBatch>,
    {
        let join_schema = infer_join_schema(&self.schema, &right.schema, how)?;
        // `Left`/`Right` must never return empty: an empty preserved side has
        // zero rows regardless (nothing to null-extend), but an empty
        // *other* side means every preserved row survives null-extended, not
        // dropped. Route those two cases to `null_extend` explicitly instead
        // of falling through to the later `concat_or_get` -> `None` ->
        // `Self::empty` path, which would incorrectly drop the preserved
        // side's rows too.
        match (how, self.len(), right.len()) {
            (JoinType::Inner | JoinType::Semi, 0, _) | (JoinType::Inner, _, 0) => {
                return Ok(Self::empty(Some(join_schema)));
            }
            (JoinType::Left, _, 0) => {
                return null_extend(self, &join_schema);
            }
            (JoinType::Right, 0, _) => {
                return null_extend(right, &join_schema);
            }
            (JoinType::Outer, 0, 0) => {
                return Ok(Self::empty(Some(join_schema)));
            }
            _ => {}
        }

        // Range statistics carry no null information: `column_stats/mod.rs:166`
        // computes `_null_count` and throws it away. So disjoint NON-NULL ranges
        // do not prove "no match" when NULL keys are allowed to compare equal —
        // the NULL/NULL pair would still match. Only short-circuit when nulls
        // are unequal.
        let nulls_never_equal = null_equals_nulls.is_none_or(|n| n.iter().all(|&x| !x));

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
                    // No match exists, so the preserved side's rows all come
                    // through null-extended. This is the bulk equivalent of
                    // the per-row no-match construction in
                    // `daft-local-execution/src/join/left_right_join.rs` — a
                    // separate execution path this change does not touch or
                    // replace.
                    JoinType::Left => return null_extend(self, &join_schema),
                    JoinType::Right => return null_extend(right, &join_schema),
                    // Outer's common columns take a supertype, so null-extension
                    // would need casts. Left on the normal path deliberately.
                    JoinType::Outer => {}
                }
            }
        }

        // TODO(Clark): Elide concatenations where possible by doing a chunk-aware local table join.
        let lt = self.concat_or_get()?;
        let rt = right.concat_or_get()?;

        match (lt, rt) {
            (None, _) | (_, None) => Ok(Self::empty(Some(join_schema))),
            (Some(lt), Some(rt)) => {
                let joined_table = table_join(&lt, &rt, left_on, right_on, how)?;
                Ok(Self::new_loaded(
                    join_schema,
                    vec![joined_table].into(),
                    None,
                ))
            }
        }
    }

    pub fn hash_join(
        &self,
        right: &Self,
        left_on: &[BoundExpr],
        right_on: &[BoundExpr],
        null_equals_nulls: Option<Vec<bool>>,
        how: JoinType,
    ) -> DaftResult<Self> {
        let null_equals_nulls = null_equals_nulls.unwrap_or_else(|| vec![false; left_on.len()]);

        let table_join = |lt: &RecordBatch,
                          rt: &RecordBatch,
                          lo: &[BoundExpr],
                          ro: &[BoundExpr],
                          _how: JoinType| {
            RecordBatch::hash_join(lt, rt, lo, ro, null_equals_nulls.as_slice(), _how)
        };

        self.join(
            right,
            left_on,
            right_on,
            Some(null_equals_nulls.as_slice()),
            how,
            table_join,
        )
    }

    pub fn sort_merge_join(
        &self,
        right: &Self,
        left_on: &[BoundExpr],
        right_on: &[BoundExpr],
        how: JoinType,
        is_sorted: bool,
    ) -> DaftResult<Self> {
        let table_join = |lt: &RecordBatch,
                          rt: &RecordBatch,
                          lo: &[BoundExpr],
                          ro: &[BoundExpr],
                          how: JoinType| {
            RecordBatch::sort_merge_join(lt, rt, lo, ro, how, is_sorted)
        };

        self.join(right, left_on, right_on, None, how, table_join)
    }

    pub fn cross_join(&self, right: &Self, outer_loop_side: JoinSide) -> DaftResult<Self> {
        let table_join =
            |lt: &RecordBatch, rt: &RecordBatch, _: &[BoundExpr], _: &[BoundExpr], _: JoinType| {
                RecordBatch::cross_join(lt, rt, outer_loop_side)
            };

        self.join(right, &[], &[], None, JoinType::Inner, table_join)
    }
}

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

    #[test]
    fn test_semi_join_short_circuits_to_empty() {
        // No left row can have a match, so a semi join keeps nothing.
        let left = mp("a", vec![Some(15), Some(16)], 10, 20);
        let right = mp("a", vec![Some(35)], 30, 40);

        let result = left
            .hash_join(
                &right,
                &on(&left, "a"),
                &on(&right, "a"),
                None,
                JoinType::Semi,
            )
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
            .hash_join(
                &right,
                &on(&left, "a"),
                &on(&right, "a"),
                None,
                JoinType::Anti,
            )
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
            .hash_join(
                &right,
                &on(&left, "a"),
                &on(&right, "a"),
                None,
                JoinType::Anti,
            )
            .unwrap();

        assert_eq!(result.len(), 1);
        let col = result.concat_or_get().unwrap().unwrap();
        let vals = col.get_column(0).i64().unwrap();
        assert_eq!(
            vals.get(0),
            Some(16),
            "anti join should keep 16, not the matched 15"
        );
    }

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
        assert_eq!(names, vec!["a", "lv", "rv"]);

        let lv_idx = names.iter().position(|n| *n == "lv").unwrap();
        let lv = batch.get_column(lv_idx);
        assert_eq!(lv.len(), 2);
        assert_eq!(lv.i64().unwrap().get(0), None);
        assert_eq!(lv.i64().unwrap().get(1), None);

        let a_idx = names.iter().position(|n| *n == "a").unwrap();
        let a = batch.get_column(a_idx).i64().unwrap();
        assert_eq!((a.get(0), a.get(1)), (Some(35), Some(36)));

        let rv_idx = names.iter().position(|n| *n == "rv").unwrap();
        let rv = batch.get_column(rv_idx).i64().unwrap();
        assert_eq!(
            (rv.get(0), rv.get(1)),
            (Some(9), Some(8)),
            "right join must preserve rv's real values, not null-fill them"
        );
    }

    #[test]
    fn test_left_join_with_empty_right_preserves_all_left_rows() {
        // Regression test: `right.len() == 0` must never make a `Left` join
        // return empty. Every left row must survive, null-extended, even
        // though nothing here is disjoint-by-stats — it's the plain
        // `right.len() == 0` guard at the top of `join`, not the range-stats
        // short-circuit, that must handle this.
        let left = mp2("a", "lv", vec![Some(15), Some(16)], vec![Some(1), Some(2)], 10, 20);
        let right = mp2("a", "rv", vec![], vec![], 10, 20);
        assert_eq!(right.len(), 0);

        let result = left
            .hash_join(
                &right,
                &on(&left, "a"),
                &on(&right, "a"),
                None,
                JoinType::Left,
            )
            .unwrap();

        assert_eq!(
            result.len(),
            2,
            "left join with an empty right must still preserve all left rows"
        );

        let batch = result.concat_or_get().unwrap().unwrap();
        let names: Vec<&str> = result.schema.into_iter().map(|f| f.name.as_ref()).collect();
        let rv_idx = names.iter().position(|n| *n == "rv").unwrap();
        let rv = batch.get_column(rv_idx).i64().unwrap();
        assert_eq!((rv.get(0), rv.get(1)), (None, None));

        let lv_idx = names.iter().position(|n| *n == "lv").unwrap();
        let lv = batch.get_column(lv_idx).i64().unwrap();
        assert_eq!((lv.get(0), lv.get(1)), (Some(1), Some(2)));
    }

    #[test]
    fn test_right_join_with_empty_left_preserves_all_right_rows() {
        // Mirror of the above: `self.len() == 0` must never make a `Right`
        // join return empty.
        let left = mp2("a", "lv", vec![], vec![], 10, 20);
        let right = mp2("a", "rv", vec![Some(35), Some(36)], vec![Some(9), Some(8)], 30, 40);
        assert_eq!(left.len(), 0);

        let result = left
            .hash_join(
                &right,
                &on(&left, "a"),
                &on(&right, "a"),
                None,
                JoinType::Right,
            )
            .unwrap();

        assert_eq!(
            result.len(),
            2,
            "right join with an empty left must still preserve all right rows"
        );

        let batch = result.concat_or_get().unwrap().unwrap();
        let names: Vec<&str> = result.schema.into_iter().map(|f| f.name.as_ref()).collect();
        let lv_idx = names.iter().position(|n| *n == "lv").unwrap();
        let lv = batch.get_column(lv_idx).i64().unwrap();
        assert_eq!((lv.get(0), lv.get(1)), (None, None));

        let a_idx = names.iter().position(|n| *n == "a").unwrap();
        let a = batch.get_column(a_idx).i64().unwrap();
        assert_eq!((a.get(0), a.get(1)), (Some(35), Some(36)));

        let rv_idx = names.iter().position(|n| *n == "rv").unwrap();
        let rv = batch.get_column(rv_idx).i64().unwrap();
        assert_eq!((rv.get(0), rv.get(1)), (Some(9), Some(8)));
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
}

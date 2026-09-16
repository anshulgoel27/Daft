use common_error::DaftResult;
use daft_core::{
    array::ops::DaftCompare,
    join::{JoinSide, JoinType},
};
use daft_dsl::{expr::bound_expr::BoundExpr, join::infer_join_schema};
use daft_recordbatch::RecordBatch;
use daft_stats::TruthValue;

use crate::micropartition::MicroPartition;

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
        match (how, self.len(), right.len()) {
            (JoinType::Inner | JoinType::Left | JoinType::Semi, 0, _)
            | (JoinType::Inner | JoinType::Right, _, 0)
            | (JoinType::Outer, 0, 0) => {
                return Ok(Self::empty(Some(join_schema)));
            }
            _ => {}
        }

        // Range statistics carry no null information: `column_stats/mod.rs:166`
        // computes `_null_count` and throws it away. So disjoint NON-NULL ranges
        // do not prove "no match" when NULL keys are allowed to compare equal —
        // the NULL/NULL pair would still match. Only short-circuit when nulls
        // are unequal.
        let nulls_never_equal = null_equals_nulls.map_or(true, |n| n.iter().all(|&x| !x));

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
                          _how: JoinType| {
            RecordBatch::sort_merge_join(lt, rt, lo, ro, is_sorted)
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
}

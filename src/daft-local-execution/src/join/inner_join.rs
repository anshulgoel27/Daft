use std::sync::Arc;

use common_error::DaftResult;
use daft_core::prelude::UInt64Array;
use daft_micropartition::MicroPartition;
use daft_recordbatch::{GrowableRecordBatch, ProbeState};

use crate::join::hash_join::HashJoinParams;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use common_error::DaftResult;
    use daft_core::join::JoinType;
    use daft_core::prelude::*;
    use daft_dsl::expr::bound_expr::BoundExpr;
    use daft_dsl::resolved_col;
    use daft_micropartition::MicroPartition;
    use daft_recordbatch::{ProbeState, RecordBatch, make_probeable_builder};
    use indexmap::IndexSet;

    use super::*;
    use crate::join::hash_join::HashJoinParams;

    fn i64_batch(name1: &str, v1: Vec<i64>, name2: &str, v2: Vec<i64>) -> RecordBatch {
        let a = Int64Array::from_vec(name1, v1).into_series();
        let b = Int64Array::from_vec(name2, v2).into_series();
        RecordBatch::from_nonempty_columns(vec![a, b]).unwrap()
    }

    fn run(build_tables: Vec<RecordBatch>) -> DaftResult<usize> {
        // build side: columns (k, v); probe side: columns (k, w); inner join on k.
        let key_schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64)]));
        let left_schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64),
            Field::new("v", DataType::Int64),
        ]));
        let right_schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64),
            Field::new("w", DataType::Int64),
        ]));
        let output_schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64),
            Field::new("v", DataType::Int64),
            Field::new("w", DataType::Int64),
        ]));
        let build_on = vec![BoundExpr::try_new(resolved_col("k"), &left_schema)?];
        let probe_on = vec![BoundExpr::try_new(resolved_col("k"), &right_schema)?];

        // Probe table is built over the build-side KEY columns.
        let mut builder = make_probeable_builder(key_schema.clone(), None, true)?;
        for t in &build_tables {
            let keys = t.eval_expression_list(&build_on)?;
            builder.add_table(&keys)?;
        }
        let probeable = builder.build();
        let probe_state = ProbeState::new(probeable, build_tables);

        let mut common = IndexSet::new();
        common.insert("k".to_string());
        let params = HashJoinParams {
            key_schema,
            build_on,
            probe_on,
            nulls_equal_aware: None,
            track_indices: true,
            join_type: JoinType::Inner,
            build_on_left: true,
            left_schema,
            right_schema: right_schema.clone(),
            common_join_cols: common,
            output_schema,
            spill_config: None,
        };

        let probe_input = MicroPartition::new_loaded(
            right_schema,
            Arc::new(vec![i64_batch("k", vec![2, 3, 4], "w", vec![200, 300, 400])]),
            None,
        );
        let out = probe_inner(&probe_input, &probe_state, &params)?;
        Ok(out.len())
    }

    #[test]
    fn inner_join_single_and_multi_table_build_match() {
        // build (k,v): k in {1,2,3}; probe k in {2,3,4} -> inner matches k=2,3 -> 2 rows.
        let single = vec![i64_batch("k", vec![1, 2, 3], "v", vec![10, 20, 30])];
        let multi = vec![
            i64_batch("k", vec![1, 2], "v", vec![10, 20]),
            i64_batch("k", vec![3], "v", vec![30]),
        ];
        assert_eq!(run(single).unwrap(), 2, "single-table build");
        assert_eq!(run(multi).unwrap(), 2, "multi-table build");
    }
}

pub(crate) fn probe_inner(
    input: &MicroPartition,
    probe_state: &ProbeState,
    params: &HashJoinParams,
) -> DaftResult<MicroPartition> {
    let build_side_tables = probe_state.get_record_batches().iter().collect::<Vec<_>>();

    let input_tables = input.record_batches();
    let result_tables = input_tables
        .iter()
        .map(|input_table| {
            let est = input_table.len();
            let join_keys = input_table.eval_expression_list(&params.probe_on)?;
            let idx_iter = probe_state.probe_indices(join_keys)?;

            let mut probe_side_idxs: Vec<u64> = Vec::with_capacity(est);
            let build_side_table = if build_side_tables.len() == 1 {
                // Single build table: collect matched build row indices and gather with one
                // vectorized take (mirrors the probe side) — no row-at-a-time growable.
                let single = build_side_tables[0];
                let mut build_row_idxs: Vec<u64> = Vec::with_capacity(est);
                for (probe_row_idx, inner_iter) in idx_iter.enumerate() {
                    if let Some(inner_iter) = inner_iter {
                        for (build_rb_idx, build_row_idx) in inner_iter {
                            debug_assert_eq!(build_rb_idx, 0, "single build table => rb idx 0");
                            build_row_idxs.push(build_row_idx);
                            probe_side_idxs.push(probe_row_idx as u64);
                        }
                    }
                }
                single.take(&UInt64Array::from_vec("", build_row_idxs))?
            } else {
                // Multiple build tables: pre-sized growable gather.
                let mut build_side_growable =
                    GrowableRecordBatch::new(&build_side_tables, false, est)?;
                for (probe_row_idx, inner_iter) in idx_iter.enumerate() {
                    if let Some(inner_iter) = inner_iter {
                        for (build_rb_idx, build_row_idx) in inner_iter {
                            build_side_growable.extend(
                                build_rb_idx as usize,
                                build_row_idx as usize,
                                1,
                            );
                            probe_side_idxs.push(probe_row_idx as u64);
                        }
                    }
                }
                build_side_growable.build()?
            };

            let probe_side_table = {
                let indices_arr = UInt64Array::from_vec("", probe_side_idxs);
                input_table.take(&indices_arr)?
            };

            let (left_table, right_table) = if params.build_on_left {
                (build_side_table, probe_side_table)
            } else {
                (probe_side_table, build_side_table)
            };

            let common_join_keys: Vec<String> = params.common_join_cols.iter().cloned().collect();
            let left_non_join_columns: Vec<String> = params
                .left_schema
                .field_names()
                .filter(|c| !params.common_join_cols.contains(*c))
                .map(ToString::to_string)
                .collect();
            let right_non_join_columns: Vec<String> = params
                .right_schema
                .field_names()
                .filter(|c| !params.common_join_cols.contains(*c))
                .map(ToString::to_string)
                .collect();

            let join_keys_table =
                daft_recordbatch::get_columns_by_name(&left_table, &common_join_keys)?;
            let left_non_join_columns =
                daft_recordbatch::get_columns_by_name(&left_table, &left_non_join_columns)?;
            let right_non_join_columns =
                daft_recordbatch::get_columns_by_name(&right_table, &right_non_join_columns)?;
            let final_table = join_keys_table
                .union(&left_non_join_columns)?
                .union(&right_non_join_columns)?;
            Ok(final_table)
        })
        .collect::<DaftResult<Vec<_>>>()?;

    Ok(MicroPartition::new_loaded(
        params.output_schema.clone(),
        Arc::new(result_tables),
        None,
    ))
}

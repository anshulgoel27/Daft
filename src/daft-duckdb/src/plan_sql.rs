use std::collections::HashSet;
use std::sync::Arc;

use common_error::{DaftError, DaftResult};
use daft_core::prelude::{DataType, Schema};
use daft_local_plan::{LocalPhysicalPlan, LocalPhysicalPlanRef, SourceId};

use crate::expr_sql::{agg_to_sql, expr_to_sql};

/// Map a Daft DataType to the DuckDB SQL type name for CAST, so aggregation results
/// have the expected type rather than DuckDB's default (e.g. SUM→HUGEINT, COUNT→BIGINT).
///
/// Returning `None` is intentional: this is a type-only best-effort mapping for the POC.
/// Non-primitive output types (Decimal, Date, etc.) are left uncasted rather than guessed,
/// so we never emit a wrong CAST silently.
fn dtype_to_duckdb_cast(dtype: &DataType) -> Option<&'static str> {
    match dtype {
        DataType::Int8 => Some("TINYINT"),
        DataType::Int16 => Some("SMALLINT"),
        DataType::Int32 => Some("INTEGER"),
        DataType::Int64 => Some("BIGINT"),
        DataType::UInt8 => Some("UTINYINT"),
        DataType::UInt16 => Some("USMALLINT"),
        DataType::UInt32 => Some("UINTEGER"),
        DataType::UInt64 => Some("UBIGINT"),
        DataType::Float32 => Some("FLOAT"),
        DataType::Float64 => Some("DOUBLE"),
        DataType::Utf8 => Some("VARCHAR"),
        _ => None,
    }
}

#[derive(Clone)]
pub struct SqlPlan {
    pub sql: String,
    pub bindings: Vec<SourceId>,
}

fn unsupported_node(node: &str) -> DaftError {
    DaftError::NotImplemented(format!("duckdb POC: unsupported plan node: {node}"))
}

/// Build a single join-key equality predicate for the ON clause.
/// When `null_equals_null` is true, Daft matches NULL == NULL, so we use DuckDB's
/// `IS NOT DISTINCT FROM`; otherwise plain `=` (where NULL never matches).
/// `l_sql`/`r_sql` are already-qualified column SQL fragments, e.g. `l."k"` / `r."k"`.
fn join_key_predicate(l_sql: &str, r_sql: &str, null_equals_null: bool) -> String {
    if null_equals_null {
        format!("l.{l_sql} IS NOT DISTINCT FROM r.{r_sql}")
    } else {
        format!("l.{l_sql} = r.{r_sql}")
    }
}

/// Comma-separated quoted output column list from a node's schema.
fn projection_list(schema: &Schema) -> String {
    schema
        .fields()
        .iter()
        .map(|f| format!("\"{}\"", f.name))
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn plan_to_sql(plan: &LocalPhysicalPlanRef) -> DaftResult<SqlPlan> {
    let mut bindings = Vec::new();
    let sql = walk(plan, &mut bindings)?;
    Ok(SqlPlan { sql, bindings })
}

fn walk(plan: &LocalPhysicalPlanRef, bindings: &mut Vec<SourceId>) -> DaftResult<String> {
    match plan.as_ref() {
        LocalPhysicalPlan::InMemoryScan(scan) => {
            bindings.push(scan.source_id);
            Ok(format!(
                "SELECT {} FROM daft_src_{}",
                projection_list(&scan.schema),
                scan.source_id
            ))
        }
        LocalPhysicalPlan::Filter(f) => {
            let child = walk(&f.input, bindings)?;
            let pred = expr_to_sql(f.predicate.inner(), &f.input.schema())?;
            Ok(format!(
                "SELECT {} FROM ({child}) WHERE {pred}",
                projection_list(&f.schema)
            ))
        }
        LocalPhysicalPlan::Project(p) => {
            let child = walk(&p.input, bindings)?;
            let input_schema = p.input.schema();
            let select_items: Vec<String> = p
                .projection
                .iter()
                .zip(p.schema.fields().iter())
                .map(|(expr, field)| {
                    let sql = expr_to_sql(expr.inner(), &input_schema)?;
                    Ok(format!("{sql} AS \"{}\"", field.name))
                })
                .collect::<DaftResult<Vec<_>>>()?;
            Ok(format!(
                "SELECT {} FROM ({child})",
                select_items.join(", ")
            ))
        }
        LocalPhysicalPlan::HashJoin(j) => {
            if j.join_type != daft_core::join::JoinType::Inner {
                return Err(unsupported_node(&format!(
                    "{:?} join (inner only)",
                    j.join_type
                )));
            }
            let left = walk(&j.left, bindings)?;
            let right = walk(&j.right, bindings)?;
            // Build a set of column names used as join keys on the LEFT side.
            // For an inner join, Daft coalesces each join-key pair into a single column
            // (taken from the left side). When we emit `SELECT ... FROM l JOIN r ON ...`,
            // the join-key column name is ambiguous unless we qualify it to one side.
            // We collect the left-side join-key column names so we can qualify them below.
            let left_key_names: HashSet<Arc<str>> = j
                .left_on
                .iter()
                .filter_map(|e| {
                    if let daft_dsl::Expr::Column(daft_dsl::Column::Bound(bc)) = e.inner().as_ref() {
                        Some(j.left.schema()[bc.index].name.clone())
                    } else {
                        None
                    }
                })
                .collect();
            let right_key_names: HashSet<Arc<str>> = j
                .right_on
                .iter()
                .filter_map(|e| {
                    if let daft_dsl::Expr::Column(daft_dsl::Column::Bound(bc)) = e.inner().as_ref() {
                        Some(j.right.schema()[bc.index].name.clone())
                    } else {
                        None
                    }
                })
                .collect();
            // Build qualified projection list: for each output column, qualify with l. or r.
            // Join keys (present in both sides) → qualify to l (matches Daft coalescing semantics).
            // Non-key left columns → l.<name>; non-key right columns → r.<name>.
            //
            // INVARIANT: at the physical layer, the ONLY same-named columns shared between the
            // left and right inputs are the coalesced join keys (Daft's logical join planner —
            // daft-logical-plan/src/ops/join.rs — renames/uniquifies any other colliding names
            // before producing the physical schema). So a name present in both `left_col_names`
            // and `right_col_names` is necessarily a coalesced join key; non-key names are unique
            // to one side. This is what makes the l/r qualification below unambiguous.
            let left_schema = j.left.schema();
            let right_schema = j.right.schema();
            let left_col_names: HashSet<Arc<str>> = left_schema
                .fields()
                .iter()
                .map(|f| f.name.clone())
                .collect();
            let right_col_names: HashSet<Arc<str>> = right_schema
                .fields()
                .iter()
                .map(|f| f.name.clone())
                .collect();
            let select_items: Vec<String> = j
                .schema
                .fields()
                .iter()
                .map(|f| {
                    let name = &f.name;
                    // If this column is a join key (coalesced from both sides), take from l.
                    if left_key_names.contains(name.as_ref()) && right_key_names.contains(name.as_ref()) {
                        format!("l.\"{name}\" AS \"{name}\"")
                    } else if left_col_names.contains(name.as_ref()) {
                        format!("l.\"{name}\" AS \"{name}\"")
                    } else if right_col_names.contains(name.as_ref()) {
                        format!("r.\"{name}\" AS \"{name}\"")
                    } else {
                        // Fallback: unqualified (shouldn't happen)
                        format!("\"{name}\"")
                    }
                })
                .collect();
            // Build the ON clause. Honor `null_equals_null` per key: when true, Daft matches
            // NULL == NULL, which plain `=` does not — emit `IS NOT DISTINCT FROM` for that key.
            // `null_equals_null` is an Option<Vec<bool>> aligned positionally with the key list;
            // None (or false) → standard `=` (NULL never matches).
            let on = j
                .left_on
                .iter()
                .zip(j.right_on.iter())
                .enumerate()
                .map(|(i, (l, r))| {
                    let l_sql = expr_to_sql(l.inner(), &j.left.schema())?;
                    let r_sql = expr_to_sql(r.inner(), &j.right.schema())?;
                    let null_eq = j
                        .null_equals_null
                        .as_ref()
                        .and_then(|v| v.get(i).copied())
                        .unwrap_or(false);
                    Ok(join_key_predicate(&l_sql, &r_sql, null_eq))
                })
                .collect::<DaftResult<Vec<_>>>()?
                .join(" AND ");
            Ok(format!(
                "SELECT {} FROM ({left}) AS l INNER JOIN ({right}) AS r ON {on}",
                select_items.join(", ")
            ))
        }
        LocalPhysicalPlan::HashAggregate(a) => {
            let child = walk(&a.input, bindings)?;
            let groups: Vec<String> = a
                .group_by
                .iter()
                .map(|g| expr_to_sql(g.inner(), &a.input.schema()))
                .collect::<DaftResult<Vec<_>>>()?;
            let select =
                aggregate_select(&a.schema, &groups, &a.aggregations, &a.input.schema())?;
            Ok(format!(
                "SELECT {select} FROM ({child}) GROUP BY {}",
                groups.join(", ")
            ))
        }
        LocalPhysicalPlan::UnGroupedAggregate(a) => {
            let child = walk(&a.input, bindings)?;
            let select = aggregate_select(&a.schema, &[], &a.aggregations, &a.input.schema())?;
            Ok(format!("SELECT {select} FROM ({child})"))
        }
        other => Err(unsupported_node(other.name())),
    }
}

/// Build the SELECT list for an aggregate: group columns first, then aggregations,
/// each aliased to the matching output-schema field name (positional).
///
/// The output schema is produced by the logical planner with group-by columns first,
/// then aggregation columns (confirmed in daft-logical-plan/src/ops/agg.rs:
/// `exprs_to_schema(&[groupby.as_slice(), aggregations.as_slice()].concat(), ...)`).
fn aggregate_select(
    out_schema: &Schema,
    groups: &[String],
    aggs: &[daft_dsl::expr::bound_expr::BoundAggExpr],
    input_schema: &Schema,
) -> DaftResult<String> {
    let mut items = Vec::new();
    let names: Vec<&str> = out_schema.fields().iter().map(|f| f.name.as_ref()).collect();
    let expected = groups.len() + aggs.len();
    if names.len() != expected {
        return Err(DaftError::ValueError(format!(
            "duckdb POC: aggregate output schema has {} fields but expected {} (group-by {} + aggregations {})",
            names.len(),
            expected,
            groups.len(),
            aggs.len()
        )));
    }
    let name_at = |idx: usize| -> DaftResult<&str> {
        names.get(idx).copied().ok_or_else(|| {
            DaftError::ValueError(format!(
                "duckdb POC: aggregate output schema field index {idx} out of bounds ({} fields)",
                names.len()
            ))
        })
    };
    for (i, g) in groups.iter().enumerate() {
        items.push(format!("{g} AS \"{}\"", name_at(i)?));
    }
    let fields: Vec<_> = out_schema.fields().iter().collect();
    for (j, agg) in aggs.iter().enumerate() {
        let out_idx = groups.len() + j;
        let sql = agg_to_sql(agg.inner(), input_schema)?;
        // CAST the aggregate result to the expected Daft output type.
        // DuckDB SUM(int) → HUGEINT (mapped to Decimal128), but Daft expects Int64.
        // DuckDB COUNT → BIGINT (Int64), but Daft expects UInt64. Cast to align.
        let cast_sql = if let Some(cast_type) = dtype_to_duckdb_cast(&fields[out_idx].dtype) {
            format!("CAST({sql} AS {cast_type})")
        } else {
            sql
        };
        items.push(format!("{cast_sql} AS \"{}\"", name_at(out_idx)?));
    }
    Ok(items.join(", "))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use daft_core::prelude::*;
    use daft_dsl::expr::AggExpr;
    use daft_dsl::{lit, resolved_col};
    use daft_local_plan::{LocalNodeContext, LocalPhysicalPlan};
    use daft_logical_plan::stats::StatsState;

    use super::*;

    fn scan(schema: SchemaRef, source_id: u32) -> daft_local_plan::LocalPhysicalPlanRef {
        LocalPhysicalPlan::in_memory_scan(
            source_id,
            schema,
            0,
            StatsState::NotMaterialized,
            LocalNodeContext::default(),
        )
    }

    #[test]
    fn filter_over_scan() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("amount", DataType::Int64),
            Field::new("region", DataType::Utf8),
        ]));
        let s = scan(schema.clone(), 7);
        let pred = daft_dsl::expr::bound_expr::BoundExpr::try_new(
            resolved_col("amount").gt(lit(100i64)),
            &schema,
        )
        .unwrap();
        let f = LocalPhysicalPlan::filter(
            s,
            pred,
            StatsState::NotMaterialized,
            LocalNodeContext::default(),
        );
        let out = plan_to_sql(&f).unwrap();
        assert_eq!(out.bindings, vec![7]);
        assert_eq!(
            out.sql,
            "SELECT \"amount\", \"region\" FROM (SELECT \"amount\", \"region\" FROM daft_src_7) WHERE (\"amount\" > 100)"
        );
    }

    #[test]
    fn hash_aggregate_group_cols_first_then_aggs() {
        use daft_dsl::expr::bound_expr::{BoundAggExpr, BoundExpr};

        // Input: region (Utf8), amount (Int64).
        let input_schema = Arc::new(Schema::new(vec![
            Field::new("region", DataType::Utf8),
            Field::new("amount", DataType::Int64),
        ]));
        let s = scan(input_schema.clone(), 3);

        // group_by [region], aggregations [sum(amount)].
        let group_by =
            vec![BoundExpr::try_new(resolved_col("region"), &input_schema).unwrap()];
        let aggregations = vec![BoundAggExpr::try_new(
            AggExpr::Sum(resolved_col("amount")),
            &input_schema,
        )
        .unwrap()];

        // Output schema: planner emits group-by cols first, then aggregations
        // (daft-logical-plan/src/ops/agg.rs:42-43). Sum(Int64) -> Int64.
        let out_schema = Arc::new(Schema::new(vec![
            Field::new("region", DataType::Utf8),
            Field::new("total", DataType::Int64),
        ]));

        let agg = LocalPhysicalPlan::hash_aggregate(
            s,
            aggregations,
            group_by,
            out_schema,
            StatsState::NotMaterialized,
            LocalNodeContext::default(),
        );

        let out = plan_to_sql(&agg).unwrap();
        assert_eq!(out.bindings, vec![3]);
        // Group col ("region") precedes the SUM(...) in the SELECT list, and GROUP BY "region".
        // The SUM is CAST to BIGINT so the result type matches Daft's Int64 output
        // (DuckDB's default SUM(int) → HUGEINT/Decimal128).
        assert_eq!(
            out.sql,
            "SELECT \"region\" AS \"region\", CAST(SUM(\"amount\") AS BIGINT) AS \"total\" \
             FROM (SELECT \"region\", \"amount\" FROM daft_src_3) GROUP BY \"region\""
        );
    }

    #[test]
    fn join_key_predicate_uses_eq_when_null_not_equal() {
        // null_equals_null = false → plain `=` (NULL never matches).
        assert_eq!(
            join_key_predicate("\"k\"", "\"k\"", false),
            "l.\"k\" = r.\"k\""
        );
    }

    #[test]
    fn join_key_predicate_uses_is_not_distinct_when_null_equal() {
        // null_equals_null = true → `IS NOT DISTINCT FROM` (NULL matches NULL).
        assert_eq!(
            join_key_predicate("\"k\"", "\"k\"", true),
            "l.\"k\" IS NOT DISTINCT FROM r.\"k\""
        );
    }

    #[test]
    fn hash_join_null_equals_null_emits_is_not_distinct_from() {
        use daft_dsl::expr::bound_expr::BoundExpr;
        use daft_core::join::JoinType;

        // Left: k (Int64), amount (Int64). Right: k (Int64), region (Utf8).
        let left_schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64),
            Field::new("amount", DataType::Int64),
        ]));
        let right_schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64),
            Field::new("region", DataType::Utf8),
        ]));
        let l = scan(left_schema.clone(), 1);
        let r = scan(right_schema.clone(), 2);

        let left_on = vec![BoundExpr::try_new(resolved_col("k"), &left_schema).unwrap()];
        let right_on = vec![BoundExpr::try_new(resolved_col("k"), &right_schema).unwrap()];

        // Coalesced inner-join output schema: k, amount, region.
        let out_schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64),
            Field::new("amount", DataType::Int64),
            Field::new("region", DataType::Utf8),
        ]));

        let join = LocalPhysicalPlan::hash_join(
            l,
            r,
            left_on,
            right_on,
            None,
            Some(vec![true]), // null_equals_null for the single key
            JoinType::Inner,
            out_schema,
            StatsState::NotMaterialized,
            LocalNodeContext::default(),
        );

        let out = plan_to_sql(&join).unwrap();
        assert!(
            out.sql.contains("l.\"k\" IS NOT DISTINCT FROM r.\"k\""),
            "expected IS NOT DISTINCT FROM in ON clause, got: {}",
            out.sql
        );
        assert!(
            !out.sql.contains("l.\"k\" = r.\"k\""),
            "should not emit plain `=` when null_equals_null is true: {}",
            out.sql
        );
        // Coalesced join key qualified to the left side, no ambiguity.
        assert!(out.sql.contains("l.\"k\" AS \"k\""), "got: {}", out.sql);
    }

    #[test]
    fn hash_join_default_null_equality_emits_eq() {
        use daft_dsl::expr::bound_expr::BoundExpr;
        use daft_core::join::JoinType;

        let left_schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64),
            Field::new("amount", DataType::Int64),
        ]));
        let right_schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64),
            Field::new("region", DataType::Utf8),
        ]));
        let l = scan(left_schema.clone(), 1);
        let r = scan(right_schema.clone(), 2);

        let left_on = vec![BoundExpr::try_new(resolved_col("k"), &left_schema).unwrap()];
        let right_on = vec![BoundExpr::try_new(resolved_col("k"), &right_schema).unwrap()];

        let out_schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64),
            Field::new("amount", DataType::Int64),
            Field::new("region", DataType::Utf8),
        ]));

        let join = LocalPhysicalPlan::hash_join(
            l,
            r,
            left_on,
            right_on,
            None,
            None, // null_equals_null unset → standard `=`
            JoinType::Inner,
            out_schema,
            StatsState::NotMaterialized,
            LocalNodeContext::default(),
        );

        let out = plan_to_sql(&join).unwrap();
        assert!(
            out.sql.contains("l.\"k\" = r.\"k\""),
            "expected plain `=` in ON clause, got: {}",
            out.sql
        );
        assert!(
            !out.sql.contains("IS NOT DISTINCT FROM"),
            "should not emit IS NOT DISTINCT FROM when null_equals_null is None: {}",
            out.sql
        );
    }
}

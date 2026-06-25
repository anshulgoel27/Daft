use common_error::{DaftError, DaftResult};
use daft_core::prelude::{CountMode, Literal, Operator, Schema};
use daft_dsl::{Column, Expr, ExprRef, expr::AggExpr};

fn unsupported(what: &str) -> DaftError {
    DaftError::NotImplemented(format!("duckdb POC: unsupported expression: {what}"))
}

pub fn expr_to_sql(expr: &ExprRef, input_schema: &Schema) -> DaftResult<String> {
    match expr.as_ref() {
        Expr::Column(Column::Bound(bc)) => {
            let name = &input_schema[bc.index].name;
            Ok(format!("\"{name}\""))
        }
        Expr::Literal(lit) => literal_to_sql(lit),
        Expr::BinaryOp { op, left, right } => {
            let l = expr_to_sql(left, input_schema)?;
            let r = expr_to_sql(right, input_schema)?;
            let sym = binop_to_sql(*op)?;
            Ok(format!("({l} {sym} {r})"))
        }
        // Alias wraps an inner expression; the caller handles the output name.
        Expr::Alias(inner, _name) => expr_to_sql(inner, input_schema),
        Expr::Not(inner) => Ok(format!("(NOT {})", expr_to_sql(inner, input_schema)?)),
        Expr::IsNull(inner) => Ok(format!("({} IS NULL)", expr_to_sql(inner, input_schema)?)),
        Expr::NotNull(inner) => Ok(format!(
            "({} IS NOT NULL)",
            expr_to_sql(inner, input_schema)?
        )),
        other => Err(unsupported(&format!("{other:?}"))),
    }
}

fn binop_to_sql(op: Operator) -> DaftResult<&'static str> {
    Ok(match op {
        Operator::Eq => "=",
        Operator::NotEq => "!=",
        Operator::Lt => "<",
        Operator::LtEq => "<=",
        Operator::Gt => ">",
        Operator::GtEq => ">=",
        Operator::And => "AND",
        Operator::Or => "OR",
        Operator::Plus => "+",
        Operator::Minus => "-",
        Operator::Multiply => "*",
        Operator::TrueDivide => "/",
        other => return Err(unsupported(&format!("operator {other:?}"))),
    })
}

fn literal_to_sql(lit: &Literal) -> DaftResult<String> {
    Ok(match lit {
        Literal::Int8(v) => v.to_string(),
        Literal::Int16(v) => v.to_string(),
        Literal::Int32(v) => v.to_string(),
        Literal::Int64(v) => v.to_string(),
        Literal::UInt8(v) => v.to_string(),
        Literal::UInt16(v) => v.to_string(),
        Literal::UInt32(v) => v.to_string(),
        Literal::UInt64(v) => v.to_string(),
        Literal::Float32(v) => {
            if v.is_finite() {
                v.to_string()
            } else {
                return Err(unsupported(&format!("non-finite float literal {v}")));
            }
        }
        Literal::Float64(v) => {
            if v.is_finite() {
                v.to_string()
            } else {
                return Err(unsupported(&format!("non-finite float literal {v}")));
            }
        }
        Literal::Boolean(v) => {
            if *v {
                "TRUE".into()
            } else {
                "FALSE".into()
            }
        }
        Literal::Utf8(s) => format!("'{}'", s.replace('\'', "''")),
        Literal::Null => "NULL".into(),
        other => return Err(unsupported(&format!("literal {other:?}"))),
    })
}

/// Translate a single aggregation expression to DuckDB SQL, e.g. SUM("x").
/// Supports SUM/COUNT/MIN/MAX/MEAN(=AVG). The argument must be a bound column.
pub fn agg_to_sql(agg_expr: &AggExpr, input_schema: &Schema) -> DaftResult<String> {
    // COUNT is mode-sensitive and must be handled separately:
    //   - Valid → COUNT(<col>) (skips nulls; Daft's default)
    //   - All   → COUNT(*)     (counts all rows incl. nulls)
    //   - Null  → no simple DuckDB equivalent in POC scope → explicit error.
    if let AggExpr::Count(c, mode) = agg_expr {
        return match mode {
            CountMode::Valid => Ok(format!("COUNT({})", expr_to_sql(c, input_schema)?)),
            CountMode::All => Ok("COUNT(*)".to_string()),
            CountMode::Null => Err(unsupported("aggregation COUNT with CountMode::Null")),
        };
    }
    let (func, child) = match agg_expr {
        AggExpr::Sum(c) => ("SUM", c),
        AggExpr::Min(c) => ("MIN", c),
        AggExpr::Max(c) => ("MAX", c),
        AggExpr::Mean(c) => ("AVG", c),
        other => return Err(unsupported(&format!("aggregation {other:?}"))),
    };
    Ok(format!("{func}({})", expr_to_sql(child, input_schema)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use daft_core::prelude::*;
    use daft_dsl::{lit, resolved_col};

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("amount", DataType::Int64),
            Field::new("region", DataType::Utf8),
        ])
    }

    // Bind an unresolved expression's columns to indices against `schema`.
    fn bound(expr: daft_dsl::ExprRef) -> daft_dsl::ExprRef {
        daft_dsl::expr::bound_expr::BoundExpr::try_new(expr, &schema())
            .unwrap()
            .inner()
            .clone()
    }

    #[test]
    fn column_becomes_quoted_name() {
        let e = bound(resolved_col("amount"));
        assert_eq!(expr_to_sql(&e, &schema()).unwrap(), "\"amount\"");
    }

    #[test]
    fn comparison_and_literal() {
        let e = bound(resolved_col("amount").gt(lit(100i64)));
        assert_eq!(expr_to_sql(&e, &schema()).unwrap(), "(\"amount\" > 100)");
    }

    #[test]
    fn string_literal_is_single_quoted() {
        let e = bound(resolved_col("region").eq(lit("us")));
        assert_eq!(expr_to_sql(&e, &schema()).unwrap(), "(\"region\" = 'us')");
    }

    #[test]
    fn unsupported_expr_errors() {
        // cast is outside the supported subset
        let e2 = bound(resolved_col("amount").cast(&DataType::Float64));
        assert!(expr_to_sql(&e2, &schema()).is_err());
    }

    #[test]
    fn not_wraps_inner() {
        let e = bound(resolved_col("amount").gt(lit(100i64)).not());
        assert_eq!(
            expr_to_sql(&e, &schema()).unwrap(),
            "(NOT (\"amount\" > 100))"
        );
    }

    #[test]
    fn is_null() {
        let e = bound(resolved_col("region").is_null());
        assert_eq!(expr_to_sql(&e, &schema()).unwrap(), "(\"region\" IS NULL)");
    }

    #[test]
    fn is_not_null() {
        let e = bound(resolved_col("region").not_null());
        assert_eq!(
            expr_to_sql(&e, &schema()).unwrap(),
            "(\"region\" IS NOT NULL)"
        );
    }

    #[test]
    fn null_literal() {
        let e = bound(lit(Option::<i64>::None));
        assert_eq!(expr_to_sql(&e, &schema()).unwrap(), "NULL");
    }

    #[test]
    fn boolean_literal() {
        let e = bound(lit(false));
        assert_eq!(expr_to_sql(&e, &schema()).unwrap(), "FALSE");
    }

    #[test]
    fn non_finite_float_literal_errors() {
        let e = bound(lit(f64::NAN));
        assert!(expr_to_sql(&e, &schema()).is_err());
    }

    // Bind an AggExpr's columns to indices against `schema`.
    fn bound_agg(agg: AggExpr) -> AggExpr {
        daft_dsl::expr::bound_expr::BoundAggExpr::try_new(agg, &schema())
            .unwrap()
            .inner()
            .clone()
    }

    #[test]
    fn count_valid_counts_column() {
        // CountMode::Valid (Daft's default) → COUNT(<col>), skipping nulls.
        let agg = bound_agg(AggExpr::Count(resolved_col("amount"), CountMode::Valid));
        assert_eq!(agg_to_sql(&agg, &schema()).unwrap(), "COUNT(\"amount\")");
    }

    #[test]
    fn count_all_counts_star() {
        // CountMode::All → COUNT(*), counting all rows including nulls.
        let agg = bound_agg(AggExpr::Count(resolved_col("amount"), CountMode::All));
        assert_eq!(agg_to_sql(&agg, &schema()).unwrap(), "COUNT(*)");
    }

    #[test]
    fn count_null_is_unsupported() {
        // CountMode::Null has no simple DuckDB equivalent → explicit error (not silent wrong SQL).
        let agg = bound_agg(AggExpr::Count(resolved_col("amount"), CountMode::Null));
        assert!(agg_to_sql(&agg, &schema()).is_err());
    }

    #[test]
    fn sum_emits_sum() {
        let agg = bound_agg(AggExpr::Sum(resolved_col("amount")));
        assert_eq!(agg_to_sql(&agg, &schema()).unwrap(), "SUM(\"amount\")");
    }
}

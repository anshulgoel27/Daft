# DuckDB Local-Engine POC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether DuckDB executing a task's `LocalPhysicalPlan` (hash-join + filter + group-by-agg) beats Daft's native swordfish engine, with verified-identical results.

**Architecture:** A new feature-gated Rust crate `daft-duckdb` holds a `DuckDbExecutor` that translates the local physical plan to DuckDB SQL (Approach A), registers each task's input partitions as Arrow (via duckdb-rs), runs one query, and returns Arrow. A pyo3 binding `PyDuckDbExecutor` mirrors `PyNativeExecutor.run` so a Python harness can drive both engines over the *same* planner-produced `(plan, inputs)` and compare row-multisets + wall-clock.

**Tech Stack:** Rust, `duckdb` crate (bundled), arrow-rs (`arrow-array`/`arrow-schema`, already in the workspace), pyo3, `daft-local-plan`, `daft-micropartition`, `daft-recordbatch`, `daft-dsl`; Python harness using `daft`, `daft.daft.LocalPhysicalPlan`, `NativeExecutor`.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-06-25-duckdb-local-engine-poc-design.md` — every task's requirements implicitly include it.
- **Scope is one query shape:** hash join (inner) + filter + group-by aggregation. Any other plan node or expression → explicit `DaftError`, **no silent native fallback**.
- **Translation = Approach A (plan → DuckDB SQL string).** Not relational API, not Substrait.
- **Explicit projection at every plan node** (`SELECT <named output cols> FROM (<child>)`, never `SELECT *`), so names stay deterministic through the join.
- **`expr_to_sql` supported subset only:** column refs, literals, comparison (`= != < <= > >=`), logical (`AND OR NOT`), arithmetic (`+ - * /`), equi-join key equality, and aggregations `SUM/COUNT/MIN/MAX/AVG`. Anything else → explicit error.
- **Measurement is compute-only:** inputs are in-memory Arrow (`InMemoryScan` leaves); no parquet I/O in the timed region. The DuckDB timing **includes** Arrow registration + SQL bind + execution.
- **Correctness gate:** results compared as a sorted **row multiset**; exact for ints/keys, epsilon for float aggregates; NULLs present in join key and filter column.
- **Branch:** `repartition_spill` (not `main`). The user asked not to commit the design artifacts during the design phase; the per-task commit steps below apply when this plan is executed — confirm commit authorization at execution time.
- **Crate is feature-gated** so the `duckdb` dependency does not affect default Daft builds.

---

## File Structure

- `src/daft-duckdb/Cargo.toml` — new crate manifest; `duckdb` dep (bundled), arrow, daft deps, `python` feature.
- `src/daft-duckdb/src/lib.rs` — crate root; re-exports `DuckDbExecutor`; registers the pyo3 module under `python`.
- `src/daft-duckdb/src/arrow_bridge.rs` — Daft `RecordBatch` ↔ arrow-rs `RecordBatch`; register Arrow into DuckDB; collect Arrow out.
- `src/daft-duckdb/src/expr_sql.rs` — `expr_to_sql` (scalar) + `agg_to_sql`.
- `src/daft-duckdb/src/plan_sql.rs` — `plan_to_sql` walking the 5 node types.
- `src/daft-duckdb/src/executor.rs` — `DuckDbExecutor::run`.
- `src/daft-duckdb/src/python.rs` — `PyDuckDbExecutor` pyo3 binding (under `python` feature).
- `tests/duckdb_poc/test_correctness.py` — Python row-multiset equivalence test.
- `tests/duckdb_poc/bench_duckdb_vs_swordfish.py` — Python benchmark harness.
- Modified: root `Cargo.toml` (workspace member), the `daft` python entry crate's module registration, `src/daft-duckdb` added to the `python` feature chain of the entry crate.

---

### Task 1: Crate scaffold + Arrow bridge

**Files:**
- Create: `src/daft-duckdb/Cargo.toml`, `src/daft-duckdb/src/lib.rs`, `src/daft-duckdb/src/arrow_bridge.rs`
- Modify: root `Cargo.toml` (add `"src/daft-duckdb"` to `[workspace] members`)

**Interfaces:**
- Produces:
  - `pub fn daft_batches_to_arrow(batches: &[daft_recordbatch::RecordBatch]) -> common_error::DaftResult<Vec<arrow_array::RecordBatch>>`
  - `pub fn arrow_to_daft_batches(arrow_batches: Vec<arrow_array::RecordBatch>) -> common_error::DaftResult<Vec<daft_recordbatch::RecordBatch>>`

- [ ] **Step 1: Create the crate manifest**

`src/daft-duckdb/Cargo.toml`:
```toml
[package]
name = "daft-duckdb"
edition = "2021"
version = "0.3.0-dev0"
publish = false

[dependencies]
common-error = {path = "../common/error", default-features = false}
daft-core = {path = "../daft-core", default-features = false}
daft-dsl = {path = "../daft-dsl", default-features = false}
daft-local-plan = {path = "../daft-local-plan", default-features = false}
daft-micropartition = {path = "../daft-micropartition", default-features = false}
daft-recordbatch = {path = "../daft-recordbatch", default-features = false}
arrow-array = {workspace = true}
arrow-schema = {workspace = true}
duckdb = {version = "1.1", features = ["bundled"]}
pyo3 = {workspace = true, optional = true}

[features]
python = [
  "dep:pyo3",
  "common-error/python",
  "daft-core/python",
  "daft-dsl/python",
  "daft-local-plan/python",
  "daft-micropartition/python",
]

[lints]
workspace = true
```

- [ ] **Step 2: Add to workspace members**

In root `Cargo.toml`, under `[workspace] members = [ ... ]`, add the line (alphabetical near the other `daft-` entries):
```toml
  "src/daft-duckdb",
```

- [ ] **Step 3: Write the failing Arrow round-trip test**

`src/daft-duckdb/src/arrow_bridge.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use daft_core::prelude::*;
    use daft_recordbatch::RecordBatch;

    fn sample_batch() -> RecordBatch {
        let a = Int64Array::from(("a", vec![1i64, 2, 3])).into_series();
        let b = Utf8Array::from(("b", vec!["x", "y", "z"].as_slice())).into_series();
        RecordBatch::from_nonempty_columns(vec![a, b]).unwrap()
    }

    #[test]
    fn round_trips_daft_arrow_daft() {
        let original = sample_batch();
        let arrow = daft_batches_to_arrow(&[original.clone()]).unwrap();
        assert_eq!(arrow.len(), 1);
        assert_eq!(arrow[0].num_rows(), 3);
        let back = arrow_to_daft_batches(arrow).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].len(), original.len());
        assert_eq!(back[0].schema, original.schema);
    }
}
```

- [ ] **Step 4: Run it to verify it fails**

Run: `cargo test -p daft-duckdb arrow_bridge`
Expected: FAIL — `daft_batches_to_arrow` not found (does not compile).

- [ ] **Step 5: Implement the bridge using Daft's existing conversions**

At the top of `src/daft-duckdb/src/arrow_bridge.rs` (Daft already implements `TryFrom<RecordBatch> for arrow_array::RecordBatch` at `daft-recordbatch/src/lib.rs:1955` and `TryFrom<&arrow_array::RecordBatch> for RecordBatch` at `:1970`):
```rust
use common_error::DaftResult;
use daft_recordbatch::RecordBatch;

/// Daft RecordBatches -> arrow-rs RecordBatches (uses Daft's TryFrom impl).
pub fn daft_batches_to_arrow(batches: &[RecordBatch]) -> DaftResult<Vec<arrow_array::RecordBatch>> {
    batches
        .iter()
        .map(|b| Ok(arrow_array::RecordBatch::try_from(b.clone())?))
        .collect()
}

/// arrow-rs RecordBatches -> Daft RecordBatches (uses Daft's TryFrom impl).
pub fn arrow_to_daft_batches(
    arrow_batches: Vec<arrow_array::RecordBatch>,
) -> DaftResult<Vec<RecordBatch>> {
    arrow_batches
        .iter()
        .map(|b| Ok(RecordBatch::try_from(b)?))
        .collect()
}
```

`src/daft-duckdb/src/lib.rs`:
```rust
pub mod arrow_bridge;
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `cargo test -p daft-duckdb arrow_bridge`
Expected: PASS (1 test).

- [ ] **Step 7: Commit**

```bash
git add src/daft-duckdb/Cargo.toml src/daft-duckdb/src/lib.rs src/daft-duckdb/src/arrow_bridge.rs Cargo.toml
git commit -m "feat(duckdb): scaffold daft-duckdb crate with Arrow bridge"
```

---

### Task 2: `expr_to_sql` — scalar expression translation

**Files:**
- Create: `src/daft-duckdb/src/expr_sql.rs`
- Modify: `src/daft-duckdb/src/lib.rs` (add `pub mod expr_sql;`)

**Interfaces:**
- Consumes: `daft_dsl::ExprRef`, `daft_core::prelude::Schema` (a node's *input* schema for column-name resolution).
- Produces: `pub fn expr_to_sql(expr: &daft_dsl::ExprRef, input_schema: &daft_core::prelude::Schema) -> common_error::DaftResult<String>`

Notes for the implementer: Daft plan exprs are `BoundExpr(ExprRef)` — call `.inner()` to get the `ExprRef`. Bound columns are `daft_dsl::Expr::Column(daft_dsl::Column::Bound(BoundColumn { index, .. }))`; resolve `index` against `input_schema.fields()` to get the column name, then quote it (`"name"`). Literals are `Expr::Literal(LiteralValue)`. Binary ops are `Expr::BinaryOp { op, left, right }`.

- [ ] **Step 1: Write failing unit tests**

`src/daft-duckdb/src/expr_sql.rs`:
```rust
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
        // a function call that is not in the supported subset
        let e = bound(resolved_col("region").is_null());
        // is_null IS supported -> use something unsupported instead:
        let e2 = bound(resolved_col("amount").cast(&DataType::Float64));
        let _ = e;
        assert!(expr_to_sql(&e2, &schema()).is_err());
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p daft-duckdb expr_sql`
Expected: FAIL — `expr_to_sql` not found.

- [ ] **Step 3: Implement `expr_to_sql`**

`src/daft-duckdb/src/expr_sql.rs` (above the test module):
```rust
use common_error::{DaftError, DaftResult};
use daft_core::prelude::Schema;
use daft_dsl::{Column, Expr, ExprRef, Operator};

fn unsupported(what: &str) -> DaftError {
    DaftError::NotImplemented(format!("duckdb POC: unsupported expression: {what}"))
}

pub fn expr_to_sql(expr: &ExprRef, input_schema: &Schema) -> DaftResult<String> {
    match expr.as_ref() {
        Expr::Column(Column::Bound(bc)) => {
            let name = &input_schema.fields()[bc.index].name;
            Ok(format!("\"{name}\""))
        }
        Expr::Literal(lit) => literal_to_sql(lit),
        Expr::BinaryOp { op, left, right } => {
            let l = expr_to_sql(left, input_schema)?;
            let r = expr_to_sql(right, input_schema)?;
            let sym = binop_to_sql(*op)?;
            Ok(format!("({l} {sym} {r})"))
        }
        Expr::Not(inner) => Ok(format!("(NOT {})", expr_to_sql(inner, input_schema)?)),
        Expr::IsNull(inner) => Ok(format!("({} IS NULL)", expr_to_sql(inner, input_schema)?)),
        Expr::NotNull(inner) => Ok(format!("({} IS NOT NULL)", expr_to_sql(inner, input_schema)?)),
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

fn literal_to_sql(lit: &daft_dsl::LiteralValue) -> DaftResult<String> {
    use daft_dsl::LiteralValue::*;
    Ok(match lit {
        Int8(v) => v.to_string(),
        Int16(v) => v.to_string(),
        Int32(v) => v.to_string(),
        Int64(v) => v.to_string(),
        UInt8(v) => v.to_string(),
        UInt16(v) => v.to_string(),
        UInt32(v) => v.to_string(),
        UInt64(v) => v.to_string(),
        Float64(v) => v.to_string(),
        Boolean(v) => if *v { "TRUE".into() } else { "FALSE".into() },
        Utf8(s) => format!("'{}'", s.replace('\'', "''")),
        Null => "NULL".into(),
        other => return Err(unsupported(&format!("literal {other:?}"))),
    })
}
```

Add to `src/daft-duckdb/src/lib.rs`:
```rust
pub mod expr_sql;
```

> Implementer note: the exact enum paths (`daft_dsl::Column::Bound`, the `BoundColumn` field name `index`, `daft_dsl::LiteralValue` variant set, `Operator` variant names, and `BoundExpr::try_new`) must be confirmed against `src/daft-dsl/src/`. If a variant name differs (e.g. `LiteralValue::Utf8` vs `Series`), adjust the match arms — the structure (column→name, binop→symbol, literal→text) does not change.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p daft-duckdb expr_sql`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/daft-duckdb/src/expr_sql.rs src/daft-duckdb/src/lib.rs
git commit -m "feat(duckdb): scalar expr -> DuckDB SQL translation"
```

---

### Task 3: `agg_to_sql` + `plan_to_sql` — plan DAG translation

**Files:**
- Create: `src/daft-duckdb/src/plan_sql.rs`
- Modify: `src/daft-duckdb/src/expr_sql.rs` (add `agg_to_sql`), `src/daft-duckdb/src/lib.rs`

**Interfaces:**
- Consumes: `daft_local_plan::{LocalPhysicalPlan, LocalPhysicalPlanRef, SourceId}`; `expr_to_sql`.
- Produces:
  - `pub fn agg_to_sql(agg: &daft_dsl::ExprRef, input_schema: &Schema) -> DaftResult<String>`
  - `pub struct SqlPlan { pub sql: String, pub bindings: Vec<SourceId> }`
  - `pub fn plan_to_sql(plan: &LocalPhysicalPlanRef) -> DaftResult<SqlPlan>` — `bindings` lists every `SourceId` referenced, each mapped to view name `daft_src_<id>`.

Each node emits `SELECT <explicit output cols from node.schema> FROM (<child sql>) <op>`. The output column list at each level is exactly `node.schema.fields()` names, quoted.

- [ ] **Step 1: Write a failing translation test (filter over scan)**

`src/daft-duckdb/src/plan_sql.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use daft_core::prelude::*;
    use daft_dsl::{lit, resolved_col};
    use daft_local_plan::LocalPhysicalPlan;
    use daft_logical_plan::stats::StatsState;
    use std::sync::Arc;

    fn scan(schema: SchemaRef, source_id: u32) -> daft_local_plan::LocalPhysicalPlanRef {
        LocalPhysicalPlan::in_memory_scan(
            source_id,
            schema,
            0,
            StatsState::NotMaterialized,
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
        let f = LocalPhysicalPlan::filter(s, pred, StatsState::NotMaterialized);
        let out = plan_to_sql(&f).unwrap();
        assert_eq!(out.bindings, vec![7]);
        assert_eq!(
            out.sql,
            "SELECT \"amount\", \"region\" FROM (SELECT \"amount\", \"region\" FROM daft_src_7) WHERE (\"amount\" > 100)"
        );
    }
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test -p daft-duckdb plan_sql`
Expected: FAIL — `plan_to_sql` not found.

- [ ] **Step 3: Implement `agg_to_sql`**

Append to `src/daft-duckdb/src/expr_sql.rs`:
```rust
/// Translate a single aggregation expression to DuckDB SQL, e.g. SUM("x").
/// Supports SUM/COUNT/MIN/MAX/MEAN(=AVG). The argument must be a bound column.
pub fn agg_to_sql(agg: &ExprRef, input_schema: &Schema) -> DaftResult<String> {
    use daft_dsl::expr::AggExpr;
    let Expr::Agg(agg_expr) = agg.as_ref() else {
        return Err(unsupported(&format!("non-aggregate in agg position: {agg:?}")));
    };
    let (func, child) = match agg_expr {
        AggExpr::Sum(c) => ("SUM", c),
        AggExpr::Count(c, _) => ("COUNT", c),
        AggExpr::Min(c) => ("MIN", c),
        AggExpr::Max(c) => ("MAX", c),
        AggExpr::Mean(c) => ("AVG", c),
        other => return Err(unsupported(&format!("aggregation {other:?}"))),
    };
    Ok(format!("{func}({})", expr_to_sql(child, input_schema)?))
}
```

> Implementer note: confirm the `daft_dsl::expr::AggExpr` variant names and the `Expr::Agg` wrapper against `src/daft-dsl/src/expr/`. `Count` carries a `CountMode`; ignore it for the POC (plain `COUNT(col)`).

- [ ] **Step 4: Implement `plan_to_sql`**

`src/daft-duckdb/src/plan_sql.rs` (above the test module):
```rust
use common_error::{DaftError, DaftResult};
use daft_core::prelude::Schema;
use daft_local_plan::{LocalPhysicalPlan, LocalPhysicalPlanRef, SourceId};

use crate::expr_sql::{agg_to_sql, expr_to_sql};

pub struct SqlPlan {
    pub sql: String,
    pub bindings: Vec<SourceId>,
}

fn unsupported(node: &str) -> DaftError {
    DaftError::NotImplemented(format!("duckdb POC: unsupported plan node: {node}"))
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
        LocalPhysicalPlan::HashJoin(j) => {
            if j.join_type != daft_core::join::JoinType::Inner {
                return Err(unsupported(&format!("{:?} join (inner only)", j.join_type)));
            }
            let left = walk(&j.left, bindings)?;
            let right = walk(&j.right, bindings)?;
            let on = j
                .left_on
                .iter()
                .zip(j.right_on.iter())
                .map(|(l, r)| {
                    Ok(format!(
                        "l.{} = r.{}",
                        expr_to_sql(l.inner(), &j.left.schema())?,
                        expr_to_sql(r.inner(), &j.right.schema())?,
                    ))
                })
                .collect::<DaftResult<Vec<_>>>()?
                .join(" AND ");
            Ok(format!(
                "SELECT {} FROM ({left}) AS l INNER JOIN ({right}) AS r ON {on}",
                projection_list(&j.schema)
            ))
        }
        LocalPhysicalPlan::HashAggregate(a) => {
            let child = walk(&a.input, bindings)?;
            let groups = a
                .group_by
                .iter()
                .map(|g| expr_to_sql(g.inner(), &a.input.schema()))
                .collect::<DaftResult<Vec<_>>>()?;
            let select = aggregate_select(&a.schema, &groups, &a.aggregations, &a.input.schema())?;
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
        other => Err(unsupported(other.name())),
    }
}

/// Build the SELECT list for an aggregate: group columns first, then aggregations,
/// each aliased to the matching output-schema field name (positional).
fn aggregate_select(
    out_schema: &Schema,
    groups: &[String],
    aggs: &[daft_dsl::expr::bound_expr::BoundAggExpr],
    input_schema: &Schema,
) -> DaftResult<String> {
    let mut items = Vec::new();
    let names: Vec<&String> = out_schema.fields().iter().map(|f| &f.name).collect();
    for (i, g) in groups.iter().enumerate() {
        items.push(format!("{g} AS \"{}\"", names[i]));
    }
    for (j, agg) in aggs.iter().enumerate() {
        let sql = agg_to_sql(agg.inner(), input_schema)?;
        items.push(format!("{sql} AS \"{}\"", names[groups.len() + j]));
    }
    Ok(items.join(", "))
}
```

Add to `src/daft-duckdb/src/lib.rs`:
```rust
pub mod plan_sql;
```

> Implementer notes:
> - `LocalPhysicalPlanRef::schema()` returns the node's `SchemaRef` (confirm the accessor name; the structs all carry a `schema` field — a `.schema()` method exists in `plan.rs`).
> - `BoundExpr::inner()` / `BoundAggExpr::inner()` return `&ExprRef` (`daft-dsl/src/expr/bound_expr.rs:73,148`).
> - `LocalPhysicalPlan::name()` gives a node label for the unsupported-node error (used elsewhere in `plan.rs`).
> - Group-by/aggregate **output column order**: Daft emits group-by columns first, then aggregations. Confirm against `translate.rs`; if the planner orders differently, the positional aliasing in `aggregate_select` must match the real `out_schema` order. The correctness test (Task 6) is the backstop.

- [ ] **Step 5: Run the test to verify it passes**

Run: `cargo test -p daft-duckdb plan_sql`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/daft-duckdb/src/plan_sql.rs src/daft-duckdb/src/expr_sql.rs src/daft-duckdb/src/lib.rs
git commit -m "feat(duckdb): LocalPhysicalPlan -> DuckDB SQL translation"
```

---

### Task 4: `DuckDbExecutor::run` — register Arrow, run SQL, return Arrow

**Files:**
- Create: `src/daft-duckdb/src/executor.rs`
- Modify: `src/daft-duckdb/src/lib.rs`, `src/daft-duckdb/src/arrow_bridge.rs` (add register/collect helpers)

**Interfaces:**
- Consumes: `plan_to_sql`, the Arrow bridge, `daft_micropartition::MicroPartition`.
- Produces:
  - `pub struct DuckDbExecutor;`
  - `pub fn DuckDbExecutor::run(plan: &LocalPhysicalPlanRef, inputs: &std::collections::HashMap<SourceId, Vec<daft_micropartition::MicroPartitionRef>>) -> DaftResult<Vec<daft_recordbatch::RecordBatch>>`

- [ ] **Step 1: Verification spike — confirm the duckdb-rs Arrow API (no test, throwaway)**

Add a throwaway binary `src/daft-duckdb/examples/arrow_spike.rs` that:
1. opens `duckdb::Connection::open_in_memory()`,
2. builds one `arrow_array::RecordBatch` (two int columns),
3. registers it as a view named `t` (the goal API: an Arrow array-stream / `RecordBatchReader` registration — in duckdb-rs this is typically `Connection::register_arrow(...)` or `appender`-free arrow view; **confirm the exact call for the pinned `duckdb = 1.1` version**),
4. runs `conn.prepare("SELECT count(*) FROM t")?.query_arrow([])?` and prints the result.

Run: `cargo run -p daft-duckdb --example arrow_spike`
Expected: prints a 1-row count of 2.

**Decision recorded by this step:** the working registration call (the "Arrow view path") to use in Step 3. **Fallback if no ergonomic zero-copy Arrow-view API exists in this version:** create a typed DuckDB table and load rows via `duckdb::Appender` (a copy; include it in the timed region in Task 7). Note the chosen path in a comment in `executor.rs`.

- [ ] **Step 2: Write the failing executor test**

`src/daft-duckdb/src/executor.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use daft_core::prelude::*;
    use daft_dsl::{lit, resolved_col};
    use daft_local_plan::LocalPhysicalPlan;
    use daft_logical_plan::stats::StatsState;
    use daft_micropartition::MicroPartition;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn filter_over_scan_runs_in_duckdb() {
        let schema = Arc::new(Schema::new(vec![Field::new("amount", DataType::Int64)]));
        let col = Int64Array::from(("amount", vec![50i64, 150, 250])).into_series();
        let rb = daft_recordbatch::RecordBatch::from_nonempty_columns(vec![col]).unwrap();
        let mp = Arc::new(MicroPartition::new_loaded(schema.clone(), Arc::new(vec![rb]), None));

        let scan = LocalPhysicalPlan::in_memory_scan(3, schema.clone(), 0, StatsState::NotMaterialized);
        let pred = daft_dsl::expr::bound_expr::BoundExpr::try_new(
            resolved_col("amount").gt(lit(100i64)),
            &schema,
        ).unwrap();
        let plan = LocalPhysicalPlan::filter(scan, pred, StatsState::NotMaterialized);

        let mut inputs = HashMap::new();
        inputs.insert(3u32, vec![mp]);

        let out = DuckDbExecutor::run(&plan, &inputs).unwrap();
        let total: usize = out.iter().map(|b| b.len()).sum();
        assert_eq!(total, 2); // 150 and 250 survive the filter
    }
}
```

- [ ] **Step 3: Implement `DuckDbExecutor::run` + Arrow register/collect helpers**

Append to `src/daft-duckdb/src/arrow_bridge.rs`:
```rust
use daft_micropartition::MicroPartitionRef;

/// Flatten a source's MicroPartitions into arrow-rs RecordBatches.
pub fn source_to_arrow(mps: &[MicroPartitionRef]) -> DaftResult<Vec<arrow_array::RecordBatch>> {
    let mut out = Vec::new();
    for mp in mps {
        out.extend(daft_batches_to_arrow(mp.get_tables()?.as_ref())?);
    }
    Ok(out)
}
```
> Implementer note: `MicroPartition::get_tables() -> DaftResult<Arc<Vec<RecordBatch>>>` (see `daft-micropartition/src/micropartition.rs`). Confirm the accessor; `record_batches()` exists but `get_tables()` returns owned/Arc — use whichever yields `&[RecordBatch]` cleanly.

`src/daft-duckdb/src/executor.rs` (above the test module):
```rust
use std::collections::HashMap;

use common_error::DaftResult;
use daft_local_plan::{LocalPhysicalPlanRef, SourceId};
use daft_micropartition::MicroPartitionRef;
use daft_recordbatch::RecordBatch;

use crate::arrow_bridge::{arrow_to_daft_batches, source_to_arrow};
use crate::plan_sql::plan_to_sql;

pub struct DuckDbExecutor;

impl DuckDbExecutor {
    pub fn run(
        plan: &LocalPhysicalPlanRef,
        inputs: &HashMap<SourceId, Vec<MicroPartitionRef>>,
    ) -> DaftResult<Vec<RecordBatch>> {
        let translated = plan_to_sql(plan)?;
        let conn = duckdb::Connection::open_in_memory()
            .map_err(|e| common_error::DaftError::External(Box::new(e)))?;

        // Register each referenced source as a DuckDB view `daft_src_<id>`.
        for source_id in &translated.bindings {
            let mps = inputs.get(source_id).ok_or_else(|| {
                common_error::DaftError::ValueError(format!(
                    "duckdb POC: no input partitions for source {source_id}"
                ))
            })?;
            let arrow_batches = source_to_arrow(mps)?;
            register_arrow_view(&conn, &format!("daft_src_{source_id}"), arrow_batches)?;
        }

        // Run the translated query and collect Arrow results.
        let mut stmt = conn
            .prepare(&translated.sql)
            .map_err(|e| common_error::DaftError::External(Box::new(e)))?;
        let arrow_iter = stmt
            .query_arrow([])
            .map_err(|e| common_error::DaftError::External(Box::new(e)))?;
        let result_batches: Vec<arrow_array::RecordBatch> = arrow_iter.collect();
        arrow_to_daft_batches(result_batches)
    }
}

/// Register `batches` as a DuckDB view named `name`. Exact call confirmed by the Task 4 spike.
fn register_arrow_view(
    conn: &duckdb::Connection,
    name: &str,
    batches: Vec<arrow_array::RecordBatch>,
) -> DaftResult<()> {
    // <-- Replace with the call confirmed in Step 1 (Arrow-view path), or the Appender fallback. -->
    conn.register_arrow(name, batches)
        .map_err(|e| common_error::DaftError::External(Box::new(e)))
}
```

Add to `src/daft-duckdb/src/lib.rs`:
```rust
pub mod executor;
pub use executor::DuckDbExecutor;
```

> Implementer notes:
> - Confirm `DaftError` has an `External(Box<dyn Error + Send + Sync>)` variant; if not, map DuckDB errors via `DaftError::ComputeError(e.to_string())`.
> - `MicroPartition::new_loaded(schema, Arc<Vec<RecordBatch>>, metadata)` — confirm signature in `daft-micropartition/src/micropartition.rs:64`.
> - `register_arrow_view` body is the one line the spike pins.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p daft-duckdb executor`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/daft-duckdb/src/executor.rs src/daft-duckdb/src/arrow_bridge.rs src/daft-duckdb/src/lib.rs
git commit -m "feat(duckdb): DuckDbExecutor runs a translated LocalPhysicalPlan"
```

---

### Task 5: `PyDuckDbExecutor` binding + module registration

**Files:**
- Create: `src/daft-duckdb/src/python.rs`
- Modify: `src/daft-duckdb/src/lib.rs`; the `daft` python entry crate's module registration (find with grep below) and its `python` feature chain.

**Interfaces:**
- Produces (Python): `daft.daft.DuckDbExecutor().run(plan, inputs, ctx) -> list[PyMicroPartition]` where `plan: LocalPhysicalPlan`, `inputs: dict[int, list[PyMicroPartition]]`, mirroring `PyNativeExecutor.run` arguments (`src/daft-local-execution/src/run.rs:213`).

- [ ] **Step 1: Locate the python module registration**

Run:
```bash
grep -rn "PyNativeExecutor\|add_class\|register_modules\|fn register" src/daft-local-execution/src/lib.rs src/lib.rs 2>/dev/null | head
```
Expected: shows where `PyNativeExecutor` is added to the module (the pattern to copy for `PyDuckDbExecutor`).

- [ ] **Step 2: Implement the binding**

`src/daft-duckdb/src/python.rs`:
```rust
use std::collections::HashMap;

use daft_local_plan::{PyLocalPhysicalPlan, SourceId};
use daft_micropartition::python::PyMicroPartition;
use pyo3::prelude::*;

use crate::executor::DuckDbExecutor;

#[pyclass(module = "daft.daft", name = "DuckDbExecutor", frozen)]
pub struct PyDuckDbExecutor;

#[pymethods]
impl PyDuckDbExecutor {
    #[new]
    pub fn new() -> Self {
        Self
    }

    /// Run `plan` over `inputs` (source_id -> list of MicroPartitions) on DuckDB,
    /// returning the result partitions. Synchronous (POC).
    pub fn run(
        &self,
        plan: &PyLocalPhysicalPlan,
        inputs: HashMap<SourceId, Vec<PyMicroPartition>>,
    ) -> PyResult<Vec<PyMicroPartition>> {
        let inputs = inputs
            .into_iter()
            .map(|(k, v)| (k, v.into_iter().map(|p| p.inner.clone()).collect()))
            .collect();
        let batches = DuckDbExecutor::run(&plan.plan, &inputs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        // Wrap each result RecordBatch as a single-batch MicroPartition.
        let result = batches
            .into_iter()
            .map(|rb| {
                let schema = rb.schema.clone();
                let mp = daft_micropartition::MicroPartition::new_loaded(
                    schema,
                    std::sync::Arc::new(vec![rb]),
                    None,
                );
                PyMicroPartition::from(std::sync::Arc::new(mp))
            })
            .collect();
        Ok(result)
    }
}

pub fn register_modules(parent: &Bound<PyModule>) -> PyResult<()> {
    parent.add_class::<PyDuckDbExecutor>()?;
    Ok(())
}
```

Add to `src/daft-duckdb/src/lib.rs`:
```rust
#[cfg(feature = "python")]
pub mod python;
```

> Implementer notes: confirm `PyLocalPhysicalPlan.plan` field (`run.rs:217,225` uses `local_physical_plan.plan`), `PyMicroPartition.inner` field and `PyMicroPartition::from(Arc<MicroPartition>)`, against `daft-micropartition/src/python.rs`.

- [ ] **Step 3: Wire the crate into the daft python entry crate**

In the entry crate identified in Step 1 (the crate building the `daft._daft`/`daft.daft` module): add `daft-duckdb = {path = "../daft-duckdb", default-features = false}` to its `[dependencies]`, add `"daft-duckdb/python"` to its `python` feature list, and call `daft_duckdb::python::register_modules(module)?;` in its module-init function next to the other `register_modules` calls.

- [ ] **Step 4: Build the python extension and import-check**

Run:
```bash
make build && /Volumes/Work/Code/Daft/.venv/bin/python -c "from daft.daft import DuckDbExecutor; print(DuckDbExecutor)"
```
Expected: prints `<class 'daft.daft.DuckDbExecutor'>` with no error.

- [ ] **Step 5: Commit**

```bash
git add src/daft-duckdb/src/python.rs src/daft-duckdb/src/lib.rs Cargo.toml src/<entry-crate>/Cargo.toml src/<entry-crate>/src/lib.rs
git commit -m "feat(duckdb): PyDuckDbExecutor binding registered in daft.daft"
```

---

### Task 6: Python correctness test (DuckDB == swordfish)

**Files:**
- Create: `tests/duckdb_poc/__init__.py` (empty), `tests/duckdb_poc/_harness.py`, `tests/duckdb_poc/test_correctness.py`

**Interfaces:**
- Consumes: `daft.daft.LocalPhysicalPlan.from_logical_plan_builder`, `NativeExecutor`, `daft.daft.DuckDbExecutor`.
- Produces: `_harness.py` helpers `build_plan_and_inputs(df)` and `run_native(plan, inputs)` / `run_duckdb(plan, inputs)` returning `daft.DataFrame`-comparable Arrow.

- [ ] **Step 1: Write the shared harness**

`tests/duckdb_poc/_harness.py`:
```python
from __future__ import annotations

import daft
from daft.context import get_context
from daft.daft import DuckDbExecutor, LocalPhysicalPlan
from daft.execution.native_executor import NativeExecutor
from daft.recordbatch import MicroPartition


def build_plan_and_inputs(df: daft.DataFrame):
    """Optimize the DataFrame and produce the (LocalPhysicalPlan, inputs) the native
    runner would execute. Mirrors native_runner.py:148-152."""
    ctx = get_context()
    builder = df._builder.optimize(ctx.daft_execution_config)
    runner = ctx.get_or_create_runner()
    psets = {
        k: [v.micropartition()._micropartition for v in parts.values()]
        for k, parts in runner._part_set_cache.get_all_partition_sets().items()
    }
    plan, inputs = LocalPhysicalPlan.from_logical_plan_builder(builder._builder, psets)
    return plan, inputs


def run_native(plan, inputs) -> daft.DataFrame:
    parts = list(NativeExecutor().run(plan, inputs, get_context(), None))
    mps = [p.partition() for p in parts]
    return _to_sorted_arrow(mps)


def run_duckdb(plan, inputs) -> "pa.Table":
    py_mps = DuckDbExecutor().run(plan, inputs)
    mps = [MicroPartition._from_pymicropartition(p) for p in py_mps]
    return _to_sorted_arrow(mps)


def _to_sorted_arrow(mps):
    import pyarrow as pa

    if not mps:
        return None
    table = pa.concat_tables([m.to_arrow() for m in mps])
    # Sort by all columns for a deterministic row-multiset comparison.
    return table.sort_by([(c, "ascending") for c in table.column_names])
```
> Implementer notes: confirm `runner._part_set_cache.get_all_partition_sets()` access (mirror of `native_runner.py:148-151`), `LocalMaterializedResult.partition()` accessor for native results, and `MicroPartition.to_arrow()`. If `_part_set_cache` is empty before execution, materialize the inputs first with `df.collect()` then rebuild psets, or construct psets directly from `daft.from_pydict` partitions.

- [ ] **Step 2: Write the failing correctness test**

`tests/duckdb_poc/test_correctness.py`:
```python
from __future__ import annotations

import daft
import pytest

from tests.duckdb_poc._harness import build_plan_and_inputs, run_duckdb, run_native


@pytest.fixture(autouse=True)
def _native_runner():
    daft.context.set_runner_native()


def _query() -> daft.DataFrame:
    build = daft.from_pydict({
        "k": [1, 2, 3, None, 2],
        "region": ["us", "eu", "us", "eu", None],
    })
    probe = daft.from_pydict({
        "k": [1, 2, 2, 3, 5],
        "amount": [10, 20, 30, 40, 999],
    })
    joined = probe.join(build, on="k")  # inner
    filtered = joined.where(daft.col("amount") > 15)
    return filtered.groupby("region").agg(
        daft.col("amount").sum().alias("total"),
        daft.col("amount").count().alias("n"),
    )


def test_duckdb_matches_swordfish():
    plan, inputs = build_plan_and_inputs(_query())
    native = run_native(plan, inputs)
    duck = run_duckdb(plan, inputs)
    assert native.schema.names == duck.schema.names
    assert native.equals(duck)
```

- [ ] **Step 3: Run it to verify it fails**

Run: `DAFT_RUNNER=native /Volumes/Work/Code/Daft/.venv/bin/python -m pytest tests/duckdb_poc/test_correctness.py -v`
Expected: FAIL initially (e.g., translation gap, column-order mismatch, or `inputs` shape) — fix the translation/harness until results match. Common fixes: aggregate output column order in `aggregate_select` (Task 3), join key column naming, null-equality handling on the join key.

- [ ] **Step 4: Iterate to green**

Adjust `plan_sql.rs`/`expr_sql.rs` (rebuild with `make build`) until the test passes. The row-multiset (sorted) comparison must hold including the `None` rows.

Run: `DAFT_RUNNER=native /Volumes/Work/Code/Daft/.venv/bin/python -m pytest tests/duckdb_poc/test_correctness.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/duckdb_poc/__init__.py tests/duckdb_poc/_harness.py tests/duckdb_poc/test_correctness.py
git commit -m "test(duckdb): correctness gate — DuckDB result == swordfish result"
```

---

### Task 7: Python benchmark harness

**Files:**
- Create: `tests/duckdb_poc/bench_duckdb_vs_swordfish.py`

**Interfaces:**
- Consumes: `_harness.build_plan_and_inputs`, `run_native`, `run_duckdb`.

- [ ] **Step 1: Write the benchmark script**

`tests/duckdb_poc/bench_duckdb_vs_swordfish.py`:
```python
from __future__ import annotations

import statistics
import time

import daft
import numpy as np

from tests.duckdb_poc._harness import build_plan_and_inputs, run_duckdb, run_native

SIZES = [10_000, 1_000_000, 10_000_000]
ITERS = 5


def make_query(n: int) -> daft.DataFrame:
    rng = np.random.default_rng(0)
    build = daft.from_pydict({
        "k": np.arange(n // 10, dtype="int64"),
        "region": rng.integers(0, 8, n // 10).astype("int64"),
    })
    probe = daft.from_pydict({
        "k": rng.integers(0, n // 10, n, dtype="int64"),
        "amount": rng.integers(0, 1000, n, dtype="int64"),
    })
    joined = probe.join(build, on="k")
    filtered = joined.where(daft.col("amount") > 100)
    return filtered.groupby("region").agg(daft.col("amount").sum().alias("total"))


def median_ms(fn, plan, inputs) -> float:
    fn(plan, inputs)  # warm-up
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn(plan, inputs)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main() -> None:
    daft.context.set_runner_native()
    print(f"{'size':>12} {'swordfish ms':>14} {'duckdb ms':>12} {'duckdb/native':>14}")
    for n in SIZES:
        plan, inputs = build_plan_and_inputs(make_query(n))
        sword = median_ms(run_native, plan, inputs)
        duck = median_ms(run_duckdb, plan, inputs)
        print(f"{n:>12,} {sword:>14.1f} {duck:>12.1f} {duck / sword:>14.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the benchmark and record the numbers**

Run: `DAFT_RUNNER=native /Volumes/Work/Code/Daft/.venv/bin/python tests/duckdb_poc/bench_duckdb_vs_swordfish.py`
Expected: a 3-row table of `(size, swordfish ms, duckdb ms, ratio)`. A `duckdb/native` ratio below 1.0 means DuckDB is faster. This is the POC's deliverable number — paste it into the spec's results section.

- [ ] **Step 3: Append results to the spec**

Add a `## Results` section to `docs/superpowers/specs/2026-06-25-duckdb-local-engine-poc-design.md` with the printed table, the machine specs, and a one-line go/no-go recommendation.

- [ ] **Step 4: Commit**

```bash
git add tests/duckdb_poc/bench_duckdb_vs_swordfish.py docs/superpowers/specs/2026-06-25-duckdb-local-engine-poc-design.md
git commit -m "bench(duckdb): swordfish vs DuckDB on hash-join+filter/agg + record results"
```

---

## Self-Review

**Spec coverage:**
- DuckdbExecutor at the real Rust seam → Tasks 4–5. ✅
- Approach A (plan → SQL) → Tasks 2–3. ✅
- Arrow handoff (C Data Interface / arrow-rs) → Task 1 + Task 4 spike. ✅ (the spike pins the exact call; arrow-rs bridge verified via Daft's existing `TryFrom`).
- Hash-join + filter + agg, inner-only, explicit projection, supported expr/agg subset, no fallback → Tasks 2–3 (errors on anything else). ✅
- Compute-only in-memory measurement → Tasks 6–7 use `daft.from_pydict` (InMemoryScan leaves), no parquet. ✅
- Row-multiset correctness incl. NULLs → Task 6. ✅
- Benchmark with sizes + ratio, not a CI gate → Task 7 (a script, not a `pytest`). ✅
- Non-goals (streaming, scheduler wiring, spilling, Substrait, cost model) → not present in any task. ✅

**Placeholder scan:** The two `> Implementer note:` blocks and the Task 4 spike are **verification actions against external/edge APIs** (duckdb-rs version, exact daft-dsl enum paths), not vague requirements — each names the exact file to confirm against and the fallback. No "TODO/handle edge cases/add validation" placeholders remain.

**Type consistency:** `DuckDbExecutor::run(plan, &HashMap<SourceId, Vec<MicroPartitionRef>>) -> Vec<RecordBatch>` is consistent across Tasks 4–5. `plan_to_sql -> SqlPlan{sql, bindings: Vec<SourceId>}` consistent across Tasks 3–4. `expr_to_sql(&ExprRef, &Schema)` / `agg_to_sql(&ExprRef, &Schema)` consistent across Tasks 2–3. `daft_src_<id>` view naming consistent between `plan_to_sql` (Task 3) and `register_arrow_view` (Task 4). ✅

**Known risk flagged for execution:** the duckdb-rs Arrow-registration call (Task 4 Step 1 spike) and the `_part_set_cache` psets access (Task 6) are the two integration points most likely to need adjustment; both have stated fallbacks.

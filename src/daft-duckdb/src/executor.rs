// Zero-copy execution: sources are registered as in-place Arrow views via
// duckdb::Connection::register_arrow (arrow-rs 59 unified across Daft and duckdb), so DuckDB
// pushes projection + filters into the scan. Results come back from query_arrow already as
// arrow_array v59 batches. See docs/superpowers/specs/2026-06-27-daft-duckdb-zerocopy-register-arrow-design.md.

use std::collections::HashMap;

use arrow_array::RecordBatch as ArrowBatch;
use common_error::{DaftError, DaftResult};
use daft_local_plan::{LocalPhysicalPlanRef, SourceId};
use daft_micropartition::MicroPartitionRef;
use daft_recordbatch::RecordBatch;

use crate::arrow_bridge::{arrow_to_daft_batches, source_to_arrow};
use crate::plan_sql::plan_to_sql;

/// Tuning for a DuckDB execution, applied as connection PRAGMAs. `None` leaves DuckDB's
/// default (all cores / ~80% RAM). On a worker running many concurrent tasks, capping
/// `threads`/`memory_limit` per task avoids core/RAM oversubscription.
#[derive(Clone, Default)]
pub struct DuckDbConfig {
    pub threads: Option<u64>,
    pub memory_limit: Option<String>,
}

pub struct DuckDbExecutor;

impl DuckDbExecutor {
    /// Run `plan` over `inputs` on DuckDB with default tuning.
    pub fn run(
        plan: &LocalPhysicalPlanRef,
        inputs: &HashMap<SourceId, Vec<MicroPartitionRef>>,
    ) -> DaftResult<Vec<RecordBatch>> {
        Self::run_with_config(plan, inputs, &DuckDbConfig::default())
    }

    /// Run `plan` over `inputs` on DuckDB with the given tuning (thread cap / memory limit).
    pub fn run_with_config(
        plan: &LocalPhysicalPlanRef,
        inputs: &HashMap<SourceId, Vec<MicroPartitionRef>>,
        config: &DuckDbConfig,
    ) -> DaftResult<Vec<RecordBatch>> {
        let translated = plan_to_sql(plan)?;
        let conn = open_conn(config)?;
        // Held until after execute returns; dropping a view unregisters its DuckDB view.
        let _views = register_sources(&conn, &translated.bindings, inputs)?;
        execute(&conn, &translated.sql)
    }

    /// Benchmark helper: register `inputs` once, then time `repeat` executions of the query.
    /// Returns `(registration_seconds, per_run_execution_seconds)`, letting a caller separate the
    /// per-task registration cost ("cold") from steady-state query compute ("warm").
    /// Each run's result is collected then discarded (timing only).
    pub fn bench_runs(
        plan: &LocalPhysicalPlanRef,
        inputs: &HashMap<SourceId, Vec<MicroPartitionRef>>,
        config: &DuckDbConfig,
        repeat: usize,
    ) -> DaftResult<(f64, Vec<f64>)> {
        let translated = plan_to_sql(plan)?;
        let conn = open_conn(config)?;

        let t_reg = std::time::Instant::now();
        let _views = register_sources(&conn, &translated.bindings, inputs)?;
        let registration_seconds = t_reg.elapsed().as_secs_f64();

        let mut run_seconds = Vec::with_capacity(repeat);
        for _ in 0..repeat {
            let t = std::time::Instant::now();
            let _ = execute(&conn, &translated.sql)?;
            run_seconds.push(t.elapsed().as_secs_f64());
        }
        Ok((registration_seconds, run_seconds))
    }
}

/// A reusable DuckDB session: one connection (config applied once) reused across many runs, with the
/// translated SQL cached per plan. Cuts the per-task fixed overhead — connection open + plan
/// re-translation — when a worker processes many partitions that share a plan.
pub struct DuckDbSession {
    conn: duckdb::Connection,
    /// (plan identity, translated SQL) — reused while the same plan repeats across partitions.
    cached: Option<(usize, crate::plan_sql::SqlPlan)>,
}

impl DuckDbSession {
    /// Open one connection and apply the config once.
    pub fn new(config: &DuckDbConfig) -> DaftResult<Self> {
        Ok(Self {
            conn: open_conn(config)?,
            cached: None,
        })
    }

    /// Run `plan` over `inputs`, reusing the connection and the cached translation.
    pub fn run(
        &mut self,
        plan: &LocalPhysicalPlanRef,
        inputs: &HashMap<SourceId, Vec<MicroPartitionRef>>,
    ) -> DaftResult<Vec<RecordBatch>> {
        // Translate once per plan identity (same Arc -> cache hit); a different plan retranslates.
        let plan_id = std::sync::Arc::as_ptr(plan) as usize;
        if self.cached.as_ref().map(|(id, _)| *id) != Some(plan_id) {
            self.cached = Some((plan_id, plan_to_sql(plan)?));
        }
        let translated = self.cached.as_ref().expect("just set").1.clone();

        // Register this partition's sources as zero-copy views, held only for this run. The prior
        // run's views were already dropped (auto-unregistered) at the end of that run; register_arrow
        // also replaces any existing view of the same name.
        let _views = register_sources(&self.conn, &translated.bindings, inputs)?;
        execute(&self.conn, &translated.sql)
    }
}

/// Open an in-memory DuckDB connection and apply the config PRAGMAs.
fn open_conn(config: &DuckDbConfig) -> DaftResult<duckdb::Connection> {
    let conn =
        duckdb::Connection::open_in_memory().map_err(|e| DaftError::External(Box::new(e)))?;
    if let Some(threads) = config.threads {
        conn.execute_batch(&format!("SET threads={threads};"))
            .map_err(|e| DaftError::External(Box::new(e)))?;
    }
    if let Some(memory_limit) = &config.memory_limit {
        // A DuckDB size string like "8GB"; reject a quote so it can't break out of the literal.
        if memory_limit.contains('\'') {
            return Err(DaftError::ValueError(format!(
                "duckdb POC: invalid memory_limit {memory_limit:?}"
            )));
        }
        conn.execute_batch(&format!("SET memory_limit='{memory_limit}';"))
            .map_err(|e| DaftError::External(Box::new(e)))?;
    }
    Ok(conn)
}

/// Register each referenced source as a zero-copy DuckDB view `daft_src_<id>` over its input
/// partitions' Arrow buffers. Returns the live `ArrowView` handles — the caller MUST keep them
/// alive until the query has finished executing (each view references its Arrow batches in place;
/// dropping a view auto-unregisters the DuckDB view).
fn register_sources<'c>(
    conn: &'c duckdb::Connection,
    bindings: &[SourceId],
    inputs: &HashMap<SourceId, Vec<MicroPartitionRef>>,
) -> DaftResult<Vec<duckdb::ArrowView<'c>>> {
    let mut views = Vec::with_capacity(bindings.len());
    for source_id in bindings {
        let mps = inputs.get(source_id).ok_or_else(|| {
            DaftError::ValueError(format!(
                "duckdb POC: no input partitions for source {source_id}"
            ))
        })?;
        let arrow_batches = source_to_arrow(mps)?;
        if arrow_batches.is_empty() {
            return Err(DaftError::ValueError(format!(
                "duckdb POC: source {source_id} has no record batches"
            )));
        }
        let view = conn
            .register_arrow(&format!("daft_src_{source_id}"), arrow_batches)
            .map_err(|e| DaftError::External(Box::new(e)))?;
        views.push(view);
    }
    Ok(views)
}

/// Prepare + run the translated SQL and collect the result as Daft RecordBatches.
/// `query_arrow` now yields `arrow_array` v59 RecordBatches directly (unified arrow), so no
/// version bridge is needed.
fn execute(conn: &duckdb::Connection, sql: &str) -> DaftResult<Vec<RecordBatch>> {
    let mut stmt = conn
        .prepare(sql)
        .map_err(|e| DaftError::External(Box::new(e)))?;
    let arrow_iter = stmt
        .query_arrow([])
        .map_err(|e| DaftError::External(Box::new(e)))?;
    let result_batches: Vec<ArrowBatch> = arrow_iter.collect();
    arrow_to_daft_batches(result_batches)
}

#[cfg(test)]
mod tests {
    use super::*;
    use daft_core::prelude::*;
    use daft_dsl::{lit, resolved_col};
    use daft_local_plan::{LocalNodeContext, LocalPhysicalPlan};
    use daft_logical_plan::stats::StatsState;
    use daft_micropartition::MicroPartition;
    use std::collections::HashMap;
    use std::sync::Arc;

    /// `filter(amount > 100)` over a 3-row in_memory_scan of [50, 150, 250], with the input
    /// MicroPartition registered under source id 3. Shared by the executor tests below.
    fn filter_plan_and_inputs() -> (LocalPhysicalPlanRef, HashMap<u32, Vec<MicroPartitionRef>>) {
        let schema = Arc::new(Schema::new(vec![Field::new("amount", DataType::Int64)]));
        let col = Int64Array::from_vec("amount", vec![50i64, 150, 250]).into_series();
        let rb = daft_recordbatch::RecordBatch::from_nonempty_columns(vec![col]).unwrap();
        let mp = Arc::new(MicroPartition::new_loaded(
            schema.clone(),
            Arc::new(vec![rb]),
            None,
        ));
        let scan = LocalPhysicalPlan::in_memory_scan(
            3,
            schema.clone(),
            0,
            StatsState::NotMaterialized,
            LocalNodeContext::default(),
        );
        let pred = daft_dsl::expr::bound_expr::BoundExpr::try_new(
            resolved_col("amount").gt(lit(100i64)),
            &schema,
        )
        .unwrap();
        let plan = LocalPhysicalPlan::filter(
            scan,
            pred,
            StatsState::NotMaterialized,
            LocalNodeContext::default(),
        );
        let mut inputs = HashMap::new();
        inputs.insert(3u32, vec![mp]);
        (plan, inputs)
    }

    #[test]
    fn filter_over_scan_runs_in_duckdb() {
        let (plan, inputs) = filter_plan_and_inputs();
        let out = DuckDbExecutor::run(&plan, &inputs).unwrap();
        let total: usize = out.iter().map(|b| b.len()).sum();
        assert_eq!(total, 2); // 150 and 250 survive the filter
    }

    #[test]
    fn run_with_config_applies_thread_and_memory_caps() {
        let (plan, inputs) = filter_plan_and_inputs();
        let config = DuckDbConfig {
            threads: Some(2),
            memory_limit: Some("1GB".to_string()),
        };
        let out = DuckDbExecutor::run_with_config(&plan, &inputs, &config).unwrap();
        let total: usize = out.iter().map(|b| b.len()).sum();
        assert_eq!(total, 2); // same result with the caps applied
    }

    #[test]
    fn bench_runs_registers_once_and_times_each_run() {
        let (plan, inputs) = filter_plan_and_inputs();
        let (reg, runs) =
            DuckDbExecutor::bench_runs(&plan, &inputs, &DuckDbConfig::default(), 3).unwrap();
        assert!(reg >= 0.0, "registration time should be non-negative");
        assert_eq!(runs.len(), 3, "one timing per repeat");
        assert!(runs.iter().all(|&t| t >= 0.0), "run times non-negative");
    }

    #[test]
    fn session_reuses_connection_across_runs() {
        let (plan, inputs) = filter_plan_and_inputs();
        let mut session = DuckDbSession::new(&DuckDbConfig::default()).unwrap();
        // Two runs on one session: each run registers fresh views (auto-unregistered at run end).
        for _ in 0..2 {
            let out = session.run(&plan, &inputs).unwrap();
            let total: usize = out.iter().map(|b| b.len()).sum();
            assert_eq!(total, 2);
        }
    }

}

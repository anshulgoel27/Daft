use std::collections::HashMap;
use std::sync::Arc;

use daft_local_plan::{PyLocalPhysicalPlan, SourceId};
use daft_micropartition::{MicroPartitionRef, python::PyMicroPartition};
use pyo3::prelude::*;

use crate::executor::{DuckDbConfig, DuckDbExecutor};

#[pyclass(module = "daft.daft", name = "DuckDbExecutor", frozen)]
pub struct PyDuckDbExecutor;

/// Unwrap the inner `Arc<MicroPartition>` out of each `PyMicroPartition`, keyed by source id.
fn to_mp_inputs(
    inputs: HashMap<SourceId, Vec<PyMicroPartition>>,
) -> HashMap<SourceId, Vec<MicroPartitionRef>> {
    inputs
        .into_iter()
        .map(|(k, v)| (k, v.into_iter().map(|p| p.inner.clone()).collect()))
        .collect()
}

/// Wrap each result `RecordBatch` as a single-batch `MicroPartition`.
fn wrap_results(batches: Vec<daft_recordbatch::RecordBatch>) -> Vec<PyMicroPartition> {
    batches
        .into_iter()
        .map(|rb| {
            let schema = rb.schema.clone();
            let mp = daft_micropartition::MicroPartition::new_loaded(schema, Arc::new(vec![rb]), None);
            PyMicroPartition::from(Arc::new(mp))
        })
        .collect()
}

#[pymethods]
impl PyDuckDbExecutor {
    #[new]
    pub fn new() -> Self {
        Self
    }

    /// Run `plan` over `inputs` (source_id -> list of MicroPartitions) on DuckDB, returning the
    /// result partitions. `threads`/`memory_limit` cap DuckDB per task (None = DuckDB defaults).
    #[pyo3(signature = (plan, inputs, threads=None, memory_limit=None))]
    pub fn run(
        &self,
        plan: &PyLocalPhysicalPlan,
        inputs: HashMap<SourceId, Vec<PyMicroPartition>>,
        threads: Option<u64>,
        memory_limit: Option<String>,
    ) -> PyResult<Vec<PyMicroPartition>> {
        let config = DuckDbConfig { threads, memory_limit };
        let inputs = to_mp_inputs(inputs);
        let batches = DuckDbExecutor::run_with_config(&plan.plan, &inputs, &config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(wrap_results(batches))
    }

    /// Register `inputs` once, then time `repeat` executions of the query. Returns
    /// `(registration_seconds, [run_seconds, ...])` — separates the per-task registration/copy
    /// cost ("cold") from steady-state query compute ("warm"). For benchmarking.
    #[pyo3(signature = (plan, inputs, repeat, threads=None, memory_limit=None))]
    pub fn bench_runs(
        &self,
        plan: &PyLocalPhysicalPlan,
        inputs: HashMap<SourceId, Vec<PyMicroPartition>>,
        repeat: usize,
        threads: Option<u64>,
        memory_limit: Option<String>,
    ) -> PyResult<(f64, Vec<f64>)> {
        let config = DuckDbConfig { threads, memory_limit };
        let inputs = to_mp_inputs(inputs);
        DuckDbExecutor::bench_runs(&plan.plan, &inputs, &config, repeat)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

pub fn register_modules(parent: &Bound<PyModule>) -> PyResult<()> {
    parent.add_class::<PyDuckDbExecutor>()?;
    Ok(())
}

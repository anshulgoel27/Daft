use std::{
    num::NonZeroUsize,
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
    vec,
};

use common_error::{DaftError, DaftResult};
use common_metrics::ops::NodeType;
use daft_core::prelude::{Schema, SchemaRef};
use daft_dsl::expr::bound_expr::BoundExpr;
#[cfg(feature = "python")]
use daft_micropartition::python::PyMicroPartition;
use daft_micropartition::{MicroPartition, partitioning::Partition};
use itertools::Itertools;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use rand::Rng;
use tracing::{Span, instrument};

use super::intermediate_op::{IntermediateOpExecuteResult, IntermediateOperator};
use crate::{
    ExecutionTaskSpawner,
    pipeline::{InputId, MorselSizeRequirement, NodeName},
};

#[derive(Clone, Debug)]
pub(crate) struct ActorHandle {
    #[cfg(feature = "python")]
    inner: Arc<Py<PyAny>>,
}

impl ActorHandle {
    /// Returns the actors that are up on this node, those up on other nodes, and how many are not up yet.
    fn get_actors_on_current_node(
        actor_handles: Vec<Self>,
    ) -> DaftResult<(Vec<Self>, Vec<Self>, usize)> {
        #[cfg(feature = "python")]
        {
            let actor_handles = actor_handles
                .into_iter()
                .map(|e| e.inner)
                .collect::<Vec<_>>();

            let (local_actors, remote_actors, num_pending) = Python::attach(|py| {
                let actor_handles = actor_handles
                    .into_iter()
                    .map(|e| e.as_ref().clone_ref(py))
                    .collect::<Vec<_>>();
                let ray_actor_pool_udf_module =
                    py.import(pyo3::intern!(py, "daft.execution.ray_actor_pool_udf"))?;
                let (local_actors, remote_actors, num_pending) = ray_actor_pool_udf_module
                    .call_method1(
                        pyo3::intern!(py, "get_ready_actors_by_location"),
                        (actor_handles,),
                    )?
                    .extract::<(Vec<Py<PyAny>>, Vec<Py<PyAny>>, usize)>()?;
                DaftResult::Ok((local_actors, remote_actors, num_pending))
            })?;

            let local_actors = local_actors
                .into_iter()
                .map(|e| Self { inner: Arc::new(e) })
                .collect::<Vec<_>>();
            let remote_actors = remote_actors
                .into_iter()
                .map(|e| Self { inner: Arc::new(e) })
                .collect::<Vec<_>>();
            Ok((local_actors, remote_actors, num_pending))
        }
        #[cfg(not(feature = "python"))]
        {
            unimplemented!(
                "ActorHandle::get_actors_on_current_node is not implemented without Python"
            );
        }
    }

    #[cfg(feature = "python")]
    async fn eval_input(&self, input: MicroPartition) -> DaftResult<MicroPartition> {
        let inner = self.inner.clone();
        let result =
            common_runtime::python::execute_python_coroutine::<_, PyMicroPartition>(move |py| {
                let coroutine = inner.call_method1(
                    py,
                    pyo3::intern!(py, "eval_input"),
                    (PyMicroPartition::from(input),),
                )?;
                Ok(coroutine.into_bound(py))
            })
            .await?;
        let mp: Arc<MicroPartition> = result.into();
        Ok(Arc::try_unwrap(mp).unwrap_or_else(|a| a.as_ref().clone()))
    }
}

#[cfg(feature = "python")]
impl From<common_py_serde::PyObjectWrapper> for ActorHandle {
    fn from(value: common_py_serde::PyObjectWrapper) -> Self {
        Self { inner: value.0 }
    }
}

/// How often an elastic pool re-checks which of its actors are up.
const ACTOR_REFRESH_INTERVAL: Duration = Duration::from_secs(5);

/// The actors a pool sends work to. A fixed pool prefers the actors on this node and falls back to remote
/// ones. An elastic pool uses every actor that is up, wherever it is: its actors come up on whichever nodes
/// gain capacity, and preferring local ones would leave the actors on other nodes unused.
fn usable_actors<T>(local: Vec<T>, remote: Vec<T>, elastic: bool) -> Vec<T> {
    if elastic {
        local.into_iter().chain(remote).collect()
    } else if local.is_empty() {
        remote
    } else {
        local
    }
}

/// The actors of an elastic pool (`actor_udf_min_ready`): some were not up when the operator was built,
/// and Ray keeps scheduling them. The operator is built once per worker and shared by every task of the
/// plan, so the usable set is refreshed here, and the actor is picked per batch instead of being fixed
/// per concurrency slot, which lets later batches use actors that came up after the first task started.
struct DynamicActorPool {
    all: Vec<ActorHandle>,
    usable: RwLock<Vec<ActorHandle>>,
    /// When to next refresh; `None` once no actor is pending.
    next_refresh: Mutex<Option<Instant>>,
    counter: AtomicUsize,
}

impl DynamicActorPool {
    fn new(all: Vec<ActorHandle>, usable: Vec<ActorHandle>) -> Self {
        let counter = AtomicUsize::new(rand::rng().random_range(0..usable.len()));
        Self {
            all,
            usable: RwLock::new(usable),
            next_refresh: Mutex::new(Some(Instant::now() + ACTOR_REFRESH_INTERVAL)),
            counter,
        }
    }

    fn next_actor(&self) -> ActorHandle {
        self.refresh_if_due();
        let usable = self.usable.read().expect("actor list lock poisoned");
        let idx = self.counter.fetch_add(1, Ordering::SeqCst) % usable.len();
        usable[idx].clone()
    }

    fn refresh_if_due(&self) {
        // One caller refreshes at a time; the others keep using the current list.
        let Ok(mut next_refresh) = self.next_refresh.try_lock() else {
            return;
        };
        match *next_refresh {
            Some(due) if Instant::now() >= due => {}
            _ => return,
        }
        match ActorHandle::get_actors_on_current_node(self.all.clone()) {
            Ok((local, remote, num_pending)) => {
                let usable = usable_actors(local, remote, true);
                if !usable.is_empty() {
                    *self.usable.write().expect("actor list lock poisoned") = usable;
                }
                *next_refresh = if num_pending == 0 {
                    None
                } else {
                    Some(Instant::now() + ACTOR_REFRESH_INTERVAL)
                };
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to refresh the UDF actor pool, keeping the current actors: {e}"
                );
                *next_refresh = Some(Instant::now() + ACTOR_REFRESH_INTERVAL);
            }
        }
    }
}

pub(crate) struct DistributedActorPoolProjectState {
    actor_handle: ActorHandle,
}

pub(crate) struct DistributedActorPoolProjectOperator {
    actor_handles: Vec<ActorHandle>,
    /// Number of actors in the pool, including ones that are not up yet.
    total_actors: usize,
    /// Set only for an elastic pool that had actors pending when the operator was built.
    dynamic: Option<Arc<DynamicActorPool>>,
    batch_size: Option<usize>,
    memory_request: u64,
    passthrough_columns: Vec<BoundExpr>, // Columns that no need to be passed to UDFActor for processing
    required_columns: Vec<usize>,        // UDF input columns
    schema: SchemaRef,                   // Output schema
    counter: AtomicUsize,
}

impl DistributedActorPoolProjectOperator {
    pub fn try_new(
        actor_handles: Vec<impl Into<ActorHandle>>,
        batch_size: Option<usize>,
        memory_request: u64,
        elastic: bool,
        passthrough_columns: Vec<BoundExpr>,
        required_columns: Vec<usize>,
        schema: SchemaRef,
    ) -> DaftResult<Self> {
        let all_actor_handles: Vec<ActorHandle> =
            actor_handles.into_iter().map(|e| e.into()).collect();
        let total_actors = all_actor_handles.len();
        let (local_actor_handles, remote_actor_handles, num_pending) =
            ActorHandle::get_actors_on_current_node(all_actor_handles.clone())?;

        let actor_handles = usable_actors(local_actor_handles, remote_actor_handles, elastic);

        // An elastic pool with actors that are not up yet keeps watching for them.
        let dynamic = (elastic && num_pending > 0).then(|| {
            Arc::new(DynamicActorPool::new(
                all_actor_handles,
                actor_handles.clone(),
            ))
        });

        let init_counter = if actor_handles.is_empty() {
            0
        } else {
            rand::rng().random_range(0..actor_handles.len())
        };

        Ok(Self {
            actor_handles,
            total_actors,
            dynamic,
            batch_size,
            memory_request,
            passthrough_columns,
            required_columns,
            schema,
            counter: AtomicUsize::new(init_counter),
        })
    }
}

impl IntermediateOperator for DistributedActorPoolProjectOperator {
    type State = DistributedActorPoolProjectState;
    type BatchingStrategy = crate::dynamic_batching::StaticBatchingStrategy;
    #[instrument(skip_all, name = "DistributedActorPoolProjectOperator::execute")]
    fn execute(
        &self,
        input: MicroPartition,
        state: Self::State,
        _runtime_stats: Arc<Self::Stats>,
        task_spawner: &ExecutionTaskSpawner,
        _input_id: InputId,
    ) -> IntermediateOpExecuteResult<Self> {
        #[cfg(feature = "python")]
        {
            let memory_request = self.memory_request;
            let required_columns = self.required_columns.clone();
            let passthrough_columns = self.passthrough_columns.clone();
            let output_schema = self.schema.clone();
            // A fixed pool keeps the actor its concurrency slot was given; an elastic one picks per batch.
            let actor_handle = match &self.dynamic {
                Some(pool) => pool.next_actor(),
                None => state.actor_handle.clone(),
            };

            let fut = task_spawner.spawn_with_memory_request(
                memory_request,
                async move {
                    // Prune input by required cols
                    let input_schema = input.schema();
                    let pruned_input = MicroPartition::new_loaded(
                        Arc::new(Schema::new(
                            required_columns
                                .iter()
                                .map(|idx| input_schema[*idx].clone()),
                        )),
                        Arc::new(
                            input
                                .record_batches()
                                .iter()
                                .map(|batch| batch.get_columns(required_columns.as_slice()))
                                .collect::<Vec<_>>(),
                        ),
                        None,
                    );
                    let input_num_rows = pruned_input.num_rows();

                    // Call the UDF with pruned input
                    let eval_output = actor_handle.eval_input(pruned_input).await?;
                    if eval_output.schema().fields().len() != 1 {
                        return Err(DaftError::InternalError(format!(
                            "UDF output schema must be a single column, but got '{}'",
                            eval_output.schema().field_names().join(", ")
                        )));
                    }

                    if input_num_rows != eval_output.num_rows() {
                        return Err(DaftError::InternalError(format!(
                            "The number of rows is mismatch between UDF input {} and output {}",
                            input_num_rows,
                            eval_output.num_rows()
                        )));
                    }

                    // Combine the eval output with passthrough columns
                    let mut output_batches = Vec::with_capacity(input.num_rows());
                    for (input_record, output_record) in input
                        .record_batches()
                        .iter()
                        .zip(eval_output.record_batches().iter())
                    {
                        let passthrough_record =
                            input_record.eval_expression_list(passthrough_columns.as_slice())?;

                        output_batches.push(passthrough_record.append_column(
                            output_schema.clone(),
                            output_record.get_column(0).clone(),
                        )?);
                    }

                    Ok((
                        state,
                        MicroPartition::new_loaded(
                            output_schema.clone(),
                            Arc::new(output_batches),
                            None,
                        ),
                    ))
                },
                Span::current(),
            );
            fut.into()
        }
        #[cfg(not(feature = "python"))]
        {
            unimplemented!(
                "DistributedActorPoolProjectOperator::execute is not implemented without Python"
            );
        }
    }

    fn name(&self) -> NodeName {
        "DistributedActorPoolProject".into()
    }

    fn op_type(&self) -> NodeType {
        NodeType::DistributedActorPoolProject
    }

    fn multiline_display(&self) -> Vec<String> {
        let mut res = vec![];
        res.push("DistributedActorPoolProject:".to_string());
        res.push(format!("BatchSize = {}", self.batch_size.unwrap_or(0)));
        res.push(format!("MemoryRequest = {}", self.memory_request));
        res
    }

    fn make_state(&self) -> Self::State {
        // Check if we need to initialize the filtered actor handles
        #[cfg(feature = "python")]
        {
            let next_actor_handle_idx =
                self.counter.fetch_add(1, Ordering::SeqCst) % self.actor_handles.len();
            let next_actor_handle = &self.actor_handles[next_actor_handle_idx];
            DistributedActorPoolProjectState {
                actor_handle: next_actor_handle.clone(),
            }
        }
        #[cfg(not(feature = "python"))]
        {
            unimplemented!(
                "DistributedActorPoolProjectOperator::make_state is not implemented without Python"
            );
        }
    }

    fn max_concurrency(&self) -> usize {
        // We set the max concurrency to be the number of actor handles * 2 to such that each actor handle has 2 workers submitting to it.
        // This allows inputs to be queued up concurrently with UDF execution.
        match &self.dynamic {
            // An elastic pool counts actors that are not up yet too: concurrency is fixed when the
            // pipeline starts, and batches simply queue on the actors that are up until the rest join.
            // Each queued batch holds its memory request, so a pool that starts far below its size
            // reserves memory for up to `total_actors * 2` batches.
            Some(_) => self.total_actors * 2,
            None => self.actor_handles.len() * 2,
        }
    }

    fn morsel_size_requirement(&self) -> Option<MorselSizeRequirement> {
        self.batch_size
            .and_then(NonZeroUsize::new)
            .map(MorselSizeRequirement::Strict)
    }
    fn batching_strategy(&self) -> DaftResult<Self::BatchingStrategy> {
        Ok(crate::dynamic_batching::StaticBatchingStrategy::new(
            self.morsel_size_requirement().unwrap_or_default(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::usable_actors;

    #[test]
    fn fixed_pool_prefers_local_actors() {
        assert_eq!(usable_actors(vec![1, 2], vec![3, 4], false), vec![1, 2]);
    }

    #[test]
    fn fixed_pool_falls_back_to_remote_actors() {
        assert_eq!(usable_actors(vec![], vec![3, 4], false), vec![3, 4]);
    }

    #[test]
    fn elastic_pool_uses_local_and_remote_actors() {
        assert_eq!(usable_actors(vec![1, 2], vec![3, 4], true), vec![1, 2, 3, 4]);
    }
}

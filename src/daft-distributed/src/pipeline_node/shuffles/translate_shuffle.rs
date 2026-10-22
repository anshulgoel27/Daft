use std::sync::Arc;

use common_display::utils::bytes_to_human_readable;
use common_error::DaftResult;
use daft_local_plan::ShuffleBackend;
use daft_logical_plan::partitioning::RepartitionSpec;
use daft_schema::schema::SchemaRef;

use crate::pipeline_node::{
    DistributedPipelineNode,
    shuffles::{
        gather::GatherNode, pre_shuffle_merge::PreShuffleMergeNode, repartition::RepartitionNode,
    },
    translate::LogicalPlanToPipelineNodeTranslator,
};

const LARGE_SHUFFLE_SIZE_BYTES_HINT_THRESHOLD: usize = 10 * 1024 * 1024 * 1024;
const LARGE_SHUFFLE_PARTITION_PRODUCT_HINT_THRESHOLD: usize = 500_000;
// Map-reduce shuffle materializes one object per (input, output) slot on the head node.
const PARTITION_SLOT_HEAD_MEMORY_BYTES: usize = 3 * 1024;

enum LargeShuffleTrigger {
    Bytes,
    PartitionProduct(usize),
}

/// Whether a shuffle is large enough to risk Ray object-store / head-node memory pressure.
fn large_shuffle_trigger(
    input_size_bytes: usize,
    input_num_partitions: usize,
    output_num_partitions: usize,
) -> Option<LargeShuffleTrigger> {
    let partition_product = input_num_partitions.saturating_mul(output_num_partitions);
    if input_size_bytes >= LARGE_SHUFFLE_SIZE_BYTES_HINT_THRESHOLD {
        Some(LargeShuffleTrigger::Bytes)
    } else if partition_product >= LARGE_SHUFFLE_PARTITION_PRODUCT_HINT_THRESHOLD {
        Some(LargeShuffleTrigger::PartitionProduct(partition_product))
    } else {
        None
    }
}

fn target_num_partitions(spec: &RepartitionSpec, input_num_partitions: usize) -> usize {
    match spec {
        RepartitionSpec::Hash(c) => c.num_partitions,
        RepartitionSpec::Random(c) => c.num_partitions,
        RepartitionSpec::Range(c) => c.num_partitions,
    }
    .unwrap_or(input_num_partitions)
}

impl LogicalPlanToPipelineNodeTranslator {
    fn flight_backend(&self) -> ShuffleBackend {
        ShuffleBackend::Flight {
            // Placeholder; each shuffle node stamps its own id in `ShuffleContext::new`.
            shuffle_id: 0,
            shuffle_dirs: self.plan_config.namespaced_flight_shuffle_dirs(),
            compression: self.plan_config.config.flight_shuffle_compression.clone(),
        }
    }

    /// Pick the shuffle backend implied by the current execution config, ignoring shuffle size.
    pub(crate) fn select_backend(&self) -> ShuffleBackend {
        if self.plan_config.uses_flight_shuffle() {
            self.flight_backend()
        } else {
            ShuffleBackend::Ray
        }
    }

    /// Like `select_backend`, but in `"auto"` mode switches to flight shuffle when the shuffle is
    /// large enough to risk Ray object-store / head-node memory pressure.
    pub(crate) fn select_backend_for_shuffle(
        &self,
        input_size_bytes: usize,
        input_num_partitions: usize,
        output_num_partitions: usize,
    ) -> ShuffleBackend {
        if self.plan_config.config.shuffle_algorithm.as_str() == "auto"
            && large_shuffle_trigger(
                input_size_bytes,
                input_num_partitions,
                output_num_partitions,
            )
            .is_some()
        {
            self.flight_backend()
        } else {
            self.select_backend()
        }
    }

    pub fn gen_repartition_node(
        &mut self,
        repartition_spec: RepartitionSpec,
        schema: SchemaRef,
        child: DistributedPipelineNode,
        input_size_bytes: usize,
    ) -> DaftResult<DistributedPipelineNode> {
        let input_num_partitions = child.config().clustering_spec.num_partitions();
        let num_partitions = target_num_partitions(&repartition_spec, input_num_partitions);
        let backend =
            self.select_backend_for_shuffle(input_size_bytes, input_num_partitions, num_partitions);
        self.gen_repartition_node_with_backend(
            repartition_spec,
            schema,
            child,
            backend,
            input_size_bytes,
        )
    }

    pub fn gen_repartition_node_with_backend(
        &mut self,
        repartition_spec: RepartitionSpec,
        schema: SchemaRef,
        child: DistributedPipelineNode,
        backend: ShuffleBackend,
        input_size_bytes: usize,
    ) -> DaftResult<DistributedPipelineNode> {
        let input_num_partitions = child.config().clustering_spec.num_partitions();
        let num_partitions = target_num_partitions(&repartition_spec, input_num_partitions);

        self.maybe_warn_large_shuffle(
            &backend,
            input_size_bytes,
            input_num_partitions,
            num_partitions,
        );

        // Flight shuffle spills to disk itself, so merging ahead of it is redundant.
        let use_pre_shuffle_merge = !matches!(backend, ShuffleBackend::Flight { .. })
            && self.should_use_pre_shuffle_merge(&child, num_partitions)?;

        let child_node = if use_pre_shuffle_merge {
            // Create merge node first
            DistributedPipelineNode::new(
                Arc::new(PreShuffleMergeNode::new(
                    self.get_next_pipeline_node_id(),
                    &self.plan_config,
                    self.plan_config.config.pre_shuffle_merge_threshold,
                    schema.clone(),
                    child,
                )),
                &self.meter,
            )
        } else {
            child
        };

        Ok(DistributedPipelineNode::new(
            Arc::new(RepartitionNode::new(
                self.get_next_pipeline_node_id(),
                &self.plan_config,
                repartition_spec,
                schema,
                num_partitions,
                backend,
                child_node,
            )?),
            &self.meter,
        ))
    }

    /// Determine if we should use pre-shuffle merge strategy
    fn should_use_pre_shuffle_merge(
        &self,
        child: &DistributedPipelineNode,
        target_num_partitions: usize,
    ) -> DaftResult<bool> {
        let input_num_partitions = child.config().clustering_spec.num_partitions();

        match self.plan_config.config.shuffle_algorithm.as_str() {
            "pre_shuffle_merge" => Ok(true),
            "map_reduce" => Ok(false),
            "flight_shuffle" => Ok(false), // Flight shuffle will be handled separately
            "auto" => {
                let total_num_partitions = input_num_partitions * target_num_partitions;
                let geometric_mean = (total_num_partitions as f64).sqrt() as usize;
                Ok(geometric_mean
                    > self
                        .plan_config
                        .config
                        .pre_shuffle_merge_partition_threshold)
            }
            _ => Ok(false), // Default to naive map_reduce for unknown strategies
        }
    }

    pub fn gen_gather_node(
        &mut self,
        input_node: DistributedPipelineNode,
        input_size_bytes: usize,
    ) -> DistributedPipelineNode {
        let input_num_partitions = input_node.config().clustering_spec.num_partitions();
        if input_num_partitions == 1 {
            return input_node;
        }

        let backend = self.select_backend_for_shuffle(input_size_bytes, input_num_partitions, 1);

        self.maybe_warn_large_shuffle(&backend, input_size_bytes, input_num_partitions, 1);

        let node_id = self.get_next_pipeline_node_id();
        DistributedPipelineNode::new(
            Arc::new(GatherNode::new(
                node_id,
                &self.plan_config,
                input_node.config().schema.clone(),
                backend,
                input_node,
            )),
            &self.meter,
        )
    }

    /// Hint the user to switch to flight_shuffle when total bytes or partition product
    /// would cause Ray object-store / head-node memory pressure. Skipped on flight_shuffle.
    pub(crate) fn maybe_warn_large_shuffle(
        &mut self,
        backend: &ShuffleBackend,
        input_size_bytes: usize,
        input_num_partitions: usize,
        output_num_partitions: usize,
    ) {
        if matches!(backend, ShuffleBackend::Flight { .. }) {
            return;
        }

        let algo = &self.plan_config.config.shuffle_algorithm;
        // Both triggers point to the same fix (switch to flight_shuffle), so emit at most one
        // hint per shuffle op even if both thresholds are crossed.
        match large_shuffle_trigger(
            input_size_bytes,
            input_num_partitions,
            output_num_partitions,
        ) {
            Some(LargeShuffleTrigger::Bytes) => {
                self.record_flight_shuffle_hint(format!(
                    "Large shuffle (~{} via `{}` shuffle algorithm)",
                    bytes_to_human_readable(input_size_bytes),
                    algo,
                ));
            }
            Some(LargeShuffleTrigger::PartitionProduct(partition_product)) => {
                let head_memory =
                    partition_product.saturating_mul(PARTITION_SLOT_HEAD_MEMORY_BYTES);
                self.record_flight_shuffle_hint(format!(
                    "Shuffle with many partitions ({} × {} = {} pieces, ~{} of head-node memory)",
                    input_num_partitions,
                    output_num_partitions,
                    partition_product,
                    bytes_to_human_readable(head_memory),
                ));
            }
            None => {}
        }
    }

    fn record_flight_shuffle_hint(&mut self, trigger: String) {
        self.record_hint(format!(
            "{trigger}. `flight_shuffle` writes shuffle data to disk and scales better. Enable: \
             daft.context.set_execution_config(shuffle_algorithm=\"flight_shuffle\", \
             flight_shuffle_dirs=[\"/path/to/fast/ssd\"])  # defaults to [\"/tmp\"]."
        ));
    }
}

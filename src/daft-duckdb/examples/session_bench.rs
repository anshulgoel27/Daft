//! Phase 3 micro-benchmark: a fresh per-partition connection vs a reused `DuckDbSession`.
//!
//! Shows the per-task FIXED-overhead saving (connection open + plan translation) when a worker
//! processes many small partitions that share a plan. It does NOT reflect the Arrow->DuckDB
//! registration copy (which dominates large partitions and is inherent — see the follow-ups doc);
//! that is why the partition here is small, so fixed overhead is the visible signal.
//!
//! Run:  cargo run -p daft-duckdb --example session_bench

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use daft_core::prelude::*;
use daft_dsl::expr::bound_expr::BoundExpr;
use daft_dsl::{lit, resolved_col};
use daft_duckdb::{DuckDbConfig, DuckDbExecutor, DuckDbSession};
use daft_local_plan::{LocalNodeContext, LocalPhysicalPlan, LocalPhysicalPlanRef, SourceId};
use daft_logical_plan::stats::StatsState;
use daft_micropartition::{MicroPartition, MicroPartitionRef};
use daft_recordbatch::RecordBatch;

/// `filter(amount > 500)` over a 1000-row in_memory_scan, registered under source id 7.
fn build() -> (LocalPhysicalPlanRef, HashMap<SourceId, Vec<MicroPartitionRef>>) {
    let schema = Arc::new(Schema::new(vec![Field::new("amount", DataType::Int64)]));
    let col = Int64Array::from_vec("amount", (0..1000i64).collect()).into_series();
    let rb = RecordBatch::from_nonempty_columns(vec![col]).unwrap();
    let mp = Arc::new(MicroPartition::new_loaded(
        schema.clone(),
        Arc::new(vec![rb]),
        None,
    ));
    let scan = LocalPhysicalPlan::in_memory_scan(
        7,
        schema.clone(),
        0,
        StatsState::NotMaterialized,
        LocalNodeContext::default(),
    );
    let pred = BoundExpr::try_new(resolved_col("amount").gt(lit(500i64)), &schema).unwrap();
    let plan = LocalPhysicalPlan::filter(
        scan,
        pred,
        StatsState::NotMaterialized,
        LocalNodeContext::default(),
    );
    let mut inputs = HashMap::new();
    inputs.insert(7u32, vec![mp]);
    (plan, inputs)
}

fn main() {
    let (plan, inputs) = build();
    let cfg = DuckDbConfig::default();
    let k: usize = 500;

    // Fresh connection per run (current DuckDbExecutor::run behavior).
    let _ = DuckDbExecutor::run(&plan, &inputs).unwrap(); // warm-up
    let t0 = Instant::now();
    for _ in 0..k {
        let _ = DuckDbExecutor::run(&plan, &inputs).unwrap();
    }
    let fresh_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Reused session (connection + cached translation).
    let mut session = DuckDbSession::new(&cfg).unwrap();
    let _ = session.run(&plan, &inputs).unwrap(); // warm-up
    let t1 = Instant::now();
    for _ in 0..k {
        let _ = session.run(&plan, &inputs).unwrap();
    }
    let reused_ms = t1.elapsed().as_secs_f64() * 1000.0;

    println!("Phase 3 — {k} runs of a small (1000-row) partition:");
    println!(
        "  fresh connection per run : {fresh_ms:8.1} ms total   {:.3} ms/run",
        fresh_ms / k as f64
    );
    println!(
        "  reused DuckDbSession     : {reused_ms:8.1} ms total   {:.3} ms/run",
        reused_ms / k as f64
    );
    println!("  speedup                  : {:.2}x", fresh_ms / reused_ms);
}

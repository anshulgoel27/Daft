//! Spike: validate **zero-copy** Arrow ingestion via the deprecated `duckdb_arrow_scan` C API,
//! in-process from Rust, against the Appender (copy) path the crate ships today.
//!
//! WHY: Phase 2b proved (in Python) that DuckDB's in-place Arrow scan collapses the per-task
//! "cold" cost onto "warm". ADBC was ruled out — its data path is a bulk-ingest *copy*, and it
//! isn't in the bundled lib (see the follow-ups doc). The genuine zero-copy mechanism is the Arrow
//! replacement scan, exposed to C as `duckdb_arrow_scan` — and the **bundled `libduckdb-sys`
//! 1.10501 we already link exports it** (plus the raw connection FFI). This spike confirms whether
//! the Phase 2b win reproduces from Rust on our existing dependency, with no ADBC and no new crate.
//!
//! TWO THINGS IT MEASURES:
//!  1. INGESTION COST (clean): a no-filter `groupby` agg, so there is no pushdown to confound the
//!     copy-vs-zero-copy comparison. Correctness is checked against the Appender path.
//!  2. FILTER-PUSHDOWN CORRECTNESS: the same query WITH a filter. This surfaced a real bug — see
//!     `demo_filter_pushdown_bug` — where DuckDB drops a pushed-down filter over a streamed arrow
//!     scan, returning WRONG results unless filter pushdown is disabled.
//!
//! SAFETY (ownership of the Arrow stream): `duckdb_arrow_scan` -> `Ingest` creates a view via
//! `TableFunction("arrow_scan", {POINTER(stream), ...})->CreateView(name, replace, temporary=false)`
//! (confirmed in the bundled `arrow-c.cpp`). The view references the stream **by raw pointer** and
//! DuckDB does NOT copy or take ownership of it. So each `FFI_ArrowArrayStream` must (a) live at a
//! stable address (boxed) and (b) outlive every query against its view; we drop the connection/db
//! (destroying the views) BEFORE dropping the streams. DuckDB never releases the stream, so the
//! Rust `Drop` (which calls release) is the sole release — no double-free.
//!
//! NOTE: `duckdb_arrow_scan` is part of DuckDB's *deprecated* Arrow C API (present and functional
//! in 1.10501; pin the version). It is, however, the only in-process zero-copy Arrow ingestion
//! DuckDB exposes to a C/Rust caller.
//!
//! Run (release strongly recommended — debug timing is not representative):
//!   cargo run -p daft-duckdb --release --example arrow_scan_spike

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use daft_core::prelude::*;
use daft_dsl::expr::AggExpr;
use daft_dsl::expr::bound_expr::{BoundAggExpr, BoundExpr};
use daft_dsl::{lit, resolved_col};
use daft_duckdb::DuckDbExecutor;
use daft_duckdb::arrow_bridge::{daft_batches_to_arrow, source_to_arrow};
use daft_duckdb::plan_sql::{SqlPlan, plan_to_sql};
use daft_local_plan::{LocalNodeContext, LocalPhysicalPlan, LocalPhysicalPlanRef, SourceId};
use daft_logical_plan::stats::StatsState;
use daft_micropartition::{MicroPartition, MicroPartitionRef};
use daft_recordbatch::RecordBatch;
use duckdb::ffi;

/// `groupby(region).sum(amount) as total` over an `n`-row scan of `amount = 0..n`, `region = i % 8`
/// (source id 7), optionally with a `filter(amount > 100)` between scan and aggregate.
fn build(n: i64, with_filter: bool) -> (LocalPhysicalPlanRef, HashMap<SourceId, Vec<MicroPartitionRef>>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("amount", DataType::Int64),
        Field::new("region", DataType::Int64),
    ]));
    let amount = Int64Array::from_vec("amount", (0..n).collect()).into_series();
    let region = Int64Array::from_vec("region", (0..n).map(|i| i % 8).collect()).into_series();
    let rb = RecordBatch::from_nonempty_columns(vec![amount, region]).unwrap();
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
    let agg_input = if with_filter {
        let pred = BoundExpr::try_new(resolved_col("amount").gt(lit(100i64)), &schema).unwrap();
        LocalPhysicalPlan::filter(
            scan,
            pred,
            StatsState::NotMaterialized,
            LocalNodeContext::default(),
        )
    } else {
        scan
    };
    // Aggregate binds against its input's output schema (unchanged: [amount, region]).
    let group_by = vec![BoundExpr::try_new(resolved_col("region"), &schema).unwrap()];
    let aggs = vec![BoundAggExpr::try_new(AggExpr::Sum(resolved_col("amount")), &schema).unwrap()];
    let out_schema = Arc::new(Schema::new(vec![
        Field::new("region", DataType::Int64),
        Field::new("total", DataType::Int64),
    ]));
    let plan = LocalPhysicalPlan::hash_aggregate(
        agg_input,
        aggs,
        group_by,
        out_schema,
        StatsState::NotMaterialized,
        LocalNodeContext::default(),
    );

    let mut inputs = HashMap::new();
    inputs.insert(7u32, vec![mp]);
    (plan, inputs)
}

/// Wrap the translated query so it returns two scalars while still scanning all inputs + running
/// the full aggregate: `(count(*), sum(total))`.
fn probe_sql(sql: &str) -> String {
    format!("SELECT count(*), CAST(COALESCE(sum(total), 0) AS BIGINT) FROM ({sql}) t")
}

/// Ground-truth `(group_count, sum_of_total)` from the Appender path's actual result rows.
fn agg_check(batches: &[RecordBatch]) -> (i64, i64) {
    let arrow = daft_batches_to_arrow(batches).unwrap();
    let mut n = 0i64;
    let mut s = 0i64;
    for b in &arrow {
        n += b.num_rows() as i64;
        let col = b.column_by_name("total").expect("result has a `total` column");
        let arr = col
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .expect("`total` is Int64");
        for v in arr.iter().flatten() {
            s += v;
        }
    }
    (n, s)
}

fn cstr_to_string(p: *const std::os::raw::c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: `p` points to a NUL-terminated C string owned by the (still-live) result.
    unsafe { std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned() }
}

/// Register each source as a zero-copy Arrow view via `duckdb_arrow_scan`, run `probe`, and return
/// `(count, sum)`. See the module-level SAFETY note for the stream-ownership argument.
///
/// `source_to_arrow` (Daft -> arrow_array) is done INSIDE the timed region so this matches what the
/// Appender path's registration also pays — the only difference left is the register mechanism.
/// `disable_pushdown` issues `SET disabled_optimizers='filter_pushdown'` (the correctness workaround
/// for the streamed-arrow-scan filter bug).
unsafe fn arrow_scan_probe(
    translated: &SqlPlan,
    inputs: &HashMap<SourceId, Vec<MicroPartitionRef>>,
    probe: &str,
    disable_pushdown: bool,
) -> Result<(i64, i64), String> {
    use arrow_array::ffi_stream::FFI_ArrowArrayStream;
    use arrow_array::{RecordBatchIterator, RecordBatchReader};

    let mut db: ffi::duckdb_database = std::ptr::null_mut();
    if ffi::duckdb_open(std::ptr::null(), &mut db) != ffi::duckdb_state_DuckDBSuccess {
        return Err("duckdb_open failed".into());
    }
    let mut conn: ffi::duckdb_connection = std::ptr::null_mut();
    if ffi::duckdb_connect(db, &mut conn) != ffi::duckdb_state_DuckDBSuccess {
        ffi::duckdb_close(&mut db);
        return Err("duckdb_connect failed".into());
    }

    let run_stmt = |conn: ffi::duckdb_connection, sql: &str| -> Result<(), String> {
        let c = std::ffi::CString::new(sql).unwrap();
        let mut r: ffi::duckdb_result = std::mem::zeroed();
        let st = ffi::duckdb_query(conn, c.as_ptr(), &mut r);
        let res = if st != ffi::duckdb_state_DuckDBSuccess {
            Err(cstr_to_string(ffi::duckdb_result_error(&mut r)))
        } else {
            Ok(())
        };
        ffi::duckdb_destroy_result(&mut r);
        res
    };

    if disable_pushdown {
        if let Err(e) = run_stmt(conn, "SET disabled_optimizers='filter_pushdown'") {
            ffi::duckdb_disconnect(&mut conn);
            ffi::duckdb_close(&mut db);
            return Err(format!("SET failed: {e}"));
        }
    }

    // Streams are kept alive (boxed, stable address) until AFTER close() — the views reference them
    // by raw pointer. We push every box before checking the call result so an error path still
    // drops them after teardown, never before.
    let mut streams: Vec<Box<FFI_ArrowArrayStream>> = Vec::new();
    let mut err: Option<String> = None;
    for source_id in &translated.bindings {
        let Some(mps) = inputs.get(source_id) else {
            err = Some(format!("missing source {source_id}"));
            break;
        };
        let batches = match source_to_arrow(mps) {
            Ok(b) if !b.is_empty() => b,
            Ok(_) => {
                err = Some(format!("source {source_id} has no batches"));
                break;
            }
            Err(e) => {
                err = Some(e.to_string());
                break;
            }
        };
        let schema = batches[0].schema();
        let reader: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
            batches.into_iter().map(Ok::<_, arrow_schema::ArrowError>),
            schema,
        ));
        let mut stream = Box::new(FFI_ArrowArrayStream::new(reader));
        let name = std::ffi::CString::new(format!("daft_src_{source_id}")).unwrap();
        let stream_ptr = (&mut *stream as *mut FFI_ArrowArrayStream) as ffi::duckdb_arrow_stream;
        let st = ffi::duckdb_arrow_scan(conn, name.as_ptr(), stream_ptr);
        streams.push(stream);
        if st != ffi::duckdb_state_DuckDBSuccess {
            err = Some(format!("duckdb_arrow_scan failed for source {source_id}"));
            break;
        }
    }

    let out = if let Some(e) = err {
        Err(e)
    } else {
        let q = std::ffi::CString::new(probe).unwrap();
        let mut result: ffi::duckdb_result = std::mem::zeroed();
        let st = ffi::duckdb_query(conn, q.as_ptr(), &mut result);
        let r = if st != ffi::duckdb_state_DuckDBSuccess {
            Err(format!(
                "query failed: {}",
                cstr_to_string(ffi::duckdb_result_error(&mut result))
            ))
        } else {
            let n = ffi::duckdb_value_int64(&mut result, 0, 0);
            let s = ffi::duckdb_value_int64(&mut result, 1, 0);
            Ok((n, s))
        };
        ffi::duckdb_destroy_result(&mut result);
        r
    };

    // Destroy the views (disconnect/close) BEFORE dropping the streams they point at.
    ffi::duckdb_disconnect(&mut conn);
    ffi::duckdb_close(&mut db);
    drop(streams);
    out
}

/// (1) Clean ingestion-cost comparison on a no-filter aggregate (no pushdown confound).
fn measure_ingestion() {
    let sizes = [100_000i64, 1_000_000, 10_000_000];
    let k = 5usize;

    // Warm up: first DuckDB open + first arrow_scan pay one-time inits we don't want on size #1.
    {
        let (plan, inputs) = build(1_000, false);
        let translated = plan_to_sql(&plan).unwrap();
        let probe = probe_sql(&translated.sql);
        let _ = DuckDbExecutor::run(&plan, &inputs).unwrap();
        unsafe { arrow_scan_probe(&translated, &inputs, &probe, false).unwrap() };
    }

    println!("(1) Zero-copy `duckdb_arrow_scan` vs Appender copy — cold per-task cost, no filter");
    println!(
        "{:>12} {:>14} {:>15} {:>9} {:>8}",
        "rows", "appender ms", "arrow_scan ms", "speedup", "groups"
    );
    for &n in &sizes {
        let (plan, inputs) = build(n, false);
        let translated = plan_to_sql(&plan).unwrap();
        let probe = probe_sql(&translated.sql);

        // Correctness: arrow_scan's (count, sum) must equal the Appender path's actual result.
        let (ref_n, ref_s) = agg_check(&DuckDbExecutor::run(&plan, &inputs).unwrap());
        let (scan_n, scan_s) = unsafe { arrow_scan_probe(&translated, &inputs, &probe, false).unwrap() };
        assert_eq!(
            (scan_n, scan_s),
            (ref_n, ref_s),
            "arrow_scan disagrees with Appender at n={n} (no filter)"
        );

        // Cold per run: open + register + query, fresh each iteration (the per-task model).
        let t0 = Instant::now();
        for _ in 0..k {
            let _ = DuckDbExecutor::run(&plan, &inputs).unwrap();
        }
        let app_ms = t0.elapsed().as_secs_f64() * 1000.0 / k as f64;

        let t1 = Instant::now();
        for _ in 0..k {
            unsafe { arrow_scan_probe(&translated, &inputs, &probe, false).unwrap() };
        }
        let scan_ms = t1.elapsed().as_secs_f64() * 1000.0 / k as f64;

        println!(
            "{:>12} {:>14.1} {:>15.1} {:>8.2}x {:>8}",
            n, app_ms, scan_ms, app_ms / scan_ms, ref_n
        );
    }
}

/// (2) Demonstrate the filter-pushdown correctness bug over a streamed arrow scan.
fn demo_filter_pushdown_bug() {
    let n = 1_000_000i64;
    let (plan, inputs) = build(n, true); // filter(amount > 100)
    let translated = plan_to_sql(&plan).unwrap();
    let probe = probe_sql(&translated.sql);

    let (_, ref_sum) = agg_check(&DuckDbExecutor::run(&plan, &inputs).unwrap()); // correct (Appender)
    let (_, scan_on) = unsafe { arrow_scan_probe(&translated, &inputs, &probe, false).unwrap() }; // pushdown ON
    let (_, scan_off) = unsafe { arrow_scan_probe(&translated, &inputs, &probe, true).unwrap() }; // pushdown OFF

    println!("\n(2) Filter-pushdown correctness over a streamed arrow_scan (filter: amount > 100)");
    println!("    Appender (reference)           sum(total) = {ref_sum}");
    println!("    arrow_scan, pushdown ON        sum(total) = {scan_on}  {}",
        if scan_on == ref_sum { "OK" } else { "<-- WRONG (filter dropped)" });
    println!("    arrow_scan, pushdown DISABLED   sum(total) = {scan_off}  {}",
        if scan_off == ref_sum { "OK (workaround)" } else { "STILL WRONG" });
}

fn main() {
    measure_ingestion();
    demo_filter_pushdown_bug();
}

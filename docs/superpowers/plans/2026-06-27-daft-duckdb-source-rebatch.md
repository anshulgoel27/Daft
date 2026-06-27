# daft-duckdb: auto-batch sources for parallel arrow scans — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split each registered source's Arrow data into ≤122,880-row zero-copy batches before `register_arrow`, so DuckDB's per-record-batch-parallel `arrow_scan` uses all cores (turns the 10M join from ~1.35× slower than swordfish into ~1.7× faster).

**Architecture:** Add a pure `rebatch` helper to `src/daft-duckdb/src/executor.rs` that slices any batch larger than a target row count into zero-copy `RecordBatch::slice` pieces, and call it in `register_sources` between `source_to_arrow` and `register_arrow`. Single-file change; nothing downstream changes.

**Tech Stack:** Rust, `arrow-array` 59 (`RecordBatch::slice`), the existing `daft-duckdb` zero-copy `register_arrow` path.

## Global Constraints

- Branch: work on `duckdb-poc` (the zero-copy integration is already merged there). Never commit to `main`.
- `TARGET_ROWS_PER_BATCH = 122_880` (DuckDB row-group/morsel size = 60 × 2048), a module constant in `executor.rs`.
- Zero-copy must be preserved: split via `RecordBatch::slice` (an offset view), never a row copy.
- Single-file change: only `src/daft-duckdb/src/executor.rs`.
- No `Co-Authored-By` trailer in the commit.
- Build/test: `cargo test -p daft-duckdb`.

---

### Task 1: `rebatch` helper + wire into `register_sources`

**Files:**
- Modify: `src/daft-duckdb/src/executor.rs` (add `TARGET_ROWS_PER_BATCH` const + `rebatch` fn; one-line call in `register_sources`; add tests to the `#[cfg(test)] mod tests` block)

**Interfaces:**
- Consumes: `ArrowBatch` (existing alias `use arrow_array::RecordBatch as ArrowBatch;`), `RecordBatch::{num_rows, slice}`; `source_to_arrow`/`register_arrow` (existing).
- Produces: `fn rebatch(batches: Vec<ArrowBatch>, target_rows: usize) -> Vec<ArrowBatch>`; `const TARGET_ROWS_PER_BATCH: usize`.

- [ ] **Step 1: Write the failing `rebatch` unit tests**

Add to the `#[cfg(test)] mod tests` block in `src/daft-duckdb/src/executor.rs`:

```rust
    #[test]
    fn rebatch_splits_large_batches_zero_copy_and_lossless() {
        use arrow_array::{Array, Int64Array, RecordBatch as AB};
        use arrow_schema::{DataType as AdT, Field as AF, Schema as ASch};
        use std::sync::Arc as A;

        let schema = A::new(ASch::new(vec![AF::new("v", AdT::Int64, false)]));
        let arr = A::new(Int64Array::from_iter_values(0..1000i64));
        let batch = AB::try_new(schema, vec![arr]).unwrap();

        // target 300 -> ceil(1000/300) = 4 slices
        let out = rebatch(vec![batch.clone()], 300);
        assert_eq!(out.len(), 4, "ceil(1000/300) slices");
        assert_eq!(out.iter().map(|b| b.num_rows()).sum::<usize>(), 1000, "rows preserved");
        // values + order preserved across the offset slices (value(i) is offset-correct)
        let mut seen = Vec::new();
        for b in &out {
            let c = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            for i in 0..b.num_rows() {
                seen.push(c.value(i));
            }
        }
        assert_eq!(seen, (0..1000i64).collect::<Vec<_>>(), "values preserved");

        // <= target -> unchanged (one batch, same rows)
        let small = rebatch(vec![batch.clone()], 5000);
        assert_eq!(small.len(), 1);
        assert_eq!(small[0].num_rows(), 1000);

        // target 0 -> no split (guard against divide-by-zero)
        let zero = rebatch(vec![batch], 0);
        assert_eq!(zero.len(), 1);

        // empty input -> empty output
        assert!(rebatch(Vec::<AB>::new(), 300).is_empty());
    }
```

- [ ] **Step 2: Run the test to verify it fails (compile error: `rebatch` not defined)**

Run:
```bash
cargo test -p daft-duckdb rebatch_splits_large_batches_zero_copy_and_lossless 2>&1 | tail -15
```
Expected: FAIL — compile error `cannot find function \`rebatch\`` (and `TARGET_ROWS_PER_BATCH` is not yet referenced).

- [ ] **Step 3: Implement `rebatch` + the constant**

In `src/daft-duckdb/src/executor.rs`, add near the top (after the imports, module scope) the constant and helper. (`ArrowBatch` is the existing `arrow_array::RecordBatch` alias.)

```rust
/// DuckDB's `arrow_scan` parallelizes at record-batch granularity, so a single large Arrow batch
/// is scanned single-threaded. Splitting each source into one batch per ~morsel lets the scan use
/// all cores. 122_880 = DuckDB's row-group/morsel size (60 * 2048), so each batch ≈ one morsel.
const TARGET_ROWS_PER_BATCH: usize = 122_880;

/// Split any batch larger than `target_rows` into ~equal slices so DuckDB's per-batch-parallel
/// arrow scan can use all cores. Batches with `<= target_rows` (or `target_rows == 0`) pass
/// through unchanged. `RecordBatch::slice` is a zero-copy offset view — no row copy — so the
/// zero-copy registration property is preserved.
fn rebatch(batches: Vec<ArrowBatch>, target_rows: usize) -> Vec<ArrowBatch> {
    if target_rows == 0 {
        return batches;
    }
    let mut out = Vec::with_capacity(batches.len());
    for batch in batches {
        let n = batch.num_rows();
        if n <= target_rows {
            out.push(batch);
            continue;
        }
        let mut offset = 0;
        while offset < n {
            let len = std::cmp::min(target_rows, n - offset);
            out.push(batch.slice(offset, len));
            offset += len;
        }
    }
    out
}
```

- [ ] **Step 4: Run the `rebatch` test to verify it passes**

Run:
```bash
cargo test -p daft-duckdb rebatch_splits_large_batches_zero_copy_and_lossless -- --nocapture
```
Expected: PASS.

- [ ] **Step 5: Wire `rebatch` into `register_sources`**

In `register_sources`, insert the `rebatch` call between the empty-source guard and `register_arrow`. The function becomes:

```rust
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
        // Split large batches so DuckDB's per-batch-parallel arrow scan uses all cores.
        let arrow_batches = rebatch(arrow_batches, TARGET_ROWS_PER_BATCH);
        let view = conn
            .register_arrow(&format!("daft_src_{source_id}"), arrow_batches)
            .map_err(|e| DaftError::External(Box::new(e)))?;
        views.push(view);
    }
    Ok(views)
}
```

- [ ] **Step 6: Add the end-to-end integration test (source > target → sliced → correct)**

Add to the `#[cfg(test)] mod tests` block:

```rust
    #[test]
    fn filter_over_large_source_is_rebatched_and_correct() {
        // 300_000-row source (> TARGET_ROWS_PER_BATCH 122_880 -> 3 batches) so register_sources
        // slices it; the filter must still return the exact rows through the zero-copy scan.
        let schema = Arc::new(Schema::new(vec![Field::new("amount", DataType::Int64)]));
        let col = Int64Array::from_vec("amount", (0..300_000i64).collect()).into_series();
        let rb = daft_recordbatch::RecordBatch::from_nonempty_columns(vec![col]).unwrap();
        let mp = Arc::new(MicroPartition::new_loaded(
            schema.clone(),
            Arc::new(vec![rb]),
            None,
        ));
        let scan = LocalPhysicalPlan::in_memory_scan(
            11,
            schema.clone(),
            0,
            StatsState::NotMaterialized,
            LocalNodeContext::default(),
        );
        // amount > 250_000 over 0..=299_999 -> values 250_001..=299_999 -> 49_999 rows.
        let pred = daft_dsl::expr::bound_expr::BoundExpr::try_new(
            resolved_col("amount").gt(lit(250_000i64)),
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
        inputs.insert(11u32, vec![mp]);

        let out = DuckDbExecutor::run(&plan, &inputs).unwrap();
        let total: usize = out.iter().map(|b| b.len()).sum();
        assert_eq!(total, 49_999);
    }
```

- [ ] **Step 7: Run the full crate test suite**

Run:
```bash
cargo test -p daft-duckdb
```
Expected: all pass — the two new tests plus every existing test (`filter_over_scan_runs_in_duckdb`, `compound_multitype_filter_pushdown`, `session_reuses_connection_across_runs`, `run_with_config_applies_thread_and_memory_caps`, `bench_runs_registers_once_and_times_each_run`, `round_trips_daft_arrow_daft`). Small-data tests are unaffected (their sources are ≤ target, so `rebatch` is a no-op).

- [ ] **Step 8: Commit**

```bash
git add src/daft-duckdb/src/executor.rs
git commit -m "feat(duckdb): auto-batch sources so arrow_scan parallelizes

Split each source into <=122,880-row zero-copy RecordBatch slices before
register_arrow so DuckDB's per-batch-parallel arrow scan uses all cores.
A single large batch otherwise scans single-threaded; this flips the 10M
join from ~1.35x slower than swordfish to ~1.7x faster. Zero-copy and
results unchanged."
```

---

## Verification (post-implementation; not a code task)

Confirm the gain lands automatically through the normal path (single-partition bench inputs now get split by `rebatch`):

```bash
make build-release
.venv/bin/python -m tests.duckdb_poc.bench_duckdb_vs_swordfish
```
Expected: the 10M `duck cold` column drops from ~250 ms to roughly ~110 ms (now faster than swordfish's ~190 ms) with `reg` still ~0.2 ms; the 10K/1M rows behave as before (10K source ≤ target → unchanged). Report the before/after table.

---

## Self-Review

**1. Spec coverage:** `rebatch` helper with target-rows-per-batch + zero-copy slice → Task 1 Step 3. `TARGET_ROWS_PER_BATCH = 122_880` → Step 3. Call in `register_sources` between `source_to_arrow` and `register_arrow` → Step 5. Unit tests (split count, row+value preservation, no-split ≤ target, `target==0` guard, empty input) → Step 1. ~300K-row executor integration test → Step 6. Existing tests green → Step 7. Bench re-run verification → Verification section. All spec items covered.

**2. Placeholder scan:** No TBD/“handle errors”. Every code step has complete code; every run step has the exact command + expected outcome.

**3. Type consistency:** `rebatch(Vec<ArrowBatch>, usize) -> Vec<ArrowBatch>` defined in Step 3 and called identically in Step 5; `ArrowBatch` is the existing `arrow_array::RecordBatch` alias; `TARGET_ROWS_PER_BATCH: usize` defined once and used in Step 5. The integration test uses the same `DuckDbExecutor::run` + Daft builders as the existing `filter_plan_and_inputs` test. Test fn names are unique (`rebatch_splits_large_batches_zero_copy_and_lossless`, `filter_over_large_source_is_rebatched_and_correct`).

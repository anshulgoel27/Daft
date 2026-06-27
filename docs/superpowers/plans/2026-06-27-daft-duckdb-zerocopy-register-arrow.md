# Daft DuckDB Zero-Copy Execution via `register_arrow` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `daft-duckdb`'s Appender copy-into-storage source registration with the zero-copy `Connection::register_arrow` from the duckdb-rs `arrow-zerocopy` branch (arrow-rs 59), giving in-place Arrow scans with filter/projection pushdown and deleting the now-redundant v58↔v59 FFI bridges + `CREATE TABLE` DDL.

**Architecture:** Point the `duckdb` dependency at the arrow-59 fork branch so Daft and duckdb share one `arrow-array` 59 in the dependency graph (`duckdb::arrow::*` becomes `arrow_array::*`). Source registration becomes `conn.register_arrow(name, batches)`, returning an `ArrowView` held for the query's lifetime; results come back from `query_arrow` already as arrow-rs 59 batches. Approach A (full replace): the Appender, the FFI bridges, and the DDL are removed.

**Tech Stack:** Rust, `cargo`, arrow-rs 59 (`arrow-array`/`arrow-schema`), duckdb-rs (fork branch, `bundled` DuckDB v1.5.4 amalgamation), Daft (`daft-recordbatch`, `daft-micropartition`, `daft-local-plan`, `daft-dsl`).

## Global Constraints

- **Branch:** all work on a new feature branch off the current `duckdb-poc` branch (where `src/daft-duckdb` lives). Never commit to `main`/`master`.
- **Dependency:** `duckdb = { git = "https://github.com/anshulgoel27/duckdb-rs", branch = "arrow-zerocopy", features = ["bundled"] }` (the `appender-arrow` feature is dropped in Task 2, once the Appender is gone).
- **Arrow must unify:** the build must resolve exactly one `arrow-array` `59.x` shared by Daft and duckdb. Verify with `cargo tree -i arrow-array` (one version) — this gates the bridge deletions.
- **Approach A (full replace):** no Appender fallback. An unhandled pushed filter/type fails loud (duckdb-rs error stream → `DaftError::External`), never a silent wrong result.
- **First build is slow:** the `bundled` feature compiles the DuckDB v1.5.4 amalgamation from the branch (several minutes). Not a hang.
- **Commit messages:** do NOT add a `Co-Authored-By` trailer (per the user's standing instruction this session).
- **Build/test:** `cargo build -p daft-duckdb` and `cargo test -p daft-duckdb` (no extra features needed for the executor tests).

---

## File Structure

- `src/daft-duckdb/Cargo.toml` — dependency swap (git branch; drop `appender-arrow`).
- `src/daft-duckdb/src/executor.rs` — the change: registration via `register_arrow` (holding `ArrowView`s), simplified `execute`, and deletion of the Appender + FFI-bridge + DDL functions and the dead bridge test. Add one pushdown test.
- `src/daft-duckdb/src/arrow_bridge.rs` — unchanged (`source_to_arrow`/`arrow_to_daft_batches` still used).
- `src/daft-duckdb/src/{plan_sql.rs,expr_sql.rs,python.rs}` — unchanged.

---

### Task 1: Branch + dependency swap to the arrow-59 fork (foundation checkpoint)

Swap the `duckdb` dependency to the fork branch **without changing executor logic yet**. The existing Appender/bridge code still compiles against the arrow-59 branch (the v58↔v59 casts become same-type no-ops under unified arrow), so the existing tests stay green. This isolates "the branch dep + arrow unification work" from the logic change.

**Files:**
- Modify: `src/daft-duckdb/Cargo.toml:16`

**Interfaces:**
- Produces: a build where `duckdb` is the fork branch and `arrow-array` resolves to a single `59.x` shared with Daft.

- [ ] **Step 1: Create the feature branch**

```bash
cd /Volumes/Work/Code/Daft
git checkout -b duckdb-zerocopy-register-arrow
git rev-parse --abbrev-ref HEAD   # expect: duckdb-zerocopy-register-arrow
```

- [ ] **Step 2: Point the `duckdb` dependency at the fork branch (keep `appender-arrow` for now)**

In `src/daft-duckdb/Cargo.toml`, replace the line:
```toml
duckdb = {version = "1.1", features = ["bundled", "appender-arrow"]}
```
with:
```toml
duckdb = {git = "https://github.com/anshulgoel27/duckdb-rs", branch = "arrow-zerocopy", features = ["bundled", "appender-arrow"]}
```

- [ ] **Step 3: Build (first build compiles the bundled amalgamation — slow)**

Run:
```bash
cargo build -p daft-duckdb
```
Expected: builds successfully. The existing executor (Appender + FFI bridges) still compiles; under unified arrow the v58↔v59 pointer casts are between identical types (harmless).

- [ ] **Step 4: Verify a single `arrow-array` version**

Run:
```bash
cargo tree -i arrow-array -p daft-duckdb
```
Expected: exactly one `arrow-array v59.x` node (shared by `daft-duckdb`, other Daft crates, and `duckdb`). If two versions appear, STOP — arrow did not unify; the bridge deletions in Task 2 are unsafe and the dependency/features must be reconciled first.

- [ ] **Step 5: Run the existing tests (still green on the new dep)**

Run:
```bash
cargo test -p daft-duckdb
```
Expected: all existing tests pass (`filter_over_scan_runs_in_duckdb`, `run_with_config_applies_thread_and_memory_caps`, `session_reuses_connection_across_runs`, `bench_runs_registers_once_and_times_each_run`, `round_trips_daft_arrow_daft`, `ffi_bridge_propagates_reader_error_cleanly`).

- [ ] **Step 6: Commit**

```bash
git add src/daft-duckdb/Cargo.toml Cargo.lock
git commit -m "build(duckdb): point daft-duckdb at the arrow-zerocopy duckdb-rs branch"
```

---

### Task 2: Replace Appender registration with `register_arrow`; delete bridges + DDL

Swap source registration to `conn.register_arrow` (zero-copy, holding `ArrowView`s for the query lifetime), simplify `execute` to consume arrow-rs 59 batches directly, and delete the now-dead Appender + FFI-bridge + DDL code (and the bridge test). Drop the `appender-arrow` feature. The whole change lands together so the crate compiles warning-clean (dead code would otherwise fail the workspace lints).

**Files:**
- Modify: `src/daft-duckdb/src/executor.rs`
- Modify: `src/daft-duckdb/Cargo.toml`

**Interfaces:**
- Consumes (from the branch): `duckdb::Connection::register_arrow(&self, name: &str, batches: Vec<arrow_array::RecordBatch>) -> duckdb::Result<duckdb::ArrowView<'_>>`; `duckdb::ArrowView<'conn>` (auto-unregisters the view on `Drop`).
- Consumes (existing, unchanged): `crate::arrow_bridge::{source_to_arrow, arrow_to_daft_batches}`; `crate::plan_sql::plan_to_sql`.
- Produces: `fn register_sources<'c>(conn: &'c duckdb::Connection, bindings: &[SourceId], inputs: &HashMap<SourceId, Vec<MicroPartitionRef>>) -> DaftResult<Vec<duckdb::ArrowView<'c>>>`; `fn execute(conn: &duckdb::Connection, sql: &str) -> DaftResult<Vec<RecordBatch>>` (unchanged signature). `DuckDbExecutor::{run,run_with_config,bench_runs}` and `DuckDbSession::{new,run}` keep their public signatures.

- [ ] **Step 1: Drop the `appender-arrow` feature from the dependency**

In `src/daft-duckdb/Cargo.toml`, change the `duckdb` line to:
```toml
duckdb = {git = "https://github.com/anshulgoel27/duckdb-rs", branch = "arrow-zerocopy", features = ["bundled"]}
```
(`register_arrow`/`ArrowView` are not gated by `appender-arrow`; removing it is what forces the Appender code out.)

- [ ] **Step 2: Rewrite `register_sources` to return held `ArrowView`s**

In `src/daft-duckdb/src/executor.rs`, replace the existing `register_sources` function with:

```rust
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
```

- [ ] **Step 3: Simplify `execute` to consume arrow-rs 59 batches directly**

Replace the existing `execute` function with:

```rust
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
```

- [ ] **Step 4: Hold views across execution in `DuckDbExecutor::run_with_config` and `bench_runs`**

Replace the body of `run_with_config` with:
```rust
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
```
(`run` is unchanged — it delegates to `run_with_config`.)

Replace the body of `bench_runs` with:
```rust
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
```
Note: bind to `_views` (a named, underscore-prefixed binding) — NOT `let _ = ...`, which would drop the views immediately and unregister the sources before `execute` runs.

- [ ] **Step 5: Update `DuckDbSession::run` to register fresh views per run (no `DROP TABLE`)**

Replace the body of `DuckDbSession::run` with:
```rust
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
```
The `_views` borrow `self.conn` only for the duration of `run()` (they are not stored on the struct — that would be self-referential and would not compile). The `DuckDbSession` struct itself (`conn`, `cached`) is unchanged.

- [ ] **Step 6: Delete the dead Appender, DDL, and FFI-bridge code**

Delete these functions from `executor.rs` entirely:
- `register_arrow_table`
- `build_create_table_ddl`
- `arrow_dtype_to_duckdb_type`
- `arrow_array_to_duck_rb`
- `v59_reader_to_duck_batches`
- `duck_rb_to_arrow_array`

Delete the now-unused imports at the top of the file:
- `use arrow_array::ffi_stream::FFI_ArrowArrayStream as Stream59;`
- `use arrow_schema::DataType as AD;`

Keep `use arrow_array::RecordBatch as ArrowBatch;` (used by `execute`). Replace the file's leading comment block (the one describing the v58/v59 ingestion bridge) with a short note:
```rust
// Zero-copy execution: sources are registered as in-place Arrow views via
// duckdb::Connection::register_arrow (arrow-rs 59 unified across Daft and duckdb), so DuckDB
// pushes projection + filters into the scan. Results come back from query_arrow already as
// arrow_array v59 batches. See docs/superpowers/specs/2026-06-27-daft-duckdb-zerocopy-register-arrow-design.md.
```

- [ ] **Step 7: Delete the dead FFI-bridge test**

In the `#[cfg(test)] mod tests` block, delete the entire `ffi_bridge_propagates_reader_error_cleanly` test function (it tested `v59_reader_to_duck_batches`, which no longer exists).

- [ ] **Step 8: Build (expect warning-clean) and run the existing tests**

Run:
```bash
cargo build -p daft-duckdb 2>&1 | tail -5
cargo test -p daft-duckdb
```
Expected: builds with no warnings (no dead code / unused imports), and these tests pass: `filter_over_scan_runs_in_duckdb` (now a filter pushed into the zero-copy scan), `run_with_config_applies_thread_and_memory_caps`, `session_reuses_connection_across_runs` (now via view drop + re-register), `bench_runs_registers_once_and_times_each_run`, `round_trips_daft_arrow_daft`.

- [ ] **Step 9: Commit**

```bash
git add src/daft-duckdb/src/executor.rs src/daft-duckdb/Cargo.toml Cargo.lock
git commit -m "feat(duckdb): zero-copy source registration via register_arrow

Replace the Appender copy-into-storage path with conn.register_arrow,
holding ArrowViews for the query lifetime so DuckDB scans Daft's Arrow
buffers in place with filter/projection pushdown. Arrow unifies to 59, so
the v58<->v59 FFI bridges and the CREATE TABLE DDL are removed."
```

---

### Task 3: Add an end-to-end compound multi-type filter pushdown test

Add one executor test that a compound predicate over mixed scalar types (int comparison + utf8 equality joined by AND) round-trips correctly through the zero-copy `register_arrow` path — proving multi-type + conjunction pushdown from a real Daft plan, beyond the single-int-column `filter_over_scan` case.

(Struct-field `STRUCT_EXTRACT` pushdown is already covered by the duckdb-rs layer's own tests; a Daft-level struct-field test additionally depends on Daft expression/SQL support for struct extraction in `expr_sql.rs`/`plan_sql.rs` — out of scope here, noted as a follow-up.)

**Files:**
- Modify: `src/daft-duckdb/src/executor.rs` (test module)

**Interfaces:**
- Consumes: `DuckDbExecutor::run`; Daft test builders (`Schema`, `Int64Array`, `Utf8Array`, `MicroPartition`, `LocalPhysicalPlan`, `BoundExpr`, `resolved_col`, `lit`) as already used by `filter_plan_and_inputs`.

- [ ] **Step 1: Add the failing test**

In the `#[cfg(test)] mod tests` block of `executor.rs`, add:

```rust
    /// `filter(amount > 100 AND region = 'us')` over a 4-row int+utf8 source, registered under
    /// source id 7. Exercises multi-type (int comparison + utf8 equality) + conjunction pushdown
    /// through the zero-copy register_arrow scan.
    fn compound_plan_and_inputs() -> (LocalPhysicalPlanRef, HashMap<u32, Vec<MicroPartitionRef>>) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("amount", DataType::Int64),
            Field::new("region", DataType::Utf8),
        ]));
        let amount = Int64Array::from_vec("amount", vec![50i64, 150, 250, 350]).into_series();
        let region =
            Utf8Array::from_slice("region", ["us", "eu", "us", "us"].as_slice()).into_series();
        let rb = daft_recordbatch::RecordBatch::from_nonempty_columns(vec![amount, region]).unwrap();
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
        // amount=[50,150,250,350], region=[us,eu,us,us]
        // amount>100 -> 150(eu),250(us),350(us); AND region='us' -> 250,350 => 2 rows.
        let pred = daft_dsl::expr::bound_expr::BoundExpr::try_new(
            resolved_col("amount")
                .gt(lit(100i64))
                .and(resolved_col("region").eq(lit("us"))),
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
        inputs.insert(7u32, vec![mp]);
        (plan, inputs)
    }

    #[test]
    fn compound_multitype_filter_pushdown() {
        let (plan, inputs) = compound_plan_and_inputs();
        let out = DuckDbExecutor::run(&plan, &inputs).unwrap();
        let total: usize = out.iter().map(|b| b.len()).sum();
        assert_eq!(total, 2); // (250, "us") and (350, "us")
    }
```

- [ ] **Step 2: Run the test**

Run:
```bash
cargo test -p daft-duckdb compound_multitype_filter_pushdown -- --nocapture
```
Expected: PASS (the filter is pushed into the zero-copy Arrow scan; 2 rows survive). If `lit("us")` or `.eq(...)`/`.and(...)` does not resolve, check `daft-dsl`'s prelude/import paths used by the existing `filter_plan_and_inputs` and match them; if `expr_sql.rs` cannot translate utf8 equality or `AND`, that is a real gap to surface (do not weaken the test) — report it.

- [ ] **Step 3: Run the full crate test suite**

Run:
```bash
cargo test -p daft-duckdb
```
Expected: all tests pass (the existing executor + arrow_bridge tests plus the new `compound_multitype_filter_pushdown`).

- [ ] **Step 4: Commit**

```bash
git add src/daft-duckdb/src/executor.rs
git commit -m "test(duckdb): end-to-end compound multi-type filter pushdown via zero-copy scan"
```

---

## Self-Review

**1. Spec coverage**
- Dependency swap to git branch + arrow unification verification → Task 1. ✓
- Replace Appender with `register_arrow` holding `ArrowView`s for the query lifetime → Task 2 Steps 2,4,5. ✓
- Delete v58↔v59 FFI bridges + `CREATE TABLE` DDL + dead bridge test → Task 2 Steps 6,7. ✓
- `DuckDbSession` drop/re-register views (no `DROP TABLE`) → Task 2 Step 5. ✓
- Keep existing executor tests green → Task 1 Step 5, Task 2 Step 8. ✓
- Add a pushdown test → Task 3 (compound multi-type; struct-field noted as duckdb-rs-covered + Daft follow-up). ✓
- Approach A / fail-loud accepted, arrow-unification gate, first-build-slow → Global Constraints + Task 1 Step 4. ✓

**2. Placeholder scan:** No TBD/TODO/"handle errors". Every code step shows complete code; every run step has the exact command + expected outcome. The one conditional (Task 3 Step 2 "if `expr_sql` can't translate…") names the exact check and the correct action (surface the gap, don't weaken the test) — not a placeholder.

**3. Type consistency:** `register_sources` returns `Vec<duckdb::ArrowView<'c>>` and is consumed as `let _views = ...` in `run_with_config`/`bench_runs`/`DuckDbSession::run` (held across `execute`). `execute` keeps its `(&Connection, &str) -> DaftResult<Vec<RecordBatch>>` signature. `ArrowBatch = arrow_array::RecordBatch` is used consistently in `execute`. `register_arrow(&str, Vec<arrow_array::RecordBatch>)` matches `source_to_arrow`'s return type under unified arrow. Deleted symbols (`register_arrow_table`, `*_duck_rb`, DDL fns, `Stream59`, `AD`) have no remaining references after Task 2.

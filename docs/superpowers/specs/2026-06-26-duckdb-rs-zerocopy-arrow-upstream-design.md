# duckdb-rs Zero-Copy Arrow Registration — Upstream/Production Design

**Date:** 2026-06-26
**Status:** Design approved; ready for implementation plan
**Related:** spike design `2026-06-26-duckdb-rs-zerocopy-arrow-shim-design.md` + [POC follow-ups Phase 2d](2026-06-25-duckdb-poc-performance-followups.md) · checkout `/Volumes/Work/Code/duckdb-rs` (v1.10504.0)

## Goal

Turn the validated spike (Phase 2d) into a production-quality, upstream-mergeable duckdb-rs feature: a `Connection::register_arrow(name, batches)` API that registers Arrow `RecordBatch`es as a **zero-copy** DuckDB view with **correct, footgun-free** projection *and* filter pushdown — matching the duckdb-python binding's behavior, with no `SET disabled_optimizers='filter_pushdown'` requirement.

Built on a **new branch `arrow-zerocopy` off duckdb-rs `main`**, keeping the existing `zerocopy-arrow-shim-spike` branch untouched.

## Non-Goals

- Replacing the existing `ArrowVTab` (the `arrow()` table function) — this is a complementary zero-copy registration path.
- Multi-batch heterogeneous schemas (all batches in one registration share a schema).
- A `Send`/`Sync` `ArrowView` (kept `!Send`, consistent with the crate's `!Send` `Connection`).
- Exhaustive coverage of every exotic `TableFilterType` on day one — unhandled kinds fail loud (never silently wrong); coverage is expandable.

## Background

The spike (Phase 2d) registers Arrow batches over DuckDB's built-in **`arrow_scan`** table function (projection + filter pushdown both enabled) via a thin C++ shim that delegates to a Rust factory (replayable; exports a fresh `FFI_ArrowArrayStream` per scan; honors projection). Its one gap: `arrow_scan` advertises filter pushdown, so DuckDB removes filters from the plan and **expects the producer to apply them** (`ArrowStreamParameters.filters`); the spike's factory ignored them, so filters silently dropped — worked around with a global `SET disabled_optimizers='filter_pushdown'`. That global setting is the only thing blocking production use.

duckdb-python solves this in C++ (`PythonTableArrowArrayStreamFactory::Produce` + `filter_pushdown_visitor.cpp`): it reads `parameters.filters`, translates the `TableFilterSet` into a filter applied to the held Arrow object, and re-exports a filtered stream. This design ports that to Rust/arrow-compute.

Verified mechanics in the bundled DuckDB source:
- `arrow_scan` is registered (`arrow.cpp:333`) with `projection_pushdown=true, filter_pushdown=true, filter_prune=true, supports_pushdown_type=ArrowPushdownType`.
- `ArrowPushdownType` (`arrow.cpp:319`) refuses to push filters on view types / non-`CanPushdown` column types → DuckDB applies those above the scan. So the producer only ever receives filters on filterable scalar columns.
- `ArrowStreamParameters { ArrowProjectedColumns projected_columns; TableFilterSet *filters; }`; `ArrowProjectedColumns { unordered_map<idx_t,string> projection_map; vector<string> columns; unordered_map<idx_t,idx_t> filter_to_col; }`.
- `enum class TableFilterType { CONSTANT_COMPARISON, IS_NULL, IS_NOT_NULL, CONJUNCTION_OR, CONJUNCTION_AND, STRUCT_EXTRACT, OPTIONAL_FILTER, IN_FILTER, DYNAMIC_FILTER, EXPRESSION_FILTER, BLOOM_FILTER }`; `TableFilterSet` holds `unordered_map<idx_t, unique_ptr<TableFilter>> filters`.

## Architecture

```
Connection::register_arrow("v", batches) ──► Box<ArrowFactory>{rust_produce, rust_get_schema, batches, schema}
   └► ddb_rs_arrow_register(conn, "v", factory) [C++]: TableFunction("arrow_scan",{factory,produce,schema})->CreateView("v")

DuckDB scans the view ── per scan ──► ShimProduce(factory, ArrowStreamParameters{projected_columns, filters}) [C++]
   ├─ flatten projected_columns.columns → const char*[] names
   ├─ flatten *filters (TableFilterSet tree) → C-ABI filter buffer (recursive; Value→typed scalar)
   └─ rust_produce(factory, names, n, filter_buf, out) [Rust]:
        project batches → evaluate filter buffer to a BooleanArray mask (arrow compute)
        → filter_record_batch → export fresh FFI_ArrowArrayStream into *out
        (on any error/unhandled kind → error_stream(schema, msg): clean scan failure, never null get_next)
```

All Arrow logic stays in Rust; the C++ shim is the trampoline (it alone can read the C++ `ArrowStreamParameters`/`TableFilterSet`) + the registration call.

## Components

### 1. C++ shim (extend `crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp`)
- `ShimProduce` additionally **flattens `parameters.filters`** into a C-ABI buffer and passes it to the extended Rust `produce` callback. Flattening walks the `TableFilterSet` map (`col_idx → TableFilter`) and, per filter, recursively serializes the node by `filter_type`:
  - `CONSTANT_COMPARISON` → `{col, op (from ExpressionType), value}` where `value` is the DuckDB `Value` serialized to a tagged scalar (int/uint/float/bool/utf8/date/timestamp/decimal as supported).
  - `IS_NULL` / `IS_NOT_NULL` → `{col, kind}`.
  - `CONJUNCTION_AND` / `CONJUNCTION_OR` → `{kind, child_count, children...}`.
  - `IN_FILTER` → `{col, value_count, values...}`.
  - `OPTIONAL_FILTER` → not required for correctness: unwrap and emit its child only if the child is a handled kind (a perf win); otherwise skip it entirely (skipping is always correct).
  - Any other kind (`STRUCT_EXTRACT`, `EXPRESSION_FILTER`, `DYNAMIC_FILTER`, `BLOOM_FILTER`) → emit an `UNHANDLED{kind}` marker (Rust decides per the correctness policy).
  - Column index mapping uses `projected_columns.filter_to_col` (filter idx → table col idx) and the projection order so masks align with the projected batch columns.
- The flattened buffer is a flat byte/record layout (length-prefixed) the Rust side decodes — chosen over passing raw C++ pointers so Rust never touches C++ objects.
- `ddb_rs_arrow_register` and the lifetime/ownership rules are unchanged from the spike (view references the factory by raw pointer; caller keeps it alive via `ArrowView`).

### 2. Rust filter evaluator (`crates/duckdb/src/arrow_zerocopy/filter.rs`)
- Decode the C-ABI filter buffer into a `FilterNode` tree (`Compare{col, op, scalar}`, `IsNull{col}`, `IsNotNull{col}`, `And(Vec)`, `Or(Vec)`, `In{col, Vec<scalar>}`, `Unhandled{kind}`).
- `evaluate(node, &RecordBatch) -> Result<BooleanArray, FilterError>`: arrow compute kernels (`cmp::eq/neq/lt/lt_eq/gt/gt_eq`, `is_null`, `is_not_null`, boolean `and`/`or`, IN via `eq` reduction or set membership). Scalar→arrow comparison handles each supported `DataType`; an unsupported scalar/column type → `FilterError`.
- **Correctness policy (grounded in duckdb-python's `filter_pushdown_visitor`):** translate `CONSTANT_COMPARISON`, `IS_NULL`, `IS_NOT_NULL`, `CONJUNCTION_AND/OR`, `IN_FILTER`; treat `OPTIONAL_FILTER`/join-reduction kinds (`DYNAMIC_FILTER`, `BLOOM_FILTER`) as **non-required** (safe to skip — the join above re-checks); any other `Unhandled` kind, or a scalar/type the evaluator can't compare → return `FilterError` so `rust_produce` exports an `error_stream` (clean scan failure naming the kind), **never silently wrong**.

### 3. Rust registration + API (`crates/duckdb/src/arrow_zerocopy/mod.rs`)
- `#[repr(C)] struct ArrowFactory { produce, get_schema, batches: Vec<RecordBatch>, schema: SchemaRef }` (callback ptrs first — same ABI contract as the spike).
- `extern "C" fn rust_produce(factory, names, n, filter_buf, filter_len, out)` — project batches, decode+evaluate the filter to a mask, `filter_record_batch`, export; on error → `error_stream` (carried over from the spike's Phase 2d fix).
- `extern "C" fn rust_get_schema(...)` — unchanged from the spike (full schema; `empty()` recovery is sound).
- `pub fn Connection::register_arrow(&self, name: &str, batches: Vec<RecordBatch>) -> Result<ArrowView>`; `pub struct ArrowView` (owns the factory box; `Drop` frees it; doc: must outlive the view; `!Send`/`!Sync`).

## Error handling

- Registration failure (C++ exception) → `Error::DuckDBFailure(..)` with a message (spike's `duckdb_err`).
- Empty batches / NUL name → `duckdb_err` / `Error::NulError` (spike).
- Produce-time failure (unhandled filter kind, unsupported scalar/type, panic) → `error_stream` (non-null `get_next` yielding an Arrow error) → DuckDB scan fails cleanly with the message. `catch_unwind` backstops panics. No segfaults, no silent wrong results.

## Type coverage

- **Data:** all Arrow types the FFI C-stream + DuckDB `arrow_scan` support (type-agnostic export).
- **Filter pushdown:** bounded by `arrow_scan`'s `ArrowPushdownType` (filters only pushed for filterable scalar column types; view/unsupported types filtered above by DuckDB). The Rust evaluator supports the comparable scalar types (signed/unsigned ints, float/double, bool, Utf8/LargeUtf8, Date32, Timestamp; Decimal best-effort); a constant type it can't map → `FilterError` (clean failure).

## Thread-safety

`ArrowView` is `!Send`/`!Sync` (holds a raw pointer), consistent with the crate's `!Send` `Connection`; documented. No cross-thread sharing.

## Packaging / upstream conventions

- New branch `arrow-zerocopy` off `main`.
- Modules: `crates/duckdb/src/arrow_zerocopy/{mod.rs, filter.rs}`; C++ in `crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp` (+ `build_bundled_cc.rs` wiring); gated `#[cfg(all(feature = "bundled", feature = "vtab-arrow"))]`.
- rustfmt + clippy clean; follow CONTRIBUTING.md; a doc-comment-level example + a CHANGELOG/PR-description entry. (Bundled-only for v1; a `loadable`/linked-build story can be a follow-up.)

## Testing

- **Correctness vs the Appender** (same query, two ingestion paths) and vs known data:
  - each filter kind: `=, !=, <, <=, >, >=` (CONSTANT_COMPARISON), `IS NULL`, `IS NOT NULL`, `AND`, `OR`, `IN`.
  - across types: int/uint widths, float/double, bool, Utf8, Date/Timestamp; NULL handling.
  - projection subset; `SELECT *`.
  - **self-join** (replayability); **inner join with a filter** (validates dynamic/bloom-skip correctness — result equals the Appender's).
  - **negative test:** a deliberately unsupported pushed filter (or a forced `Unhandled`) surfaces as a clean query error — not a wrong result, not a crash.
- Benchmark: reuse the spike's copy-dominated example; confirm pushdown perf (filtered + projected).

## Staging (always-correct intermediates — for the plan)

1. **Registration + projection + `CONSTANT_COMPARISON`**, erroring (`error_stream`) on every other pushed filter kind. Correct, end-to-end, the FFI/flatten/evaluate skeleton in place.
2. **Expand kinds:** `IS_NULL`/`IS_NOT_NULL`, `CONJUNCTION_AND/OR`, `IN_FILTER`; skip `OPTIONAL_FILTER`/`DYNAMIC`/`BLOOM`.
3. **Type breadth + null semantics** across the supported scalar types.
4. **API polish, docs, benchmark, conventions** (rustfmt/clippy/CHANGELOG).

Each stage is correct (errors loud rather than wrong); later stages widen coverage.

## Risks

- **Filter-translation correctness** is the central risk. Mitigation: ground semantics in duckdb-python's `filter_pushdown_visitor`; fail-loud on anything outside the handled set; test join queries (the dynamic/bloom-skip path) against the Appender oracle.
- **C++↔Rust filter-buffer ABI** (the flattened layout): a focused, versioned encoding decoded in one place; round-trip tested.
- **DuckDB-internals coupling** (`TableFilterSet`/`Value`/`ExpressionType` in the shim): version-pinned to the bundled amalgamation (same constraint duckdb-python lives with); the C++ surface is confined to the shim.
- **`Value` scalar extraction breadth** in C++ — start with the supported scalar types; unhandled → `Unhandled`/error, not wrong.

## Decision log

- Filter strategy: **full pushdown (duckdb-python parity)** — translate `TableFilterSet`, pre-filter Arrow in the factory.
- Correctness: translate-known / skip-optional+join-reduction / **fail-loud-on-unknown**, grounded in duckdb-python's `filter_pushdown_visitor`.
- API: `Connection::register_arrow(name, Vec<RecordBatch>) -> ArrowView`; `ArrowView` `!Send`, owns data, outlives the view.
- Branch: `arrow-zerocopy` off duckdb-rs `main`; bundled-only; `vtab-arrow` feature.
- Logic split: all Arrow + filter evaluation in Rust; C++ is trampoline + `TableFilterSet` flattening + registration.

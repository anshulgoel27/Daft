# duckdb-rs Zero-Copy Arrow Scan Shim — Spike Design

**Date:** 2026-06-26
**Status:** Design approved; ready for implementation plan
**Related:** [DuckDB POC performance follow-ups](2026-06-25-duckdb-poc-performance-followups.md) (Phases 1–2c) · local checkout `/Volumes/Work/Code/duckdb-rs` (v1.10504.0)

## Goal

Prove that **correct, production-shaped zero-copy Arrow ingestion** is achievable from Rust by registering DuckDB's native `arrow_scan` table function with a custom, Rust-driven factory — via a thin C++ shim added to `libduckdb-sys`. "Correct" means the two things the deprecated `duckdb_arrow_scan` path (Phase 2c) could not do:

1. **Replayable** — the registered view can be scanned more than once (self-joins, repeated queries), because each scan produces a *fresh* Arrow stream from held data.
2. **Projection pushdown** — only the columns DuckDB asks for are streamed.

Then **measure** the cold per-task cost on a copy-dominated workload vs the Appender, to decide whether to pursue a fork/upstream (Option 2 full) later.

This is a **spike** in the local `duckdb-rs` checkout, not an upstream PR.

## Non-Goals

- **Filter pushdown into the scan** (translating DuckDB's `TableFilterSet` to apply to Arrow batches). Out of scope — see "Filter handling" for the lean alternative. This is the bulk of duckdb-python's `filter_pushdown_visitor.cpp`; deferred.
- **Full Arrow type coverage.** Spike supports primitives + `Utf8`/`LargeUtf8` (what the POC query needs); other types error cleanly.
- **A general/ergonomic public API** or upstream-quality tests/docs. Minimal surface to run the validation.
- **Changing Daft.** The benchmark/correctness harness lives in the `duckdb-rs` checkout (a Rust example + tests), not in Daft.

## Background (why a C++ shim is required)

Established in Phases 2b–2c (see the follow-ups doc):

- The zero-copy mechanism is DuckDB's C++ `ArrowTableFunction` (DuckDB vectors *alias* Arrow buffers in place). The **C API exposes no way** to do this: a VTab makes you fill DuckDB vectors (a copy), and `duckdb_vector_*` cannot wrap an external buffer.
- The deprecated C API `duckdb_arrow_scan` *is* zero-copy but hardcodes a factory (`FactoryGetNext` in `arrow-c.cpp`) that **ignores `ArrowStreamParameters`** (projection + filters) and **reuses a one-shot stream** → dropped filters + not replayable.
- duckdb-python works because its C++ factory (`PythonTableArrowArrayStreamFactory::Produce`) reads `ArrowStreamParameters`, applies projection, and re-exports a **fresh** stream each call from a held pyarrow object.

So a correct factory must be C++ (it returns a C++ `unique_ptr<ArrowArrayStreamWrapper>` and consumes a C++ `ArrowStreamParameters&`). The minimal way to get it from Rust is a thin C++ trampoline that delegates all Arrow logic back to Rust.

Relevant C++ declarations (`duckdb/function/table/arrow.hpp`, confirmed in the bundled tarball):

```cpp
struct ArrowProjectedColumns {
    unordered_map<idx_t, string> projection_map;
    vector<string> columns;             // projected column names, in order
    unordered_map<idx_t, idx_t> filter_to_col;
};
struct ArrowStreamParameters {
    ArrowProjectedColumns projected_columns;
    TableFilterSet *filters;
};
typedef unique_ptr<ArrowArrayStreamWrapper> (*stream_factory_produce_t)(uintptr_t ptr, ArrowStreamParameters &params);
typedef void (*stream_factory_get_schema_t)(ArrowArrayStream *ptr, ArrowSchema &schema);
```

## Architecture

All Arrow logic stays in Rust; the C++ shim is a trampoline + registration call.

```
Rust: register_arrow_zerocopy(conn, "daft_src_7", batches)
  │  build Box<ArrowFactory> (first fields = rust_produce, rust_get_schema fn ptrs)
  ▼
C++ ddb_rs_arrow_register(duckdb_connection, name, factory_ptr) -> int
  │  reinterpret conn as duckdb::Connection*
  │  Connection::TableFunction("arrow_scan",
  │      { Value::POINTER(factory_ptr), Value::POINTER(&ShimProduce), Value::POINTER(&ShimGetSchema) })
  │    ->CreateView(name, /*replace*/true, /*temporary*/true)
  ▼
DuckDB executes a query that scans the view  ── per scan ──►  ShimProduce(factory_ptr, params)
                                                                │ extract params.projected_columns.columns
                                                                │   → const char* const* names, size_t n
                                                                │ call rust_produce(factory_ptr, names, n)
                                                                │   → ArrowArrayStream*  (fresh, projected)
                                                                │ wrap in ArrowArrayStreamWrapper (owns it)
                                                                └ ShimGetSchema → rust_get_schema(factory_ptr, ArrowSchema*)
```

### Components

**1. C++ shim — `crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp` (~50 lines)**
- The Rust callback pointers live in **factory state** (not globals), so the C++ static produce/schema functions (which must match the C++ typedefs and take no extra args) reach them through `factory_ptr`. The shim declares a prefix struct matching the Rust factory's first fields:
  ```cpp
  typedef ArrowArrayStream *(*RustProduceFn)(void *factory, const char *const *names, size_t n);
  typedef void (*RustGetSchemaFn)(void *factory, struct ArrowSchema *out);
  struct RustCallbacks { RustProduceFn produce; RustGetSchemaFn get_schema; }; // == ArrowFactory prefix
  ```
- `extern "C" int ddb_rs_arrow_register(duckdb_connection conn, const char *name, void *factory_ptr)` — reinterpret `conn` as `duckdb::Connection*` and run `TableFunction("arrow_scan", { Value::POINTER(factory_ptr), Value::POINTER(&ShimProduce), Value::POINTER(&ShimGetSchema) })->CreateView(name, true, true)`. Returns 0 on success, 1 on C++ exception (caught).
- `static unique_ptr<ArrowArrayStreamWrapper> ShimProduce(uintptr_t factory_ptr, ArrowStreamParameters &params)`: read the callback via `reinterpret_cast<RustCallbacks*>(factory_ptr)->produce`; copy `params.projected_columns.columns` into a `vector<const char*>` (via `c_str()`); call `produce((void*)factory_ptr, names.data(), names.size())` → `ArrowArrayStream*`; then `auto w = make_uniq<ArrowArrayStreamWrapper>(); w->arrow_array_stream = *stream; stream->release = nullptr;` (transfer ownership to the wrapper, like duckdb-python) and return `w`. Ignores `params.filters` (see Filter handling).
- `static void ShimGetSchema(ArrowArrayStream *factory_ptr, ArrowSchema &schema)`: read `reinterpret_cast<RustCallbacks*>(factory_ptr)->get_schema` and call it with `&schema`.

**2. Rust FFI + callbacks — `crates/libduckdb-sys` (or `duckdb` crate)**
- `extern "C" { fn ddb_rs_arrow_register(conn: duckdb_connection, name: *const c_char, factory: *mut c_void) -> c_int; }`
- `type RustProduceFn = extern "C" fn(factory: *mut c_void, names: *const *const c_char, n: usize) -> *mut FFI_ArrowArrayStream;`
- `type RustGetSchemaFn = extern "C" fn(factory: *mut c_void, out: *mut FFI_ArrowSchema);`
- `extern "C" fn rust_produce(...)`: cast `factory` → `&ArrowFactory`; resolve `names` against the held schema → column indices (all columns if `n == 0`); `RecordBatch::project(indices)` each held batch (zero-copy); build a `RecordBatchIterator` → `FFI_ArrowArrayStream::new(reader)`; `Box::into_raw`/leak the stream so the C++ wrapper owns it (it nulled release on its copy; the wrapper releases on destruct). Return the raw pointer.
- `extern "C" fn rust_get_schema(...)`: export the held schema to the out `FFI_ArrowSchema`.

**3. Rust API (spike surface) — `crates/duckdb/src/arrow_zerocopy.rs`**
- `#[repr(C)] pub struct ArrowFactory { produce: RustProduceFn, get_schema: RustGetSchemaFn, batches: Vec<RecordBatch>, schema: SchemaRef }` — `#[repr(C)]` with the two fn pointers first so the C++ `RustCallbacks` prefix matches the layout. `register_arrow_zerocopy` sets `produce = rust_produce`, `get_schema = rust_get_schema`.
- `pub fn register_arrow_zerocopy(conn: &Connection, name: &str, batches: Vec<RecordBatch>) -> Result<ArrowRegistration>` — boxes an `ArrowFactory`, calls `ddb_rs_arrow_register`, returns a handle owning the box. **The handle must outlive the view** (the view holds the factory pointer); dropping it after the connection/queries are done frees the batches.
- The caller (validation harness) opens a dedicated connection, runs `SET disabled_optimizers='filter_pushdown'` (see below), registers, queries, then drops the handle.

**4. Build wiring — `crates/libduckdb-sys/build_bundled_cc.rs`**
- Add the shim to the same `cc::Build` that compiles the amalgamation: `cfg.file("src/arrow_zerocopy_shim.cpp")` after the manifest `cpp_files` are added (so it shares the include dirs `{out_dir}/duckdb/...` and `-std=c++11`/`/EHsc`). It compiles against `duckdb.hpp` and links with the amalgamation in one unit set.

## Data flow & lifetime

- **Register:** the boxed `ArrowFactory` (holding Arc'd batches) is handed to DuckDB by raw pointer; DuckDB stores it in the view's bind data.
- **Each scan:** DuckDB calls `ShimProduce` → `rust_produce` builds a **fresh** `FFI_ArrowArrayStream` over (projected) clones of the held batches (Arc bumps, zero-copy). Fresh-per-call = replayable.
- **Stream ownership:** `rust_produce` leaks the `FFI_ArrowArrayStream` to the C++ side; `ShimProduce` moves it into the `ArrowArrayStreamWrapper` and nulls the source's release; the wrapper releases it when the scan completes. No double-free, no per-scan leak.
- **Factory lifetime:** the `ArrowRegistration` handle owns the box; it must be dropped **after** the connection (or at least after all queries), since the view references it by pointer. Same rule the Phase 2c spike already validated for the stream.

## Filter handling (Lean — approved)

The built-in `arrow_scan` advertises filter pushdown, so DuckDB removes filters from the plan expecting the factory to apply them. The factory ignores `params.filters`, so we must prevent the drop: the validation harness runs **`SET disabled_optimizers='filter_pushdown'`** on the dedicated per-task connection. DuckDB then applies filters *above* the scan → correct results; projection pushdown still works. Cost: the scan reads all rows then filters (no row-skipping) — acceptable for the spike and for copy-dominated workloads. Upgrading to in-scan filter pushdown (translate `TableFilterSet`) is the "Option 2 full" follow-up.

## Error handling

- C++ entrypoint wraps the `TableFunction/CreateView` in try/catch → returns non-zero; Rust maps to `Error`.
- `rust_produce`/`rust_get_schema` must not unwind across the FFI boundary: wrap bodies in `catch_unwind`; on error, return a null stream / an empty (released) schema so the C++ side fails the scan rather than UB.
- Unsupported Arrow type during projection/export → return null stream (scan fails with a DuckDB error) rather than panic.

## Validation plan

**Correctness** (`crates/duckdb` test): zero-copy `arrow_scan` view result == Appender result, on
1. **filter + group-by aggregate** (the POC shape) — proves pushdown-disabled filtering is correct; and
2. **self-join** of the registered view (`SELECT ... FROM v a JOIN v b ON ...`) — proves **replayability** (the case the deprecated one-shot path cannot do).

**Performance** (`crates/duckdb/examples/arrow_zerocopy_bench.rs`): cold per-task (open + register + query) for zero-copy vs Appender on a **copy-dominated** workload — a wide table (several columns) with light compute and a projection that selects a subset (so both *skipped copy* and *projection pushdown* show). Report ms/run at a few sizes. Expectation per Phase 2c: modest on compute-heavy, larger as ingestion dominates.

## Risks

- **C++ ABI / version pinning:** the shim uses internal C++ headers (`Connection`, `ArrowStreamParameters`, `ArrowArrayStreamWrapper`); it is pinned to the bundled DuckDB version. Acceptable for a spike (same constraint duckdb-python lives with). `ArrowArrayStreamWrapper`'s definition must be confirmed includable from `duckdb.hpp`; if it lives in a non-umbrella header, include it directly.
- **Callback-pointer plumbing:** storing the Rust fn pointers in factory state (preferred) vs file-statics; the plan should implement factory-state to stay reentrant.
- **Unwind safety** across FFI (mitigated by `catch_unwind`).
- **Win may be modest** — this spike measures it; a negative/modest result is itself a valid outcome that closes Option 2.

## Decision log

- **Deliverable:** spike in the local `duckdb-rs` checkout (not fork/upstream yet).
- **Filter handling:** Lean (disable filter pushdown; filter above) — minimal C++; in-scan pushdown deferred.
- **Types:** primitives + Utf8/LargeUtf8; others error.
- **Logic split:** all Arrow logic in Rust; C++ is a trampoline + registration.

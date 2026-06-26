# duckdb-rs Zero-Copy Arrow Scan Shim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a thin C++ shim to the local `duckdb-rs` checkout that registers DuckDB's native `arrow_scan` table function with a Rust-driven, replayable, projection-honoring factory — giving correct zero-copy Arrow ingestion from Rust — then validate correctness and benchmark it vs the Appender.

**Architecture:** A C++ trampoline (`arrow_zerocopy_shim.cpp`) compiled into `libduckdb-sys` exposes one `extern "C"` entrypoint that calls `Connection::TableFunction("arrow_scan", {...})->CreateView(...)`. Its `produce`/`get_schema` statics read the Rust callback pointers from the factory's first fields and call back into Rust, where all Arrow logic lives (hold batches, project, export a fresh `FFI_ArrowArrayStream` per scan). Filters are handled "lean": the caller disables filter pushdown so DuckDB applies filters above the scan.

**Tech Stack:** Rust + C++ (FFI), DuckDB bundled C++ amalgamation (v1.10504.0), arrow-rs v58 (`ffi`, `ffi_stream`), `cc` build.

## Global Constraints

- All work is in the **local duckdb-rs checkout** `/Volumes/Work/Code/duckdb-rs` (separate git repo from Daft). Create a feature branch there first; commit there. The Daft repo holds only this plan/spec.
- **duckdb-rs version:** `1.10504.0`; build with the `bundled` feature (compiles the amalgamation via `build_bundled_cc.rs`).
- **arrow-rs v58** (`crates/duckdb` workspace dep `arrow = "58"`, features `prettyprint`, `ffi`). Use `arrow::ffi_stream::FFI_ArrowArrayStream`, `arrow::ffi::FFI_ArrowSchema`, `arrow::record_batch::{RecordBatch, RecordBatchIterator}`, `arrow::datatypes::SchemaRef`.
- **All Arrow logic in Rust; C++ is a trampoline + registration only.**
- **Factory layout:** `#[repr(C)] struct ArrowFactory` whose **first two fields are the Rust callback fn pointers** (`produce`, then `get_schema`), so the C++ `RustCallbacks` prefix struct aliases them via the factory pointer. No global/static callback pointers.
- **Lean filter handling:** the registering connection runs `SET disabled_optimizers='filter_pushdown'`; filters are applied above the scan (correct, no row-skipping). Do NOT translate `TableFilterSet`.
- **Type scope:** primitives + `Utf8`/`LargeUtf8`. Other types may error (projection/export returns an empty stream → DuckDB fails the scan); do not add broad type handling.
- **Lifetime rule:** the `ArrowRegistration` handle owns the factory box and frees it on drop; it MUST outlive the view (drop the connection / finish all queries before dropping the handle).
- **Feature gating:** the Rust module is `#[cfg(all(feature = "bundled", feature = "vtab-arrow"))]` (needs the shim symbol from the bundled build + arrow). Tests/bench run with `--features "bundled appender-arrow"` (`appender-arrow` ⊃ `vtab-arrow`).
- **`duckdb_connection` → `duckdb::Connection*`:** the shim does `reinterpret_cast<duckdb::Connection *>(conn)`, mirroring the in-tree `arrow-c.cpp` `Ingest`.
- **Commits:** conventional-commit messages ending with the line `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## File Structure

- **Create** `crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp` — C++ trampoline + `ddb_rs_arrow_register` entrypoint.
- **Modify** `crates/libduckdb-sys/build_bundled_cc.rs` — compile the shim into the same `cc::Build`.
- **Create** `crates/duckdb/src/arrow_zerocopy.rs` — `ArrowFactory`, `rust_produce`/`rust_get_schema`, the `ddb_rs_arrow_register` extern, `Connection::register_arrow_zerocopy`, `ArrowRegistration`, and `#[cfg(test)] mod test`.
- **Modify** `crates/duckdb/src/lib.rs` — add `mod arrow_zerocopy;` (gated) and re-export `ArrowRegistration`.
- **Create** `crates/duckdb/examples/arrow_zerocopy_bench.rs` — copy-dominated perf benchmark vs Appender.

---

### Task 1: Zero-copy registration vertical slice (shim + build + Rust + round-trip/projection tests)

This is the irreducible FFI slice: the C++ shim cannot be tested without the Rust side and vice versa, so it ships together with the basic correctness gate.

**Files:**
- Create: `crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp`
- Modify: `crates/libduckdb-sys/build_bundled_cc.rs` (the `cfg` build block, around lines 100–108)
- Create: `crates/duckdb/src/arrow_zerocopy.rs`
- Modify: `crates/duckdb/src/lib.rs`

**Interfaces:**
- Produces (Rust, used by Tasks 2–3):
  - `Connection::register_arrow_zerocopy(&self, name: &str, batches: Vec<arrow::record_batch::RecordBatch>) -> duckdb::Result<ArrowRegistration>`
  - `pub struct ArrowRegistration` (opaque; owns the factory box; `Drop` frees it). Re-exported from the crate root.
- Produces (C ABI, used by the build): `extern "C" int ddb_rs_arrow_register(duckdb_connection conn, const char *name, void *factory_ptr)` returning 0 on success, 1 on C++ exception.

- [ ] **Step 1: Create a feature branch in the duckdb-rs checkout**

```bash
git -C /Volumes/Work/Code/duckdb-rs checkout -b zerocopy-arrow-shim-spike
git -C /Volumes/Work/Code/duckdb-rs status
```
Expected: switched to a new branch, clean tree.

- [ ] **Step 2: Write the C++ shim**

Create `crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp`:

```cpp
// Zero-copy Arrow registration shim.
//
// Registers DuckDB's native `arrow_scan` table function with a Rust-driven, replayable,
// projection-honoring factory. All Arrow logic lives in Rust; this file is a trampoline
// (translate the C++ ArrowStreamParameters that Rust cannot read into plain C) plus the
// TableFunction/CreateView registration call.
//
// Ownership: `factory_ptr` is a Rust Box<ArrowFactory>; its first two fields are the Rust
// callback pointers (matched by RustCallbacks below, #[repr(C)] on the Rust side). DuckDB
// stores `factory_ptr` in the view's bind data and calls ShimProduce per scan and
// ShimGetSchema at bind time. The view references the factory by pointer; the Rust side keeps
// the box alive until the view is gone (see ArrowRegistration).

#include <cstddef>
#include <cstdint>
#include <vector>

#include "duckdb.h"                                   // duckdb_connection
#include "duckdb.hpp"                                 // duckdb::Connection, Value, Relation
#include "duckdb/function/table/arrow.hpp"           // ArrowStreamParameters, stream factory typedefs
#include "duckdb/common/arrow/arrow_wrapper.hpp"     // ArrowArrayStreamWrapper

using duckdb::ArrowArrayStreamWrapper;
using duckdb::ArrowStreamParameters;

// Must match the first two fields of the Rust #[repr(C)] ArrowFactory.
typedef void (*RustProduceFn)(void *factory, const char *const *names, size_t n, ArrowArrayStream *out);
typedef void (*RustGetSchemaFn)(void *factory, ArrowSchema *out);
struct RustCallbacks {
    RustProduceFn produce;
    RustGetSchemaFn get_schema;
};

// Produce a fresh stream for one scan, honoring projection. Called by DuckDB possibly many
// times (self-joins, repeated scans) — replayability comes from Rust building a new stream
// from the held batches each call.
static duckdb::unique_ptr<ArrowArrayStreamWrapper> ShimProduce(uintptr_t factory_ptr,
                                                               ArrowStreamParameters &params) {
    auto *cbs = reinterpret_cast<RustCallbacks *>(factory_ptr);
    std::vector<const char *> names;
    names.reserve(params.projected_columns.columns.size());
    for (auto &col : params.projected_columns.columns) {
        names.push_back(col.c_str());
    }
    auto wrapper = duckdb::make_uniq<ArrowArrayStreamWrapper>();
    // Rust fills wrapper->arrow_array_stream in place; the wrapper owns + releases it.
    cbs->produce(reinterpret_cast<void *>(factory_ptr), names.data(), names.size(),
                 &wrapper->arrow_array_stream);
    // params.filters intentionally ignored (lean handling; pushdown disabled by the caller).
    return wrapper;
}

// DuckDB passes the same registered factory pointer here, typed as ArrowArrayStream* per the
// typedef; reinterpret it back to the factory.
static void ShimGetSchema(ArrowArrayStream *factory_ptr, ArrowSchema &schema) {
    auto *cbs = reinterpret_cast<RustCallbacks *>(factory_ptr);
    cbs->get_schema(reinterpret_cast<void *>(factory_ptr), &schema);
}

extern "C" int ddb_rs_arrow_register(duckdb_connection conn, const char *name, void *factory_ptr) {
    try {
        auto *connection = reinterpret_cast<duckdb::Connection *>(conn);
        duckdb::vector<duckdb::Value> values;
        values.push_back(duckdb::Value::POINTER(reinterpret_cast<uintptr_t>(factory_ptr)));
        values.push_back(duckdb::Value::POINTER(reinterpret_cast<uintptr_t>(&ShimProduce)));
        values.push_back(duckdb::Value::POINTER(reinterpret_cast<uintptr_t>(&ShimGetSchema)));
        connection->TableFunction("arrow_scan", values)->CreateView(name, /*replace*/ true, /*temporary*/ true);
        return 0;
    } catch (...) {
        return 1;
    }
}
```

- [ ] **Step 3: Wire the shim into the cc build**

In `crates/libduckdb-sys/build_bundled_cc.rs`, find the block that adds the amalgamation `.cpp` files (the `for f in cpp_files_vec ... { cfg.file(f); }` loop, ~line 104–107). Immediately AFTER that loop, add:

```rust
    // Zero-copy Arrow registration shim (compiled with the amalgamation so it links against the
    // DuckDB C++ API and shares its include dirs).
    cfg.file("src/arrow_zerocopy_shim.cpp");
    println!("cargo:rerun-if-changed=src/arrow_zerocopy_shim.cpp");
```

- [ ] **Step 4: Write the failing round-trip + projection tests**

Create `crates/duckdb/src/arrow_zerocopy.rs` with the module skeleton + tests (implementation comes in Step 6, so this must currently fail to compile/link, which is the failing state):

```rust
//! Zero-copy Arrow registration via DuckDB's native `arrow_scan` table function.
//!
//! Registers a set of held Arrow `RecordBatch`es as a DuckDB view whose scans reference the Arrow
//! buffers in place (no copy into DuckDB native storage), unlike the `Appender`. The factory is
//! replayable (a fresh stream per scan, so self-joins work) and honors projection pushdown.
//!
//! Filters: DuckDB's `arrow_scan` advertises filter pushdown, so the caller must run
//! `SET disabled_optimizers='filter_pushdown'` on the connection; filters are then applied above
//! the scan. This module does NOT translate `TableFilterSet`.

use std::ffi::{CString, c_char, c_void};
use std::os::raw::c_int;

use arrow::datatypes::SchemaRef;
use arrow::ffi::FFI_ArrowSchema;
use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow::record_batch::{RecordBatch, RecordBatchIterator};

use crate::error::Error;
use crate::inner_connection::InnerConnection;
use crate::{Connection, Result};

type ProduceFn = extern "C" fn(*mut c_void, *const *const c_char, usize, *mut FFI_ArrowArrayStream);
type GetSchemaFn = extern "C" fn(*mut c_void, *mut FFI_ArrowSchema);

// SAFETY: layout matched by the C++ `RustCallbacks` prefix; the two fn pointers MUST stay first.
#[repr(C)]
struct ArrowFactory {
    produce: ProduceFn,
    get_schema: GetSchemaFn,
    batches: Vec<RecordBatch>,
    schema: SchemaRef,
}

unsafe extern "C" {
    fn ddb_rs_arrow_register(conn: crate::ffi::duckdb_connection, name: *const c_char, factory: *mut c_void) -> c_int;
}

#[cfg(test)]
mod test {
    use super::*;
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn sample() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("region", DataType::Utf8, true),
        ]));
        let k = Int64Array::from(vec![1, 2, 3, 4]);
        let region = StringArray::from(vec![Some("us"), Some("eu"), None, Some("us")]);
        RecordBatch::try_new(schema, vec![Arc::new(k), Arc::new(region)]).unwrap()
    }

    #[test]
    fn round_trip_select_star() {
        let conn = Connection::open_in_memory().unwrap();
        let batch = sample();
        let reg = conn.register_arrow_zerocopy("v", vec![batch.clone()]).unwrap();
        // count
        let n: i64 = conn
            .query_row("SELECT count(*) FROM v", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 4);
        // sum of k and a string round-trip
        let sum_k: i64 = conn
            .query_row("SELECT sum(k) FROM v", [], |r| r.get(0))
            .unwrap();
        assert_eq!(sum_k, 10);
        let region2: String = conn
            .query_row("SELECT region FROM v WHERE k = 2", [], |r| r.get(0))
            .unwrap();
        assert_eq!(region2, "eu");
        drop(reg); // handle dropped after queries (lifetime rule)
    }

    #[test]
    fn projection_subset() {
        let conn = Connection::open_in_memory().unwrap();
        let reg = conn.register_arrow_zerocopy("v", vec![sample()]).unwrap();
        // Select only `region` — DuckDB pushes a single-column projection into the factory.
        let count_eu: i64 = conn
            .query_row("SELECT count(*) FROM (SELECT region FROM v) WHERE region = 'us'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count_eu, 2);
        drop(reg);
    }
}
```

- [ ] **Step 5: Run the tests to verify they fail**

Run: `cargo test -p duckdb --features "bundled appender-arrow" arrow_zerocopy 2>&1 | tail -20`
Expected: compile error — `register_arrow_zerocopy` not found on `Connection` and `mod arrow_zerocopy` not declared. (This is the failing state.)

- [ ] **Step 6: Implement the Rust callbacks + register API**

Append to `crates/duckdb/src/arrow_zerocopy.rs` (above the `#[cfg(test)]` module):

```rust
/// A live zero-copy Arrow registration. Owns the factory box (held batches + schema). It MUST
/// outlive the registered view — drop it only after the connection is closed or all queries that
/// scan the view have completed, since the view references the factory by raw pointer.
pub struct ArrowRegistration {
    factory: *mut ArrowFactory,
}

impl Drop for ArrowRegistration {
    fn drop(&mut self) {
        // SAFETY: `factory` came from Box::into_raw in register_arrow_zerocopy and is freed once.
        unsafe { drop(Box::from_raw(self.factory)) };
    }
}

extern "C" fn rust_produce(
    factory: *mut c_void,
    names: *const *const c_char,
    n: usize,
    out: *mut FFI_ArrowArrayStream,
) {
    let result = std::panic::catch_unwind(|| unsafe {
        let f = &*(factory as *const ArrowFactory);
        // Resolve projected column names -> indices (n == 0 means "all columns").
        let indices: Vec<usize> = if n == 0 {
            (0..f.schema.fields().len()).collect()
        } else {
            let name_ptrs = std::slice::from_raw_parts(names, n);
            name_ptrs
                .iter()
                .map(|&p| {
                    let name = std::ffi::CStr::from_ptr(p).to_str().expect("utf8 column name");
                    f.schema.index_of(name).expect("projected column exists")
                })
                .collect()
        };
        let projected: Vec<RecordBatch> = f
            .batches
            .iter()
            .map(|b| b.project(&indices).expect("project batch"))
            .collect();
        let proj_schema: SchemaRef = match projected.first() {
            Some(b) => b.schema(),
            None => std::sync::Arc::new(f.schema.project(&indices).expect("project schema")),
        };
        let reader = RecordBatchIterator::new(
            projected.into_iter().map(Ok::<RecordBatch, arrow::error::ArrowError>),
            proj_schema,
        );
        let stream = FFI_ArrowArrayStream::new(Box::new(reader));
        std::ptr::write(out, stream);
    });
    if result.is_err() {
        // Leave an empty (released) stream so DuckDB fails the scan instead of reading garbage.
        unsafe { std::ptr::write(out, FFI_ArrowArrayStream::empty()) };
    }
}

extern "C" fn rust_get_schema(factory: *mut c_void, out: *mut FFI_ArrowSchema) {
    let result = std::panic::catch_unwind(|| unsafe {
        let f = &*(factory as *const ArrowFactory);
        let schema = FFI_ArrowSchema::try_from(f.schema.as_ref()).expect("export schema");
        std::ptr::write(out, schema);
    });
    if result.is_err() {
        unsafe { std::ptr::write(out, FFI_ArrowSchema::empty()) };
    }
}

/// Build a generic DuckDB-failure error with a message (the crate's `Error` has no plain
/// string variant; this mirrors `error.rs`'s `Error::DuckDBFailure(ffi::Error::new(code), msg)`).
fn duckdb_err(msg: impl Into<String>) -> Error {
    Error::DuckDBFailure(
        crate::ffi::Error::new(crate::ffi::duckdb_state_DuckDBError),
        Some(msg.into()),
    )
}

impl Connection {
    /// Register `batches` as a zero-copy Arrow view named `name`.
    ///
    /// The view scans the Arrow buffers in place (no copy into DuckDB storage). The returned
    /// [`ArrowRegistration`] owns the data and MUST outlive the view (drop it after the connection
    /// or after all queries scanning the view). Run `SET disabled_optimizers='filter_pushdown'`
    /// before querying if the query filters the view (see module docs).
    pub fn register_arrow_zerocopy(
        &self,
        name: &str,
        batches: Vec<RecordBatch>,
    ) -> Result<ArrowRegistration> {
        // Fail (without allocating the factory) before Box::into_raw so nothing can leak.
        let cname = CString::new(name).map_err(Error::NulError)?;
        let schema = batches
            .first()
            .map(|b| b.schema())
            .ok_or_else(|| duckdb_err("register_arrow_zerocopy: empty batches"))?;
        let factory_ptr = Box::into_raw(Box::new(ArrowFactory {
            produce: rust_produce,
            get_schema: rust_get_schema,
            batches,
            schema,
        }));
        match self
            .db
            .borrow_mut()
            .register_arrow_zerocopy(cname.as_ptr(), factory_ptr.cast::<c_void>())
        {
            Ok(()) => Ok(ArrowRegistration { factory: factory_ptr }),
            Err(e) => {
                // SAFETY: registration failed, reclaim the box we just leaked.
                unsafe { drop(Box::from_raw(factory_ptr)) };
                Err(e)
            }
        }
    }
}

impl InnerConnection {
    fn register_arrow_zerocopy(&mut self, name: *const c_char, factory: *mut c_void) -> Result<()> {
        // SAFETY: `self.con` is a live duckdb_connection; the shim returns 0 on success.
        let rc = unsafe { ddb_rs_arrow_register(self.con, name, factory) };
        if rc != 0 {
            return Err(duckdb_err(
                "ddb_rs_arrow_register failed (C++ exception in arrow_scan/CreateView)",
            ));
        }
        Ok(())
    }
}
```

Then wire the module in `crates/duckdb/src/lib.rs` — add near the other `mod` declarations:

```rust
#[cfg(all(feature = "bundled", feature = "vtab-arrow"))]
mod arrow_zerocopy;
#[cfg(all(feature = "bundled", feature = "vtab-arrow"))]
pub use arrow_zerocopy::ArrowRegistration;
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `cargo test -p duckdb --features "bundled appender-arrow" arrow_zerocopy 2>&1 | tail -20`
Expected: `round_trip_select_star` and `projection_subset` PASS. (First build recompiles the amalgamation + shim; allow several minutes.)

- [ ] **Step 8: Commit**

```bash
cd /Volumes/Work/Code/duckdb-rs
git add crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp crates/libduckdb-sys/build_bundled_cc.rs crates/duckdb/src/arrow_zerocopy.rs crates/duckdb/src/lib.rs
git commit -m "feat(arrow): zero-copy arrow_scan registration via C++ shim

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Replayability (self-join) + lean filter correctness

**Files:**
- Modify: `crates/duckdb/src/arrow_zerocopy.rs` (extend `#[cfg(test)] mod test`)

**Interfaces:**
- Consumes: `Connection::register_arrow_zerocopy`, `ArrowRegistration` (Task 1).

- [ ] **Step 1: Write the failing self-join test**

Add to `mod test` in `crates/duckdb/src/arrow_zerocopy.rs`:

```rust
    #[test]
    fn self_join_is_replayable() {
        // A self-join scans the view twice → the factory's produce is called more than once.
        // Each call must yield fresh data (the one-shot deprecated path returns empty on the 2nd).
        let conn = Connection::open_in_memory().unwrap();
        let reg = conn.register_arrow_zerocopy("v", vec![sample()]).unwrap();
        let pairs: i64 = conn
            .query_row(
                "SELECT count(*) FROM v a JOIN v b ON a.k = b.k",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(pairs, 4); // each of 4 distinct k matches itself exactly once
        drop(reg);
    }
```

- [ ] **Step 2: Run to verify it passes (replayability already implemented in Task 1)**

Run: `cargo test -p duckdb --features "bundled appender-arrow" arrow_zerocopy::test::self_join_is_replayable 2>&1 | tail -15`
Expected: PASS. If it returns 0 rows / fails, the factory is not producing fresh streams — stop and fix `rust_produce` (it must build a new `RecordBatchIterator` each call; it does, since it reads `f.batches` fresh).

- [ ] **Step 3: Write the failing filter-correctness test**

Add to `mod test`:

```rust
    #[test]
    fn filter_correct_with_pushdown_disabled() {
        // DuckDB's arrow_scan advertises filter pushdown; without disabling it the factory (which
        // ignores filters) would return unfiltered rows. With it disabled, DuckDB filters above
        // the scan → correct results.
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("SET disabled_optimizers='filter_pushdown'").unwrap();
        let reg = conn.register_arrow_zerocopy("v", vec![sample()]).unwrap();
        // k > 2 keeps k in {3,4} → sum = 7
        let sum_hi: i64 = conn
            .query_row("SELECT sum(k) FROM v WHERE k > 2", [], |r| r.get(0))
            .unwrap();
        assert_eq!(sum_hi, 7);
        drop(reg);
    }

    #[test]
    fn filter_dropped_without_disabling_pushdown() {
        // Documents the bug the lean workaround addresses: with pushdown ON, the filter is dropped
        // and the sum is the UNFILTERED total (10), not 7. This asserts the known-bad behavior so a
        // future DuckDB fix (filter honored) makes this test fail loudly and prompt removal.
        let conn = Connection::open_in_memory().unwrap();
        let reg = conn.register_arrow_zerocopy("v", vec![sample()]).unwrap();
        let sum_hi: i64 = conn
            .query_row("SELECT sum(k) FROM v WHERE k > 2", [], |r| r.get(0))
            .unwrap();
        assert_eq!(sum_hi, 10, "expected the known filter-drop bug (unfiltered total)");
        drop(reg);
    }
```

- [ ] **Step 4: Run to verify both pass**

Run: `cargo test -p duckdb --features "bundled appender-arrow" arrow_zerocopy 2>&1 | tail -15`
Expected: all `arrow_zerocopy::test::*` PASS (4 from Task 1+2 plus the 2 new). If `filter_dropped_without_disabling_pushdown` does NOT see 10, the bundled DuckDB may have changed behavior — note it and adjust the assertion/comment.

- [ ] **Step 5: Commit**

```bash
cd /Volumes/Work/Code/duckdb-rs
git add crates/duckdb/src/arrow_zerocopy.rs
git commit -m "test(arrow): zero-copy self-join replayability + lean filter handling

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Copy-dominated perf benchmark vs Appender

**Files:**
- Create: `crates/duckdb/examples/arrow_zerocopy_bench.rs`

**Interfaces:**
- Consumes: `Connection::register_arrow_zerocopy`, `ArrowRegistration` (Task 1); the Appender API (`Connection::appender`, `Appender::append_record_batch`, feature `appender-arrow`).

- [ ] **Step 1: Write the benchmark example**

Create `crates/duckdb/examples/arrow_zerocopy_bench.rs`:

```rust
//! Copy-dominated benchmark: zero-copy `register_arrow_zerocopy` vs the Appender (copy), cold
//! per-task (open connection + ingest + one light query). The table is wide (8 int64 columns) with
//! light compute and a single-column projection, so both the skipped ingest copy AND projection
//! pushdown show. Run:
//!   cargo run -p duckdb --features "bundled appender-arrow" --release --example arrow_zerocopy_bench

use std::sync::Arc;
use std::time::Instant;

use duckdb::Connection;
use duckdb::arrow::array::Int64Array;
use duckdb::arrow::datatypes::{DataType, Field, Schema};
use duckdb::arrow::record_batch::RecordBatch;

const COLS: usize = 8;

fn make_batch(n: i64) -> RecordBatch {
    let fields: Vec<Field> = (0..COLS).map(|c| Field::new(format!("c{c}"), DataType::Int64, false)).collect();
    let schema = Arc::new(Schema::new(fields));
    let cols: Vec<Arc<dyn duckdb::arrow::array::Array>> = (0..COLS)
        .map(|c| Arc::new(Int64Array::from_iter_values((0..n).map(|i| i + c as i64))) as _)
        .collect();
    RecordBatch::try_new(schema, cols).unwrap()
}

fn ddl() -> String {
    let cols: Vec<String> = (0..COLS).map(|c| format!("c{c} BIGINT")).collect();
    format!("CREATE TABLE t ({})", cols.join(", "))
}

// Light query that scans all rows but projects one column (so projection pushdown matters).
const QUERY: &str = "SELECT sum(c0) FROM t";
const QUERY_V: &str = "SELECT sum(c0) FROM v";

fn appender_cold(batch: &RecordBatch) -> i64 {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(&ddl()).unwrap();
    {
        let mut app = conn.appender("t").unwrap();
        app.append_record_batch(batch.clone()).unwrap();
    }
    conn.query_row(QUERY, [], |r| r.get(0)).unwrap()
}

fn zerocopy_cold(batch: &RecordBatch) -> i64 {
    let conn = Connection::open_in_memory().unwrap();
    let reg = conn.register_arrow_zerocopy("v", vec![batch.clone()]).unwrap();
    let out: i64 = conn.query_row(QUERY_V, [], |r| r.get(0)).unwrap();
    drop(reg);
    out
}

fn main() {
    let sizes = [100_000i64, 1_000_000, 10_000_000];
    let k = 5;

    // Warm up (first connection + first arrow_scan pay one-time inits).
    let warm = make_batch(1000);
    assert_eq!(appender_cold(&warm), zerocopy_cold(&warm), "warm-up results must agree");

    println!("Copy-dominated cold per-task: zero-copy arrow_scan vs Appender (wide table, {COLS} cols)");
    println!("{:>12} {:>14} {:>15} {:>9}", "rows", "appender ms", "zero-copy ms", "speedup");
    for &n in &sizes {
        let batch = make_batch(n);
        assert_eq!(appender_cold(&batch), zerocopy_cold(&batch), "results must agree at n={n}");

        let t0 = Instant::now();
        for _ in 0..k { let _ = appender_cold(&batch); }
        let app_ms = t0.elapsed().as_secs_f64() * 1000.0 / k as f64;

        let t1 = Instant::now();
        for _ in 0..k { let _ = zerocopy_cold(&batch); }
        let zc_ms = t1.elapsed().as_secs_f64() * 1000.0 / k as f64;

        println!("{:>12} {:>14.1} {:>15.1} {:>8.2}x", n, app_ms, zc_ms, app_ms / zc_ms);
    }
}
```

- [ ] **Step 2: Build the example (debug) to catch errors**

Run: `cargo build -p duckdb --features "bundled appender-arrow" --example arrow_zerocopy_bench 2>&1 | tail -15`
Expected: builds. Fix any API mismatches (e.g. `Int64Array::from_iter_values`, `appender`/`append_record_batch` signatures) against the crate's actual arrow v58 / appender API before proceeding.

- [ ] **Step 3: Run the benchmark (release) and record results**

Run: `cargo run -p duckdb --features "bundled appender-arrow" --release --example arrow_zerocopy_bench 2>&1 | tail -8`
Expected: a table; the in-run `assert_eq!`s confirm correctness (zero-copy == Appender) at every size. Record the numbers — they are the spike's deliverable (expect modest at small sizes, larger as ingestion dominates, per Phase 2c).

- [ ] **Step 4: Commit**

```bash
cd /Volumes/Work/Code/duckdb-rs
git add crates/duckdb/examples/arrow_zerocopy_bench.rs
git commit -m "bench(arrow): copy-dominated zero-copy arrow_scan vs Appender

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Post-implementation

After Task 3, record the benchmark numbers + correctness outcome back in the Daft follow-ups doc (`docs/superpowers/specs/2026-06-25-duckdb-poc-performance-followups.md`) as a "Phase 2d (duckdb-rs shim spike)" section: whether zero-copy is now correct (filters via lean workaround, self-join replayable) and the measured cold-vs-Appender speedup, then the decision on whether Option 2 (fork/upstream) is worth pursuing.

## Self-Review notes (for the implementer)

- **Arrow API drift:** the exact arrow v58 calls (`FFI_ArrowArrayStream::new`, `FFI_ArrowSchema::try_from(&Schema)`, `RecordBatch::project`, `Schema::index_of`, `Int64Array::from_iter_values`, `Appender::append_record_batch`) are mirrored from in-repo usage (`crates/duckdb/src/raw_statement.rs`, `vtab/arrow.rs`, `appender/arrow.rs`); if a signature differs in 1.10504, adapt to the in-repo form rather than guessing.
- **`Error` construction:** the crate `Error` has no plain-string variant; custom errors use the `duckdb_err(msg)` helper (`Error::DuckDBFailure(ffi::Error::new(duckdb_state_DuckDBError), Some(msg))`) and `Error::NulError` for the name — both verified against `crates/duckdb/src/error.rs` (edition 2024, so `unsafe extern "C"` blocks are correct).
- **Lifetime:** every test/bench drops `ArrowRegistration` only after its queries (and the `Connection` is dropped at scope end after the handle in tests — acceptable since no query runs between). Keep this ordering in any new test.
- **First build is slow** (compiles the amalgamation + shim). Subsequent edits to only Rust/the shim are fast.

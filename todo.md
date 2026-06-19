# Task: Read Avro from tar.gz → Parquet via Daft Distributed

**Goal**: `daft.read_avro_tar(s3_glob)` → `df.write_parquet(output)` using Daft's distributed (Ray) execution.
**Branch**: `sbhatti/gz_avro_support` on `anshulgoel27/Daft`
**Based on**: Cherry-pick of `Lucas61000:issue-6901` (PR #7009 — Avro support, still open upstream)

---

## Phase 1 — Branch Setup & Cherry-pick PR #7009

- [x] **1.1** Fetch upstream and create feature branch
  ```
  git fetch upstream
  git checkout -b sbhatti/gz_avro_support upstream/main
  ```

- [x] **1.2** Add Lucas61000's fork as remote and fetch the PR branch
  ```
  git remote add lucas61000 https://github.com/Lucas61000/Daft.git
  git fetch lucas61000 issue-6901
  ```

- [x] **1.3** Get the list of commits on the PR branch to cherry-pick
  ```
  git log --oneline upstream/main..lucas61000/issue-6901
  ```

- [x] **1.4** Cherry-pick all PR commits onto `sbhatti/gz_avro_support`
  - Likely conflict zones: `Cargo.toml`, `Cargo.lock`, `src/lib.rs`
  - Resolve conflicts, `git add`, `git cherry-pick --continue`

- [ ] **1.5** Run `make build` ← **USER RUNS THIS**
  - Compiles the new `src/daft-avro/` Rust crate and rebuilds the Python wheel

- [ ] **1.6** Verify cherry-pick with existing Avro tests
  ```
  DAFT_RUNNER=native make test EXTRA_ARGS="-v tests/io/test_avro.py tests/io/test_avro_write.py"
  ```

---

## Phase 2 — Inspect PR's PyO3 Binding API

- [ ] **2.1** After build, check what the PyO3 binding exposes
  - File: `src/daft-avro/src/python.rs` (or `lib.rs`)
  - Key question: does `read_avro` accept **file paths only**, or also **raw bytes/buffer**?
  - This determines how `AvroTarSourceTask.read()` passes extracted `.avro` bytes to the Rust reader:
    - **Path only** → write extracted bytes to `tempfile.NamedTemporaryFile(suffix='.avro')`
    - **Bytes** → pass directly to the binding

---

## Phase 3 — Implement `read_avro_tar()`

**Reference pattern**: `daft/io/mcap/_mcap.py` → `MCAPSource(DataSource)` + `MCAPSourceTask(DataSourceTask)`

### Files to create

- [ ] **3.1** `daft/io/avro_tar/__init__.py`
  - Just re-exports `read_avro_tar`

- [ ] **3.2** `daft/io/avro_tar/_avro_tar.py`
  ```
  AvroTarSource(DataSource)
    schema()          → download first tar.gz, extract first .avro, infer schema
    get_tasks()       → S3 glob expand → one AvroTarSourceTask per .tar.gz file

  AvroTarSourceTask(DataSourceTask)
    read()            → download tar.gz bytes from S3
                     → tarfile.open(fileobj=BytesIO(bytes), mode='r:gz')
                     → for each .avro member: extract → read → yield RecordBatch

  read_avro_tar(path, io_config, schema, file_path_column) → DataFrame
  ```
  - Parallelism unit: **one task per tar.gz file** (Ray distributes across workers)

### Files to modify

- [ ] **3.3** `daft/io/__init__.py`
  - Add `from daft.io.avro_tar import read_avro_tar`
  - Add `read_avro_tar` to `__all__`

- [ ] **3.4** `daft/__init__.py`
  - Add `read_avro_tar` to the `from daft.io import (...)` block
  - Add `read_avro_tar` to top-level `__all__`

---

## Phase 4 — Tests

- [ ] **4.1** Create `tests/io/test_avro_tar.py`

  | Test | What it verifies |
  |------|-----------------|
  | `test_single_tar_gz_multiple_avros` | Basic read: row count + schema |
  | `test_multiple_tar_gz_glob` | Glob over multiple `.tar.gz` files |
  | `test_schema_inference` | Auto-detected schema matches expected |
  | `test_column_projection` | Only requested columns returned |
  | `test_avro_tar_to_parquet` | Full pipeline: `read_avro_tar` → `write_parquet` |
  | `test_ray_distributed` | Skipped if `DAFT_RUNNER != ray` |
  | `test_empty_tar_gz` | Edge case: archive with no `.avro` files |
  | `test_various_avro_types` | Nullable, nested, numeric types |

  - Fixtures use `fastavro` to generate `.avro` bytes + Python `tarfile` to package them

- [ ] **4.2** Run tests
  ```
  DAFT_RUNNER=native make test EXTRA_ARGS="-v tests/io/test_avro_tar.py"
  DAFT_RUNNER=ray  make test EXTRA_ARGS="-v tests/io/test_avro_tar.py"
  ```

---

## Phase 5 — Branch Prep & Push

- [ ] **5.1** Commit with conventional format
  ```
  feat: add read_avro_tar() for distributed Avro-in-tar.gz to Parquet conversion
  ```

- [ ] **5.2** Rebase onto latest upstream/main
  ```
  git fetch upstream && git rebase upstream/main
  ```

- [ ] **5.3** Push to fork
  ```
  git push origin sbhatti/gz_avro_support
  ```

- [ ] **5.4** Verify CI passes on `anshulgoel27/Daft`

---

## Key Files

| File | Action |
|------|--------|
| `src/daft-avro/` | Cherry-picked Rust Avro crate (from PR #7009) |
| `daft/io/_avro.py` | `read_avro()` Python API (from PR #7009) |
| `daft/io/avro_tar/__init__.py` | NEW |
| `daft/io/avro_tar/_avro_tar.py` | NEW — core implementation |
| `daft/io/__init__.py` | MODIFY — add export |
| `daft/__init__.py` | MODIFY — add to public API |
| `tests/io/test_avro_tar.py` | NEW — 8 tests |
| `daft/io/mcap/_mcap.py` | REFERENCE pattern |
| `daft/io/source.py` | REFERENCE — DataSource/DataSourceTask ABCs |

---

## Notes / Decisions

- One `AvroTarSourceTask` per tar.gz → Ray distributes across workers (not one task per `.avro` within archive)
- Python `tarfile` handles extraction — no new Rust code needed for the tar layer
- PR #7009 is still open upstream; apply it via cherry-pick, not via merge from upstream/main
- Open review items on PR #7009 (defer for now):
  - OOM risk on large remote files (streaming fix pending Arrow 59 upgrade in PR #7141)
  - Redundant custom schema converter (should use `TryFrom<&ArrowSchema>`)
- S3 integration tests: skip if `TEST_S3_BUCKET` env var not set; use local tmpdir fixtures for CI

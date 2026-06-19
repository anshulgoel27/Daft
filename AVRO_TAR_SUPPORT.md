# Avro + tar.gz Support for Daft

## What Was Built

This branch (`sbhatti/gz_avro_support`) adds two capabilities to Daft:

1. **Native Avro read/write** — cherry-picked from [Lucas61000/Daft PR #7009](https://github.com/Lucas61000/Daft/pull/7009) (`daft.read_avro`, `daft.write_avro`)
2. **`daft.read_avro_tar()`** — reads Avro files packed inside `.tar.gz` archives, including from S3/GCS/Azure, with full distributed (Ray) support

---

## `daft.read_avro_tar()` API

```python
import daft

df = daft.read_avro_tar(
    path,                    # str, list[str], or glob pattern — local or s3://
    io_config=None,          # daft.io.IOConfig for S3/GCS/Azure credentials
    column_names=None,       # list[str] — column projection (read subset of columns)
    file_path_column=None,   # str — adds a column with the source tar.gz URI
)
```

### Examples

```python
# Single tar.gz on S3
df = daft.read_avro_tar("s3://my-bucket/data/records.tar.gz")

# Glob pattern across many archives
df = daft.read_avro_tar("s3://my-bucket/data/part-*.tar.gz")

# With credentials and column projection
from daft.io import IOConfig, S3Config
io_config = IOConfig(s3=S3Config(region_name="us-east-1"))
df = daft.read_avro_tar(
    "s3://my-bucket/data/*.tar.gz",
    io_config=io_config,
    column_names=["id", "timestamp", "value"],
    file_path_column="source_file",
)

# Convert to Parquet
df.write_parquet("s3://my-bucket/output/")

# Distributed execution on Ray
import ray
ray.init()
daft.context.set_runner_ray()
df = daft.read_avro_tar("s3://my-bucket/data/*.tar.gz")
df.write_parquet("s3://my-bucket/parquet/")
```

---

## Architecture

- **`daft/io/avro_tar/_avro_tar.py`** — core implementation
  - `AvroTarSource(DataSource)` — one task per `.tar.gz` file for Ray parallelism
  - `AvroTarSourceTask(DataSourceTask)` — streams the archive, extracts each `.avro` member to a temp file, calls the Rust Avro reader
  - Schema is inferred from the first `.avro` entry in the first archive
- **`src/daft-avro/`** — Rust crate (from PR #7009) using `arrow-avro` for OCF read/write
- Follows the same `DataSource`/`DataSourceTask` pattern as `daft/io/mcap/`

---

## Commits on This Branch

| Commit | Description |
|--------|-------------|
| Cherry-pick of PR #7009 | Native Avro read/write Rust crate + Python bindings |
| `feat: add Avro read/write support and read_avro_tar()` | Full `read_avro_tar()` implementation + tests |
| `fix: restore correct into_remainder() pattern in daft-minhash` | Fixes build regression introduced by the geo spatial join commit (`c6480b6be`) |

---

## Tests

```bash
# Avro read/write tests (from cherry-picked PR)
DAFT_RUNNER=native make test EXTRA_ARGS="-v tests/io/test_avro.py tests/io/test_avro_write.py"

# read_avro_tar() tests
DAFT_RUNNER=native make test EXTRA_ARGS="-v tests/io/test_avro_tar.py"
```

All tests pass (10/10 native, 1 Ray test skipped without `DAFT_RUNNER=ray`).

---

## Pending

- **Push to fork**: Waiting for `anshulgoel27` to grant collaborator access to one of:
  - `devilpreet` (GitHub)
  - `sarbpreet-bhatti_prcly` (GitHub)

  Once granted:
  ```bash
  git push origin sbhatti/gz_avro_support
  gh pr create --base main --head sbhatti/gz_avro_support \
    --title "feat: add read_avro_tar() for distributed Avro/tar.gz ingestion" \
    --repo anshulgoel27/Daft
  ```

---

## Build Notes

- After any Rust change: `make build`
- `daft-minhash` build regression: fixed in this branch — `into_remainder()` returns `Option<IntoIter<u64, N>>`, must unwrap before iterating (`if let Some(r) = ... { for hash in r { } }`)
- `Cargo.toml` has both geo deps (`geo`, `geohash`, `h3o`, `wkt`) and upstream's `half` crate — both required

---

## Context Transfer Prompt (for the udp/consumer project)

> Paste the following prompt when switching to the project that consumes this Daft fork.

---

I am working on a project that uses a custom Daft fork as a dependency. Here is the full context:

**The Daft fork (`sbhatti/gz_avro_support` branch):**
- Adds `daft.read_avro` and `daft.read_avro_tar()` — reads Avro files and Avro files packed inside `.tar.gz` archives (local, S3, GCS, Azure)
- Built and tested locally on macOS arm64 — `read_avro` round-trip confirmed working
- Produces a Linux wheel: `daft-0.3.0.dev0-cp310-abi3-manylinux_2_28_aarch64.whl` (or x86_64 depending on target)
- Fork repo: local at `/Users/sarbpreet/repos/Daft`, branch `sbhatti/gz_avro_support`

**Current state:**
- The Linux wheel is being built right now via `make build-whl`
- Once complete, the `.whl` will be at `target/wheels/daft-0.3.0.dev0-cp310-abi3-*.whl`
- This wheel needs to be installed into the udp project's Docker image

**This project (udp) uses Daft on a Ray cluster running on Kubernetes on AWS.**

**What needs to be implemented in the udp project:**

1. **Dockerfile** — replace the current `pip install daft` with the custom wheel:
   ```dockerfile
   COPY wheels/daft-0.3.0.dev0-cp310-abi3-manylinux_2_28_aarch64.whl /tmp/
   RUN pip install /tmp/daft-0.3.0.dev0-cp310-abi3-manylinux_2_28_aarch64.whl
   ```

2. **Verify `read_avro_tar` is available** in the image:
   ```bash
   python -c "import daft; print(hasattr(daft, 'read_avro_tar'))"  # must print True
   python -c "import daft; print(hasattr(daft, 'read_avro'))"      # must print True
   ```

3. **Dev cycle improvements to implement:**
   - Split Dockerfile so the `pip install daft` layer is cached independently from app code — Python-only changes should rebuild in < 30 sec
   - Add a GitHub Actions workflow that: on push to the branch, pulls the pre-built wheel from S3 or a GitHub release, builds the Docker image, and pushes to ECR
   - Use Ray `runtime_env` for rapid iteration without rebuilding Docker:
     ```python
     ray.init(runtime_env={"pip": ["s3://your-bucket/daft-0.3.0.dev0-cp310-abi3-manylinux_2_28_aarch64.whl"]})
     ```
   - Store the Daft wheel in S3 so CI and local builds pull from one place — no one builds the wheel locally again

4. **Key insight to preserve:** Most Daft fork changes are Python-only. Never trigger a full Rust recompile for a Python change. The wheel only needs rebuilding when Rust code in the fork changes (rare). Everything else is just copying the Python files or swapping the wheel.

**The `read_avro_tar` API:**
```python
import daft
df = daft.read_avro_tar(
    "s3://my-bucket/data/records.tar.gz",  # or glob pattern
    io_config=None,        # daft.io.IOConfig for credentials
    column_names=None,     # list[str] for column projection
    file_path_column=None, # str — adds source path as a column
)
df.write_parquet("s3://my-bucket/output/")
```

Please help me implement the Dockerfile change and dev cycle improvements described above.

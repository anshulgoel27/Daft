# Parquet `CorruptFile` on distributed reads — split scan tasks + HEAD-skip fast path

**Status:** Root-caused
**Severity:** High — silently fails otherwise-valid Parquet imports/reads with a misleading "corrupt file" error
**Scope:** Fork only (regression introduced by fork commit `eb6cfecdf`); upstream Daft is unaffected

---

## Symptom

Distributed (Ray runner) reads of large Parquet files fail with:

```
DaftError::CorruptFile Failed to read parquet metadata for
s3://raw-snowflake-consumption/raw/geo_addressing_2025-12-04.parquet
using the caller-supplied file size of 140789350 bytes (the HEAD request was
skipped because the scan task reported an exact size); a stale or approximate
scan-task size is the likely cause:
... is not a valid parquet file. Incorrect footer magic: [50, 26, 58, 11]
```

Key detail: the **caller-supplied file size (`140789350` ≈ 134 MB) is smaller than the
actual file** (`geo_addressing_2025-12-04.parquet` is 966 MB). The reader used the
smaller size to locate the footer and read from the wrong offset, so the "footer magic"
bytes were actually mid-file data (`[50, 26, 58, 11]`, i.e. arbitrary column bytes,
not `PAR1`).

The file itself is **not** corrupt — `aws s3 cp` + `pyarrow` read it fine, and upstream
Daft reads it fine.

---

## Root cause

A two-part interaction between pre-existing upstream behavior and a fork-added
optimization.

### Part 1 — a split scan task's `size_bytes` is the *chunk* size (upstream, benign)

`split_by_row_groups` (`src/daft-scan/src/scan_task_iters/mod.rs`) splits a large Parquet
file into multiple scan tasks, one per contiguous group of row groups. Each split task's
`size_bytes` is set to the **sum of that split's row-group compressed sizes** — i.e. the
*chunk* size, not the whole-file size:

```rust
*chunk_spec = Some(ChunkSpec::Parquet(curr_row_group_indices));
new_source.size_bytes = Some(curr_size_bytes as u64); // sum of THIS split's row-group sizes
```

This is **inherited from upstream** and is correct there: upstream uses `size_bytes` only
as a *scheduling / memory estimate*. It never uses it to locate the Parquet footer.

### Part 2 — the fork's HEAD-skip fast path trusts `size_bytes` as the *whole-file* size

Fork commit `eb6cfecdf` ("feat(parquet): skip per-file HEAD when file size is already
known") added an optimization: if the scan task already knows the file size, skip the
per-file S3 `HEAD` request and fetch the footer directly with an exact byte range.

`prepare_remote_chunk_source` (`src/daft-parquet/src/reader/chunk_source.rs`):

```rust
let (parquet_metadata_res, file_size) = if let Some(known_size) = opts.size_bytes {
    // trust known_size → skip HEAD, fetch footer at known_size - footer_len
    let metadata_res = read_parquet_metadata(uri, Some(known_size), ...);
    (metadata_res, known_size)
} else {
    // real HEAD to discover the true file size
    ...
};
```

This assumes `size_bytes` is the **exact whole-file size**. But for a split scan task
(Part 1) it is the **chunk size**. The footer lives at `whole_file_size - footer_len`;
reading at `chunk_size - footer_len` lands in the middle of the file → bad footer magic →
`CorruptFile`.

### Why it only shows up on the cluster

- **`count_rows` locally bypasses it.** Row-count queries read footers directly via
  `read_parquet_metadata(uri, ...)` with no `size_bytes` hint, so they never hit the
  fast path. (This is why the initial local repro with `count_rows` passed.)
- **The attached metadata is dropped over Ray.** The split also attaches a pre-read
  `parquet_metadata: Arc<DaftParquetMetadata>` to the source, which *could* avoid the
  footer read — but the arrowrs reader currently ignores it (`TODO` in `read.rs`), and it
  is an in-memory `Arc` that is not serialized when `ScanTask`s are shipped to Ray
  workers. So a distributed full read reconstructs the task with only `size_bytes`
  (chunk) + `chunk_spec`, then falls into the fast path and reads the footer at the wrong
  offset.

So the failure requires: a **large, splittable Parquet file** (so `split_by_row_groups`
fires) **+ a distributed full read** (so the fast path runs without the in-memory
metadata). Small files (below the split threshold) and metadata-only reads are unaffected.

---

## Reproduction

`scripts/repro_split_merge.py` (in the pipeline repo) drives this directly.

- Upstream Daft `0.7.20`: reads fine with `enable_scan_task_split_and_merge=True`, even
  forcing ~60 tiny 16 MB split tasks over the 966 MB file → **PASS** (no HEAD-skip path).
- Fork `0.3.0-dev0` full read of the split scan tasks over S3 → **`CorruptFile`**.

A minimal unit-level repro: build a `ScanTask` for a multi-row-group Parquet file, run it
through `split_by_row_groups`, take one split (whose `size_bytes` < file size), and read
it via `prepare_remote_chunk_source` — the footer read fails.

---

## Attribution / history

| Piece | Commit | Author | Origin |
|---|---|---|---|
| Split sets `size_bytes` = chunk size | `22b6a7716` refactor(io): move common source fields to own type (#6451) | R. C. Howell | Upstream |
| **HEAD-skip trusts `size_bytes` for the footer** | `eb6cfecdf` feat(parquet): skip per-file HEAD when file size is already known | Anshul Goel | **Fork** |
| Known-size error-message hardening | `a4d93ce7e` fix(parquet): harden known-size error classification on the scan fast path | Anshul Goel | Fork |

The bug is a **fork-introduced regression**: `eb6cfecdf` added a fast path that is
incompatible with the long-standing upstream convention that a split scan task's
`size_bytes` is the chunk size.

---

## Open questions

- Should `size_bytes` be split into two distinct concepts — a scheduling/memory estimate
  vs. an exact whole-file size usable for footer location — so the two can never be
  confused again?
- Is there any other consumer that trusts `size_bytes` as an exact whole-file size while a
  `chunk_spec` / `row_groups` subset is present (e.g. the local-file reader path)?
- The split-attached `parquet_metadata` `Arc` is dropped when `ScanTask`s are serialized
  to Ray workers — is it worth persisting it (or the exact file size) across that boundary
  so subset reads never need to re-derive the footer location?

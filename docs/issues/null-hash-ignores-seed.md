# Multi-column `distinct()` / `groupby()` / joins hang when a key column is NULL — NULLs in primitive columns hash without the seed

**Status:** Root-caused; reproduced locally. A fix is proposed below and has **not** been built or tested.
**Severity:** High — a query silently runs for hours (or effectively forever) on large inputs, with no error
**Scope:** The kernel code comes from upstream Daft and is present in this fork; the flag involved first appears in upstream commit
`2427a22d1` (perf: reduce binary size by de-monomorphizing sort and dispatch paths, #6541). Whether it behaved this way before that
commit was not checked.

---

## Symptom

A Ray job running `df.distinct()` over about 153M rows did not finish in more than 5 hours. In the plan, the stage
`ShuffleRead->Dedup` reached 2105 of 2106 tasks within minutes; the last task then ran for hours:

- CPU-bound on one worker (30+ CPU-hours, 8 compute threads at 100%), ~7.4 GiB RSS, no I/O wait.
- The cluster's CPU counter showed 0 used, because swordfish actors do not declare CPU to Ray, so nothing looked busy.
- The input data was **not** skewed: a 7% sample had one row per entity and no hot key.

The frame being deduplicated was `(entity_id: string, elevation: float64, elevation_high: float64, elevation_low: float64)`, where the
last two columns were constant NULL.

## Reproduction

Pure Daft, no cluster, no data files. Native runner, 300,000 rows:

```python
import time
import daft
from daft import DataType, col, lit

n = 300_000
df = daft.from_pydict({"id": [f"id-{i:07d}" for i in range(n)], "v": [float(i % 5000) for i in range(n)]})

ok = df.select(col("id"), col("v"))
bad = df.select(
    col("id"), col("v"),
    lit(None).cast(DataType.float64()).alias("hi"),
    lit(None).cast(DataType.float64()).alias("lo"),
)

t = time.time(); ok.distinct().count_rows();  print("no NULL key columns:", time.time() - t)   # 0.5 s
t = time.time(); bad.distinct().count_rows(); print("two constant-NULL key columns:", time.time() - t)  # still running after 120 s
```

Measured on a 427,406-row sample of the real data, with a 90 s cap per variant:

| `distinct()` over | Result |
|---|---|
| `entity_id` | 0.2 s |
| `entity_id`, `elevation` | 0.3 s |
| `entity_id`, `elevation`, two constant-NULL float columns | did not finish in 90 s |

A native stack sample (macOS `sample`) of the stalled process showed the time inside `RecordBatch::to_idx_hash_table` (a hashbrown table
with an identity hasher), the multi-column equality kernel (`build_multi_array_is_equal_from_arrays`) and string/float comparison: a hash
table in which every entry collides.

## Direct check of the hash

Hash an all-NULL column with a per-row seed (`seed = hash(id)`) and count distinct results over 20,000 rows:

| dtype | distinct hashes | seed used? |
|---|---|---|
| float64, float32, int64, int32, bool, date, timestamp, decimal128(10,2) | **1** | no |
| string, binary | 20,000 | yes |

```python
h = col("id").hash()
df.with_column("n", lit(None).cast(DataType.float64())).select(col("n").hash(seed=h)).to_pydict()  # one distinct value
```

## Root cause

`src/daft-core/src/kernels/hashing.rs`.

The engine hashes a multi-column key by hashing the first column, then each following column with the previous result as its seed.
A NULL in a **primitive** column breaks that chain:

- `trait HashImpl` has `const SEEDED_NULL_USES_DEFAULT: bool` (line ~68). It is `true` for `Xxh32HashImpl`, `Xxh64HashImpl` and
  `Xxh3_64HashImpl` (lines ~197, 209, 221) and `false` for murmur and sha1 (lines ~173, 185).
- `scalar_null_hash(seed)` (line ~77) returns `default_null_hash()` when the flag is `true`, i.e. the hash of the empty byte string,
  **ignoring the seed**.
- `hash_primitive_inner` (line ~462), and the boolean, decimal and timestamp variants, call `scalar_null_hash(seed)` for a NULL element
  when a seed is present. `Expression.hash()` and the engine's row hashing use `XxHash3_64`, so they all take this path.
- The string and binary kernels (`hash_large_string` line ~601, `hash_large_binary` line ~567) map NULL to the empty string/bytes and
  then call `hash_seeded(bytes, seed)`, so the seed is kept. That is why only strings and binary behave.

Effect on a key `(c1, c2, ..., ck)`: for every row where a primitive column `cj` is NULL, the hash accumulated from `c1..c(j-1)` is
discarded and replaced by a constant. Rows that agree on all columns **after** `cj` therefore share one hash regardless of their earlier
columns. If the trailing columns are NULL for most rows (as with constant-NULL columns), nearly all rows share one hash.

Two consequences:

1. **One partition gets everything.** A hash-partitioned shuffle (`distinct`, `groupby`, hash join, repartition by columns) assigns the
   partition from the hash, so all colliding rows go to the same partition. That is the single straggler task above.
2. **Quadratic work inside the partition.** The in-partition hash table (`to_idx_hash_table`, identity hasher) puts all colliding rows in
   one chain, so each insert compares against every earlier row.

## Who is affected

Any hashed key of two or more columns where a primitive column is NULL for many rows, worst when NULL columns come last:

- `distinct()` on a frame with such columns (with no arguments it uses every column).
- `groupby(a, b, ...)`, multi-column joins, `repartition(n, a, b, ...)`.

Not affected: single-column keys (there is no seed to ignore, and NULL is one correct group), and keys made only of string/binary columns.

This is a performance problem, not a correctness one, as far as the code shows: the hash only picks the partition and the table bucket,
and the equality check still tells rows apart. No colliding case was observed running to completion, so that is a reading of the code,
not a measured result.

## Workaround

Keep NULLs out of the key columns of primitive type: fill them with a sentinel that cannot be a real value, deduplicate, then restore
the NULLs.

```python
SENTINEL = -1.0e9   # must lie outside the column's valid range
keyed = df.select("id", *[col(c).fill_null(SENTINEL).alias(c) for c in float_cols])
out = keyed.distinct().select("id", *[when(col(c) == SENTINEL, lit(None).cast(DataType.float64())).otherwise(col(c)).alias(c) for c in float_cols])
```

Dropping constant-NULL columns from the key and re-adding them afterwards also works. For approximate use, a string-typed surrogate key
(for example an md5 of the columns) avoids the problem entirely.

## Proposed fix (not built, not tested)

1. In `hashing.rs`, set `SEEDED_NULL_USES_DEFAULT = false` for `Xxh32HashImpl`, `Xxh64HashImpl` and `Xxh3_64HashImpl`, so a NULL hashes
   as `hash_seeded(b"", seed)` — what murmur and sha1 already do, and what the string path already does.
2. **Find out why it was `true`.** The reason is not known. One possibility is wanting `hash(NULL)` to be one stable value for any seed.
   Check the upstream history and issue tracker before changing it.
3. **Impact to assess.** Hash values change for key chains that contain a NULL, so shuffle partition assignment changes for those rows.
   Anything that stores or compares hash values across versions (user columns built with `Expression.hash(seed=...)`, persisted
   hashes) would see different values for NULL-containing rows. Search the repo and downstream code for seeded `.hash(` and `hash_with(`.
4. **Tests to add.**
   - Rust, in `hashing.rs`: for each primitive dtype, `hash(NULL, seed=a) != hash(NULL, seed=b)` when `a != b`, with a null mask in the
     middle of a column.
   - Python: the reproduction above, with a time bound of a few seconds.
   - A multi-column `groupby` and a multi-column hash join with trailing NULL columns.
   - Then the full test suite, since partition assignment changes.
5. **Roll out.** The change needs a Rust rebuild (`make build`) and a new image. Run the reproduction on the cluster before relying on it.

## How to find it in a running job

- The signature in a Ray job log is a progress bar stuck one task short (here `ShuffleRead->Dedup`, 2105/2106) while the cluster looks idle.
- Confirm with the process's CPU time rather than Ray's CPU counter: `ps -o time,pcpu -p <worker pid>` on the node hosting the
  `RUNNING` task from `ray list tasks`, or the pod's CPU in `kubectl top`.
- Confirm the cause by checking the key for NULLs in primitive columns, then by running the hash check above on those columns.

# duckdb-rs Arrow Filter Pushdown — v1.5.4 Full-Coverage Design

**Date:** 2026-06-26
**Status:** Design approved; ready for implementation plan
**Supersedes:** [exprfilter port (Phase B, main bump)](2026-06-26-duckdb-rs-arrow-exprfilter-port-design.md) and its plan. The DuckDB-`main` bump is abandoned — see "Why v1.5.4, not main" below.
**Branch:** base off `arrow-zerocopy` (42492c6, bundle = `v1.5.4` tag `08e34c44`); new work branch `arrow-zerocopy-fullfilter`.
**Reference:** duckdb-python `/Volumes/Work/Code/duckdb-python/src/duckdb_py/arrow/filter_pushdown_visitor.cpp` (`TransformExpression`/`TransformFilter`).

## Goal

Reach **filter-pushdown coverage parity with duckdb-python** for `Connection::register_arrow`, while staying on the **released DuckDB `v1.5.4`** bundle that duckdb-rs already ships. Close the three gaps that the existing structured-filter walker leaves open: the `STRUCT_EXTRACT` filter kind, scalar-type breadth (Date / Time / Timestamp / Decimal128, plus float NaN), and the `EXPRESSION_FILTER` catch-all (a full port of duckdb-python's expression-tree walk, adapted to v1.5.4's classic bound-expression API).

## Why v1.5.4, not main

The earlier plan bumped the bundle to DuckDB `main` (`06eb6b6`) on the belief that v1.5.4 used the structured `TableFilter` model and only `main` had `EXPRESSION_FILTER`. Verification disproved that premise:

- `06eb6b6` is upstream `duckdb/duckdb` **main**, `git describe` = `v1.5.4-8021-g06eb6b6`. v1.5.4 was tagged 2026-06-16, only 9 days earlier — they are two **divergent** lines (stable release branch vs. main), not "v1.5.4 + a few commits."
- The **released v1.5.4 already has `EXPRESSION_FILTER` (=9)** in `TableFilterType` *and* the structured kinds. For arrow scans v1.5.4 emits the **structured** kinds (the working spike receives `CONSTANT_COMPARISON` etc.); `EXPRESSION_FILTER` is the catch-all for predicates that don't fit a structured kind.
- `main` renamed the structured constants to `LEGACY_*` and flipped arrow-scan pushdown to emit `EXPRESSION_FILTER`; that rename — not any missing capability — is what broke the shim and forced the bump. The bump also forced build hacks (`-std=c++17`, jemalloc exclusion) and pinned us to an unreleased, moving target.

Staying on v1.5.4 gives the **same filter coverage** on a **stable, tagged release** with a clean build, and reuses the proven structured walker. When a future DuckDB release removes the `LEGACY_*` kinds, the expression front-end we add here becomes the sole path — so this is forward-compatible.

## Verified v1.5.4 API (in `crates/libduckdb-sys/duckdb-sources` at tag `v1.5.4`)

- `enum class TableFilterType`: `CONSTANT_COMPARISON=0, IS_NULL=1, IS_NOT_NULL=2, CONJUNCTION_OR=3, CONJUNCTION_AND=4, STRUCT_EXTRACT=5, OPTIONAL_FILTER=6, IN_FILTER=7, DYNAMIC_FILTER=8, EXPRESSION_FILTER=9, BLOOM_FILTER=10` (clean names; no `LEGACY_` prefix; no `PREFIX_RANGE`).
- `StructFilter : TableFilter` (`TYPE = STRUCT_EXTRACT`): `{ idx_t child_idx; string child_name; unique_ptr<TableFilter> child_filter; }` — descend to child `child_idx`, apply `child_filter` (which may itself be another `StructFilter`, a `ConstantFilter`, etc.).
- `ExpressionFilter : TableFilter` (`TYPE = EXPRESSION_FILTER`): `{ unique_ptr<Expression> expr; }`.
- `BoundComparisonExpression : Expression` (classic): `{ unique_ptr<Expression> left, right; }`; comparison op = `GetExpressionType()` (`COMPARE_EQUAL=25, COMPARE_NOTEQUAL=26, COMPARE_LESSTHAN=27, COMPARE_GREATERTHAN=28, COMPARE_LESSTHANOREQUALTO=29, COMPARE_GREATERTHANOREQUALTO=30`). **Not** a `BoundFunctionExpression` (that is main's shape).
- `BoundConjunctionExpression : Expression`: `{ vector<unique_ptr<Expression>> children; }`; type `CONJUNCTION_AND` / `CONJUNCTION_OR`.
- `BoundOperatorExpression : Expression`: `{ vector<unique_ptr<Expression>> children; }`; type `OPERATOR_IS_NULL` / `OPERATOR_IS_NOT_NULL` / `COMPARE_IN` (children[0] = column side, children[1..] = IN constants).
- `BoundReferenceExpression : Expression`: `{ storage_t index; }` (physical column index into the scan's projected output).
- `BoundConstantExpression`: `GetValue() -> Value` (constant side, expression type `VALUE_CONSTANT`).
- struct_extract in an expression tree: a `BoundFunctionExpression` whose function name is `struct_extract` (also `struct_extract_at`), with `bind_info` = `StructExtractBindData { idx_t index; }` giving the child index; `children[0]` is the inner column expression.
- **`table_filter_functions.hpp` does NOT exist on v1.5.4.** Optional / dynamic / bloom filters are *structured kinds* (`OPTIONAL_FILTER`, `DYNAMIC_FILTER`, `BLOOM_FILTER`), already handled by the structured front-end. The expression walker therefore does **not** need duckdb-python's function-name matching (`IsTableFilterFunction`, `OptionalFilterScalarFun::NAME`) — that is main-only, post-migration logic.
- `CanPushdown` (v1.5.4) pushes: bool, all int widths, DATE, TIME (+ NS), TIMESTAMP (+ MS/NS/SEC/TZ), unsigned, FLOAT, DOUBLE, VARCHAR/BLOB (non-view), DECIMAL (INT128 physical only), STRUCT (when all children pushable).

## Architecture (Approach A — dual front-ends, one wire format, one evaluator)

```
arrow_scan view scanned ── per scan ──► ShimProduce(factory, ArrowStreamParameters{projected_columns, filters})
  for each (projected_col, TableFilter) in *filters:
     if filter is EXPRESSION_FILTER:
        emit_expression(ef.expr, root_path={projected_col}, buf)   // ports TransformExpression (v1.5.4 API)
     else:
        emit_filter(filter, root_path={projected_col}, buf)        // structured walker, +STRUCT_EXTRACT
  → Rust rust_produce: decode → evaluate(node, batch) → BooleanArray mask → filter_record_batch
```

- Still registers over the built-in `arrow_scan` table function (projection pushdown + filter pushdown ON); the producer pre-applies the filter before handing batches back.
- **Structured front-end** (`emit_filter`, existing): `CONSTANT_COMPARISON`, `IS_NULL`/`IS_NOT_NULL`, `CONJUNCTION_AND`/`OR`, `IN_FILTER`, `OPTIONAL_FILTER` (unwrap child), `DYNAMIC_FILTER`/`BLOOM_FILTER` (skip). **New: `STRUCT_EXTRACT`** — push `child_idx` onto the column path and recurse on `child_filter`.
- **Expression front-end** (`emit_expression`, new): handles only the `EXPRESSION_FILTER` kind, walking `ef.expr`. Covers comparison / conjunction(AND·OR) / operator(IS NULL · IS NOT NULL · IN) / struct_extract column chains / bound-ref / constant. Anything else → UNHANDLED (fail-loud).
- Both front-ends share `emit_scalar` and `emit_column_path`, and emit the **same** node tags.

## Components

### 1. Wire format extension (`arrow_zerocopy_shim.cpp` ↔ `crates/duckdb/src/arrow_zerocopy/filter.rs`)

Column references become **paths** (this is the only breaking change to existing nodes):
- COMPARE / IS_NULL / IS_NOT_NULL / IN replace their single `col:u32` with `depth:u32` followed by `depth × u32` indices. `index[0]` = projected root column position; `index[1..]` = struct child indices (depth ≥ 1).

Node tags (unchanged set; only the column encoding changes):
- COMPARE: `path`, `op:u8` (0=EQ,1=NE,2=LT,3=GT,4=LE,5=GE), `scalar`
- IS_NULL: `path` · IS_NOT_NULL: `path`
- AND: `count:u32`, children · OR: `count:u32`, children
- IN: `path`, `count:u32`, `count × scalar`
- UNHANDLED (tag 255): `raw_filter_type:u32` (fail-loud sentinel)

Scalar tags — existing: signed-int (`i64`), unsigned-int (`u64`), double (`f64`), float (`f32`), bool (`u8`), varchar (`len:u32` + utf8 bytes). **New:**
- Date32: `i32` (days since epoch)
- Time: `i64` value + `unit:u8` (µs / ns)
- Timestamp: `i64` value + `unit:u8` (s / ms / µs / ns; TZ carried as µs UTC)
- Decimal128: `i128` (little-endian 16 bytes) + `scale:u8`
- Blob: `len:u32` + raw bytes
- A float/double constant that is **NaN** emits a dedicated **NaN-compare** scalar marker so the evaluator applies DuckDB's NaN ordering rather than a literal value compare.

### 2. C++ shim (`crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp`)

- Extend `emit_filter` with a `STRUCT_EXTRACT` branch: cast to `StructFilter`, append `child_idx` to the path, recurse `emit_filter(child_filter, path)`.
- Add `emit_expression(const Expression&, vector<idx_t> path, buf)` — port of `TransformExpression` on the v1.5.4 classic API:
  - `BoundComparisonExpression`: identify the constant side (`VALUE_CONSTANT`) vs. column side; if the constant is on the left, flip the op; resolve the column side via `resolve_column_path` (BOUND_REF → leaf; `struct_extract`/`struct_extract_at` `BoundFunctionExpression` chain via `StructExtractBindData.index` → append child indices, innermost-first); emit COMPARE. A NaN float/double constant emits the NaN-compare marker.
  - `BoundOperatorExpression`: `OPERATOR_IS_NULL` → IS_NULL{path}; `OPERATOR_IS_NOT_NULL` → IS_NOT_NULL{path}; `COMPARE_IN` → IN{path, scalars from children[1..]}.
  - `BoundConjunctionExpression`: `CONJUNCTION_AND` → AND of translatable children (drop a child that is UNHANDLED — the engine re-applies it above the scan); `CONJUNCTION_OR` → OR, but if **any** child is untranslatable, emit UNHANDLED for the whole OR (a stricter pushed predicate would be wrong).
  - anything else → UNHANDLED{expression-class}.
- Widen `emit_scalar` to the new types listed above; detect NaN for FLOAT/DOUBLE constants.
- `resolve_column_path` (shared helper): BOUND_REF → `{root path}`; `struct_extract` function → recurse into `children[0]`, then append `StructExtractBindData.index`. Non-ref, non-struct_extract column side → throw/return UNHANDLED.

### 3. Rust evaluator (`crates/duckdb/src/arrow_zerocopy/filter.rs`)

- Decode each node's path into `Vec<usize>`.
- `resolve_path(batch, path) -> ArrayRef`: `path[0]` selects the projected column; each subsequent index descends into a `StructArray` child.
- New scalar arms: Date32, Time32/64 (by unit), Timestamp (by unit; TZ compared as UTC µs), Decimal128 (value + scale; compare against the column's decimal scale), Blob.
- NaN-compare semantics matching DuckDB's float **total order** (NaN sorts highest; `NaN == NaN` is true): `col = NaN` → true only where col is NaN; `col <> NaN` → true where col is non-null and not NaN; `col < NaN` / `<= NaN` / `> NaN` / `>= NaN` follow the total order. Nulls never satisfy a comparison.
- `compare` / `is_null` / `is_not_null` / `in` / `and` / `or` reused, now operating on the path-resolved leaf array.
- Fail-loud unchanged: unresolved path / unsupported scalar or column type / UNHANDLED node → `FilterError` → `error_stream` (the scan query errors cleanly; never a silently wrong row set).

### 4. API / registration

Unchanged from the `arrow-zerocopy` branch: `Connection::register_arrow(name, Vec<RecordBatch>) -> ArrowView<'conn>`, auto-unregister-in-Drop, projection pushdown, all Arrow data types carried in the stream.

### 5. Branch & cleanup

- Work on `arrow-zerocopy-fullfilter`, branched from `arrow-zerocopy` (v1.5.4). `arrow-zerocopy-duckdb-main` (the bump) is abandoned — left in place for the record, not built on.
- This spec's `Supersedes` header points back at the Phase B spec/plan; add a one-line "Superseded by …" note atop those two Daft docs.

## Correctness policy

Identical invariant to the existing walker: never an over/under-inclusive row set for a **translated** filter. `AND` may drop untranslatable conjuncts (engine re-applies above the scan); `OR` with any untranslatable branch → don't push the whole disjunction; `OPTIONAL`/`DYNAMIC`/`BLOOM` safe-skip; everything else → `error_stream`. Grounded in duckdb-python's `TransformExpression` semantics and the v1.5.4 structured model.

## Testing (Appender oracle + known data, native runner)

- All 6 comparison ops; `IS [NOT] NULL`; `IN`; `AND` / `OR`; nested `AND`/`OR`.
- **Struct field filters via both paths:** a `struct.field = x` that arrives as a structured `STRUCT_EXTRACT`, and one that arrives inside an `EXPRESSION_FILTER` as a `struct_extract` chain — single and nested struct depth.
- Scalar types: int widths, unsigned, float/double (incl. NaN), bool, Utf8, Blob, Date, Time, Timestamp (multiple units, incl. TZ), Decimal128 (with scale).
- **EXPRESSION_FILTER arbitrary predicate** that pushes down correctly — e.g. `k + 1 = 3` or `length(s) > 2` (whatever v1.5.4 routes through `EXPRESSION_FILTER`); confirmed against the oracle.
- Projection subset; **multi-column projection + filter** (the prior critical filter↔column mapping regression); self-join (replayability); inner join with a filter (dynamic/bloom skip stays correct).
- Negative: a genuinely untranslatable expression → clean query error (fail-loud), not a wrong result or crash; plus direct `filter::evaluate` unit tests for the new arms (Date/Time/Timestamp/Decimal128/NaN, path resolution).
- The whole `duckdb` crate test suite stays green on v1.5.4.

## Staging (always-correct increments — for the plan)

1. **Column paths + `STRUCT_EXTRACT`.** Migrate the wire format from `col:u32` to path encoding; add the `STRUCT_EXTRACT` structured branch; add Rust `resolve_path`. Existing filters keep working; struct-field filters now push. Branch green.
2. **Scalar breadth.** Widen `emit_scalar` and the Rust scalar arms to Date / Time / Timestamp / Decimal128 / Blob; implement NaN-compare semantics. Green.
3. **`emit_expression` (EXPRESSION_FILTER port).** Walk the v1.5.4 bound-expression tree (comparison / conjunction / operator / struct_extract / ref / const); fail-loud on the rest. Green.
4. **Docs / benchmark / lint / full `duckdb` suite green** + CHANGELOG entry.

Each stage is independently correct (UNHANDLED → error, never wrong); later stages widen coverage.

## Risks

- **NaN + Decimal128 scale** correctness — covered by explicit evaluator unit tests and oracle tests.
- **Column-path index semantics** (projected position vs. struct child index) — guarded by the multi-column and nested-struct tests; the prior critical bug here means the regression test is mandatory.
- **EXPRESSION_FILTER coverage breadth** — fail-loud keeps unknown expression shapes safe; coverage can widen later without correctness risk.
- **EXPRESSION_FILTER may be rarely emitted on v1.5.4.** The optimizer routes most predicates to structured kinds, so Stage 3 must first find a query that actually produces an `EXPRESSION_FILTER` for an arrow scan (e.g. an arithmetic/function predicate). If filter pushdown converts everything to structured kinds, the catch-all still stands as correctness-safe, but the test must construct a qualifying predicate — or the stage documents that none was found and the front-end remains a guarded fallback.
- **`struct_extract` naming** — handle both `struct_extract` and `struct_extract_at` on v1.5.4.
- **Possible further `duckdb` crate friction** — none expected (we are on the released bundle the crate already ships), but Stage 4 runs the full suite to confirm.

## Decision log

- Stay on released **v1.5.4** (drop the `main` bump) — chosen by the user after verification showed v1.5.4 already provides the same filter coverage with a stable build. Supersedes the Phase B / main-bump direction.
- **Approach A** (dual front-ends, unified column-path wire format, single Rust evaluator) over converting structured filters to expressions via `ToExpression` (B) or per-path wire formats (C) — reuses the proven structured walker, isolates the new expression walk, keeps one evaluator, and is forward-compatible.
- **Full** `EXPRESSION_FILTER` port (not safe-skip) — chosen by the user for true duckdb-python coverage parity.

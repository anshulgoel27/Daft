# duckdb-rs Arrow Filter Pushdown — EXPRESSION_FILTER Port (Phase B) Design

> **⛔ SUPERSEDED (2026-06-26)** by [v1.5.4 full-coverage design](2026-06-26-duckdb-rs-arrow-v154-fullfilter-design.md). The DuckDB-`main` bump this depends on is abandoned: verification showed released **v1.5.4 already provides the same filter coverage** (it has both the structured kinds and `EXPRESSION_FILTER`) on a stable build, so we extend the structured walker on v1.5.4 instead of porting onto unreleased `main`. Kept for the record only.

**Date:** 2026-06-26
**Status:** Superseded — do not implement
**Related:** [upstream design](2026-06-26-duckdb-rs-zerocopy-arrow-upstream-design.md) · branch `arrow-zerocopy-duckdb-main` (foundation commit `911fc15`) · reference: duckdb-python `/Volumes/Work/Code/duckdb-python/src/duckdb_py/arrow/filter_pushdown_visitor.cpp`

## Goal

Reach **filter-pushdown coverage parity with duckdb-python** for `Connection::register_arrow`, now that the bundled DuckDB is bumped to `main` (`06eb6b6`). On main, the structured `TableFilter` kinds are gone — every pushed filter arrives as an `EXPRESSION_FILTER` wrapping a `BoundExpression` tree. So the shim must be rewritten to walk that tree (a port of duckdb-python's `TransformExpression`), covering: constant comparisons, `IS [NOT] NULL`, `IN`, `AND`/`OR`, and **nested `struct_extract` column paths**; skipping optional/dynamic/bloom; failing loud on anything else. Plus broaden scalar coverage to all pushable types (Date/Time/Timestamp/Decimal128).

## Background

The foundation (`911fc15`) bumped libduckdb-sys to DuckDB `main` and fixed the build (C++17, jemalloc-exclude). On main: `TableFilterType` no longer has `CONSTANT_COMPARISON`/`IS_NULL`/`CONJUNCTION_*`/`IN_FILTER`/`OPTIONAL_FILTER`; `ConstantFilter`/`ConjunctionAndFilter`/`InFilter`/`OptionalFilter` are removed. `TransformFilter` (duckdb-python) handles **only** `EXPRESSION_FILTER`, delegating to `TransformExpression(const Expression&, …)`. Our current shim (structured-filter walker) therefore does not compile against main. The Rust evaluator (`compare`/`is_null`/`and`/`or`/`in` over a `BooleanArray` mask) is largely reusable; it gains a column-**path** concept for struct extract.

Verified main API (in `crates/libduckdb-sys/duckdb-sources`):
- `ExpressionFilter { unique_ptr<Expression> expr; }`.
- `BoundComparisonExpression::IsComparison(type)`, `::Left(BoundFunctionExpression&)`, `::Right(...)` — comparisons are `BoundFunctionExpression`s; one side is `VALUE_CONSTANT` (`BoundConstantExpression`), the other the column side (flip op if the constant is on the left).
- `BoundOperatorExpression` → `OPERATOR_IS_NULL`, `OPERATOR_IS_NOT_NULL`, `COMPARE_IN`.
- `BoundConjunctionExpression` → `CONJUNCTION_AND` / `CONJUNCTION_OR` over `GetChildren()`.
- `BoundReferenceExpression { idx_t index; }` (column-side leaf).
- `BoundConstantExpression::GetValue() -> Value`.
- `TryGetStructExtractChildIndex(const BoundFunctionExpression&, idx_t&)` (struct_extract column chains).
- `TableFilterFunctions::IsTableFilterFunction(name)` (dynamic/bloom/prefix-range/perfect-hash → skip); `OptionalFilterScalarFun::NAME` / `SelectivityOptionalFilterScalarFun::NAME` (wrap a child predicate in `BindInfo`).
- `CanPushdown` (main) pushes: bool, all int widths, DATE, TIME(+NS), TIMESTAMP(+MS/NS/SEC/TZ), unsigned, FLOAT, DOUBLE, VARCHAR/BLOB(non-view), DECIMAL **only when INT128**, and STRUCT (if all children pushable).

## Architecture

```
arrow_scan view scanned ── per scan ──► ShimProduce(factory, ArrowStreamParameters{projected_columns, filters})
  for each filter in *filters (each is an ExpressionFilter):
     emit_expression(expr_filter.expr, root_path={}, buf)   // ports TransformExpression
        BOUND_FUNCTION + IsComparison → resolve column path + constant → COMPARE  (NaN → NaN-compare)
        BOUND_FUNCTION + optional/selectivity-optional → emit child if translatable, else skip
        BOUND_FUNCTION + IsTableFilterFunction (dynamic/bloom/…) → skip
        BOUND_OPERATOR  IS_NULL / IS_NOT_NULL / COMPARE_IN → emit path-node
        BOUND_CONJUNCTION AND/OR → recurse (AND drops untranslatable children; OR fails whole)
        else → UNHANDLED
  → Rust rust_produce: decode → evaluate(node, batch) → BooleanArray mask → filter_record_batch
```
Still registers over the built-in `arrow_scan` (projection + filter pushdown on); the producer pre-applies the filter. Unhandled → `error_stream` (fail-loud, never silently wrong). Column references are **paths** (root projected column + zero or more struct child indices).

## Components

### 1. C++ shim rewrite (`crates/libduckdb-sys/src/arrow_zerocopy_shim.cpp`)
Replace the structured `TableFilterSet`/`ConstantFilter`/… walking with:
- `ShimProduce`: each entry in `*params.filters` is an `ExpressionFilter`; call `emit_expression(*ef.expr, root_path, buf)`. (The `TableFilterSet` is still keyed by projected column position; that key seeds the root path. Verify against `filter_to_col` as before — the multi-column regression test must pass.)
- `emit_expression(const Expression&, vector<idx_t> path, buf)` — port of `TransformExpression`:
  - `BoundComparisonExpression::IsComparison` → resolve the column side via `ResolveColumnPath` (BOUND_REF → path leaf; struct_extract `BoundFunctionExpression` chain via `TryGetStructExtractChildIndex` → append child indices), extract the constant `Value`; emit COMPARE{path, op, scalar}. NaN constant → emit a NaN-compare marker.
  - `BoundOperatorExpression` IS_NULL / IS_NOT_NULL → emit IS_NULL/IS_NOT_NULL{path}; COMPARE_IN → emit IN{path, scalars}.
  - `BoundConjunctionExpression` AND/OR → emit AND/OR with translatable children (AND: skip a child that throws NotImplemented; OR: if any child untranslatable, emit UNHANDLED for the whole OR — pushing a stricter predicate would be wrong).
  - `BoundFunctionExpression` whose name is optional/selectivity-optional → recurse into the `BindInfo` child if present+translatable, else skip (emit empty AND child); name `IsTableFilterFunction` (dynamic/bloom/…) → skip.
  - anything else → UNHANDLED{class}.
- `emit_scalar` widened to all `CanPushdown` types: bool, all int widths (i64/u64), FLOAT/DOUBLE (f64), VARCHAR (utf8), DATE (date_t.days → i32), TIME/TIMESTAMP variants (i64, unit tag), DECIMAL128 (i128 + scale). NaN floats flagged.

### 2. Wire format extension (column path + new scalars)
- Replace single `col:u32` in COMPARE / IS_NULL / IS_NOT_NULL / IN with a **path**: `depth:u32` then `depth × u32` indices (depth ≥ 1; index[0] = projected root column, index[1..] = struct child indices).
- Scalars gain: `stype 6 = i128` (Decimal128, + `scale:u8`), date/timestamp carried as their physical int (i32/i64) with a unit tag where needed; a `nan:u8` flag (or a dedicated NaN-compare node) for float NaN comparisons.

### 3. Rust evaluator (`crates/duckdb/src/arrow_zerocopy/filter.rs`)
- Decode the path (`Vec<usize>`) in each node; `resolve_path(batch, path) -> ArrayRef` walks `StructArray` children for `path[1..]`.
- `compare`/`is_null`/`is_not_null`/`in` operate on the resolved leaf array (reuse existing logic). Add scalar arms for Date32, Time32/64, Timestamp(unit), Decimal128(scale); NaN-compare semantics matching duckdb-python (`= NaN` → false; `!= NaN` → true, etc.). And/Or reused unchanged.
- Fail-loud unchanged: unresolved path / unsupported scalar/type / UNHANDLED → `FilterError` → `error_stream`.

### 4. API / registration
Unchanged from the upstream branch: `Connection::register_arrow(name, Vec<RecordBatch>) -> ArrowView<'conn>`, auto-unregister in Drop, projection pushdown, all Arrow data types in the stream.

## Correctness policy
Identical invariant: never an over/under-inclusive row set for a translated filter; OR with an untranslatable child → don't push (whole OR → UNHANDLED); AND may drop untranslatable children (engine re-checks above); optional/dynamic/bloom safe-skip; anything else → `error_stream`. Grounded in duckdb-python's `TransformExpression` semantics.

## Testing
Against the Appender oracle + known data:
- comparisons (all 6 ops), `IS [NOT] NULL`, `IN`, `AND`, `OR`, nested `AND/OR`.
- **struct column**: filter on `struct.field` (`struct_extract` path) — single and nested.
- scalar types: int widths, unsigned, float/double (incl. NaN), bool, Utf8, Date, Timestamp, Decimal128.
- projection subset; **multi-column** projection+filter (the prior critical-bug regression test); self-join (replayability); inner join with filter (dynamic/bloom skip stays correct).
- negative: an untranslatable expression → clean query error (fail-loud), not wrong/crash; plus direct `filter::evaluate` unit tests for arms DuckDB applies above the scan.
- The whole `duckdb` test suite still green on main (the bump didn't regress non-arrow paths).

## Staging (always-correct increments — for the plan)
1. **Shim skeleton on main**: rewrite `ShimProduce` + `emit_expression` for `BoundComparison` (column = BOUND_REF only) + constant; reuse Rust `compare`. Erroring (UNHANDLED) on every other expression class. Branch compiles + green on main; the structured code is gone.
2. **Null / IN / conjunction**: `BoundOperator` (IS_NULL/IS_NOT_NULL/IN), `BoundConjunction` (AND/OR with the drop/fail policy), optional/dynamic/bloom skips.
3. **Struct-extract column paths**: `ResolveColumnPath` + wire-format path + Rust `resolve_path`.
4. **Scalar type breadth**: Date/Time/Timestamp/Decimal128 + NaN-compare.
5. **Docs / benchmark / lint / full-suite green** + CHANGELOG.

Each stage is correct (UNHANDLED→error, never wrong); later stages widen coverage.

## Risks
- **Main is unreleased + moving**: pinned to `06eb6b6`; not upstream-mergeable as-is (documented; this is an exploration to reach python parity). Re-pin/rebase as needed.
- **Expression API drift on main** between this commit and any later re-pin.
- **NaN + Decimal128 scale** correctness — explicit tests.
- **Column-path index semantics** (projected position vs struct child) — the multi-column + struct tests guard it.
- Possible **further `duckdb` crate breakage on main** beyond the shim (run the full suite in stage 5).

## Decision log
- Bump to DuckDB `main` (`06eb6b6`, matching duckdb-python) — chosen by the user over adapting on stable.
- Port duckdb-python's `TransformExpression` coverage exactly (compare/null/in/and-or/struct-extract; skip optional/dynamic/bloom; fail-loud rest).
- Keep `register_arrow`/`ArrowView`/auto-unregister + the wire-format/Rust-evaluator architecture; extend with column paths + new scalar types.

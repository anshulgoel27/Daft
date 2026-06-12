# Local Setup and Troubleshooting Notes (macOS arm64)

This guide records a real setup and troubleshooting session so new contributors can set up quickly and diagnose common local failures.

## Scope

- Repository: Daft fork
- OS: macOS arm64
- Primary goal: get local development working with `make .venv`, `make build`, and targeted tests

## Recommended Setup Path

1. Install prerequisites:
   - `uv`
   - Rust toolchain (`rustup`)
   - Node.js
   - `cmake`
   - `protoc`
2. Create the environment:
   - `make .venv`
3. Build locally:
   - `make build`
4. Run tests locally:
   - `DAFT_RUNNER=native make test EXTRA_ARGS="-v <test-path-or-node>"`

## Commands Used During Debugging

### Environment and repo checks

- `git remote -v`
- `test -d .venv && echo '.venv exists' || echo '.venv missing'`
- `command -v uv`
- `command -v rustup`
- `command -v node`
- `command -v cmake`
- `command -v protoc`

### Dependency and setup commands

- `uv venv .venv -p python3.11`
- `uv sync --no-install-project --all-extras --all-groups`
- `uv sync --no-install-project --all-extras --all-groups --no-group dev`

### Build and validation commands

- `make build`
- `PYO3_PYTHON=.venv/bin/python .venv/bin/maturin develop --group lint`
- `python -c "import daft; print(daft.__version__)"`

## Errors, Root Cause, and Fixes

### 1) `nvidia-cudnn-frontend` wheel resolution failure on macOS arm64

Observed during:

- `uv sync --no-install-project --all-extras --all-groups`
- `make .venv` (same underlying sync path)

Root cause:

- `vllm==0.21.0` in the dev dependency group pulled Linux and Windows CUDA transitive packages.
- On macOS arm64, those wheels are unavailable, so resolution failed.

Fix:

- Add a platform marker in `pyproject.toml`:
  - from: `"vllm==0.21.0"`
  - to: `"vllm==0.21.0; sys_platform == 'linux'"`

Why this fix:

- Keeps build logic generic in Makefile.
- Encodes platform compatibility at dependency definition level.
- Preserves Linux behavior while unblocking macOS local setup.

### 2) `tokenspeed-mla==0.1.2` no matching wheel during `make build`

Observed during:

- `make build`

Root cause:

- Same dependency chain from `vllm` in dev dependencies.

Fix:

- Same marker change in `pyproject.toml`:
  - `"vllm==0.21.0; sys_platform == 'linux'"`

Result:

- `make build` can resolve dependencies on macOS arm64.

### 3) Rust compile failure (`E0308`) in minhash remainder handling

Observed in:

- `src/daft-minhash/src/lib.rs`
- Around `chunks.into_remainder()`

Root cause:

- Remainder handling needed to match current iterator behavior for this toolchain.

Fix:

- Replace direct loop over `chunks.into_remainder()` with an `if let Some(remainder)` guard and iterate inside it.

Result:

- Minhash crate compiles successfully with the pinned toolchain.

## Code Changes from Session

1. `pyproject.toml`
   - `vllm` marked Linux-only in dev dependencies.
2. `src/daft-minhash/src/lib.rs`
   - Remainder handling updated for iterator compatibility.

## Verification Checklist

1. `make .venv` completes.
2. `make build` completes.
3. `python -c "import daft; print(daft.__version__)"` prints a valid dev version.
4. At least one targeted native test runs:
   - `DAFT_RUNNER=native make test EXTRA_ARGS="-v <test-path-or-node>"`

## Quick Troubleshooting Playbook

1. If setup or build fails with CUDA, vllm, or tokenspeed resolution on macOS:
   - Verify `pyproject.toml` includes `"vllm==0.21.0; sys_platform == 'linux'"`.
2. If build fails in minhash with remainder type mismatch:
   - Verify `src/daft-minhash/src/lib.rs` uses `if let Some(remainder) = chunks.into_remainder() { ... }`.
3. Re-run in order:
   - `make .venv`
   - `make build`
   - targeted test command

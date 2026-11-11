---
name: verify
description: Runs the checks for the current branch of sciml-rs, chosen by what it changed - format, Clippy, tests, docs, the datagen and metal checks, spikes, and the bit-identity protection of an objective's constraint 1 - and reports exactly what ran. Use before calling any task done, and from /next-task.
---

# Verify

Runs what root `CLAUDE.md` ("Checks") requires for the change and reports each command with
its result. An unrun check is never reported as passing.

Adapted from the `/verify` skill of `tbetcke/fmm` (MIT or Apache-2.0).

## 1. What changed

`git diff --name-only main...HEAD` and the uncommitted files. If a task is in progress,
read its plan (`docs/<objective>/tasks/<ID>.md`): "Checks" and "VM runs" add to what
follows. A change to Markdown only needs the link check of §3 and nothing else.

## 2. Always (any Rust change)

```sh
cargo fmt -- --check
cargo clippy --no-deps -- -D warnings
cargo clippy --no-deps --examples -- -D warnings
cargo clippy --all-targets -- -D warnings
cargo test
cargo doc --no-deps
```

## 3. By what changed

- **`datagen` code** (`chebyshev/` behind it, `pde/`, `examples/generate/`, `metrics/pod`)
  or anything they call: `cargo clippy --no-deps --features datagen --all-targets -- -D
  warnings` and `cargo test --features datagen`.
- **Backend-sensitive code** (tensor ops in `layers/`, `models/`, `utils/fft.rs`, the
  loaders' device path): the touched tests with `--features metal`, and
  `cargo test --features metal --test gpu_parity`. Say whether each ran.
- **`mlflow` code** (`tracking/`): `cargo clippy --no-deps --features mlflow --all-targets
  -- -D warnings` and `cargo test --features mlflow`.
- **cuda-only paths**: on the VM (`ssh cricket`, `source ~/sciml-gpu.sh`), the tests the
  plan names with `--no-default-features --features cuda`; pending until they have run.
- **Spikes**: their own tests (`cargo test -p <spike>` for workspace members, else in the
  spike's directory), reported apart.
- **`Cargo.toml` or `Cargo.lock`**: `cargo tree -e normal` shows no unexpected new
  dependency; the Burn fork pin unchanged unless the plan says so.
- **Docs**: every relative Markdown link in the changed files resolves.
- **Exact test filters** need the full module path; check the "running N tests" line, as a
  short path runs 0 tests and still reports `ok`.

## 4. Constraint 1

- **`laplace3d`** (every 2D output bit for bit unchanged): if the change touches code the 2D
  objective runs (`chebyshev/`, `metrics/poisson`, `pde/poisson/`, `data/loaders/poisson/`,
  the Poisson examples), the bit-identity tests the plan names, on fixed 2D inputs, against
  the values on `main`.
- **`operators`** (every existing FNO output bit for bit unchanged): if the change touches
  `layers/`, `models/`, `training/`, `runs/` or `losses/`, the existing FNO tests unedited
  and the bit-identity tests on a fixed FNO forward and on one Phase 5-shaped training epoch
  with the default options, against `main`.

## 5. Report

A table: command | where | result (passed; failed, with the first error; not run, and
why). Fix failures and run again. Never weaken, skip or delete a test to get green; if a
test seems wrong, explain why and ask.

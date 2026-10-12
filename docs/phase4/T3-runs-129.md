# T3: production runs at 129² on cuda

Phase 4, task 3. Branch `phase4/T3-runs-129`. Depends on T2 (the chosen p).

## Read first
- Design §5.3 (stage 3, storage), §7, §9 (backends, the `chebyshev` feature on Linux),
  §10 (Phase 4), §12 decisions 16, 17 and 18.
- `examples/generate/poisson.rs` (`DATASETS`), `examples/train/poisson.rs`,
  `examples/predict/poisson.rs`; `datasets/README.md`.
- `CLAUDE.md` Burn (backends, `BURN_DEVICE`); README "Backends" (the cuda command line).

## Goal
The first production runs: train at 129² on an NVIDIA GPU and report accuracy and
performance with the five errors.

## Deliverables
1. **Stage selection.** `train_poisson` and `predict_poisson` take the stage from an
   argument, `-- [<padding>] [<n>]` for training (n ∈ {65, 129, 257}, default 65),
   reading file names from one small table (n, K, train file, test file, test sidecar)
   in each example. `predict_poisson` takes n from the run's `data_cfg.json`. Stage 2
   behaviour and the existing commands do not change.
2. **Stage 3 dataset** (n = 129, K = 64): generated with `--features chebyshev`
   (locally, then copied, or on HPC); label check and error 2 recorded.
3. **Runs at 129² on cuda** (`--no-default-features --features cuda`) at T2's p, full
   length, Darcy hyperparameters with `modes [12, 12]` unchanged. At least one run;
   a second seed if time allows. Train / test relative L², s/epoch and the GPU model.
4. **`predict_poisson`** on the 129² run(s): the five errors and the boundary verdict.
5. A short note in `README.md` or `datasets/README.md` on running on HPC (build flags,
   where the datasets go, `BURN_DEVICE`), only as far as it was actually needed.

## Tests
Examples have no test harness, so the stage table stays a small `const` in each
example and is checked by running stage 2 unchanged. If the lookup moves into the
library, it gets a unit test there. `train_poisson_smoke` still passes.

## Acceptance
- The T3 rows of `docs/phase4/README.md` Results filled in.
- If the 129² error is not below or near the 65² error, it is recorded in design §12
  before anything is retuned.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`; the cuda
build (`cargo build --release --no-default-features --features cuda --examples`) on the
HPC machine. Say which checks ran where.

## Do not
- Change modes, width, layers or the optimiser: a tuned 129² model is out of scope.
- Commit datasets, runs or HPC job scripts with account names.
- Start the resolution study (T4).

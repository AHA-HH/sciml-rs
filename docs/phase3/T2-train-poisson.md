# T2: `train_poisson` and the first 65² run

Phase 3, task 2. Branch `phase3/T2-train-poisson`. Depends on Phase 2 (the stage 2
dataset and the loader); can run in parallel with T1.

## Read first
- Design §5.3 (stages), §7 (training, padding), §10 (Phase 3), §11 (the pipeline row),
  §12 decisions 2, 14, 15 and 16.
- `docs/CONVENTIONS.md` §3 (padding), §10 (normalisation) and §11 (`LpLoss`).
- `examples/train/darcy.rs` and `src/neural_operators/training/trainers/darcy.rs`, the
  model to follow; `training/trainer.rs` (`TrainingConfig`, `build_training_components`,
  `training_loop`); `metrics/io.rs` (`write_run_artifacts`).
- `src/neural_operators/data/loaders/poisson.rs` (`PoissonConfig`, `load_poisson_uniform`,
  `PoissonNormalizers`) and its tests' `.npz` fixtures.
- `datasets/README.md` (Poisson section) and `CLAUDE.md` (Burn: the autodiff device note).

## Goal
The first FNO trained on Chebyshev data: the stage 2 set at 65², transferred to the
uniform 64² grid, with padding chosen by measurement, reaching test relative L² ≤ 2e-2
on metal and flex.

## Deliverables
1. **`src/neural_operators/training/trainers/poisson.rs`**: `train_poisson<T: HostFloat>(
   train, test, y_normalizer, model_cfg, train_cfg, device) -> (FNO<4>, Vec<EpochMetrics>)`,
   the same as `train_darcy` apart from its name and docs: decode both sides in training,
   only the prediction in evaluation (CONVENTIONS §10); `R = 4`, `RM1 = 3`. Registered in
   `trainers/mod.rs`.
2. **`examples/train/poisson.rs`**, registered in `Cargo.toml` as `train_poisson` with no
   required features:
   - Literals at the top: `N = 65`, 1000 train and 200 test (the whole test split), the
     dataset paths `datasets/poisson/poisson_n65_K32_{train,test}.npz`, the Darcy model
     and training hyperparameters (design §7) and `PADDING`, the chosen p.
   - The padding can be overridden by an optional first argument
     (`cargo run --release --example train_poisson -- 8`), so the sweep needs no edits.
     Document this in the file header.
   - Asserts that both files exist, pointing to `datasets/README.md`.
   - Loads with `load_poisson_uniform::<f32>` and trains on
     `Device::default().autodiff()`.
   - Writes `runs/poisson_fno_p{p}_<unix_secs>/` with `write_run_artifacts` (the
     `PoissonConfig` is the data config), both normalizers and the weights, as
     `train_darcy` does.
3. **The padding sweep.** p ∈ {0, 4, 8, 16}, 500 epochs each, on metal. The lowest final
   test relative L² wins, and values within 2% of each other go to the smaller p
   (decision 16). Set `PADDING` to the winner.
4. **The flex run.** The chosen p, full length, on flex (`BURN_DEVICE` or a build without
   `metal`).
5. Docs: the trainer and the example in the Layout and Commands sections of `CLAUDE.md`.

## Tests (`#[cfg(test)] mod tests` in `trainers/poisson.rs`)
- `train_poisson_smoke`: write a tiny synthetic stage (n = 9, 4 train and 2 test
  samples) to a fresh directory under `std::env::temp_dir()`, following the loader tests'
  fixtures (their helpers are private; write a small one here). Load it, train 2 epochs
  on Flex's autodiff device with a 2-layer, width-4, modes [2, 2] FNO, and check that
  every metric is finite and that two epochs were recorded. Keep it under a second in
  debug.

## Acceptance
- All tests pass under `cargo test` and `cargo test --features chebyshev`, on macOS and
  in both CI jobs.
- Final test relative L² of the chosen p ≤ 2e-2 on metal and on flex (the §11 pipeline
  row). If either misses, stop: record the numbers in the design's §12 and ask before
  retuning anything.
- The T2 rows of Results in `docs/phase3/README.md` are filled in: every sweep value with
  backend and wall time, the chosen p, both full runs with time per epoch, and the test
  wall time. Say which run directories T3 should evaluate.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`, then the
same clippy and test commands with `--features chebyshev`. Give the exact test filter
(`neural_operators::training::trainers::poisson::tests::train_poisson_smoke`) and check
the "running N tests" line. Training is backend-sensitive: run the smoke test with
`--features metal` as well, and report which backend each training run used.

## Do not
- Change the FNO, `trainer.rs`, the Darcy trainer or example, the loss or the loader.
- Change the Darcy hyperparameters other than padding, or tune after seeing the threshold
  result without recording it first.
- Add evaluation on the Chebyshev grid or read the raw `u` (T3).
- Commit datasets or run directories.

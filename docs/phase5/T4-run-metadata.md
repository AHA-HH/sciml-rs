# T4: run metadata and costs

Phase 5, task 4. Branch `phase5/T4-run-metadata`. Depends on T3.

## Read first
- Design §13.3 (Measurements) and §13.9 (Metadata and output files).
- `src/neural_operators/runs/` (`RunDir`, `EpochMetrics`, the CSV writer and reader).
- `training/trainer.rs` (`training_loop`, `EpochObserver`, `train_fno`) and
  `training/config.rs` (`TrainingConfig::seed`).
- `examples/train/poisson.rs`, `examples/predict/poisson.rs`.
- Burn's `Module::num_params` in the fork (`~/.cargo/git/checkouts/burn-*/ddfa9af/`):
  confirm it exists before relying on it.

## Goal
Every Phase 5–7 run records what it was (grids, transfers, seeds) and what it cost, so the
T6 comparison reads files, not terminal logs.

## Deliverables
1. **`metrics.csv`** gains `epoch_time_s`, from the `Duration` the observer already
   receives. `RunDir::metrics()` still reads the old five-column files (the field is
   `Option` or defaults to NaN; document which).
2. **`costs.json`,** written and read only through `RunDir`:
   - dataset generation time (the train and test sidecars' `wall_time_s`);
   - load-and-transfer time, total training time;
   - inference time on the test split, warm (written by `predict_poisson`; absent until
     then);
   - host-to-device bytes, computed: N × s² × (in + out channels) × 4 per split;
   - peak GPU memory, or the string `"unavailable"`; never estimated. On cuda the value
     may be filled from `nvidia-smi` by hand and the PR says so;
   - backend features (`backend_features`), device name, git commit;
   - trainable parameter count (`num_params`).
3. **`data_cfg.json`** carries n, K, s, the restriction and its parameters, P's d, and the
   train and test file names with their sidecar git commits. Prefer fields on
   `PoissonConfig` (T3) over a second file.
4. **`train_poisson -- [<padding>] [<n>] [<s>] [<seed>]`:** the seed sets
   `TrainingConfig::seed` (default 42) and appears in the run name when not 42.
5. `CLAUDE.md`: the command line and the `runs` Layout line.

## Tests
- `metrics()` reads an old five-column CSV and a new six-column one.
- `costs.json` round-trips through `RunDir`.
- `run_artifacts_without_gnuplot` still passes.

## Acceptance
- A 2-epoch stage 2 run writes every item above; record the files in the PR.
- `predict_poisson` on `runs/poisson_fno_p4_1791922918` (or another existing run) still
  prints its five errors unchanged.
- The T4 Results row filled in.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`.

## Do not
- Add checkpointing or resume (Phase 9).
- Rename or remove any existing run file or column.
- Measure GPU memory by estimate.

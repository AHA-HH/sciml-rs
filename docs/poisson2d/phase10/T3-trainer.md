# T3: trainer integration

Phase 10, task 3. Branch `phase10/T3-trainer`. Depends on T2 and Phase 9 T1 (done: `checkpoint.bin`).

## Read first
- This phase's README (G1, G3–G7, "Module layout") and design §13.17.
- `src/neural_operators/training/{config,trainer,step}.rs` (`TrainingConfig`,
  `TrainingComponents`, `train_epoch`, `training_loop`, `forward_loss`, `flatten_pair`).
- `src/neural_operators/runs/{mod,checkpoint,costs,csv}.rs` (`RunDir`, `checkpoint.bin`,
  `RunCosts`, `peak_gpu_memory`, `RunDir::artifacts`, which `tracking/builder.rs` uploads).
- `examples/train/poisson.rs` (literals, `TRAIN_DEVICES`, `--resume`).
- `docs/CONVENTIONS.md` §9–§11.

## Goal
`train_poisson` trains the FNO with Gauss–Newton when asked, records what the
experiments need, and resumes; with nothing asked, everything is as before, bit for bit.

## Deliverables
1. **`TrainingConfig.optimizer: Option<OptimizerConfig>`**, an enum with `Adam` (today's
   settings) and `GaussNewton(GaussNewtonConfig)`. `None` means Adam, so every saved
   `training_cfg.json` still loads (the `FNOConfig::padding` pattern).
2. **`FnoResidual`** in `training/`, implementing `LeastSquares<FNO<R>>` with
   `Batch = Batch<R, RM1>`: forward, `flatten_pair`, decode, rᵢ = (ûᵢ − uᵢ)/‖uᵢ‖, scaled
   by 1/√B (G1); the per-sample weight kept as one place a later arm (C) can change.
3. **`train_epoch_gn`** beside `train_epoch`: per batch, one `GaussNewton::step`, the
   baseline loss L accumulated for `metrics.csv`'s `train_l2` (and `train_mse`) as today;
   with a gradient batch larger than the loader's (G3, 200), consecutive loader batches
   are concatenated or the loader built at that size (pick one, say why). The branch in
   `training_loop` picks it; `train_epoch` is not edited. GN with replicas panics with a
   clear message (data-parallel GN is out of scope).
4. **`gn_steps.csv`**: one row per GN step (every `GnStep` field, epoch, step, wall
   time), written by `RunDir::append_gn_step`, the only writer, and listed in
   `RunDir::artifacts` so the MLflow export uploads it.
5. **`costs.json`**: forward-equivalent passes (counted, Adam included: forward plus
   backward per step). Old files still read. `peak_gpu_memory` stays as Phases 5 and 7
   record it: `nvidia-smi` sampled beside the run on cuda, filled in afterwards (no crate
   code; `runs/costs.rs`), so GN and Adam runs are measured the same way.
6. **Checkpoints:** the GN state (T2's record) appended to `checkpoint.bin` as an optional
   section; Adam checkpoints and old files decode unchanged.
7. **`train_poisson`:** GN literals at the top of the file; `OPTIMIZER=gauss-newton`
   (environment, like `TRAIN_DEVICES`) selects GN; the run name gains `_gn`. Positional
   arguments unchanged. `--resume` reads the optimiser from the saved config.
8. **CONVENTIONS §11:** the GN residual (A) and its weights, noted in the PR (§11 does not
   bump `CONVENTION_VERSION`).
9. **cuda f64 check (G7):** on the VM, whether the tiny FNO runs in f64 end to end; the
   answer in the PR.

## Tests (flex)
- An old `training_cfg.json` (without `optimizer`) loads as Adam, and a run from it
  trains bit for bit as before (`metric_bits` against a fixed run).
- Every existing trainer, run-dir and checkpoint test passes unedited.
- `FnoResidual`'s ½‖r‖² summed over a batch equals (1/2B) Σᵢ eᵢ² from `LpLoss` with
  `Reduction::None`, to round-off.
- A GN smoke run (`train_poisson_smoke`'s fixture) writes `metrics.csv`,
  `gn_steps.csv` and `costs.json`; stopped and resumed, it matches an uninterrupted GN
  run bit for bit on flex.

## Acceptance
- On the VM (one A100): a 2-epoch GN run of stage 3, s = 64, N = 100; `predict_poisson`
  evaluates it unchanged; report `gn_steps.csv`'s first rows and `costs.json`.
- One Adam run of the baseline for a few epochs on the Phase 10 commit, `metrics.csv`
  compared with the same epochs of a Phase 7 run's (cuda round-off).

## Checks
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`; the
trainer tests with `--features metal` if available; the cuda build on the VM. Say which
ran.

## Do not
- Edit `train_epoch`, `forward_loss`, the Adam settings or the evaluation.
- Change `metrics.csv`'s columns.
- Run the T4 or T5 experiments here.

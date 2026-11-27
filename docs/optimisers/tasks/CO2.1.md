# CO2.1 The L-BFGS arm

Started 2026-11-27 on `optimisers/co2-1`. Roadmap: "`OptimizerConfig::Lbfgs`,
`train_epoch_lbfgs` with the closure over Φ on the full batch, passes into `costs.json`, the
checkpoint section, `OPTIMIZER=lbfgs` in `train_poisson`." Proposal: §10 (O2); §1.4, §2
(Burn's L-BFGS), §3 (passes, D7), §4, §6, §8 (R1, R2), §9 (L-BFGS); decisions D3, D4, D7,
D13.

## Goal
Train the FNO with Burn's `LBFGS` on Gauss–Newton's objective Φ (+ ½μ‖θ‖²), full batch, as
an arm of `train_poisson` that checkpoints, resumes exactly and counts its passes.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)
"tests of proposal §9 pass; a flex smoke run and its resume; constraint 1 shown."
Proposal §9: "the trainer integration on a small least-squares problem (it converges, the
passes add up, resume is exact); Burn's own tests cover the algorithm." §9, constraint 1:
"the existing suite unedited, … an Adam and a Gauss–Newton run on flex bit for bit equal
before and after each change."

## Read first
`training/{config,gn,trainer}.rs`, `runs/{checkpoint,gn_steps,mod}.rs`,
`examples/train/{common,poisson}.rs`; Burn's `burn-optim-0.23.0-pre.2/src/optim/lbfgs.rs`
(`LBFGSConfig`, `LBFGS::step`, `to_record`/`load_record`).

## Changes
- `training/config.rs`: `LbfgsArm { lbfgs: LBFGSConfig, learning_rate (1), weight_decay
  (μ, 0) }`, `OptimizerConfig::Lbfgs(LbfgsArm)`, `TrainingConfig::lbfgs()`. Old configs load
  unchanged (a new variant only).
- `training/lbfgs.rs` (new): `train_epoch_lbfgs`, one `LBFGS::step` per epoch on the one
  full batch (D4; panics if the loader yields more than one batch), the closure Φ =
  ½‖r‖² of `FnoResidual` (D3) plus ½μ‖θ‖² when μ > 0 (D13), counting its evaluations;
  the epoch's train metrics at the incoming parameters as `train_epoch_gn` measures them
  (one forward without autodiff); `last_lr` is the arm's learning rate.
- `training/trainer.rs`: `TrainingComponents::lbfgs`, built from the config; the loop
  dispatches to `train_epoch_lbfgs` (one device; no hybrid), checkpoints and resumes its
  state. The Adam and Gauss–Newton branches untouched. `run_loop`'s signature is kept (an
  existing test calls it); the step-row sinks go to an inner function.
- `runs/lbfgs_steps.rs` (new): `lbfgs_steps.csv`, one row per step (= epoch): epoch, Φ
  before and after, forward and backward passes (D7: one each per closure evaluation), wall
  time; `RunDir::{append_lbfgs_step, truncate_lbfgs_steps, lbfgs_passes}` (summing reuses
  `gn_steps`' column reader, truncation `csv::truncate_metrics_csv`); in `artifacts()`.
- `runs/checkpoint.rs`: an L-BFGS checkpoint writes an empty fourth (Gauss–Newton) blob and
  a fifth, `LBFGS::to_record`'s bytes; `OperatorCheckpoint::lbfgs`. Adam and Gauss–Newton
  checkpoints are byte for byte as before, and old files decode.
- `examples/train/common.rs`: `OPTIMIZER=lbfgs` (where the example allows it), its
  literals (`Literals::lbfgs`, `None` for Burgers and Darcy), `LBFGS_HISTORY`,
  `LBFGS_MAX_ITER`, `LBFGS_WEIGHT_DECAY`, `TIME_BUDGET_S` allowed; `run_costs`: steps from
  `lbfgs_steps.csv`, passes = its sums plus one metrics forward per step (as Gauss–Newton);
  `reopen` truncates `lbfgs_steps.csv`.
- `examples/train/poisson.rs`: `lbfgs` allowed, literals (strong Wolfe, lr 1, history 100,
  `max_iter` 20: D5's recommendation pending the CO2.2 pilot; μ = 0), the full batch = the
  run's N, the run name `_lbfgs`. Minimal edits (CO0.4 will touch this file).
- `examples/train/{burgers,darcy}.rs`: `lbfgs: None` in their `Literals` only.
- Conventions: none.

"Updates" (inner iterations that move θ) are not counted: Burn's `LBFGS` keeps its
iteration count private, and reading it through `to_record` would copy the whole history
to the host each step. `optimisation_steps` is the number of `step` calls (epochs); a
question for CO2.2.

## Tests that prove it
- Config: an `Lbfgs` arm round-trips through `train_cfg.json`; old configs still load as
  Adam (existing test, unedited).
- `train_epoch_lbfgs` on the tiny fixture: Φ falls every step and by a large factor over a
  few epochs (strong Wolfe guarantees decrease); with no line search and `max_iter` k each
  step costs exactly k + 1 closure evaluations, so the recorded passes are (k + 1) per
  step, both forward and backward; Φ with μ > 0 equals Φ at μ = 0 plus ½μ‖θ‖² (`ParamVec`
  norm) to f32 round-off; more than one batch panics.
- Smoke and resume (training loop, `train_poisson_smoke`'s fixture): 3 epochs uninterrupted
  against killed in epoch 1 and resumed from epoch 0's checkpoint: `metrics.csv`,
  `lbfgs_steps.csv` (but wall times) and the weights bit for bit equal on flex; the
  checkpoint holds the L-BFGS section and no Gauss–Newton state; replicas panic.
- Checkpoint: an Adam and a Gauss–Newton checkpoint encode as before (existing tests
  unedited); an L-BFGS record round-trips.
- Constraint 1: the existing suite unedited; `train_poisson` Adam (`EPOCHS=2`, N = 20) and
  Gauss–Newton (N = 20, `GN_BATCH_SIZE=10`, CG cap 5, one epoch) on flex at `origin/main`
  (`655fabc`) and at the branch head: `metrics.csv`, `gn_steps.csv` (but wall times),
  configs, `costs.json` counts, `checkpoint.bin` and the weights' tensor data equal.

## Checks
`/verify` (fmt, the clippy runs, `cargo test`, `cargo doc`, datagen clippy and tests); the
flex smoke run `OPTIMIZER=lbfgs EPOCHS=3 train_poisson -- 4 65 64 42 20 0` and its resume
(killed after its first checkpoint with a checkpoint interval lowered for the test, or the
library test above standing for it).

## VM runs
None (CO2.2 pilots the settings on the VM).

## Constraints and risks
- Constraint 1 (§1.4): additive variant, new module, the existing branches unchanged; shown
  as above.
- R1 (f32 stalls): Φ is evaluated in f32 and passed to Burn as f64; reported, not fixed.
- R2 (memory): the history is `history_size` × 2 vectors of the parameter count on the
  device, and each checkpoint copies it to the host (1.9 GB at history 100 for the 2D FNO).
- The training loader shuffles the one batch each epoch, so Φ's sum order changes and its
  value moves at f32 round-off between epochs; deterministic, so resume stays exact.

## Pending runs
None.

## Open questions
None yet.

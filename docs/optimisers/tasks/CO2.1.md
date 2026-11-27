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

## Outcome (2026-11-27)

All on the Mac (M2 Pro), flex, release builds; scratch run directories, not kept.

- **Tests:** `training::lbfgs::tests` (converges under strong Wolfe, Φ below 0.1× its start in
  five steps; `max_iter` + 1 evaluations per step without a line search, and Burn's
  `max_eval` cap; Φ + ½μ‖θ‖² against the `ParamVec` norm; mini-batches refused),
  `training::trainer::tests::{lbfgs_smoke_run_resumes_exactly, lbfgs_with_replicas_panics}`,
  `runs::lbfgs_steps::tests::rows_sum_and_truncate`,
  `training::config::tests::lbfgs_arm_round_trips`. The same eight pass with `--features
  metal`.
- **Smoke run:** `OPTIMIZER=lbfgs EPOCHS=3 train_poisson -- 4 65 16 42 20 0` (s = 16 rather
  than the plan's 64, to keep flex's evaluation short): Φ 0.541 → 0.119 → 0.030 → 0.019,
  train L² 1.007 → 0.235, test L² 0.532 → 0.310; `costs.json` 3 steps, 69 backward and 72
  forward passes (69 closure evaluations and 3 metrics forwards). Reproduced exactly at
  `bc1b290`.
- **Resume:** `OPTIMIZER=lbfgs EPOCHS=27 LBFGS_MAX_ITER=3 LBFGS_MAX_EVAL=10`, the same
  arguments, uninterrupted against killed (`kill -9`) after its epoch-25 checkpoint and
  resumed with `--resume`: `metrics.csv` and `lbfgs_steps.csv` equal but for the times, the
  weights' tensor data equal (only the 2,408-byte `param_id` header differs), `costs.json`
  equal but for the times and `resumes` (1). Φ was still falling over the resumed epochs
  (0.01139 → 0.01015).
- **A stall found:** the first resume attempt, at `LBFGS_MAX_ITER=2`, stalled for good at
  epoch 4 (Φ 0.343 at every later step). Burn's default `max_eval`, ⌊1.25 `max_iter`⌋ = 2,
  leaves the strong Wolfe search one evaluation; when it fails it returns t = 0, the
  curvature pair is zero and never stored, and every later step repeats the same search.
  At `max_iter` 1 the default leaves it none. `LBFGS_MAX_EVAL` (and the literal) were added;
  the CO2.2 line now asks for `max_eval` with each cell.
- **Constraint 1:** `train_poisson` Adam (`EPOCHS=2 … -- 4 65 64 42 20 0`) and Gauss–Newton
  (`OPTIMIZER=gauss-newton GN_BATCH_SIZE=10 GN_CG_MAX_ITERS=5 TIME_BUDGET_S=0.001`, the same
  arguments, one epoch of two steps) at `655fabc` and at `94d4736` and `bc1b290`:
  `metrics.csv` and `gn_steps.csv` equal but for the times; configs and normalisers
  byte-identical; `costs.json` equal but for times and the commit; the weights' tensor data
  equal; `checkpoint.bin` decoded blob by blob: state, weights, scheduler equal, Adam's
  record equal as a multiset of (state, shape, data) (its layout follows Burn's random
  `ParamId`s), the Gauss–Newton state equal with its `ParamId`s zeroed. The existing suite
  passes unedited (no removed or changed line in a test module).
- **Checks:** `/verify`: fmt, the three clippy runs, `cargo test`, `cargo doc` (no
  warnings), datagen clippy and tests pass at `bc1b290`; `cargo test --features metal --test
  gpu_parity` passed (9) at `9b5f68b`, before the review fixes, which touch no tensor op.

## Pending runs
None.

## Open questions
1. `max_eval` for CO2.2 (D5): Burn's default stalls strong Wolfe at `max_iter` 1 and 2.
   Proposed: the pilot sets it per cell, e.g. 25 at `max_iter` 20 (the default) and 10 at
   `max_iter` 1.
2. "Updates" (inner iterations) are not counted: Burn keeps the count private. Proposed:
   CO2.2 reports steps, passes and time, and asks for the count only if the comparison needs
   it (an upstream accessor, or a copy of `LBFGS::step`).

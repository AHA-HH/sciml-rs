# CO1.2 Gauss–Newton on Burgers and Darcy

Started 2026-11-26 on `optimisers/co1-2`. Roadmap: "CO1.2 Gauss–Newton on Burgers and
Darcy. `OPTIMIZER=gauss-newton` and the `GN_*` variables in `train_burgers` and
`train_darcy`, as in `train_poisson`; `FnoResidual` tested on a rank-3 batch;
`gn_steps.csv` and passes written; `SEED` and `TIME_BUDGET_S` in both examples, which
CO1.3's cells need (CO1.1 added neither); the examples' `env_var` and `RunCosts` assembly,
then in three examples, shared rather than copied a third time (CO1.1's code-quality
review, F5). *Done when:* a flex smoke run of G0 and G1 on Burgers and of G0 on Darcy, a
resume of one; constraint 1 shown. *Needs:* CO1.1, D12." Proposal: §1.4, §2, §3.1 (O1's
cells and budgets), §9 (O1 and constraint 1), §10; D11, D12.

## Goal
`train_burgers` and `train_darcy` run every O1 cell (A0, A1, G0, G1, G0-200, at equal time
and at the long budget) from environment variables alone, with checkpoints, resume and
`costs.json`, sharing their helpers with `train_poisson`.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)
- "a flex smoke run of G0 and G1 on Burgers and of G0 on Darcy, a resume of one;
  constraint 1 shown."
- §9, O1: "`FnoResidual` on a rank-3 batch gives ½‖r‖² = (1/2B) Σ eᵢ², half the mean
  squared relative error, as its 2D test does; … a Gauss–Newton Burgers smoke run on flex
  and its resume."
- §9, constraint 1: "the existing suite unedited, `train_burgers` and `train_darcy` with no
  new variable set bit for bit equal before and after, … and an Adam and a Gauss–Newton run
  on flex bit for bit equal before and after each change."

## Read first
`examples/train/{burgers,darcy,poisson}.rs`; `training/{trainer,gn,config}.rs`
(`build_training_components`, `train_components`, `training_passes`, `budget_spent`);
`runs/` (`RunCosts`, `checkpoint::<R>`, `gn_passes`, `truncate_*`); CO1.1's plan and its
code-quality findings F5 and F6.

## Changes
- **`examples/train/common.rs`** (new; `mod common;` from the three examples, not an example
  itself): what the three share, every item used by all three (no `dead_code` allows):
  `env_var`; `training_config(allowed, seed, &Literals)` (with `GnLiterals`), the one
  parser of `OPTIMIZER` (`adam`, `adamw`, `gauss-newton`, `hybrid`, each example naming the
  ones it takes), of `GN_PRECONDITIONER`, `GN_CG_MAX_ITERS`, `GN_WEIGHT_DECAY`,
  `GN_BATCH_SIZE`, `WEIGHT_DECAY` and `TIME_BUDGET_S` (Gauss–Newton only, as today in
  `train_poisson`), building the whole `TrainingConfig`, a hybrid run's Adam epochs
  included; `CHECKPOINT_EVERY` and `session::<R, C>`, a new run or one reopened from its
  checkpoint with its saved configs (`--resume`); `run_costs` with `Measured`,
  `costs.json`'s steps and passes (Adam, Gauss–Newton and hybrid, from `train_poisson`) and
  the fields every example fills alike. (Amended after the code-quality review, 2026-11-26.)
- **`train_burgers`, `train_darcy`**: `OPTIMIZER=adam|adamw|gauss-newton` (Darcy gains
  `adamw` too: CO1.4 runs the same four cells on Darcy); `WEIGHT_DECAY`; `SEED` (default 42,
  `_seed<n>` in the run name); `EPOCHS` (default 500; AdamW halves every `EPOCHS / 5`
  epochs, so 100 at 500 and 600 at 3,000, D11's long budget; the author, 2026-11-26);
  `TIME_BUDGET_S`; the `GN_*` variables with D12's literals (full batch, CG cap 100, the
  Hutchinson diagonal with 10 probes, λ₀ 1, μ = 1e-4 for G1, `GN_WEIGHT_DECAY=0` for G0,
  `GN_BATCH_SIZE=200` for G0-200; run name `_gn`); checkpoints every 25 epochs and
  `-- --resume runs/<dir>`; `costs.json` with steps and passes (new for Darcy). Configs are
  saved before training (for `--resume`); the plot from every session's metrics.
- **`train_poisson`**: its copies replaced by `common` (behaviour unchanged).
- **`training/gn.rs`**: a test, `FnoResidual` on a rank-3 (Burgers-shaped) batch.
- No library API, no convention, no dependency changes.

## Tests that prove it
- `gn::tests::half_squared_residual_is_the_mean_squared_relative_error_rank_3`: as the 2D
  test, on `tiny_dataset` (`[5, 8, 1]`), no decoder (Burgers trains in physical units):
  ½‖r‖² against `LpLoss` `Reduction::None`, relative 1e-5.
- Constraint 1, each on flex, `origin/main` (`6ea636b`) against the branch, epoch literal
  temporarily set to 2 on `main` and `EPOCHS=2` on the branch, `metrics.csv` compared
  without `epoch_time_s`: `train_burgers` (nothing set), `train_darcy` (nothing set),
  `train_poisson` (Adam, stage 2); `train_poisson OPTIMIZER=gauss-newton` on a 20-sample
  subset, `gn_steps.csv` compared without `wall_time_s`; the configs, `model_weights.bpk`
  byte for byte, and `costs.json` on steps, passes and bytes. The existing suite unedited.
  Each tree built in a target directory of its own (one shared directory served one tree's
  binary to the other).
- Smoke runs (flex, the Mac): Burgers `RECIPE=petscml OPTIMIZER=gauss-newton` G0
  (`GN_WEIGHT_DECAY=0`) and G1, one epoch each (`TIME_BUDGET_S=1`), full batch;
  Darcy G0 one epoch at a reduced batch and CG cap (full batch at 85 × 85 does not fit a
  flex run on 16 GB in useful time); `gn_steps.csv` and the passes in `costs.json` checked.
- Resume: a Burgers G0 run with `EPOCHS=27`, a small CG cap, stopped after its epoch-25
  checkpoint and resumed, against the same run uninterrupted: `metrics.csv` (without
  `epoch_time_s`) and `gn_steps.csv` (without `wall_time_s`) bit for bit.

## Checks
`/verify` (fmt, the three clippy runs, `cargo test`, `cargo doc`, datagen), with
`CARGO_TARGET_DIR=~/sciml-rs-target/co1-2`; the runs above.

## VM runs
None. CO1.3 (the Burgers 2 × 2) and CO1.4 (Darcy) are the VM tasks; their cells, e.g.:
`RECIPE=petscml OPTIMIZER=gauss-newton GN_WEIGHT_DECAY=0 SEED=42 TIME_BUDGET_S=<A1 mean>
cargo run --release --no-default-features --features cuda --example train_burgers`
(G0), and `RECIPE=petscml OPTIMIZER=adamw EPOCHS=3000 SEED=42 …` (A1 at the long budget:
6 × the epochs, D11, not exactly 6 × A1's time). `EPOCHS` stays the cap of a Gauss–Newton
run under `TIME_BUDGET_S`, so G0-200 (5 steps per epoch) at the long budget needs a large
`EPOCHS` for the budget to end it. Run names do not tell the cells apart (A0 and A1, or
G0, G1 and G0-200, share one): CO1.3 groups runs by their `train_cfg.json`.

## Constraints and risks
- Constraint 1 (§1.4): no library code path changes; unset variables give today's configs.
  The switch from `train_fno` to `train_components` with checkpoints adds `checkpoint.bin`
  only (Phase 9 T1: checkpoints do not change the training); the constraint 1 runs show it.
- `train_poisson` keeps every output; only its helpers move.
- The 2D objective: unaffected (no library change).

## Pending runs
None. Every run above is done (2026-11-26); results in `progress.md` and the PR.

## Open questions
- Answered 2026-11-26 by the author: `EPOCHS` in both examples, AdamW halving every
  `EPOCHS / 5` epochs.

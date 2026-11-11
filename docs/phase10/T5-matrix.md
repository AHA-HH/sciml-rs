# T5: the N = 1000 matrix (draft)

Phase 10, task 5. Branch `phase10/T5-matrix`. Its code depends on T3 only; its runs on T4's
go (order amended by the author 2026-11-11, decision 31: code first, every VM run at the end
of the phase). **Draft** for the runs: T4's preconditioner choice and any default changes
are filled in when the go is recorded.

## Read first
- This phase's README ("Experiment matrix", G3–G6, "Diagnostics") and T4's results.
- `examples/predict/poisson_matrix.rs` (the `--by-n` table, the grouping it does today).
- `docs/phase7/results.md` (the baseline rows and their format).

## Goal
Answer H10a: at equal wall-clock on one A100, does Gauss–Newton reach a lower fine-grid
error than Adam on the Phase 5 baseline, and at what cost per update and in memory?

## Runs
Stage 3, s = 64, p = 0, N = 1000, seeds 42, 43, 44, one A100. Each run evaluated by
`predict_poisson`.

| Arm | Optimiser | Batch | Preconditioner | Budget |
| --- | --- | --- | --- | --- |
| Adam | baseline settings, 500 epochs (G6) | 20 | – | its own; sets the budget |
| GN20 | GN, residual (A) | 20 | none | Adam's mean training time |
| GN20-D | GN | 20 | Hutchinson diagonal | same |
| GN200 | GN | 200 | T4's choice | same |
| Hybrid | Adam for 250 epochs (cosine over those epochs' steps), then GN | 20 | T4's choice | same |

Hybrid's Adam half is counted in epochs, not seconds: 250 is half the Adam reruns' 500, and
an Adam epoch at batch 20 costs the same in either run, so it is half the budget while the
schedule's length is known before training (the cosine needs it). Gauss–Newton then trains
until the budget.

Commands (`TIME_BUDGET_S` the Adam reruns' mean training time; `GN_PRECONDITIONER` `none`
or `hutchinson`, T4's choice for GN200 and Hybrid):
```sh
cargo run --release --no-default-features --features cuda --example train_poisson -- 0 129 64 <seed>
OPTIMIZER=gauss-newton GN_PRECONDITIONER=none TIME_BUDGET_S=<t> cargo run ... -- 0 129 64 <seed>
OPTIMIZER=gauss-newton GN_PRECONDITIONER=hutchinson TIME_BUDGET_S=<t> cargo run ... -- 0 129 64 <seed>
OPTIMIZER=gauss-newton GN_BATCH_SIZE=200 GN_PRECONDITIONER=<T4> TIME_BUDGET_S=<t> cargo run ... -- 0 129 64 <seed>
OPTIMIZER=hybrid GN_PRECONDITIONER=<T4> TIME_BUDGET_S=<t> cargo run ... -- 0 129 64 <seed>
```

The Adam reruns run first: their mean training time is the GN budget (G4), and their
error 4 must fall within Phase 7's 1.041e-2–1.095e-2 widened by ±2 %, 1.020e-2–1.117e-2 (G6, amended
2026-11-11). If one does not, stop and
report before the GN arms.

## Deliverables
0. Code, before the runs (2026-11-11): `TrainingConfig::adam_epochs` and the switch in
   `training_loop` (the Hybrid arm; checkpoints carry both optimisers' states);
   `OPTIMIZER=hybrid` (`HYBRID_ADAM_EPOCHS`) and `GN_PRECONDITIONER` (`none` |
   `hutchinson`, `GN_HUTCHINSON_PROBES`) in `train_poisson`, a hybrid run's Adam steps
   in `costs.json`; `poisson_matrix --by-optimizer` (item 2).
1. The 15 runs on the VM, copied back to the main checkout's `runs/`.
2. **`poisson_matrix -- --by-optimizer docs/phase10/results.md runs/...`**: a table per arm
   (mean and min–max over seeds) of error 3, error 4, final train and test L², updates,
   epochs, forward-equivalent passes, training time and peak memory; plots of error 4
   against wall-clock, updates and passes (gnuplot). The existing modes are unchanged
   (`--by-n` regenerates `docs/phase7/results.md` identically).
3. `docs/phase10/results.md` with the table, the plots, H10a answered, and the
   generalisation gap per arm.
4. The T5 Results row in the README.

## Checks
All of `CLAUDE.md`'s checks for the code of item 0, and
`cargo clippy --all-targets -- -D warnings`.

## Do not
- Change the budget, seeds or arms after seeing test errors.
- Cite a multi-GPU time (Phase 9's rule); every run is one A100.

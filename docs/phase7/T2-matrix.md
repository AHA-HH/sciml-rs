# T2: the training matrix

Phase 7, task 2. Branch `phase7/T2-matrix`. Depends on T1. Runs on the cuda A100 VM.

## Read first
- Design §13.8 (the Phase 7 matrix), §13.5, decision 30's 2026-10-29 amendments.
- `docs/phase5/T5-matrix.md` and the Phase 5 T5 Results rows (the cuda workflow, the
  cost-first rule, `--no-default-features --features cuda`).
- This phase's README (Planning decisions); `CLAUDE.md` Commands (`train_poisson` after
  T1).

## Goal
Train the Phase 5 baseline on every subset size, with enough subset and initialisation
seeds to separate the effect of N from seed noise.

## Deliverables
1. **Cost first.** Time the first epochs of one run per N ∈ {25, 50, 100, 250, 500}
   (subset seed 0, seed 42), extrapolate each to 500 epochs and the matrix to 45 runs,
   and state it in the PR before the full runs start; the author approves it.
2. **The matrix:** stage 3, s = 64, p = 0, modes [12, 12], N ∈ {25, 50, 100, 250, 500},
   subset seeds 0, 1, 2, initialisation seeds 42, 43, 44, 500 epochs (D6), the Phase 5
   hyperparameters. 45 runs.
3. **N = 1000:** none trained; the Phase 5 s = 64 runs with seeds 42, 43, 44 are the row
   (README, Inputs).
4. Every run in the main checkout's `runs/`, with its `costs.json`. A table in the PR:
   run directory, N, subset seed, seed, optimisation steps, final train / test relative
   L², s/epoch, wall time.

## Tests
None new. The runs are the deliverable.

## Acceptance
- 45 runs, all 500 epochs, all recording N, subset seed, index hash and steps.
- The T2 parts of the Results rows filled in (costs and the coarse error; the fine-grid
  error comes in T3).
- A run that diverges or is much worse than its neighbours is reported, not rerun
  silently.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`, locally;
the cuda build (`cargo build --release --no-default-features --features cuda --examples`)
on the VM. Say which ran where.

## Do not
- Retune hyperparameters, the epoch count, the model or the evaluation cadence for small
  N (D6: fixed epochs; a step budget is deferred).
- Retrain N = 1000.
- Commit runs, datasets or job scripts with account names.

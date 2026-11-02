# T6: the full-length flex run

Phase 9, task 6. Branch `phase9/T6-flex-run`. Depends on nothing; runs at any time
(decision 21).

## Read first
- Design decisions 17 and 21; §11 (the pipeline row, threshold test relative
  L² ≤ 2e-2).
- `docs/phase7/README.md` (the flex smoke run: s = 64, p = 0, N = 25, 2.4 s/epoch).
- This phase's README (P6).

## Goal
Show the baseline trains to the §11 threshold on the pure-Rust CPU backend, so the
pipeline row is met on flex as it is on metal and cuda.

## Deliverables
1. **Timing first:** 3 epochs of the baseline (stage 3, s = 64, p = 0, N = 1000,
   seed 42) on flex; the machine, the s/epoch and the projected 500-epoch cost stated in
   the PR and approved by the author before the full run.
2. **The full run** on flex, then `predict_poisson` on it (on flex).
3. **Comparison** with the cuda run on the same seed (Phase 5's `…_1793188648`): final
   test relative L², error 4, and the per-epoch curves; a difference beyond round-off is
   explained or recorded.
4. If T2 is merged, the run uses checkpoints, so a stop costs at most one cadence.
5. Decision 21 closed in design §12 (dated), and the T6 Results row.

## Acceptance
- Test relative L² ≤ 2e-2 on flex, or the shortfall recorded against decision 21.

## Checks to run and report
None beyond those of any code change made (normally none).

## Do not
- Change any hyperparameter, the padding or the data to make flex faster.
- Run at p = 4 unless the author picks it over P6.

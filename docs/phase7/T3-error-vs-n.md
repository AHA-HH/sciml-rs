# T3: error and cost against N, and close-out

Phase 7, task 3. Branch `phase7/T3-error-vs-n`. Depends on T2.

## Read first
- `examples/predict/poisson_matrix.rs` (grouping by padding and s, the tables, the
  plots) and `examples/predict/poisson.rs` (`errors.json`).
- `docs/phase5/T6-evaluate.md` and `docs/phase5/results.md` (the table this extends).
- Design §13.5 (Reports, Acceptance), §13.13 (H7), §13.15 (D7); this phase's README.

## Goal
Measure the fine-grid error and the cost of every Phase 7 run against N, find the
smallest N that meets the threshold fixed at planning, and close Phase 7.

## Deliverables
1. **Evaluate** every T2 run and the three N = 1000 runs with `predict_poisson` (flex),
   so each has `errors.json`.
2. **`poisson_matrix` by N:** runs that record a subset group by N (Phase 5 runs with no
   subset count as N = 1000); per N, mean [min–max] over all seeds of error 3, error 4,
   final train and test L², optimisation steps, s/epoch and training time; the spread by
   subset seed (mean over initialisation seeds) and by initialisation seed (mean over
   subset seeds). Phase 5 output for runs without a subset unchanged. Plots, skipped
   without gnuplot: error 4 against N (log–log, min–max band, the D7 line) and training
   time against N. Writes `docs/phase7/results.md`.
3. **The smallest N meeting D7:** mean error 4 over the 9 runs ≤ 2e-2 (3 at N = 1000),
   with the spread, and whether every run meets it too.
4. **Cost per additional reduction in error** between successive N: extra training time
   and extra samples per unit fall in mean error 4.
5. **H7 against the results** (design §13.13), noting that under D6 small N also trains
   for fewer steps.
6. Any result that contradicts the design, recorded in design §12 (dated amendment to
   decision 30); the README Results and exit checklist completed; the "Current phase"
   bullet in `CLAUDE.md` updated.

## Tests
- `poisson_matrix`: a unit test that groups synthetic run records by N, counting a run
  with no subset as N = 1000, and leaves the Phase 5 grouping unchanged.

## Acceptance
- Every row of Results has a value and every exit-checklist item is ticked.
- `docs/phase5/results.md` regenerates unchanged from the Phase 5 runs.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`.

## Do not
- Change D7 or how it is applied after seeing the results. If it gives an unusable
  answer, say so in §12 and let the author decide.
- Train new runs (a fixed-step arm is deferred research, not this task).

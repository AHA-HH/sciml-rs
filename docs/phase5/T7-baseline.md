# T7: the Phase 7 baseline and close-out

Phase 5, task 7. Branch `phase5/T7-baseline`. Depends on T6. Docs only.

## Read first
- `docs/phase5/README.md` (Planning decisions: the baseline rule; Results).
- `docs/phase5/results.md` (T6), design §13.5 and §13.13 (H5).

## Goal
Fix the spatial baseline Phases 6 and 7 use, by the rule set at planning, and close
Phase 5.

## Deliverables
1. **The baseline s:** the smallest s whose mean fine-grid error over seeds is within
   10% of the s = 128 main-arm mean (the rule fixed before T5). State the numbers that
   decide it.
2. **The full baseline:** s, R and its parameters, P and d, p = 0, modes by D2, width,
   layers, hyperparameters, seeds. Recorded in design §12 (a dated amendment to decision
   30) and in the README Results.
3. **H5 against the results** (design §13.13): whether an s* exists below which the floor
   dominates, and the time per epoch against s².
4. Any result that contradicts the design, recorded in design §12, dated.
5. The exit checklist ticked; the "Current phase" bullet in `CLAUDE.md` updated (Phase 7
   next; Phase 6 may already be under way).

## Acceptance
Every row of Results has a value and every exit-checklist item is ticked.

## Checks to run and report
`cargo test` and `cargo doc --no-deps` (docs-only change).

## Do not
- Change the rule after seeing the results. If it gives an unusable answer, say so in §12
  and let the author decide.
- Start Phase 7.

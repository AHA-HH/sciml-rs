# T2: padding sweep and the flex run at 65²

Phase 4, task 2. Branch `phase4/T2-padding-sweep`. Depends on T1 (the regenerated p = 0
run is the sweep's baseline).

## Read first
- Design §7 (padding), §11 (the pipeline row), §12 decisions 16, 17 and 18.
- `docs/CONVENTIONS.md` §3 (padding) and §4 (FFT lengths, Bluestein).
- `docs/phase3/T2-train-poisson.md` and the Phase 3 Results; the T1 row of
  `docs/phase4/README.md`.
- `examples/train/poisson.rs` (`PADDING`, the padding argument) and
  `examples/predict/poisson.rs`.

## Goal
Choose p by the decision 16 rule, and meet the 2e-2 pipeline threshold on flex as well
as on metal, closing the two items decision 17 carried over.

## Before running
State the cost and where each run goes, and confirm with the author:
- p ∈ {4, 8, 16} at 500 epochs on metal, locally: about 40 h (p = 8 measured at about
  101 s/epoch), on mains power. The sweep runs on metal before anything runs on cuda
  (decision 18); cuda comes in with T3.
- The flex run at the chosen p: measure the first 3 epochs, extrapolate, and report it
  before committing to the full run. At p > 0 expect about 520 s/epoch (about 72 h).
- If the author asks for a Bluestein fix in Burn first, stop: it is its own task
  (decision 17), planned then.

## Deliverables
1. **The sweep:** `train_poisson -- <p>` for p = 4, 8, 16, full length, on metal
   (`--features metal`) for all three, runs kept in the main checkout's `runs/`. The
   final test relative L² of each, with s/epoch and wall time, against T1's p = 0 run.
2. **The choice** by the decision 16 rule. If p ≠ 0, `PADDING` changes in
   `examples/train/poisson.rs` and its doc comment drops "provisional"; if p = 0, only
   the comment changes.
3. **The flex run:** full length at the chosen p, `--no-default-features --features flex`
   (or the default build), with train / test relative L² and s/epoch. The threshold
   2e-2 is checked.
4. **If p ≠ 0:** `predict_poisson` on the chosen metal run; the five errors and the
   boundary verdict recorded.
5. Design §11's pipeline row and §10's Phase 4 text updated to read "met on metal and
   flex" once both hold.

## Tests
No new tests: this task changes a constant and runs experiments. `train_poisson_smoke`
must still pass.

## Acceptance
- The T2 rows of `docs/phase4/README.md` Results filled in.
- If the threshold is missed on flex, or a p > 0 wins by less than the run-to-run spread
  seen between the two p = 0 runs (1.16e-2 against 1.116e-2), it is recorded in design
  §12 before anything is retuned.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`. Report
the backend and Burn device of every run, and whether each ran on mains power.

## Do not
- Change the hyperparameters other than `PADDING`, the model or the trainer.
- Shorten runs to save time: the rule is full length.
- Patch Burn or the FFT path in this task.

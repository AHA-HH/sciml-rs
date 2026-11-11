# T2: Learner showcase tightened

Phase 4s, task 2. Branch `phase4s/T2-learner-showcase`. Depends on T1.
- Can run in parallel with T3–T8.
- Shares `training/trainer.rs` with T3, so whichever merges second rebases. Suggested order:
  T3 first.

> **Refresh at start:** if T3 has merged, re-read `trainer.rs`. The new shared loss step goes
> next to `train_fno` and `train_epoch`, wherever they ended up.

## Read first
- Review §6 T2-1 (revised 2026-10-15) and §7 step A1. Design §12, decision 25 ("The Learner
  example stays").
- `src/neural_operators/training/learner.rs`, especially `learner_forward` :32-48 and the
  test `train_step_gradients_match_hand_written_loop` :113-153.
- `src/neural_operators/training/trainer.rs`: `flatten_pair` :88-108, `eval_epoch`
  :112-138, `train_epoch` :151-221.
- `examples/train/burgers_learner.rs` and `examples/train/burgers.rs`.

## Goal
`train_burgers_learner` stays as the showcase of Burn's `Learner` (TUI, metric logging,
summary), but its loss must not drift from `train_epoch`, and the runs it writes must be
readable by `predict_burgers`.

## Deliverables
1. **One shared loss step** in `trainer.rs`: a documented `pub fn` that does forward +
   `flatten_pair` + the summed relative L² and returns the loss tensor (any shapes the
   caller needs come with it).
   - `train_epoch` (the line `loss_fn.rel(out, target)` at :184) and `learner_forward`
     (`learner.rs:37`, which builds its own `LpLoss::new(1, 2, Reduction::Sum)`) both call
     it.
   - `eval_epoch` (:127) also calls it if the post-processing hook fits. If it doesn't, leave
     `eval_epoch` alone; T13 reworks it.
2. **`examples/train/burgers_learner.rs` runs:**
   - Today it writes to `runs/burgers_fno_learner`, relative to the current directory
     (`ARTIFACT_DIR` :31), with no timestamp, no `data_cfg.json` and no `metrics.csv`.
   - Write a timestamped `runs/burgers_fno_learner_<unix_secs>/` under
     `env!("CARGO_MANIFEST_DIR")`, with `model_cfg.json`, `train_cfg.json`, `data_cfg.json`
     and `model_weights`, the same set `train_burgers` writes.
   - Burn's own Learner artifacts go into the same directory.
   - Reuse `metrics::io::write_run_artifacts` if the Learner's metrics can become
     `Vec<EpochMetrics>`. If not, write the three configs directly and say in the PR that
     `metrics.csv` is missing because the Learner doesn't expose per-epoch metrics. T10
     moves this onto `RunDir` either way.
3. **Fix the comment at `burgers_learner.rs:68`:** it says "Fewer epochs than
   `train_burgers`", but both use 500 (:70 and burgers.rs:92). Either correct the comment
   or lower the count and say so.
4. **Mark it as a showcase:**
   - `learner.rs` module doc: "`train_epoch` is the canonical loop; this is the Burgers-only
     (`FNO<3>`) Learner equivalent." Add one sentence on why `TrainStep` cannot be generic
     over R here.
   - The same line in CLAUDE.md "Commands", next to `train_burgers_learner`.

## Tests
- `train_step_gradients_match_hand_written_loop` stays and still passes. If both sides now
  call the shared step, simplify the hand-written side so the test still compares something
  independent: keep one hand-written `LpLoss` call in the test as the oracle.
- Add a unit test for the shared step: same output as `LpLoss::new(R-2, 2, Sum).rel` after
  `flatten_pair` on a tiny `FNO<3>` and `FNO<4>`.

## Acceptance
- Run `train_burgers_learner` briefly. It needs a real terminal for the TUI; temporarily
  lower the epoch count locally if you need to, without committing that.
- Evaluate the resulting run with `predict_burgers -- runs/burgers_fno_learner_<secs>` and
  record the test L² in the README Results row "`train_burgers_learner` run evaluated by
  `predict_burgers`".
- If no real terminal is available, say so plainly and leave the row empty.

## Checks to run and report
The six from the README. The chebyshev pair is not needed.

## Do not
- Delete the Learner example or its module (design §12, decision 25).
- Make the Learner generic over R, or add Learner features.
- Introduce `RunDir` (T10) or change `Postprocess` / `identity` (T17).

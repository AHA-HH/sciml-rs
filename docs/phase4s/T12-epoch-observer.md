# T12: epoch observer in `training_loop`

Phase 4s, task 12. Branch `phase4s/T12-epoch-observer`. Depends on T10.
- Can run in parallel with T11. Both edit `training/trainer.rs`, so whichever merges second
  rebases.
- Must merge before Phase 5 T2 (checkpoint/resume).

> **Refresh at start:** re-read `training_loop` and T3's `train_fno` on `main`, and T10's
> `RunDir::save_metrics`.

## Read first
- Review §6 T3-2, §7 step B5. Design §10, Phase 5 T2 (what checkpointing will need).
- `training/trainer.rs::training_loop` (:315-366 before T3): it has a hard-coded
  `println!` and returns metrics only at the end.

## Goal
Without a hook, Phase 5 would add `if epoch % k == 0` branches and `Option` parameters
inside the loop. Give the loop one per-epoch callback now, while nothing depends on it.

## Deliverables
1. **An observer parameter** on `training_loop`. Choose one of:
   - `on_epoch: &mut dyn FnMut(&EpochMetrics, &FNO<R>)`
   - a small trait `EpochObserver { fn on_epoch(&mut self, m: &EpochMetrics, model:
     &FNO<R>); }` with blanket impls

   Pick whichever keeps the call sites short, and document why. The model reference is
   there for Phase 5 checkpoints and is unused for now.
2. **A default printing observer** that reproduces today's `println!` line exactly, so the
   console output doesn't change. `training_loop` itself no longer prints.
3. **A `RunDir` observer** that appends one row of `metrics.csv` per epoch (header on the
   first epoch). Training then always writes its metrics through `RunDir`.
   - `RunDir::save_metrics` at the end either stays, writing the plot only, or becomes
     "plot from the CSV". Keep the plot an end-of-run action.
   - The CSV columns and number format are unchanged.
4. **Combining observers:** `train_fno` passes the default plus the run-dir observer.
   - That means a tuple impl, a `Vec<Box<dyn …>>`, or two calls in a closure; keep it
     simple.
   - The train examples need the `RunDir` before training starts, so create it first.
     That moves `RunDir::create` ahead of training in the three train examples. Say so in
     the PR.

## Tests
- `incremental_csv_equals_end_of_run_csv`: train a tiny model for 3 epochs with the
  run-dir observer, then compare the file byte-for-byte with what `save_metrics` would
  have written from the returned `Vec<EpochMetrics>`.
- `default_observer_prints_same_line`: if capturing stdout is awkward, test the formatting
  function instead.
- The existing trainer tests pass.

## Acceptance
A short real training run (`train_darcy` or `train_burgers` with the epoch count lowered
locally, not committed) writes `metrics.csv` row by row. Check this with `wc -l` while it
runs, or after an interrupted run.

## Checks to run and report
The six from the README. The chebyshev pair is not needed.

## Do not
- Implement checkpointing, resume or early stopping. Those belong to Phase 5.
- Change the learning-rate schedule, the loss or the metric definitions.

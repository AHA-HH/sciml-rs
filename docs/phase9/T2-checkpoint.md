# T2: checkpoint and resume

Phase 9, task 2. Branch `phase9/T2-checkpoint`. Depends on Phase 4s T10/T12 (merged);
parallel with T1.

## Read first
- `src/neural_operators/training/trainer.rs` (`TrainingComponents`,
  `build_training_components`, `training_loop`, `EpochObserver`, `train_fno`).
- `src/neural_operators/runs/mod.rs` (`RunDir`: `save_weights`, `load_fno`,
  `append_metrics`, `metrics`) and `runs/csv.rs`.
- `src/neural_operators/data/device_batcher.rs` (the seeded shuffle).
- Burn: `burn-optim/src/optim/module/module_optimizer.rs` (`to_record`, `load_record`,
  `save`, `load`) and `lr_scheduler/module_lr_scheduler.rs`, `cosine.rs`.
- `examples/train/poisson.rs`. This phase's README (P2, P3).

## Goal
A training run that is killed can be resumed from its last checkpoint and finishes with
the same result as if it had never stopped.

## Deliverables
1. **Checkpoint contents:** model weights, optimiser record, scheduler record, the next
   epoch number, and whatever the shuffle needs to give that epoch's order (see Tests).
   Written every k epochs (default 25) and after the last, into the run directory,
   atomically (write to a temporary name, then rename). One checkpoint kept.
2. **`RunDir`** owns the file names: `save_checkpoint` and `checkpoint` (reader,
   `Ok(None)` when absent), following the existing `save_*` / reader pairs.
3. **`training_loop`** takes the checkpoint cadence and an optional starting state. The
   observer either gains access to the optimiser and scheduler, or the loop saves
   itself; pick one and say why in the PR.
4. **Resume:** `train_poisson --resume runs/<run_dir>` reopens the run, reads its configs
   (not the literals at the top of the file), loads the checkpoint, truncates
   `metrics.csv` to the checkpoint's epoch, and continues. `costs.json` records the
   total training time across sessions and the number of resumes.
5. **Shuffle across resume:** the epoch-k order after a resume equals the epoch-k order
   of an uninterrupted run. Either advance the loader's RNG k epochs, or seed each
   epoch from (seed, epoch); the second changes today's order, so it needs the author's
   approval and a note that old runs are not reproducible bit for bit by new code.

## Tests
- On flex, a tiny synthetic run of 6 epochs with cadence 2, stopped after epoch 3 and
  resumed: final weights and every `metrics.csv` row (except `epoch_time_s`) equal the
  uninterrupted run's, bit for bit.
- A checkpoint round-trips the Adam moments and the cosine `current_iter`.
- `train_poisson_smoke` (`training/trainer.rs`) extended to stop after epoch 1 and
  resume to epoch 2 through the checkpoint (this absorbs T5, closed 2026-11-02).
- A run directory without a checkpoint reports `None`; a truncated checkpoint file is an
  error, not a panic.
- Every existing trainer and run-dir test unchanged.

## Acceptance
- Interrupt a short real run (stage 2 or 3, a few epochs) with Ctrl-C and resume it;
  report both `metrics.csv` files.
- The T2 Results row filled in.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`; the
resume test with `--features metal` if available, saying whether it ran.

## Do not
- Change the training step, the loss, the evaluation cadence or the hyperparameters.
- Keep more than one checkpoint per run, or add a "best epoch" checkpoint.
- Save checkpoints from `train_burgers_learner` (Burn's Learner has its own).

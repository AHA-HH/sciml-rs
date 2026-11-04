# T2: data-parallel training on several GPUs

Phase 9, task 2 (task 3 before the renumbering of 2026-11-04). Branch
`phase9/T2-multi-gpu`. Depends on T1 (checkpoints), and on a machine with at least two
GPUs of one kind (P5: a 2× A100 VM, available 2026-11-04).

## Read first
- The profiling finding (this phase's README, "Closed at planning"): at s = 64 a batch costs about 31 ms
  whatever its size, so a split batch may not run faster. Expect gains at s = 128 or
  stage 4, not at the baseline.
- `src/neural_operators/training/trainer.rs` (`train_epoch`, `training_loop`,
  `build_training_components`) and `step.rs` (`forward_loss`).
- `src/neural_operators/data/device_batcher.rs` (`device_loader`).
- Burn: `burn-tensor/src/tensor/distributed.rs` (`DistributedContext`, `all_reduce`),
  `burn-cubecl/src/ops/distributed.rs`, and
  `burn-train/src/learner/supervised/strategies/ddp/` (README, `worker.rs`, `epoch.rs`)
  as a reference for how Burn syncs gradients.
- Design §13.11; this phase's README (P4, facts on `all_reduce`).

## Goal
Train one run on D GPUs such that the effective batch, the learning-rate schedule and
the number of optimisation steps are those of the single-GPU run, and the result matches
it within a stated round-off.

## Deliverables
1. **A data-parallel step** beside `train_epoch`: one model replica per device, each
   batch of 20 split into D shards (equal when D divides the batch; the PR states the
   rule for a partial last batch), forward and backward per device, gradients
   mean-reduced with `all_reduce` (weighted by shard size if shards differ), one
   optimiser step applied identically on every replica (or on one, then broadcast; say
   which and why).
2. **Data placement:** each device holds its own copy of the split, or its shard, via
   `device_loader`; state the memory cost. The global batch order equals the
   single-device order for the same seed.
3. **Evaluation** on the first device only, as today.
4. **`train_poisson --devices <list>`** (or `BURN_DEVICE`-style), defaulting to one
   device: today's path, unchanged and bit-identical.
5. **Communication backend:** find out and report whether cubecl's `all_reduce` on cuda
   uses NCCL, peer-to-peer copies or the host. No new dependency without the author's
   approval.
6. Works with T1's checkpoints (save from the first replica).

## Tests
- Gradient equivalence on one GPU split into D = 2 "devices" is not possible on flex
  (`all_reduce` is `unimplemented!()` there), so: a GPU test, `#[ignore]`d without
  the feature, that compares the mean-reduced gradient of two shards with the gradient
  of the whole batch on one device, within round-off.
- With one device, `train_epoch`'s output is unchanged (existing tests).

## Acceptance
- On 2 GPUs: the baseline (s = 64, p = 0, N = 1000, seed 42) for 500 epochs; final test
  relative L² and error 4 within a stated round-off of the single-GPU run on the same
  seed (Phase 5's `…_1793188648`), and the per-epoch curves overlaid in the PR.
- The T2 Results row filled in.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`; the GPU
test with `--features cuda`; `--test gpu_parity` with `--features metal` if available.
Say which ran.

## Do not
- Use the Learner's `MultiDevice` or DDP strategy for the Poisson trainer (P4).
- Change the batch size, the learning rate or the schedule to "scale" with D.
- Cite any speedup in a Phase 5–7 result (design §13.11).

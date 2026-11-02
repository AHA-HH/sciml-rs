# T1: profile the baseline epoch

Phase 9, task 1. Branch `phase9/T1-profiling`. Depends on nothing; parallel with T2.

**Status: closed 2026-11-02 from existing data (Phase 5 T5 costs, Phase 7 T2 fit); no
branch needed.** Deliverables 1 (coarse split) and 4 (the test pass) are answered in
this phase's README, Results. Deliverables 2 and 3 (GPU utilisation, trace) move to T4.
The brief below is kept as history.

## Read first
- `src/neural_operators/training/trainer.rs` (`train_epoch`, `training_loop`) and
  `step.rs` (`forward_loss`, `evaluate`).
- `src/neural_operators/data/device_batcher.rs` (`device_loader`).
- `src/neural_operators/layers/spectral_convolution.rs` and `utils/fft.rs`.
- `docs/phase7/README.md` (Results: the matrix cost and its per-epoch fit).
- This phase's README (P1).

## Goal
Say where one epoch of the Phase 5 baseline spends its time on cuda (A100), with numbers,
so T3 knows what data parallelism can and cannot speed up.

## Deliverables
1. **Wall-clock split per epoch**, from the baseline run (stage 3, s = 64, p = 0,
   N = 1000, seed 42), a few epochs after warm-up: the training steps (forward, backward,
   optimiser step, each if separable with a device sync), the test pass, and anything
   else (loader, host copies, metric reads). Measured with explicit device syncs, behind
   a timing switch that is off by default and costs nothing when off.
2. **GPU utilisation** over the same epochs (`nvidia-smi dmon` or nsys).
3. **One trace or flamegraph**: `nsys profile` on cuda if available, else `cargo
   flamegraph`; attached to the PR, not committed.
4. **The test-pass question:** Phase 7's fit puts about 0.12 s of a 1.69 s epoch in a
   fixed cost. Confirm or correct that it is the per-epoch test pass, and how much a
   larger evaluation batch would save. Report only; do not change the cadence.
5. **A short findings section** in this phase's README (the T1 Results row): the split,
   the three largest kernels or ops, and what T3 should expect to scale.

## Acceptance
- The split sums to the measured epoch time within 5 %.
- The timing switch off gives the same `metrics.csv` as `main` on flex for a short run.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`. Say
which backend each number comes from; no cuda, no T1 numbers.

## Do not
- Optimise anything in this task. Findings only; fixes are their own tasks.
- Change the evaluation cadence, batch size or any hyperparameter.
- Profile the 129² p = 4 run (P1); the Bluestein cost is already measured.

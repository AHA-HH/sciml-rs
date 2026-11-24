# CL0.1 Rank-5 FNO cost

Started 2026-11-24 on `laplace3d/cl0-1`. Roadmap: "**CL0.1 Rank-5 FNO cost.** Peak memory
and time per epoch of `FNO<5>` on random data at s = 32 and 64, modes 8³ and 12³, width 32,
on flex, metal and cuda, at the batch sizes that fit; a zip64 round trip of an npz above
4 GiB." Proposal: §10; §7 (cost and memory model), §8 R1 and R6, §12 D4 and D5.

## Goal
Measure what a rank-5 FNO and a 3D n = 65 split cost, so that the author can take D4 (model
size and batch) and D5 (stages, sizes, names, storage) on evidence.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)
- Roadmap: "a table of both, with parameter counts and activation memory against §7; the
  zip64 result; D4 (with the batch) and D5 proposed."
- Proposal §7: "L0 measures memory and time per epoch on metal and cuda. The activation
  figures scale the 2D measurement by the 64× point count and are the ones that decide R1,
  D4 and D8." "Files above 4 GiB need zip64 in the npz writer and reader: CL0.1 checks it,
  or the split is stored in several files."

## Read first
- Proposal §7, §8 (R1, R6), §12; roadmap D4, D5.
- `src/neural_operators/training/trainer.rs` (`build_training_components`, `train_epoch`),
  `training/step.rs` (`evaluate`), `data/device_batcher.rs` (`device_loader`).
- `src/neural_operators/pde/poisson/dataset.rs` (`write_split`: `NpzWriter::new`, stored,
  no zip64 option) and `data/io/readers/npz.rs` (`NpzFileReader`).
- ndarray-npy 0.10 `NpzWriter::new` (zip 6.0.0, `large_file` false) and zip's
  "Large file option has not been set" checks (one member above 4 GiB fails; a file above
  4 GiB made of smaller members is the case to test).

## Changes
- New spike `spikes/fno3d_cost/` (package `fno3d-cost-spike`), a non-default member of the
  root workspace like `spikes/transfer_floor` (root `Cargo.toml` `members` gains it; the
  lock gains the package entry, no new external crate). Features `metal` and `cuda` pass
  through to `sciml-rs`; flex is always built, so `BURN_DEVICE` picks the backend.
  - binary `fno3d_cost -- <s> <modes> <batch> <n_train> <n_test>`: builds the real
    training components (`build_training_components::<5, 4, f32>`, width 32, 4 layers,
    Adam, the cosine schedule) on random data `[N, s, s, s, 1]`, runs one warm-up epoch,
    then times one `train_epoch` and one `evaluate`; prints the backend, parameter count,
    the times, and per-sample times extrapolated to 1000 train / 200 test.
  - binary `zip64_roundtrip -- <path>`: writes f and u `[1000, 65, 65, 65]` f64 (the
    n = 65 train split of D5, 4.39 GB) with `NpzWriter::new` exactly as `write_split` does,
    reads both back with `NpzFileReader`, and checks every value.
  - `run.sh` (one run per config line, peak memory read outside the process), the config
    lists `metal.cfg`, `flex.cfg`, `cuda.cfg`, and `mac.log` (the Mac output verbatim).
    The machine, commands and method are in the results file rather than a spike
    `REPORT.md`.
- Peak memory is read outside the binary: `/usr/bin/time -l` ("peak memory footprint",
  which counts Metal buffers on Apple silicon) on the Mac; `nvidia-smi` per-process
  sampling on the VM. Activation memory per sample is the slope of peak memory between two
  batch sizes at fixed data size, so the resident dataset and the model cancel.
- Results `docs/laplace3d/results/cl0-1.md`: the table, the zip64 result, D4 and D5
  proposed.
- No change under `src/`, `examples/`, `tests/`; no convention change.

## Tests that prove it
- `spikes/fno3d_cost` unit test `param_counts_match_section_7`: `FNO<5>::num_params()` at
  modes 8³ and 12³, width 32, 4 layers equals §7's closed form (16 785 953 and 56 631 841)
  exactly.
- The zip64 round trip checks every value bit for bit against the values written
  (`f[i] = i`, `u[i] = −i` as f64, exact up to 2^53).
- Timings are measurements, asserted nowhere.

## Checks
- `/verify` (the branch touches `Cargo.toml`, `Cargo.lock` and a spike; `src/` untouched).
- `cargo test -p fno3d-cost-spike`, `cargo clippy -p fno3d-cost-spike --all-targets -- -D warnings`
  (and with `--features metal`).
- The measurement runs below, release builds.

## VM runs
cuda numbers (A100 80 GB), awaiting the author's go; no ssh from this session. About
30 minutes in all (build 5–10 min; s = 64 epochs of 1000 samples at batch 20 are the long
part). Commands in "Pending runs". They decide D4's batch at s = 64 and the cuda column of
the table.

## Constraints and risks
- Constraint 1: nothing under `src/` changes, so every 2D output is unchanged; the root
  `Cargo.toml` change only adds a non-default workspace member (default members stay `.`).
- The Mac has 16 GB of unified memory: s = 64 at batch 20 (§7: 35–40 GB) cannot run on
  metal or flex; the Mac columns use the batches that fit and the per-sample slope.
- flex at s = 64 is slow: Mac epochs use a smaller N and are extrapolated per sample
  (stated in the table).

## Pending runs
- **cuda matrix and zip64 on the VM** (not run: this session has no ssh; awaiting the
  author's go). tmux session `cl0-1`, log `~/sciml-rs/logs/cl0-1.log`, worktree
  `~/sciml-rs-wt/cl0-1`, about one hour. Decides the cuda column, batch 20 at s = 64 (D4)
  and the zip64 check on Linux. Commands:
  ```sh
  ssh cricket
  cd ~/sciml-rs && git fetch origin
  git worktree add --detach ~/sciml-rs-wt/cl0-1 origin/laplace3d/cl0-1
  cd ~/sciml-rs-wt/cl0-1
  source ~/sciml-gpu.sh
  export CARGO_TARGET_DIR=~/sciml-rs-target/laplace3d
  cargo build --release -p fno3d-cost-spike --features cuda
  tmux new -d -s cl0-1 "($CARGO_TARGET_DIR/release/zip64_roundtrip ~/sciml-rs-wt/cl0-1/zip64.npz; CUDA_VISIBLE_DEVICES=0 bash spikes/fno3d_cost/run.sh cuda $CARGO_TARGET_DIR/release/fno3d_cost < spikes/fno3d_cost/cuda.cfg) 2>&1 | tee ~/sciml-rs/logs/cl0-1.log"
  ```
  No datasets are needed, so the dataset links of the skill's step 6 are skipped.

## Open questions
- D4 and D5 as proposed in `docs/laplace3d/results/cl0-1.md` (author's sign-off).
- metal's peak footprint is contaminated by autotune and is not used for D4; the cuda
  reading replaces it. Accept flex's slope as the activation figure until then?

# CO0.3 Cost profile of an Adam step

Started 2026-11-27 on `optimisers/co0-3` (the author's request of 2026-11-27, through the
coordinating session). Roadmap: "CO0.3 Cost profile of an Adam step. Time per phase of one
Adam step at batch 20 (forward, backward, the optimiser update, the loader, the per-epoch
evaluation and checkpoint) on Burn 0.23 against the fork, to explain why CO0.1's Adam epoch
took about 3.6× the fork's time (3,067 s for 500 epochs against 859.4 s) while a GN200 step
slowed about 24 %." Proposal: §1.2 (O0), §1.4, §2.

## Goal
Measure where the time of one Adam step and one epoch goes on Burn 0.23 and on the fork
(A100, cuda, f32), on CO0.1's setting, and say where the extra time goes, without changing
any library code.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)
"the profile table and where the extra time goes (nothing changed here; a fix, if any, is a
later task, since it would move the equal-time budget)." Proposal §1.2: O0 profiles the
reference arms; changes it suggests are later work.

## Read first
- `src/neural_operators/training/trainer.rs` (`train_epoch_with_loss`: the step's loop:
  loader `next`, `forward_loss`, the MSE and the two metric accumulators, `backward`,
  `GradientsParams::from_grads`, `scheduler.step`, `optim.step`; the epoch's two
  `into_scalar`s; `run_loop`'s `evaluate` and `save_checkpoint` every 25 epochs).
- `src/neural_operators/training/step.rs` (`forward_loss`, crate-private: the forward,
  `flatten_pair`, the decode, `LpLoss::new(2, Sum).rel`; `evaluate`).
- `examples/profile/gn_profile.rs` (CO0.2: `timed`, `median`, the report's layout).
- `docs/optimisers/results/CO0.1.md` ("Adam's 3.4× slowdown": the bisect, fork `4541f02`
  1.723 s per epoch, #119 `684c141` 5.930 s) and `docs/optimisers/results/CO0.2.md`.
- `cubek.toml` (TF32 off since #119: one of the changes #119 made besides the Burn pin).

## Changes
- **No library code.** A new example `adam_profile` (`examples/profile/adam_profile.rs`,
  registered in `Cargo.toml` with `test = true`), built only from public API that is the
  same at `4541f02` (the fork, last commit before #119) and on `main`:
  `build_training_components`, `train_epoch`, `evaluate`, `FNO::forward`, `flatten_pair`,
  `FlatDecoder`, `LpLoss`, `RunDir::create` / `save_checkpoint`, `load_poisson_uniform`,
  `catalogue::stage`, `Device::sync`, `backend_features`, `git_commit`. The configs come
  from builders (`FNOConfig::new`, `TrainingConfig::new().with_*`), never struct literals,
  so the fields `main` added since the fork keep their defaults. One source file serves
  both builds.
- Setting: CO0.1's Adam (Phase 5): stage 3, n = 129, s = 64, p = 0, modes [12, 12], width
  32, 4 layers, N = 1,000, batch 20 (50 steps per epoch), lr 1e-3 cosine over 500 epochs,
  min 1e-5, weight decay 1e-4, seed 42, the 200 test samples at the test batch size; a
  freshly initialised model (the cost of a step does not depend on the weights). Literals at
  the top of the file.
- `adam_profile -- <out.md>`:
  1. **Warm-up**: `WARMUP_EPOCHS` (3) epochs of the library's `train_epoch`, untimed
     (autotune, allocator).
  2. **The library's epoch**: `EPOCHS` (3) epochs of `train_epoch`, each synced at its ends,
     then `evaluate` on the test split and a `save_checkpoint` (each synced): the real
     cost, comparable with `metrics.csv`'s `epoch_time_s` (training + evaluation).
  3. **The step, inline, unsynced**: `EPOCHS` epochs of a copy of `train_epoch`'s loop,
     host time (`Instant`, no sync) around each phase: loader, forward and loss, metrics
     (the MSE and the accumulators), backward (`backward` and `from_grads`), the scheduler,
     the optimiser update; the epoch's closing host reads timed with them. This is the time
     the host spends issuing each phase; its epoch total against step 2's shows the copy
     is faithful.
  4. **The step, inline, synced**: the same loop with `Device::sync` before and after each
     phase: the time each phase takes to finish on the device.
  5. **The report**: per phase, the mean per step (and per epoch, ×50) unsynced and synced,
     with the share of the synced step; the epoch's host reads, evaluation and checkpoint
     (÷25, as `run_loop` checkpoints every 25 epochs); the library's epoch; the gap between
     the synced sum and the library epoch. If the unsynced (host) times match the synced
     ones, the step is host-bound: the device waits on kernel launches, not on work.
- The comparison is made by the report `docs/optimisers/results/CO0.3.md` from the three
  VM reports (below): per phase, Burn 0.23 against the fork and the ratio; where the extra
  time goes; candidate fixes for a later task.
- **The fork build.** No branch and no patch: on the VM a detached worktree at `4541f02`
  takes the example file from this branch (`git checkout origin/optimisers/co0-3 --
  examples/profile/adam_profile.rs`) and one `[[example]]` entry appended to its
  `Cargo.toml`; nothing is committed there. Checked on the Mac before the PR: the example
  builds and its smoke run completes at `4541f02` (flex), so the VM build is the only
  unknown left.
- Conventions: none.

## Tests that prove it
- Unit test in the example (run by `cargo test`): the per-phase summary (`Phases::mean`,
  the per-epoch scaling and the shares) against numbers by hand.
- The measurement validates itself: the unsynced inline epoch against the library's
  `train_epoch` epoch (the copy is faithful within 10 %, reported); the synced sum against
  the library epoch (the sync's own cost, reported, not a pass mark).
- Constraint 1: no library file changes (`git diff --stat origin/main` lists only the
  example, `Cargo.toml` and docs); the existing suite runs unedited.

## Checks
`/verify` (fmt, clippy with `--examples` and `--all-targets`, `cargo test`, `cargo doc`).
Smoke runs on the Mac (flex, `EPOCHS`/warm-up as built; timings not reported) with
`CARGO_TARGET_DIR=~/sciml-rs-target/co0-3`:
- this branch: `cargo run --release --example adam_profile -- target/co0-3-smoke.md`;
- the fork, in a scratch worktree at `4541f02` with the example copied in as above:
  `cargo run --release --example adam_profile -- target/co0-3-fork-smoke.md`.

## VM runs
Three runs, each a few minutes after its build (setup and stage 3's load under a minute;
9 epochs at up to about 12 s on Burn 0.23 synced, about 2–4 s on the fork). **Run alone,
the other GPU idle**: the Adam step is host-bound (Phase 9 T3), so any other process on the
machine moves the numbers. The orchestrator runs them (no session here uses ssh):

```sh
ssh cricket
tmux ls; nvidia-smi --query-compute-apps=pid,name --format=csv   # nothing else running
mkdir -p ~/sciml-rs/logs/co0-3
cd ~/sciml-rs && git fetch origin

# A: Burn 0.23 (this branch; TF32 off by cubek.toml, as CO0.1 ran)
git worktree add --detach ~/sciml-rs-wt/co0-3 origin/optimisers/co0-3
cd ~/sciml-rs-wt/co0-3
find ~/sciml-rs/datasets -mindepth 1 -maxdepth 1 ! -name README.md -exec ln -s {} datasets/ \;
source ~/sciml-gpu.sh
export CARGO_TARGET_DIR=~/sciml-rs-target/co0-3
tmux new -d -s co0-3 "CUDA_VISIBLE_DEVICES=0 cargo run --release --no-default-features --features cuda --example adam_profile -- ~/sciml-rs/logs/co0-3/burn023.md 2>&1 | tee ~/sciml-rs/logs/co0-3/burn023.log"

# B: the fork (after A has finished): 4541f02 with the example copied in, uncommitted
cd ~/sciml-rs
git worktree add --detach ~/sciml-rs-wt/co0-3-fork 4541f02
cd ~/sciml-rs-wt/co0-3-fork
git checkout origin/optimisers/co0-3 -- examples/profile/adam_profile.rs
printf '\n[[example]]\nname = "adam_profile"\npath = "examples/profile/adam_profile.rs"\n' >> Cargo.toml
find ~/sciml-rs/datasets -mindepth 1 -maxdepth 1 ! -name README.md -exec ln -s {} datasets/ \;
source ~/sciml-gpu.sh
export CARGO_TARGET_DIR=~/sciml-rs-target/co0-3-fork   # its own cache: another Burn, and the same binary name
tmux new -d -s co0-3-fork "CUDA_VISIBLE_DEVICES=0 cargo run --release --no-default-features --features cuda --example adam_profile -- ~/sciml-rs/logs/co0-3/fork.md 2>&1 | tee ~/sciml-rs/logs/co0-3/fork.log"

# C: Burn 0.23 with TF32 on (after B): #119 also turned TF32 off; this isolates that change
cd ~/sciml-rs-wt/co0-3
sed -i 's/allow_tf32 = false/allow_tf32 = true/' cubek.toml && grep allow_tf32 cubek.toml
source ~/sciml-gpu.sh
export CARGO_TARGET_DIR=~/sciml-rs-target/co0-3
tmux new -d -s co0-3-tf32 "CUDA_VISIBLE_DEVICES=0 cargo run --release --no-default-features --features cuda --example adam_profile -- ~/sciml-rs/logs/co0-3/burn023-tf32.md 2>&1 | tee ~/sciml-rs/logs/co0-3/burn023-tf32.log"
git checkout cubek.toml
```

The fork's target dir is a sibling of the one asked for (`co0-3-fork`): both builds name
their binary `release/examples/adam_profile`, and another Burn shares nothing with the
first build anyway. Afterwards the three `.md` reports (and logs) come back to the Mac; the
two worktrees are removed (`git -C ~/sciml-rs worktree remove ~/sciml-rs-wt/co0-3` and
`... co0-3-fork --force`, the latter holding the uncommitted copy). The runs decide nothing
on their own: they fill the profile table and the "where the extra time goes" list.

## Constraints and risks
- Constraint 1 (proposal §1.4): binds; met by changing no library code.
- The inline copy of `train_epoch`'s loop could drift from the library's; step 3's
  comparison against step 2 measures that.
- Syncing every phase removes the overlap of host and device, so the synced phases sum to
  more than the epoch; the unsynced host times and the library epoch bound the truth from
  both sides.
- The fork's Burn source comes from GitHub (`ax1s-x1zz/burn` at `ddfa9af`); the VM built it
  for CO0.1's bisect, so it is in its cargo cache.
- 2D objective: no output changes; nothing reads or writes a run directory but the
  example's own scratch checkpoint (in the system temp directory, removed at the end).

## Pending runs
A, B and C above (not started; the orchestrator runs them).

## Open questions
None blocking.

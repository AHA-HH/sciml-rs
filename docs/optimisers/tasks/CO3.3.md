# CO3.3 The SOAP arm and pilot

Started 2026-11-27 on `optimisers/co3-3`. Roadmap: "`OptimizerConfig::Soap`,
`OPTIMIZER=soap`; resume exact; then the pilot of D6 on seed 42 at the equal-time budget."
Proposal: §10 (O3); §1.4, §2 (constraints), §3 (protocol, passes), §5.3 (D6), §5.5 (code),
§6, §9 (SOAP); decisions D6 (open, this task proposes it), D7, D13 (signed off).

## Goal
Train the FNO with SOAP (`optim/soap/`, CO3.2) as an arm of `train_poisson` that
checkpoints, resumes exactly and counts its passes, then pilot D6's grid on the VM.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)
"a flex smoke run and its resume, constraint 1 shown, the pilot table, D6 proposed." From
CO3.2's questions (the author, 2026-11-25): the resume bit for bit on flex; on the GPU the
drift of a resumed run recorded, not asserted zero; the pilot logs each factor's eigenvalue
ratio at the first refresh for one seed (ratios near 1e7–1e8 go back to the author as an
amendment of D9). Proposal §9: "the state's record round trip and a resumed run bit for bit
equal to an uninterrupted one"; constraint 1: "the existing suite unedited, … an Adam and a
Gauss–Newton run on flex bit for bit equal before and after each change". §5.5: "an
`OptimizerConfig::Soap(SoapConfig)` variant builds `Soap` where Adam's `ModuleOptimizer` is
built; nothing else in the Adam path changes. `OPTIMIZER=soap` and the `_soap` run name in
`train_poisson`."

## Read first
`training/{config,trainer}.rs`, `training/trainer/tests.rs` (`resumed_run_matches_uninterrupted_run`,
`lbfgs_with_replicas_panics`), `optim/soap/mod.rs`, `examples/train/{common,poisson}.rs`,
`docs/optimisers/co0_1_runs.sh`, `docs/optimisers/results/CO0.1.md` (B = 2,952 s, pairing).

## Changes
- `training/config.rs`: `OptimizerConfig::Soap(SoapConfig)` (a new variant: old configs load
  unchanged); `TrainingConfig::soap()`. SOAP's weight decay is the `SoapConfig`'s own
  (decoupled, D13); `TrainingConfig`'s learning rate and schedule drive it as Adam's.
- `training/trainer.rs`: `build_components` builds `soap.init()` in place of Adam's
  `ModuleOptimizer`; everything after is Adam's path (`train_epoch`, the checkpoint's
  optimiser record, resume). One assertion: SOAP with replicas panics (data-parallel SOAP is
  untested; one device, as L-BFGS). Adam's, AdamW's, Gauss–Newton's and L-BFGS's branches
  untouched.
- `optim/soap/mod.rs`: `SoapConfig::log_first_refresh` (default false): at a parameter's
  first refresh, print each factor's dimension, size, and extreme eigenvalues and their
  ratio (f64 Jacobi on the host copy of Lₖ the refresh already reads). The author's pilot
  log; it changes no arithmetic.
- `examples/train/common.rs`: `OPTIMIZER=soap` (where the example allows it); `SOAP_FREQUENCY`
  (f, literal 10), `SOAP_LOG_FIRST_REFRESH=1`, `LEARNING_RATE` (any Adam-path arm; the pilot's
  lr); `WEIGHT_DECAY` is SOAP's decay; `TIME_BUDGET_S` allowed for SOAP (equal time);
  `run_costs`: SOAP's steps are the epochs `metrics.csv` holds × steps per epoch (a budget may
  stop it), one forward and one backward each (D7; its first step, which updates nothing,
  included).
- `examples/train/poisson.rs`: `soap` allowed, run name `_soap`, one device; literals: D6's
  start (lr 1e-3 cosine, weight decay 1e-4, β (0.95, 0.95), f 10).
- `examples/train/{burgers,darcy}.rs`: unchanged (their `allowed` lists stay).
- `docs/optimisers/co0_1_runs.sh`: a `soap:<seed>:<lr>:<f>:<budget_s>` spec, clearing the
  new variables before each run.
- Conventions: none. Dependencies: none.

## Tests that prove it
- Config: a `Soap` arm round-trips through `train_cfg.json`; old configs load as Adam
  (existing tests unedited).
- Trainer: a SOAP run of the tiny fixture (`resumed_run_matches_uninterrupted_run`'s, f = 2 so
  refreshes fall inside and across epochs) killed in epoch 2 and resumed from epoch 1's
  checkpoint equals the uninterrupted run (metrics and weights bit for bit on flex), and its
  trajectory differs from Adam's (the arm is really SOAP); SOAP with replicas panics.
- `optim/soap`: the Jacobi eigenvalues of a known symmetric matrix against its closed form
  (1e-12).
- Examples: `run_costs` is example code; the smoke run's `costs.json` is checked by hand
  (steps = epochs × ⌈N/20⌉, passes equal).
- Constraint 1: the existing suite unedited; `tests/fno_bit_identity.rs`'s four dumps
  (forward, Adam, Gauss–Newton, L-BFGS) equal on `origin/main` and the branch (`/verify`).

## Checks
`/verify` (fmt, the clippy runs, `cargo test`, `cargo doc`, datagen clippy and tests, the
bit-identity dumps); the trainer tests with `--features metal`. Flex smoke run
`OPTIMIZER=soap EPOCHS=3 train_poisson -- 4 65 16 42 20 0` and a resume:
`OPTIMIZER=soap EPOCHS=27`, the same arguments, `kill -9` after the epoch-25 checkpoint,
`--resume`, against the uninterrupted run.

## VM runs (the orchestrator runs them; this session never ssh-es)
The pilot (D6): seed 42, stage 3 (`0 129 64 42`: p 0, n 129, s 64, N 1,000), equal time
B = 2,952 s (CO0.1), weight decay 1e-4, EPOCHS 500 as cap (the cosine spans 500 epochs, so a
budget that stops SOAP early leaves its schedule unfinished, as with any timed arm); the grid
lr ∈ {1e-3, 3e-3} × f ∈ {10, 50}, four cells, two per GPU side by side (pairing accepted
2026-11-26, CO0.1), about 2 × 52 min. Every cell logs the first refresh's eigenvalue ratios
(seed 42 is the one seed). After the VM is free (booked to ~14:30 UTC 2026-11-28 for CL4.2):

```sh
cd ~/sciml-rs && git fetch origin
git worktree add --detach ~/sciml-rs-wt/co3-3 origin/optimisers/co3-3
cd ~/sciml-rs-wt/co3-3
find ~/sciml-rs/datasets -mindepth 1 -maxdepth 1 ! -name README.md -exec ln -s {} datasets/ \;
mkdir -p ~/sciml-rs/logs/co3-3
tmux new -d -s co33-g0 "source ~/sciml-gpu.sh; export CARGO_TARGET_DIR=~/sciml-rs-target/co3-3; \
  LOGDIR=~/sciml-rs/logs/co3-3 GPU=0 bash docs/optimisers/co0_1_runs.sh \
  soap:42:1e-3:10:2952 soap:42:1e-3:50:2952 2>&1 | tee ~/sciml-rs/logs/co3-3/g0.log"
sleep 120   # the first instance builds; the second then finds the binaries up to date
tmux new -d -s co33-g1 "source ~/sciml-gpu.sh; export CARGO_TARGET_DIR=~/sciml-rs-target/co3-3; \
  LOGDIR=~/sciml-rs/logs/co3-3 GPU=1 bash docs/optimisers/co0_1_runs.sh \
  soap:42:3e-3:10:2952 soap:42:3e-3:50:2952 2>&1 | tee ~/sciml-rs/logs/co3-3/g1.log"
```

The script passes `--no-default-features --features cuda`. Run directories land in the
worktree's `runs/` (`poisson_fno_n129_s64_p0_soap_<secs>`, two started in one second get
CO0.4's suffix); `~/sciml-rs/logs/co3-3/progress.txt` gets one `done` line per cell. Then per
run directory, on GPU 0:

```sh
cd ~/sciml-rs-wt/co3-3 && source ~/sciml-gpu.sh && export CARGO_TARGET_DIR=~/sciml-rs-target/co3-3
for d in $(sed -n 's/^done soap[^ ]* \([^ ]*\) .*/\1/p' ~/sciml-rs/logs/co3-3/progress.txt); do
  CUDA_VISIBLE_DEVICES=0 cargo run --release --no-default-features --features cuda \
    --example predict_poisson -- "$d" 2>&1 | tee -a ~/sciml-rs/logs/co3-3/predict.log
done
grep -h "first refresh" ~/sciml-rs/logs/co3-3/soap_42_1e-3_10_2952.log | sort -u > ~/sciml-rs/logs/co3-3/spectra.txt
```

What comes back: each run's `metrics.csv`, `costs.json`, `errors.json`, the logs and
`spectra.txt`. The pilot table: per cell, epochs reached, training time, final train and test
L², error 4, steps and passes, median epoch time against Adam's 5.97 s; the largest
eigenvalue ratio per factor shape. D6 proposed from it (the cell with the lowest error 4;
ties to the cheaper f). A GPU resume drift check (recorded, not asserted): none planned
beyond the flex resume unless the author asks; the pilot does not resume.

## Constraints and risks
- Constraint 1 (§1.4): a new variant and its arm; Adam's construction and loop are reached
  unchanged when `optimizer` is anything but `Soap`. Shown by the existing suite and the
  bit-identity dumps.
- `LEARNING_RATE` is new for every arm of the three examples' `common.rs`; unset, the literal.
- SOAP's per-step host refresh (every f steps, every factor to the host and back) may make
  its epoch slower than Adam's 5.97 s; at equal time it then trains fewer epochs and its cosine
  ends unfinished. Recorded in the table; if large, a question for the author (an `EPOCHS`
  matched to the budget instead).
- R3 (§8): the host refresh is f64 and the factors f32 (D9); the eigenvalue log is the check.
- `poisson_matrix --by-optimizer` labels a SOAP run as Adam until CO4.1 extends it; the pilot
  table is built from the run files, not from it.

## Pending runs
The pilot (above): waiting for the VM.

## Open questions
None yet.

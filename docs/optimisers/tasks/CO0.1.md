# CO0.1 Reference arms on Burn 0.23

Started 2026-11-25 on `optimisers/co0-1`. Roadmap: "**CO0.1 Reference arms on Burn 0.23.**
Adam (Phase 5 settings) on seeds 42–46, 500 epochs, giving the equal-time budget; Adam at
5,400 s on seeds 42–44; GN200-D at μ = 0 with CG cap 50 and 150 on seed 42 at both budgets,
then the chosen cap on seeds 43–46 at equal time and 43–44 at 5,400 s (D2). Timed runs side
by side on the two GPUs only after a check that the epoch times match a run alone (the step
is host-bound, Phase 9 T3)." Proposal: §10; §1.4, §2, §3 (protocol), §7, §12 (D1, D2).

## Goal
Re-measure Phase 10's reference arms (Adam, and GN200-D at μ = 0) on Burn 0.23, so every
later arm of the objective is compared on the same Burn, and set the equal-time budget.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)
"a results table (`docs/optimisers/results/CO0.1.md`) against Phase 10's fork-era numbers,
and D1 proposed." *Sign-off:* D1.

Proposal §3: "*equal time*, the mean training time of O0's Adam reruns (Phase 10's was
859.4 s); and *long*, 5,400 s … Every arm runs at both, so the long budget also answers
whether Adam improves with 6.3× the time". §12, D1: "GN200-D at μ = 0; CG cap 50 or 150
from CO0.1". Reported per run (§3): error 4 and error 3, final train and test L² and the
gap, updates, forward and backward passes, training time, peak memory.

## Read first
- `docs/poisson2d/phase10/results.md` (the fork-era table and follow-ups this is compared
  with), `docs/poisson2d/phase9/results.md` (host-bound step).
- `examples/train/poisson.rs` (`OPTIMIZER`, `GN_*`, `TIME_BUDGET_S`, `WEIGHT_DECAY`).
- `examples/predict/poisson_matrix.rs` (`--by-optimizer`, `optimizer_arm`).
- `docs/poisson2d/phase7/t2_matrix.sh` (the run-script pattern: one GPU, peak memory by
  `nvidia-smi`, resumable progress file).

## Changes
1. **`EPOCHS` for Adam's long budget** (`examples/train/poisson.rs` only). Adam takes no
   time budget, by design: its cosine schedule spans `EPOCHS`, and a budget would stop it
   partway down the schedule. Adam at 5,400 s is therefore Adam given more epochs, with the
   schedule stretched over them: `EPOCHS=<E>` overrides the literal (saved in
   `train_cfg.json`, so `--resume` keeps it), for Adam runs only (a hybrid run's
   `HYBRID_ADAM_EPOCHS` is a fixed half of the literal, and Gauss–Newton runs stop by
   `TIME_BUDGET_S`). E = round(5,400 × 500 / B), B the equal-time budget, so the run takes
   about 5,400 s at the equal-time runs' epoch time; the training time it actually took is
   reported, not E. On the fork's numbers, B = 859.4 s gives E = 3,142.
2. **`docs/optimisers/co0_1_runs.sh`**, the run script (Phase 7's pattern): runs a list of
   run specs one after another on `GPU`, writes peak memory into each `costs.json`, logs
   to `~/sciml-rs/logs/co0-1/`, skips specs already done, and records spec → run directory
   in `progress.txt`.
3. **`docs/optimisers/results/CO0.1.md`**, written from the runs: one
   `poisson_matrix --by-optimizer` table per group (the tool labels arms by batch and
   preconditioner only, so cap 50 and 150, and the two budgets, would merge into one arm;
   extending the labels is CO4.1's), assembled with Phase 10's rows beside them.

No library change, no public API, no conventions, no dependency.

## Tests that prove it
- Constraint 1: with `EPOCHS` unset, `train_poisson` saves the same configs as on
  `origin/main` (`train_cfg.json`, `model_cfg.json`, `data_cfg.json` byte for byte, flex,
  stage 3, `-- 0 129 64`, stopped after the configs are written); the configs are the
  example's only input to training, so the trained outputs are unchanged. No test file: the
  example has none, and the change is one defaulted variable.
- `EPOCHS=3` with Adam on flex: the run's `train_cfg.json` has `epochs: 3`; with
  `OPTIMIZER=gauss-newton` it panics with the message.
- The runs themselves are the measurement; no timing is asserted.

## Checks
`/verify` (fmt, clippy with examples, `cargo test`, docs); `bash -n` on the script.

## VM runs
All on the VM (one A100 per run, cuda, f32, TF32 off by `cubek.toml`), stage 3, s = 64,
p = 0, N = 1000, from the worktree `~/sciml-rs-wt/co0-1` at this branch's head. **Nothing
starts before the author's go, and nothing timed starts while CL0.1's timing run (tmux
`co31-cl01`, GPU 0) or any other timed run is on the VM.**

Setup (once):
```sh
cd ~/sciml-rs && git fetch origin
git worktree add --detach ~/sciml-rs-wt/co0-1 origin/optimisers/co0-1
cd ~/sciml-rs-wt/co0-1
find ~/sciml-rs/datasets -mindepth 1 -maxdepth 1 ! -name README.md -exec ln -s {} datasets/ \;
```
The script builds `train_poisson` and `predict_poisson` itself and runs copies of them in
`~/sciml-rs/logs/co0-1/bin-<commit>/`, so another worktree building into the shared
`CARGO_TARGET_DIR` cannot swap the binary between specs (the run directories land in this
worktree's `runs/`, `CARGO_MANIFEST_DIR` being fixed at compile time). Each tmux command
sources the Spack environment itself, since a running tmux server does not pass on the
client's environment; the script clears every optimiser variable before each run.

Run specs (the script's arguments): `adam:<seed>` (500 epochs), `adam:<seed>:<E>` (E
epochs), `gn:<seed>:<cap>:<budget_s>` (GN200-D: `OPTIMIZER=gauss-newton GN_BATCH_SIZE=200
GN_PRECONDITIONER=hutchinson GN_WEIGHT_DECAY=0 GN_CG_MAX_ITERS=<cap>
TIME_BUDGET_S=<budget_s>`). Every run is `train_poisson -- 0 129 64 <seed>`.

**Stage 1, Adam at equal time, and the side-by-side check** (about 0.75 h wall, 1.2 GPU-h):
```sh
tmux new -d -s co01-g0 "source ~/sciml-gpu.sh; export CARGO_TARGET_DIR=~/sciml-rs-target/optimisers; cd ~/sciml-rs-wt/co0-1; GPU=0 bash docs/optimisers/co0_1_runs.sh adam:42"
# when it is done, the pair:
tmux new -d -s co01-g0 "source ~/sciml-gpu.sh; export CARGO_TARGET_DIR=~/sciml-rs-target/optimisers; cd ~/sciml-rs-wt/co0-1; GPU=0 bash docs/optimisers/co0_1_runs.sh adam:43"
tmux new -d -s co01-g1 "source ~/sciml-gpu.sh; export CARGO_TARGET_DIR=~/sciml-rs-target/optimisers; cd ~/sciml-rs-wt/co0-1; GPU=1 bash docs/optimisers/co0_1_runs.sh adam:44"
# if the check holds, the next pair:
tmux new -d -s co01-g0 "source ~/sciml-gpu.sh; export CARGO_TARGET_DIR=~/sciml-rs-target/optimisers; cd ~/sciml-rs-wt/co0-1; GPU=0 bash docs/optimisers/co0_1_runs.sh adam:45"
tmux new -d -s co01-g1 "source ~/sciml-gpu.sh; export CARGO_TARGET_DIR=~/sciml-rs-target/optimisers; cd ~/sciml-rs-wt/co0-1; GPU=1 bash docs/optimisers/co0_1_runs.sh adam:46"
```
The check: the median `epoch_time_s` of seeds 43 and 44 (`metrics.csv`) each within 3 %
of seed 42's (alone). Phase 10's lone Adam runs spread ±1.5 % in total time. If it holds,
45 and 46 run as a pair; if not, 43 and 44 run again alone (their progress lines removed
first), and every timed run of this task runs alone (about 13 h wall instead of 8). Then
B = mean `training_time_s` of the five, E = round(5,400 × 500 / B).

**Stage 2, seed 42 at both caps, and Adam long** (about 3.5 h wall, 6.5 GPU-h):
```sh
# equal time, alone, one after the other: each is the pass-rate reference of its cap
tmux new -d -s co01-g0 "source ~/sciml-gpu.sh; export CARGO_TARGET_DIR=~/sciml-rs-target/optimisers; cd ~/sciml-rs-wt/co0-1; GPU=0 bash docs/optimisers/co0_1_runs.sh gn:42:50:<B> gn:42:150:<B>"
# then the long runs as pairs
tmux new -d -s co01-g0 "source ~/sciml-gpu.sh; export CARGO_TARGET_DIR=~/sciml-rs-target/optimisers; cd ~/sciml-rs-wt/co0-1; GPU=0 bash docs/optimisers/co0_1_runs.sh gn:42:50:5400 adam:42:<E>"
tmux new -d -s co01-g1 "source ~/sciml-gpu.sh; export CARGO_TARGET_DIR=~/sciml-rs-target/optimisers; cd ~/sciml-rs-wt/co0-1; GPU=1 bash docs/optimisers/co0_1_runs.sh gn:42:150:5400 adam:43:<E>"
```
GN's side-by-side check comes after the fact: passes per second of each long GN run
within 3 % of its cap's equal-time run alone (`costs.json`), and the Adam long runs' median
epoch time within 3 % of stage 1's. A run that fails is run again alone (up to 3 GPU-h at
risk). Unlike Adam's, the GN check is not made before the first GN pair, and the overlaps
of different arms (a GN run ending beside an Adam run that has started) are covered only
by these after-the-fact checks; the PR says so.

**Stop: D1.** The cap is proposed to the author from seed 42 (lower error 4 at equal time
first, as the matrix's primary budget; the long budget second), and the author chooses.

**Stage 3, the chosen cap C** (about 3.5 h wall, 5.5 GPU-h):
```sh
tmux new -d -s co01-g0 "source ~/sciml-gpu.sh; export CARGO_TARGET_DIR=~/sciml-rs-target/optimisers; cd ~/sciml-rs-wt/co0-1; GPU=0 bash docs/optimisers/co0_1_runs.sh gn:43:C:<B> gn:45:C:<B> gn:43:C:5400"
tmux new -d -s co01-g1 "source ~/sciml-gpu.sh; export CARGO_TARGET_DIR=~/sciml-rs-target/optimisers; cd ~/sciml-rs-wt/co0-1; GPU=1 bash docs/optimisers/co0_1_runs.sh gn:44:C:<B> gn:46:C:<B> gn:44:C:5400 adam:44:<E>"
```
The equal-time GN pairs rely on stage 2's GN check; a slowdown there would show as fewer
passes, which the same check on their `costs.json` catches.

**Evaluation**, only once no timed run is on the VM (not timed itself), with the copied
binary:
```sh
cd ~/sciml-rs-wt/co0-1; source ~/sciml-gpu.sh
P=$(ls -d ~/sciml-rs/logs/co0-1/bin-* | tail -1)/predict_poisson
for d in $(awk '/^done /{print $3}' ~/sciml-rs/logs/co0-1/progress.txt); do "$P" "$d"; done
mkdir -p ~/sciml-rs/runs/co0-1
for d in $(awk '/^done /{print $3}' ~/sciml-rs/logs/co0-1/progress.txt); do cp -r "$d" ~/sciml-rs/runs/co0-1/; done
```
Then the run directories go to the Mac (`scp -r cricket:sciml-rs/runs/co0-1 runs/`), and
the tables are made there by `poisson_matrix --by-optimizer`, one call per group (Adam
equal, Adam long, each cap × budget), with `progress.txt` mapping specs to directories.

**Total: about 13.2 GPU-hours on the fork's times** (Adam 5 × 0.24 h + 3 × 1.5 h; GN
6 × 0.25 h + 4 × 1.5 h), about 8 h wall on the two A100s if the side-by-side checks hold,
13 h if not. **Superseded by stage 1:** Adam on Burn 0.23 takes about 3,070 s, so about 20
GPU-hours with the 5,400 s budget kept, 47 with it scaled (open question 1).

What it decides: B and E for every later pilot and the matrix (CO2.2, CO3.3, CO4.1); D1;
whether Burn 0.23 reproduces Phase 10's arms.

## Constraints and risks
- Constraint 1: the only code change is a defaulted variable in an example; unset, the
  configs are identical (test above). Adam and Gauss–Newton paths in `training/` untouched.
- CM0.4 runs in parallel in `models/`, `runs/`, `tracking/`: no overlap.
- Side by side may slow the host-bound step; the checks above decide, at the cost of
  rerunning a pair alone.
- Adam's long runs end near, not at, 5,400 s (E from B's epoch time); reported as taken.
- One seed decides the cap (D1), as the roadmap plans; the decision is the author's.

## Pending runs
- **Stage 1 done 2026-11-26** (seeds 42 and 44 at `29a21e2`, 43, 45, 46 at `911a64e`;
  the copy race of 2026-11-25 fixed in `9609232`): `docs/optimisers/results/CO0.1.md`.
  B = 2,952 s (wall-clock mean; `costs.json` gives 2,949.1 s), E = 915.
- **Stage 2 running** since 2026-11-26 11:29:56 UTC on the VM (tmux `co01-s2`, driver
  `~/sciml-rs/logs/co01-s2.sh`, commit `911a64e`, started by the coordinating session at
  the author's go): `gn:42:50:2952` then `gn:42:150:2952` alone on GPU 0; then
  `gn:42:50:5400` and `adam:42:915` on GPU 0 beside `gn:42:150:5400` and `adam:43:915` on
  GPU 1. About 4.5 h. Decides D1 (proposed from seed 42), and Adam's long budget on 42–43.
- **Stage 3** after D1.

## Open questions
2. **Answered 2026-11-26 by the author: pairing accepted.** Stage 1's paired runs were
   2.9–3.1 % faster than seed 42 alone (seed 43 outside the 3 % band by 0.14 %, on the
   fast side), and the lone runs spread over 10 %; the rule is there to catch a slowdown,
   and there is none.
1. **Answered 2026-11-25 by the author: keep 5,400 s** (D2 amended; option A below), and
   the Adam step profile added to the roadmap as CO0.3. **Adam is about 3.6× slower on Burn 0.23.** Seed 42: 3,067 s for 500 epochs (6.1 s
   per epoch at the end, peak 3,048 MiB), against 859.4 s on the fork (Phase 10, same
   settings, A100); final test L² 9.63e-3 (the fork's seed 42: 1.095e-2). CO0.2's first
   attempt put a GN200 step at cap 50 only about 24 % slower. So B ≈ 3,070 s, not 860 s,
   and 5,400 s is now 1.76 × B, not 6.3 × B; at equal time GN gets about 3.6 × the fork's
   work against Adam. What it means for D2's long budget, put to the author before stage 2
   (one seed so far; seed 44's run, alone on GPU 1, will confirm the epoch time):
   - **A (recommended): keep 5,400 s.** The long GN runs stay comparable with Phase 10's
     5,400 s runs (3.03e-3 at cap 150, 5.58e-3 at cap 50), since GN's speed barely moved;
     E = round(5,400 × 500 / B) ≈ 880 epochs. The long budget then asks only how far each
     arm goes in 1.5 h, no longer "Adam with 6.3× the time". About 20 GPU-hours in all
     (equal time 11 runs × 0.85 h = 9.4; long 7 × 1.5 h = 10.5), about 11 h wall paired.
   - **B: scale to 6.3 × B ≈ 19,300 s** (5.4 h per run), keeping D2's ratio: about 47
     GPU-hours (long 7 × 5.4 h = 37.6), about 24 h wall paired; E stays 3,142 epochs.
   - Either way, the slowdown is Burn 0.23's per-step overhead at batch 20 (GN at batch
     200 barely slowed), so equal time now depends on the Burn version; worth a profile of
     an Adam step beside CO0.2's GN profile (a roadmap line if the author wants it).

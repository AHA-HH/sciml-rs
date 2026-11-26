# CO1.3 The Burgers 2 × 2

Started 2026-11-26 on `optimisers/co1-3`. Roadmap: "CO1.3 The Burgers 2 × 2. A0, A1, G0, G1
and G0-200 (G0 at batch 200, D12) on seeds 42–46 at A1's mean time and 42–44 at 6 × it
(proposal §3.1, D2); the coupled-Adam cells if D11 keeps them." Proposal: §3.1, §7 (O1),
§10; decisions D2, D10, D11, D12 (all signed off).

## Goal
Run the seven Burgers cells (A0, A1, G0, G1, G0-200 and D11's coupled-Adam C0, C1) at both
budgets on the VM and write the verdict on PETScML's gap: A1 → G0 split into A1 → A0 (weight
decay) and A0 → G0 (the optimiser).

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)
"`docs/optimisers/results/CO1.3.md`: test L² per cell (mean and range), against epochs,
passes, PETScML's oracle calls and time; A1 against D10's band; the verdict on A1 → G0
split into A1 → A0 (weight decay) and A0 → G0 (the optimiser)."

Proposal §3.1: "A1's mean test L² over seeds 42–46 within a factor 2 of PETScML's 0.0014
counts as reproducing their reference. Outside that band, O1 reports the gap, checks the
data split and the parameter count first, and goes on"; "Every cell at A1's mean training
time (equal time, seeds 42–46) and at 6 × that (long, seeds 42–44) … at the long budget
AdamW runs 6 × the epochs with the halving every 6 × 100 (D11)"; "the verdict on PETScML's
gap uses G0"; the coupled-Adam cells "tell whether weight decay helps a stochastic
first-order method whether coupled or decoupled (D13's hypothesis)".

## Read first
- Proposal §3.1 (the table, the 2 × 2, acceptance, budgets), §7 (O1: 40 runs, 56 with the
  coupled-Adam cells).
- `docs/optimisers/results/CO1.1.md` (A1 seed 42 on metal: 7.11e-4 final, 7.07e-4 best).
- `docs/optimisers/tasks/CO1.2.md` "VM runs" (the cells' variables, `EPOCHS` as GN's cap).
- `examples/train/burgers.rs`, `examples/train/common.rs` (the variables).
- CO0.1's plan on `optimisers/co0-1` (#139): its run script and the side-by-side check,
  which this task's script follows.

## Changes
- `docs/optimisers/co1_3_runs.sh` (new): CO0.1's script adapted to `train_burgers`, one
  spec per cell: `a1|a0|c1|c0:<seed>[:<epochs>]`, `g1|g0|g0b200:<seed>:<budget_s>`. Every
  run has `RECIPE=petscml`; every training variable is cleared first; GN runs get
  `EPOCHS=100000` so only the budget stops them (CO1.2's note); a run whose directory
  another done line names is marked FAILED (the RunDir collision, until CO0.4).
- `docs/optimisers/results/CO1.3.md` (after the runs).
- No library or example code: CO1.2 already runs every cell. Conventions: none.

| cell | variables (besides `RECIPE=petscml SEED=<seed>`) |
| --- | --- |
| A1 | `OPTIMIZER=adamw WEIGHT_DECAY=1e-4` (long: `EPOCHS=3000`, halving every 600) |
| A0 | `OPTIMIZER=adamw WEIGHT_DECAY=0` (long: `EPOCHS=3000`) |
| C1 | `OPTIMIZER=adam WEIGHT_DECAY=1e-4` (coupled L2, cosine; long: `EPOCHS=3000`) |
| C0 | `OPTIMIZER=adam WEIGHT_DECAY=0` (long: `EPOCHS=3000`) |
| G1 | `OPTIMIZER=gauss-newton GN_WEIGHT_DECAY=1e-4 TIME_BUDGET_S=<B> EPOCHS=100000` |
| G0 | `OPTIMIZER=gauss-newton GN_WEIGHT_DECAY=0 TIME_BUDGET_S=<B> EPOCHS=100000` |
| G0-200 | G0 and `GN_BATCH_SIZE=200` |

B is T (A1's mean `training_time_s` over seeds 42–46, from `costs.json`) at equal time and
6T at the long budget. The A and C cells are budgeted by epochs (500 and 3,000), not time:
Adam's schedules span the epochs (`TIME_BUDGET_S` is Gauss–Newton only).

## Tests that prove it
- No code, so no new Rust tests. The script: `bash -n`, and a dry run on the Mac with a stub
  binary and a stub `nvidia-smi` (done 2026-11-26): each cell's variables reach the binary,
  a `WEIGHT_DECAY` and a `GN_CG_MAX_ITERS` set in the caller's environment do not, an
  unknown cell and a reused run directory are each marked FAILED.
- The results: every run's `train_cfg.json` checked against its cell (optimizer, weight
  decay, μ, batch size, epochs, seed) before it enters a table, since run names do not tell
  the cells apart; every run directory named by exactly one done line.

## Checks
`/verify` (a docs and script change: the CI path filter skips the jobs; fmt and clippy have
nothing to check); `bash -n docs/optimisers/co1_3_runs.sh`.

## VM runs
All on the VM, one A100 per run, cuda, f32, TF32 off (`cubek.toml`), from the worktree
`~/sciml-rs-wt/co1-3` at this branch's head. **Nothing starts before the author's go, and
nothing timed starts while CO0.1's stage 3 is on the VM (tmux `co01-s3g0`, `co01-s3g1`,
until about 03:40 UTC 2026-11-27).**

Setup (once):
```sh
cd ~/sciml-rs && git fetch origin
git worktree add --detach ~/sciml-rs-wt/co1-3 origin/optimisers/co1-3
cd ~/sciml-rs-wt/co1-3
find ~/sciml-rs/datasets -mindepth 1 -maxdepth 1 ! -name README.md -exec ln -s {} datasets/ \;
mkdir -p ~/sciml-rs/logs
```
The script builds `train_burgers` into `~/sciml-rs-target/co1-3` (this task's own build
directory) and runs a copy from `~/sciml-rs/logs/co1-3/bin-<commit>/`; logs, progress and
peak memory go to `~/sciml-rs/logs/co1-3/`. Each tmux command sources the Spack environment
itself. The tmux wrappers `tee` to `~/sciml-rs/logs/co1-3-<stage>-g<gpu>.log`.

**Run-directory collisions.** Directories are `burgers_fno[_seed<n>][_gn]_<unix_secs>`
(seed 42 has no tag), and a directory that exists is reused. The queues below never run the
same name on both GPUs at once: within a stage, the two GPUs take different seeds, and
where a seed is on both (44) one GPU takes it first and the other last, never overlapping;
GPU 1 also starts 60 s after GPU 0. The script marks any reused directory FAILED.

T ≈ 2,500 s is a guess for the cost estimates below (25,000 steps of batch 20, host-bound
like CO0.1's Poisson Adam at about 2,950 s on Burn 0.23; A1 seed 42 took 498 s on the Mac's
metal); stage 1 measures it.

**Stage 1: the A and C cells at equal time** (20 runs, 20T ≈ 14 GPU-h, wall 10T ≈ 7 h):
```sh
tmux new -d -s co13-s1g0 "source ~/sciml-gpu.sh; export CARGO_TARGET_DIR=~/sciml-rs-target/co1-3; cd ~/sciml-rs-wt/co1-3; GPU=0 bash docs/optimisers/co1_3_runs.sh a1:42 a1:43 a0:42 a0:43 c1:42 c1:43 c0:42 c0:43 a0:44 c0:44 2>&1 | tee -a ~/sciml-rs/logs/co1-3-s1-g0.log"
sleep 60
tmux new -d -s co13-s1g1 "source ~/sciml-gpu.sh; export CARGO_TARGET_DIR=~/sciml-rs-target/co1-3; cd ~/sciml-rs-wt/co1-3; GPU=1 bash docs/optimisers/co1_3_runs.sh a1:44 c1:44 a1:45 a1:46 a0:45 a0:46 c1:45 c1:46 c0:45 c0:46 2>&1 | tee -a ~/sciml-rs/logs/co1-3-s1-g1.log"
```
Then T = mean `training_time_s` of the five A1 runs, rounded to 0.1 s. Side-by-side check
(as CO0.1's): the median `epoch_time_s` of the A1 runs on GPU 1 within 3 % of those on
GPU 0; if it fails, the author decides whether the timed GN runs go alone.

**Stage 2: the Gauss–Newton cells at equal time** (15 runs, about 15T ≈ 10.5 GPU-h, wall 8T
≈ 5.6 h; each run ends after the first epoch at or over T, a full-batch step past it):
```sh
tmux new -d -s co13-s2g0 "source ~/sciml-gpu.sh; export CARGO_TARGET_DIR=~/sciml-rs-target/co1-3; cd ~/sciml-rs-wt/co1-3; GPU=0 bash docs/optimisers/co1_3_runs.sh g0:42:<T> g1:42:<T> g0b200:42:<T> g0:43:<T> g1:43:<T> g0b200:43:<T> g0:44:<T> 2>&1 | tee -a ~/sciml-rs/logs/co1-3-s2-g0.log"
sleep 60
tmux new -d -s co13-s2g1 "source ~/sciml-gpu.sh; export CARGO_TARGET_DIR=~/sciml-rs-target/co1-3; cd ~/sciml-rs-wt/co1-3; GPU=1 bash docs/optimisers/co1_3_runs.sh g1:44:<T> g0b200:44:<T> g0:45:<T> g1:45:<T> g0b200:45:<T> g0:46:<T> g1:46:<T> g0b200:46:<T> 2>&1 | tee -a ~/sciml-rs/logs/co1-3-s2-g1.log"
```

**Stage 3: every cell at the long budget, seeds 42–44** (21 runs, 126T ≈ 88 GPU-h, wall
66T ≈ 46 h; GPU 0 takes seed 42 and seed 44's GN cells, GPU 1 seed 43 and seed 44's A and C
cells, so no name is ever on both):
```sh
tmux new -d -s co13-s3g0 "source ~/sciml-gpu.sh; export CARGO_TARGET_DIR=~/sciml-rs-target/co1-3; cd ~/sciml-rs-wt/co1-3; GPU=0 bash docs/optimisers/co1_3_runs.sh a1:42:3000 a0:42:3000 c1:42:3000 c0:42:3000 g0:42:<6T> g1:42:<6T> g0b200:42:<6T> g0:44:<6T> g1:44:<6T> g0b200:44:<6T> 2>&1 | tee -a ~/sciml-rs/logs/co1-3-s3-g0.log"
sleep 60
tmux new -d -s co13-s3g1 "source ~/sciml-gpu.sh; export CARGO_TARGET_DIR=~/sciml-rs-target/co1-3; cd ~/sciml-rs-wt/co1-3; GPU=1 bash docs/optimisers/co1_3_runs.sh a1:43:3000 a0:43:3000 c1:43:3000 c0:43:3000 g0:43:<6T> g1:43:<6T> g0b200:43:<6T> a1:44:3000 a0:44:3000 c1:44:3000 c0:44:3000 2>&1 | tee -a ~/sciml-rs/logs/co1-3-s3-g1.log"
```

In all: 56 runs, about 161T ≈ 112 GPU-h, about 84T ≈ 58 h of wall time on the two A100s
(at T = 2,500 s; it scales with T). The long budget is 78 % of it. Stages 2 and 3 can be
queued on a GPU as soon as T is known; each run is checkpointed every 25 epochs, and a
rerun of the same command skips the specs already done (a run cut off mid-way starts again
from scratch through the script; `--resume runs/<dir>` by hand continues it).

**Fetch.** When a stage is done: copy `~/sciml-rs-wt/co1-3/runs/burgers_fno*` to
`~/sciml-rs/runs/co1-3/`, and `~/sciml-rs/logs/co1-3/` (progress, logs, memory samples) to
the Mac's `runs/co1-3/`.

## The results (after the runs)
`docs/optimisers/results/CO1.3.md`, from each run's `metrics.csv` (`test_l2`, the mean
relative L², PETScML's eq. 7), `costs.json` (`training_time_s`, `optimisation_steps`,
passes, peak memory), `gn_steps.csv` and `train_cfg.json`:
- per cell and budget: final test L² and best so far (PETScML's legend metric), mean and
  min–max over seeds; train L² and the gap; time, epochs, steps, forward and backward
  passes, PETScML's oracle calls, peak memory;
- the curves of each cell's mean test L² against epochs, passes, oracle calls and time (a
  table at fixed fractions of the budget; plots if gnuplot is at hand);
- **oracle calls** by PETScML's rule (a gradient 2, a Gauss–Newton product 2, both on the
  whole training set): every gradient and every J v product pair contains exactly one
  backward pass, so oracle calls = 2 × backward passes × B / N (B the batch, N = 1,000).
  Our central-difference J v spends two forwards where their forward mode spends one, so our
  passes overstate a GN step's cost against theirs; the Hutchinson probes (one VJP each) are
  counted as oracle calls too;
- D10: A1's mean final test L² over seeds 42–46 against [7e-4, 2.8e-3];
- the verdict: A1 → A0 (weight decay) and A0 → G0 (the optimiser), at both budgets, with
  G1 against G0 (Phase 10's trap), G0-200 against G0, and C0/C1 against A0/A1 (coupled
  against decoupled weight decay, D13's hypothesis);
- per GN run: NaN or rejected steps, ρ outliers, λ's range (`gn_steps.csv`).

## Constraints and risks
- Constraint 1: no library or example code changes, so every output is unchanged.
- The 2D objective: untouched.
- `training/`: untouched (CO2.1, CO1.5, CM0.6 and CO3.3 may take it).
- Risks: (1) T is a guess; the cost scales with it. (2) A full-batch Burgers GN step on
  cuda has not been run (CO1.2 ran it on flex): stage 2's first run shows its step time
  and memory. (3) Two runs per GPU pair may slow each other (the step is host-bound);
  stage 1's check covers the A1 runs, the GN runs only after the fact, as in CO0.1. (4) The
  A and C cells at 3,000 epochs take about 6T only if the epoch time stays constant.

## Pending runs
Stages 1–3 above, none started; waiting for the author's go and for CO0.1's stage 3 to end.

## Open questions
1. The go for stage 1 (about 14 GPU-h), and whether stages 2 and 3 follow on the author's
   go after T is known, or start on their own.

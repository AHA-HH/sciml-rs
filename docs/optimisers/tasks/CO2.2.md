# CO2.2 L-BFGS pilot

Started 2026-11-27 on `optimisers/co2-2`. Roadmap: "CO2.2 L-BFGS pilot. Seed 42 at the
equal-time budget: history ∈ {10, 100}, `max_iter` ∈ {1, 20}, strong Wolfe. `max_eval` is set
with each cell (`LBFGS_MAX_EVAL`) … *Done when:* the pilot table and D5 proposed. *Needs:*
CO2.1, CO0.1. *Machines:* the VM. *Sign-off:* D5." Proposal (`docs/design/optimisers.md`):
§3 (tuning budget: one seed, the equal-time budget, the grid of D5), the L-BFGS settings
(D5: strong Wolfe, lr 1, history and `max_iter` from the pilot), the memory estimate
(2 × 100 history vectors ≈ 1.9 GB), §12 D5.

## Goal
Run L-BFGS's four-cell pilot on the VM at CO0.1's equal-time budget and propose D5 from it.

## Acceptance
Roadmap *Done when*, verbatim: "the pilot table and D5 proposed." Proposal: "the new arms get
a fixed pilot (one seed, the equal-time budget, a short grid named in D5 and D6) before the
matrix"; D5 "strong Wolfe, lr 1; history and `max_iter` from the pilot".

## Read first
- `docs/optimisers/tasks/CO2.1.md` (the arm, the smoke run, the `max_eval` stall).
- `docs/optimisers/tasks/CO0.1.md` and its results (B = 2,952 s, Adam's and GN200-D's
  errors at B, the run script).
- `examples/train/poisson.rs` (the `LBFGS_*` literals: history 100, `max_iter` 20,
  `max_eval` Burn's default, μ 0) and `examples/train/common.rs` (`LBFGS_*`, `EPOCHS`,
  `TIME_BUDGET_S`).
- Burn's `LBFGS::step` (`burn-optim-0.23.0-pre.2/src/optim/lbfgs.rs`): `max_eval` counts
  the step's first closure evaluation and passes `max_eval − 1` to the strong Wolfe search
  as `max_ls`; the search makes one trial before its loop and up to `max_ls` more, so at
  `max_iter` 1 it gets up to `max_eval` evaluations and the step up to `max_eval` + 1.
  `max_eval` is not a hard cap per step.

## Changes
- No Rust. `OPTIMIZER=lbfgs` and the `LBFGS_*` variables (CO2.1, on `main`) cover every cell.
- `docs/optimisers/co0_1_runs.sh`: a spec `lbfgs:<seed>:<history>:<max_iter>:<max_eval>:<budget_s>`
  (μ 0 and lr 1 from the literals; `EPOCHS=20000` so the budget, not the 500-epoch literal,
  stops it: at `max_iter` 1 a step is a few closure evaluations, so 500 epochs could end
  before B; 20,000 rather than more because the trainer sizes a `Vec` by `epochs`). The
  `LBFGS_*` variables join the cleared list. The new case arm sits first in the `case`, the
  usage line before `adam`'s and the `-u` line after `env`'s first, so it does not touch the
  lines CO3.3's `soap` arm adds or edits: `git merge-tree` against `origin/optimisers/co3-3`
  merges the script without conflict.
- This plan, the results (`docs/optimisers/results/CO2.2.md`, after the runs), the roadmap
  and progress entries.
- Conventions: none.

### The grid
| cell | history | `max_iter` | `max_eval` | why that `max_eval` |
|---|---|---|---|---|
| H10-I1 | 10 | 1 | 10 | the search gets up to 10 evaluations (Burn's default, 1, gives it one trial at t = 1 and no bracketing; a failed trial returns t = 0, the stall) |
| H100-I1 | 100 | 1 | 10 | as H10-I1 |
| H10-I20 | 10 | 20 | 25 | Burn's default ⌊1.25 · 20⌋, made explicit |
| H100-I20 | 100 | 20 | 25 | as H10-I20 |

These are CO2.1's proposed values (its open question 1); the author's answer (2026-11-27)
was to set `max_eval` explicitly with every cell, so they are taken as named, not tuned.

## Tests that prove it
No code changes, so no new tests. The script's argument parsing was checked by hand for the
four forms (a well-formed `lbfgs` spec, one with a field missing, `gn`, `adam`), and
`bash -n` passes. The pilot's numbers are the evidence; each is read from the run files
(`metrics.csv`, `lbfgs_steps.csv`, `costs.json`, `errors.json`), not from logs.

## Checks
`/verify` (docs and a shell script: the CI `changes` filter skips the jobs; nothing in
`src/`, `examples/`, `tests/` changes). `bash -n docs/optimisers/co0_1_runs.sh`.

## VM runs (the orchestrator runs them; this session never ssh-es)
The pilot (D5): seed 42, stage 3 (`0 129 64 42`: p 0, n 129, s 64, N 1,000), full batch
(D4), strong Wolfe, lr 1, μ 0 (D3; the literal), equal time B = 2,952 s (CO0.1), four cells
(the grid above), two per GPU one after the other, both GPUs at once (pairing accepted
2026-11-26, CO0.1), about 2 × 52 min. Each GPU gets one history-10 and one history-100
cell, so memory is alike on both. After CL4.2 s = 64 and CO3.3's SOAP pilot (about 16:30 UTC
2026-11-28):

```sh
cd ~/sciml-rs && git fetch origin
git worktree add --detach ~/sciml-rs-wt/co2-2 origin/optimisers/co2-2
cd ~/sciml-rs-wt/co2-2
find ~/sciml-rs/datasets -mindepth 1 -maxdepth 1 ! -name README.md -exec ln -s {} datasets/ \;
mkdir -p ~/sciml-rs/logs/co2-2
tmux new -d -s co22-g0 "source ~/sciml-gpu.sh; export CARGO_TARGET_DIR=~/sciml-rs-target/co2-2; \
  LOGDIR=~/sciml-rs/logs/co2-2 GPU=0 bash docs/optimisers/co0_1_runs.sh \
  lbfgs:42:10:1:10:2952 lbfgs:42:100:20:25:2952 2>&1 | tee ~/sciml-rs/logs/co2-2/g0.log"
sleep 120   # the first instance builds; the second then finds the binaries up to date
tmux new -d -s co22-g1 "source ~/sciml-gpu.sh; export CARGO_TARGET_DIR=~/sciml-rs-target/co2-2; \
  LOGDIR=~/sciml-rs/logs/co2-2 GPU=1 bash docs/optimisers/co0_1_runs.sh \
  lbfgs:42:100:1:10:2952 lbfgs:42:10:20:25:2952 2>&1 | tee ~/sciml-rs/logs/co2-2/g1.log"
```

A fresh `CARGO_TARGET_DIR` builds from scratch (several minutes); if the first instance has
not finished building after the `sleep`, the second waits on Cargo's lock and then finds the
binaries up to date, so nothing is lost but the wait. The script passes
`--no-default-features --features cuda`. Run directories land in the worktree's `runs/`
(`poisson_fno_n129_s64_p0_lbfgs_<secs>`; two started in one second get CO0.4's suffix);
`~/sciml-rs/logs/co2-2/progress.txt` gets one `done` line per cell. Then per run directory,
on GPU 0:

```sh
cd ~/sciml-rs-wt/co2-2 && source ~/sciml-gpu.sh && export CARGO_TARGET_DIR=~/sciml-rs-target/co2-2
for d in $(sed -n 's/^done lbfgs[^ ]* \([^ ]*\) .*/\1/p' ~/sciml-rs/logs/co2-2/progress.txt); do
  CUDA_VISIBLE_DEVICES=0 cargo run --release --no-default-features --features cuda \
    --example predict_poisson -- "$d" 2>&1 | tee -a ~/sciml-rs/logs/co2-2/predict.log
done
```

What comes back: each run's `metrics.csv`, `lbfgs_steps.csv`, `costs.json`, `errors.json`,
`train_cfg.json`, and the logs. The pilot table (`docs/optimisers/results/CO2.2.md`), per
cell: steps (epochs) reached, training time, closure evaluations (= forward and backward
passes, D7) and their mean and maximum per step, median step time, final Φ, train and test
L², errors 3 and 4, peak GPU memory, the checkpoint share (below), and stalled steps (`phi_after == phi_before`, the CO2.1 failure); beside
them CO0.1's seed-42 errors 4 at B: Adam 9.637e-3, GN200-D (cap 150) 5.846e-3
(`docs/optimisers/results/CO0.1.md`). Updates are not counted
(Burn hides the inner iterations; accepted 2026-11-27), so the table says only whether steps
and passes were enough to compare the cells.

D5 is proposed from it as history, `max_iter` and `max_eval` together (the literal
`LBFGS_MAX_EVAL` is `None`, Burn's default, which stalls at `max_iter` 1, so `max_eval` must
travel with the decision): the cell with the lowest error 4; among cells within 5 % of the
lowest, the smaller history (less memory) and then the smaller `max_iter`. The proposal names
no rule; this one follows CO3.3's. One seed cannot resolve 5 % (Adam's error 4 at B spreads
about 14 % over seeds 42–46, CO0.1), which the D5 question states as a limitation. No GPU resume check: the pilot
does not resume (flex resume shown in CO2.1).

## Constraints and risks
- Constraint 1 (every existing output, Adam and Gauss–Newton included, bit for bit): no
  library or example code changes; the script's `adam` and `gn` arms behave as before (the
  new `-u` variables are unset in their runs too, and none of them was set by those arms).
- Checkpoints count against the budget (the trainer's clock runs through them), and a
  history-100 checkpoint carries about 1.9 GB of curvature pairs every 25 epochs. At
  `max_iter` 1 epochs are short, so H100-I1 may spend a visible share of B writing
  checkpoints. Reported, not fixed (fixing would touch `training/`, which CO3.3 holds).
  `epoch_time_s` is taken before the checkpoint write, so the share is `costs.json`'s
  `training_time_s` minus Σ `metrics.csv` `epoch_time_s` (setup aside), per cell; whether
  that biases D5 is an author's question once it is known.
- R1 of CO2.1 (Φ in f32, passed to Burn as f64): line searches may end on rounding at small
  Φ; stalls are counted, not fixed.
- Memory: history 100 at the full batch of 1,000; the proposal's estimate (1.9 GB history,
  the full-batch forward and backward below Phase 10's 12.6–17.2 GiB) fits an A100. Two
  cells never share a GPU.
- The 2D objective: no effect.

## Pending runs
The four cells and their `predict_poisson` evaluations, above ("VM runs"): tmux `co22-g0`
and `co22-g1`, logs `~/sciml-rs/logs/co2-2/`, worktree `~/sciml-rs-wt/co2-2`. They decide
D5.

## Open questions
None yet; D5 is asked once the table exists.

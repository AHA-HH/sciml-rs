# CO0.1 results: reference arms on Burn 0.23

Plan: [`../tasks/CO0.1.md`](../tasks/CO0.1.md). Stage 3, n = 129, K = 64, s = 64, p = 0,
modes [12, 12], width 32, 4 layers, N = 1000, the 200 test samples. Every run is on the VM
(one A100, cuda, f32, TF32 off by `cubek.toml`), Burn 0.23.0-pre.2, trained by
`docs/optimisers/co0_1_runs.sh` and evaluated by `predict_poisson`. Error 3 is the coarse
error on the learning grid, error 4 the fine-grid error on the CGL grid; the gap is final
test L² minus final train L². Run directories: `runs/co0-1/` (git-ignored), mapped to run
specs by its `progress.txt`.

**Status:** done 2026-11-27: stages 1–3, D1 signed off (cap 150). The two arms at a
glance are under "Final tables".

## Final tables

Error 4 (mean over the test set), GN200-D at μ = 0 with CG cap 150 against Adam (Phase 5
settings):

| seed | Adam, equal time | GN, equal time | GN against Adam | Adam, 5,400 s | GN, 5,400 s | GN against Adam |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 9.637e-3 | 5.846e-3 | −39.3 % | 8.585e-3 | 2.469e-3 | −71.2 % |
| 43 | 1.096e-2 | 4.863e-3 | −55.6 % | 9.410e-3 | 6.070e-3 | −35.5 % |
| 44 | 1.105e-2 | 4.483e-3 | −59.4 % | 9.486e-3 | 3.593e-3 | −62.1 % |
| 45 | 9.936e-3 | 1.008e-2 | +1.5 % | – | – | – |
| 46 | 9.846e-3 | 6.427e-3 | −34.7 % | – | – | – |
| **mean** [min–max] | **1.029e-2** [9.637e-3–1.105e-2] | **6.341e-3** [4.483e-3–1.008e-2] | −38 % | **9.160e-3** [8.585e-3–9.486e-3] | **4.044e-3** [2.469e-3–6.070e-3] | −56 % |
| training time, mean (s) | 2,949.1 | 3,045.5 | | 5,397.9 | 5,494.5 | |

- **At equal time, GN at μ = 0 is ahead on average** (6.34e-3 against 1.03e-2) and on 4
  of 5 seeds. On seed 45 it is level with Adam: 1.008e-2 against 9.94e-3, +1.5 %. GN's
  spread across seeds (4.5e-3–1.0e-2) is wider than Adam's (9.6e-3–1.1e-2). On the fork
  (Phase 10, 859 s), GN at μ = 0 was ahead on average on three seeds but not on all of
  them. Here the budget is 3.4× longer, because Adam slowed.
- **At 5,400 s GN is ahead on all three seeds**, at 44 % of Adam's mean.
- **GN gets slightly more time.** A GN run stops at the end of the epoch that crosses the
  budget, which put it 3.3 % over Adam's mean at equal time and 1.8 % at 5,400 s.
- **The long budget is 1.83 × B**, not Phase 10's 6.3×: D2 kept 5,400 s.



## Stage 1: Adam, 500 epochs, seeds 42–46

Phase 5 settings: batch 20, lr 1e-3 cosine to 1e-5, weight decay 1e-4.

| seed | GPU | alone or paired | training time (s) | median epoch (s, from epoch 4) | final train L² | final test L² | gap | error 3 mean | error 4 mean / max | peak memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 0 | alone | 3,064.9 | 5.972 | 3.108e-3 | 9.633e-3 | 6.53e-3 | 9.633e-3 | 9.637e-3 / 3.489e-2 | 3,048 MiB |
| 43 | 0 | paired with 45 | 2,912.2 | 5.785 | 3.378e-3 | 1.095e-2 | 7.57e-3 | 1.095e-2 | 1.096e-2 / 3.836e-2 | 1,128 MiB |
| 44 | 1 | alone | 3,035.7 | 6.039 | 3.218e-3 | 1.105e-2 | 7.83e-3 | 1.105e-2 | 1.105e-2 / 5.472e-2 | 983 MiB |
| 45 | 1 | paired with 43 | 2,919.2 | 5.799 | 3.372e-3 | 9.933e-3 | 6.56e-3 | 9.933e-3 | 9.936e-3 / 3.641e-2 | 983 MiB |
| 46 | 0 | alone | 2,813.5 | 5.477 | 3.155e-3 | 9.841e-3 | 6.69e-3 | 9.841e-3 | 9.846e-3 / 3.159e-2 | 1,128 MiB |
| **mean** [min–max] | | | **2,949.1** [2,813.5–3,064.9] | | 3.246e-3 | 1.028e-2 | 7.04e-3 | 1.028e-2 | **1.029e-2** [9.637e-3–1.105e-2] | |

Each run: 25,000 updates, 25,000 forward and 25,000 backward passes. Peak memory is the
whole device's, sampled each second; seed 42's 3,048 MiB is not explained by its own run
(the others read 983–1,128 MiB, Phase 10's Adam 971–1,701 MiB) and is likely another
process's memory on GPU 0 at its start (CO0.2's profile exited ten seconds before).

### Against Phase 10 (the fork `ddfa9af`, commit `9dbeb91`)

| | runs | error 4 mean [min–max] | final train L² | gap | training time (s) |
| --- | --- | --- | --- | --- | --- |
| Phase 10, Adam | 42–44 | 1.061e-2 [1.041e-2–1.095e-2] | 3.185e-3 | 7.42e-3 | 859.4 [845.9–869.7] |
| Burn 0.23, seeds 42–44 | 42–44 | 1.055e-2 [9.637e-3–1.105e-2] | 3.235e-3 | 7.31e-3 | 3,004.3 [2,912.2–3,064.9] |
| Burn 0.23, seeds 42–46 | 42–46 | 1.029e-2 [9.637e-3–1.105e-2] | 3.246e-3 | 7.04e-3 | 2,949.1 [2,813.5–3,064.9] |

- **Accuracy is reproduced.** Over the same seeds the mean error 4 is within 1 % of the
  fork's (1.055e-2 against 1.061e-2), with a wider range; seed by seed the runs differ
  (seed 42: 9.64e-3 against 1.095e-2), as two Burns are not bit-comparable (proposal §2).
  Two more seeds lower the mean to 1.029e-2.
- **Training is 3.4× slower** (2,949 s against 859 s; 5.5–6.0 s per epoch against
  1.72 s). A bisect on the VM (2026-11-25; 113 and 28 epochs per build, median epoch from
  epoch 4) puts it on the move to Burn 0.23 (#119): the fork at `4541f02` 1.723 s per epoch;
  #119 `684c141` 5.930 s; `e327200` 5.911 s; CM0.3 `7ae5c8a` 5.930 s; `main` `752eb90`
  5.935 s. No later change of ours moved it. A GN200 step at cap 50 slowed only about
  24 % (CO0.2), so the extra cost is per step at batch 20, not per sample: CO0.3 profiles
  it.

## The budgets

- **B, the equal-time budget:** **2,952 s**, the mean of the five runs' wall-clock seconds
  in `progress.txt` (3,067, 2,916, 3,039, 2,924, 2,814), the value stage 2 runs with. The
  mean of `costs.json`'s `training_time_s`, the plan's definition, is 2,949.1 s; the two
  differ by 0.1 %, and 2,952 s is kept for every later arm so that all runs share it.
- **E, Adam's epochs at the long budget:** round(5,400 × 500 / 2,952) = **915** (916 from
  2,949.1). The long budget stays 5,400 s (D2, the author, 2026-11-25), now 1.83 × B
  rather than Phase 10's 6.3 ×.

## Running side by side

The roadmap allows timed runs on both GPUs at once only after a check that a paired run
keeps a lone run's epoch time (the step is host-bound, Phase 9 T3). The plan's check: each
paired seed's median epoch within 3 % of seed 42's alone.

| | median epoch (s) | against seed 42 alone |
| --- | --- | --- |
| 42, alone | 5.972 | – |
| 44, alone | 6.039 | +1.1 % |
| 46, alone | 5.477 | −8.3 % |
| 43, paired | 5.785 | −3.1 % |
| 45, paired | 5.799 | −2.9 % |

The paired runs were faster, not slower: seed 43 left the 3 % band by 0.14 % on the fast
side. The lone runs themselves spread over 10 % (5.48–6.04 s), more than the band, so
seed 42 alone is a noisy reference; the paired runs lie inside the lone runs' range.
**Pairing accepted by the author 2026-11-26**: the rule exists to catch a slowdown, and
there is none.

## Data-parallel on both GPUs (a separate measurement, not part of B)

At the author's request, seed 46 trained on both A100s at once (`TRAIN_DEVICES=0,1`, the
same binary, each batch of 20 split across them, Phase 9 T2): **5,869.4 s** (median
epoch 11.556 s), error 4 **1.009e-2** (mean; max 2.93e-2) against 9.85e-3 on one GPU,
final test L² 1.008e-2. That is 2.09 × the one-GPU run's time (0.48 × its speed), as Phase 9 T3 found on
the fork (0.44 × at s = 64): the step is host-bound, and splitting a batch of 20 adds the
all-reduce without adding work per device. Run directory
`poisson_fno_n129_s64_p0_seed46_1795627012`; no peak memory recorded.

## Stage 2: Adam at the long budget, seeds 42 and 43

915 epochs (E), the cosine schedule over all of them, otherwise stage 1's settings; seed 42
on GPU 0 beside seed 43 on GPU 1, commit `911a64e`.

| seed | training time (s) | median epoch (s, from epoch 4) | final train L² | final test L² | gap | error 4 mean / max | at 500 epochs (stage 1) | change | peak memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 5,322.1 | 5.823 | 2.168e-3 | 8.582e-3 | 6.41e-3 | 8.585e-3 / 2.893e-2 | 9.637e-3 | −10.9 % | 1,128 MiB |
| 43 | 5,335.1 | 5.837 | 2.226e-3 | 9.407e-3 | 7.18e-3 | 9.410e-3 / 3.016e-2 | 1.096e-2 | −14.1 % | 983 MiB |

Each run: 45,750 updates, forward and backward passes. With 1.8× the time Adam gains
11–14 % in error 4, and the train error falls more than the test error (the gap stays
6.4–7.2e-3). Both median epochs are within 3 % of seed 42's alone in stage 1 (−2.5 %,
−2.3 %), so the pair kept a lone run's pace; the training times, 1.80–1.81 × B, fall
1.2–1.4 % short of 5,400 s since E came from B's wall-clock seconds and these epochs were
faster.

## Stage 2: Gauss–Newton, seed 42

GN200-D: batch 200, μ = 0, the Hutchinson diagonal, warm-started CG with Martens' stop,
seed 42, commit `911a64e`. The equal-time runs ran alone on GPU 0, one after the other.
The long runs were paired (cap 50 on GPU 0, cap 150 on GPU 1, started 60 s apart). They
are the rerun of 2026-11-26 16:52 UTC: the first pair had shared a run directory and is
void (plan, "Pending runs").

| cap | budget | training time (s) | steps (accepted) | passes fwd / bwd | passes per s | final train L² | final test L² | gap | error 4 mean / max | peak memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 2,952 s | 3,000.6 | 220 (129) | 32,407 / 10,571 | 14.32 | 5.032e-3 | 6.124e-3 | 1.09e-3 | 6.129e-3 / 2.029e-2 | 4,650 MiB |
| 150 | 2,952 s | 3,123.8 | 115 (80) | 33,500 / 11,078 | 14.27 | 4.790e-3 | 5.841e-3 | 1.05e-3 | **5.846e-3** / 2.575e-2 | 4,406 MiB |
| 50 | 5,400 s | 5,439.5 | 405 (266) | 62,188 / 20,418 | 15.19 | 3.599e-3 | 4.575e-3 | 9.8e-4 | 4.579e-3 / 2.006e-2 | 5,016 MiB |
| 150 | 5,400 s | 5,453.1 | 170 (143) | 60,828 / 20,245 | 14.87 | 2.046e-3 | 2.465e-3 | 4.2e-4 | **2.469e-3** / 1.209e-2 | 4,261 MiB |

A run stops after the first epoch (5 steps of 200) that ends at or over the budget. At
cap 150 an epoch takes about 135 s, so the equal-time cap-150 run trained 5.8 % over B
(cap 50: 1.6 %).

### Against Adam and against Phase 10 (seed 42)

| | equal time | long budget |
| --- | --- | --- |
| Adam (stage 1; stage 2's 915 epochs) | 9.637e-3 | 8.585e-3 |
| GN cap 50 | 6.129e-3 | 4.579e-3 |
| GN cap 150 | 5.846e-3 | 2.469e-3 |
| Phase 10 (fork), GN cap 50, μ = 0 | 1.200e-2 (at 859.4 s) | 5.584e-3 (475 steps) |
| Phase 10 (fork), GN cap 150, μ = 0 | – | 3.030e-3 (230 steps) |

- **GN now beats Adam at equal time on seed 42**, by 36 % (cap 50) and 39 % (cap 150). On
  the fork it lost (1.200e-2 against Adam's 1.095e-2). Adam's step slowed 3.4× on
  Burn 0.23 and GN's only about 24 % (CO0.2), so the same budget buys GN far more work.
  At the long budget, cap 150 is 3.5× below Adam.
- **The long runs match or beat Phase 10's on fewer steps:** 405 against 475 at cap 50 and
  170 against 230 at cap 150, so GN's step is slower here too. Two Burns are not
  bit-comparable and this is one seed, so the difference is no finding.
- **The train/test gap stays small for GN** (4.2e-4–1.1e-3, against Adam's 6.4–7.2e-3), as
  in Phase 10.

### Side by side

Passes per second over the first 2,952 s of each long run, against its cap's run alone
(summed from `gn_steps.csv`): cap 50 15.16 against 14.50 (+4.5 %), cap 150 14.85 against
14.30 (+3.9 %). As with Adam's pairs, the paired runs were faster, not slower, and outside
the 3 % band on the fast side.

### Wild steps (open question 6)

A step is wild here when |ρ| > 1e3:

| run | wild steps | first at step (φ there) | accepted |
| --- | --- | --- | --- |
| cap 50, 2,952 s | 43 of 220 | 47 (1.5e-4) | none |
| cap 150, 2,952 s | 19 of 115 | 47 (1.1e-4) | none |
| cap 50, 5,400 s | 108 of 405 | 53 (6.6e-5) | none |
| cap 150, 5,400 s | 0 of 170 | – | – |

- **No NaN in any of the four runs.** Every wild step was rejected; φ is unchanged on
  every rejected step, and λ was raised.
- **The quadratic model breaks down rather than being noisy.** The median predicted
  reduction of a wild step is 48 to 2.8e4, while φ itself is below 1.5e-4 by then. A step
  that the model says lowers φ by 10⁵–10⁸ times φ can only be a step along directions
  where G + λI is almost singular. At μ = 0 nothing holds those directions, and λ at
  the wild steps was small (median 4e-7–3e-5 by run, down to 1e-8). Example: the equal-time cap-150 run's last step had norm 6.8e3,
  predicted reduction 136.8 and ρ −4.4e14.
- **The cost is wasted steps, not wrong ones.** Wild steps are 20–27 % of cap
  50's steps and up to 17 % of cap 150's. At the longer cap, CG's stop on progress ends
  most solves early (73 of 170 steps at 5,400 s).
- **The void directory's 88 NaN rows** (from step 58 on) cannot be tied to a run. Its
  rerun at the same settings has none, but GPU runs are not bit-reproducible, so these
  settings could still produce NaN on some run. Stage 3 reports any NaN row.

### D1, proposed: CG cap 150

The plan's rule is the lower error 4 on seed 42 at equal time first, then at the long
budget. Cap 150 wins both:
- 5.846e-3 against 6.129e-3 at equal time (4.6 % lower, within one seed's noise);
- 2.469e-3 against 4.579e-3 at 5,400 s (46 % lower).

It also has fewer wild steps (19 and 0, against 43 and 108) and the lower peak memory
(4.3–4.4 GiB against 4.7–5.0). Phase 10 found the same at μ = 0 (3.03e-3 against
5.58e-3 at 5,400 s). The caveat: at equal time the margin is small, and cap 150 trained
5.8 % over B against cap 50's 1.6 %. The long budget is the clear separation. **Signed
off 2026-11-26: cap 150**, given with stage 3's go through the coordinating session.

## Stage 3: GN cap 150 on seeds 43–46; Adam seed 44 at the long budget

Commit `911a64e` (binary `bin-911a64e`). The runs were paired across the GPUs, with
pairing accepted as for stages 1 and 2:
- GPU 0: 43 and 45 at B, then 43 at 5,400 s.
- GPU 1, started 20 s later: 44 and 46 at B, then 44 at 5,400 s and Adam 44 at 915 epochs.

| seed | budget | training time (s) | steps (accepted) | passes per s | final train L² | final test L² | error 4 mean / max | wild steps (\|ρ\| > 1e3) | peak memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 43 | 2,952 s | 3,136.4 | 135 (103) | 14.92 | 4.462e-3 | 4.850e-3 | 4.863e-3 / 1.855e-2 | 3 | 4,406 MiB |
| 44 | 2,952 s | 2,988.5 | 125 (91) | 15.00 | 4.463e-3 | 4.473e-3 | 4.483e-3 / 1.721e-2 | 0 | 4,261 MiB |
| 45 | 2,952 s | 3,002.1 | 125 (85) | 15.09 | 9.383e-3 | 1.008e-2 | 1.008e-2 / 3.711e-2 | 0 | 4,406 MiB |
| 46 | 2,952 s | 2,976.6 | 125 (90) | 15.16 | 5.706e-3 | 6.421e-3 | 6.427e-3 / 1.945e-2 | 0 | 4,261 MiB |
| 43 | 5,400 s | 5,507.1 | 295 (183) | 15.17 | 5.776e-3 | 6.058e-3 | 6.070e-3 / 2.004e-2 | 35 | 4,406 MiB |
| 44 | 5,400 s | 5,523.2 | 205 (137) | 15.16 | 3.230e-3 | 3.585e-3 | 3.593e-3 / 1.150e-2 | 22 | 4,261 MiB |

Adam seed 44, 915 epochs: 5,536.5 s, median epoch 6.040 s (+1.1 % against seed 42 alone in
stage 1), final train L² 2.335e-3, test L² 9.483e-3, error 4 9.486e-3 / 3.871e-2 (−14.2 %
against 500 epochs), 983 MiB.

- **No NaN in any run, and no wild step was accepted.** All six GN runs ran 14.9–15.2
  passes per second, faster than stage 2's cap-150 run alone (14.27), so pairing slowed
  nothing.
- **Seed 45 at equal time** is GN's worst run (1.008e-2), level with Adam's seed 45. It
  had no wild steps. Its train L² (9.38e-3) is twice the other seeds', so it optimised
  slowly rather than overfitting.
- **Seed 43's long run ended worse than its equal-time run** (6.07e-3 against 4.86e-3), with
  a higher train L² too (5.78e-3 against 4.46e-3). Training longer did not cause this: the
  two runs are different trajectories from the start. Runs with the same seed and settings
  agree only to f32's last digits after the first step on cuda (φ 0.49425319 against
  0.49425325 at step 1; seeds 42 and 44 the same). Gauss–Newton amplifies that, so by
  epoch 4 the long run's train L² was 6.75e-2 against 9.60e-2, and at 2,952 s its test
  L² was about 9.3e-3 against 4.85e-3. Its damping swung widely from step 100 on (4e-5 to
  7.6, sampled every 15 steps), with 112 of its 295 steps rejected (35 of them wild).
- **What follows for CO4.1:** with GN at μ = 0, an equal-time run and a long run on one
  seed are independent draws, not one run read at two times. The spread by seed includes
  this run-to-run spread; Adam's own run-to-run spread was not measured. Five seeds per arm, as D2 has
  them, are what absorb it.

## Provenance

- Seeds 42 and 44 ran the binary built at `29a21e2` (the build log and `progress.txt`'s
  start line), seeds 43, 45 and 46 the one built at `911a64e`; the two commits differ only
  in this task's docs and the run script, so all five trained the same code.
- `costs.json` of seeds 42 and 44 records `f869110` and `f893620` (CO0.2's branch), which
  is wrong: `git_commit()` reads the source checkout's HEAD when the run ends, and the VM's
  `~/sciml-rs-wt/co0-1` had been moved to CO0.2's commits by then (neither contains the run
  script, so neither can have run it). Use the commits above.

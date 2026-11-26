# CO0.1 results: reference arms on Burn 0.23

Plan: [`../tasks/CO0.1.md`](../tasks/CO0.1.md). Stage 3, n = 129, K = 64, s = 64, p = 0,
modes [12, 12], width 32, 4 layers, N = 1000, the 200 test samples. Every run is on the VM
(one A100, cuda, f32, TF32 off by `cubek.toml`), Burn 0.23.0-pre.2, trained by
`docs/optimisers/co0_1_runs.sh` and evaluated by `predict_poisson`. Error 3 is the coarse
error on the learning grid, error 4 the fine-grid error on the CGL grid; the gap is final
test L² minus final train L². Run directories: `runs/co0-1/` (git-ignored), mapped to run
specs by its `progress.txt`.

**Status:** stage 1 (Adam at equal time) done 2026-11-26. Stage 2 (GN200-D at μ = 0,
seed 42, caps 50 and 150; Adam at the long budget, seeds 42 and 43): the equal-time GN runs
done, the long Adam runs running, the two long GN runs void (they shared a run directory,
plan "Pending runs") and to be rerun. Stage 3 after D1.

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

## Provenance

- Seeds 42 and 44 ran the binary built at `29a21e2` (the build log and `progress.txt`'s
  start line), seeds 43, 45 and 46 the one built at `911a64e`; the two commits differ only
  in this task's docs and the run script, so all five trained the same code.
- `costs.json` of seeds 42 and 44 records `f869110` and `f893620` (CO0.2's branch), which
  is wrong: `git_commit()` reads the source checkout's HEAD when the run ends, and the VM's
  `~/sciml-rs-wt/co0-1` had been moved to CO0.2's commits by then (neither contains the run
  script, so neither can have run it). Use the commits above.

# Phase 7: training-set size and sample efficiency

As of 2026-10-29; planned, T1–T3 (decision 30, amended 2026-10-29 with the Phase 7
planning decisions). Design: `docs/design/2d-chebyshev-poisson-fno.md` (§10 Phase 7,
§13.5). `docs/CONVENTIONS.md` wins on any conflict.

Phase 7 asks how many forcing–solution pairs the FNO needs to reach the acceptance
threshold, at the spatial baseline Phase 5 fixed. It trains the baseline on nested
subsets of the stage 3 training split, N ∈ {25, 50, 100, 250, 500, 1000}, always
evaluating on the same 200 test samples, and reports:
- the coarse error (learning grid) and the fine-grid error (after prolongation) against N;
- the spread across subset seeds and initialisation seeds;
- training time and the number of optimisation steps against N;
- the smallest N whose mean fine-grid error over seeds meets the threshold (D7);
- the cost per additional reduction in error between successive N.

Sample efficiency is measured directly: Phase 6's low ranks describe the data, not how
many samples the model needs (design §13.6). The model, the transfers and the
hyperparameters do not change in this phase.

## Read first

- Design §13.5 (this phase), §13.6 (confounders), §13.8 (the Phase 7 matrix), §13.9
  (metadata: N, the subset seed and the index hash in `data_cfg.json`), §13.13 (H7),
  §13.14 (known limitations), §13.15 (D6, D7), §12 decision 30 and its amendments.
- `docs/CONVENTIONS.md` §10 (normalisers fitted on the train split) and §12 (the
  transfers).
- `docs/phase5/README.md` (Results: the baseline, the matrix cost, the s = 64 runs) and
  `docs/phase5/results.md`.
- `CLAUDE.md`: Working agreement, Checks, Commands (`train_poisson`, `predict_poisson`,
  `poisson_matrix`).

## Inputs from Phase 5

| Item | Value | Source |
| --- | --- | --- |
| Stage 3 dataset | n = 129, K = 64; 1000 train / 200 test; `datasets/poisson/poisson_n129_K64_{train,test}.{npz,json}` (main checkout) | Phase 4 T3 |
| Spatial baseline | s = 64; R = unfiltered T_cu, P = Floater–Hormann d = 2 (D1); p = 0 (D3); modes [12, 12] (D2); width 32, 4 Fourier layers | Phase 5 T7 |
| Training baseline | relative L² loss, Adam (1e-3, weight decay 1e-4), cosine annealing, batch 20, 500 epochs; initialisation seeds 42, 43, 44 (D4) | Phase 5 T7 |
| Baseline error at N = 1000, mean [min–max] over seeds 42–46 | error 4 (fine grid) 1.080e-2 [1.041e-2–1.127e-2]; error 3 (coarse) 1.080e-2; final train L² 2.9e-3 to 3.5e-3 | Phase 5 T6 |
| Baseline cost at N = 1000 (cuda, A100) | 1.69 s/epoch, about 851 s per 500-epoch run; per-epoch time flat from s = 16 to 64, so a fixed per-epoch cost dominates | Phase 5 T5 |
| The N = 1000 runs | `runs/poisson_fno_n129_s64_p0_1793188648` (seed 42), `…_seed43_1793192445`, `…_seed44_1793196221` (main checkout) | Phase 5 T5 |
| Code in place | `PoissonConfig` (n, s, `source`); `load_poisson_uniform` (the first `n_train` samples); `normalize_split_pair` (fits on what it is given); `train_poisson -- <p> <n> <s> <seed>`; `predict_poisson` (saved normalisers, `errors.json`); `poisson_matrix` (groups by padding and s) | Phase 5 |

## Facts verified at `8c1739a`

- **The loader takes a prefix.** `load_poisson_uniform` reads `values.slice(s![..count, ..])`
  with `count = n_train` (`data/loaders/poisson/uniform.rs`). A prefix of the file is
  nested but not random: sample seeds are `base_seed + i`, so it is a fixed draw, and
  every "subset seed" would give the same subset. A seeded permutation is needed.
- **Normalisers follow the subset for free.** `normalize_split_pair` fits on the train
  array it receives (`data/loaders/common.rs`), so selecting the subset before it keeps
  CONVENTIONS §10 with no other change.
- **Evaluation uses the saved normalisers.** `predict_poisson` loads them from the run
  (`RunDir::normalizers`) and reads only the test split, so it evaluates a subset run
  unchanged. `poisson_resolution` refits on the full train split and is not used here.
- **No ungated RNG.** `rand_chacha` (pinned `=0.10.0`, in `Cargo.lock`) is optional,
  behind `datagen`; the loader is ungated.
- **The trainer budgets by epochs.** The cosine schedule runs over
  `epochs × ceil(N / batch)` steps and the test split is evaluated after every epoch
  (`training/trainer.rs`). At 500 epochs N = 25 gets 1,000 optimisation steps and
  N = 1000 gets 25,000.
- **N = 1000 needs no new runs.** Every permutation of the 1000 train samples selects all
  of them, so the subset seed changes only the order before the training shuffle. The
  Phase 5 s = 64 runs with seeds 42, 43, 44 are the N = 1000 row.

## Planning decisions

Recorded in design §12, decision 30 (amendment of 2026-10-29). Chosen by the author
(AHA-HH), 2026-10-29.

- **D6, training budget: fixed epochs, 500** for every N, as in Phase 5. Each run reports
  its total optimisation steps, 500 × ⌈N / 20⌉. Small N therefore gets fewer updates,
  so its error mixes too little data with too little training; this is a stated
  limitation (design §13.14), not a measured effect. **Deferred:** a fixed step budget
  (25,000 steps, equal updates for every N), as later research.
- **D7, threshold:** fine-grid (error 4) test relative L² ≤ 2e-2, met by the mean over all
  seeds of an N, with the min–max spread reported. Fixed now, before any Phase 7 run, and
  not re-chosen after the results.
- **Seeds:** 3 subset seeds (0, 1, 2) × 3 initialisation seeds (42, 43, 44) at each
  N ∈ {25, 50, 100, 250, 500}: 45 new runs. N = 1000 reuses Phase 5's three s = 64 runs
  (above). Seeds 45 and 46 at N = 1000 are reported but not used in the N = 1000 mean, so
  every N has the same initialisation seeds.
- **Evaluation:** after every epoch, as today. **Deferred:** evaluation every fixed
  number of steps, which a fixed step budget would need (N = 25 would otherwise run
  12,500 epochs, each with a 200-sample test pass).
- **Subsets:** nested prefixes of a seeded permutation of the 1000 train indices, by
  ChaCha8 (`rand_chacha`, made non-optional; no new dependency). The selected indices are
  sorted before loading, so the tensors keep file order, and `data_cfg.json` records N,
  the subset seed and a hash of the sorted indices (design §13.9). With no subset, the
  loader reads the prefix as today.
- **Compute:** the matrix trains on the cuda A100 VM, as Phase 5 T5 did, so costs compare
  with the N = 1000 runs. Evaluation runs on flex. The matrix cost is timed from the
  first epochs at each N and stated before the full runs start.
- **Outputs:** `docs/phase7/results.md`, written by `poisson_matrix` extended to group by
  N; gnuplot plots of error and cost against N attached to the PR (`*.png` is ignored).

## Tasks

| Task | Brief | Delivers | Depends on |
| --- | --- | --- | --- |
| T1 | [T1-subsets.md](T1-subsets.md) | Nested seeded subsets in `PoissonConfig` and the loader; normalisers on the subset; N and subset-seed arguments to `train_poisson`; N, seed, index hash and total steps in the run files; D7 recorded | Phase 5 T7 (merged) |
| T2 | [T2-matrix.md](T2-matrix.md) | The Phase 7 training matrix on cuda: cost timed first, then 45 runs | T1 |
| T3 | [T3-error-vs-n.md](T3-error-vs-n.md) | Every run evaluated; `poisson_matrix` by N; error and cost against N; the smallest N under the threshold; H7; phase close-out | T2 |

Task numbers are execution order (design §10). One task per branch and PR:
`phase7/T<k>-<name>`. Each task is worked from its brief alone, plus the files the brief
lists under "Read first".

## Module layout

- `data/loaders/poisson/` (T1): `PoissonConfig` gains an optional subset (seed and index
  hash; N stays `base.n_train`); `uniform.rs` selects the subset before the transfer and
  the normaliser fit.
- `training/config` or `runs/` (T1): the total optimisation steps recorded with the run.
- `examples/train/poisson.rs` (T1): N and subset-seed arguments; `examples/predict/poisson_matrix.rs` (T3): grouping and plots by N.
- No change to the model, the transfers, the trainer's loop or CONVENTIONS.

## Results

Filled in as tasks merge. Phase 7 is done when every row has a value.

| Item | Value | Source |
| --- | --- | --- |
| Subsets nested and reproducible; normalisers fitted on the subset; old runs still open | Done. Unit tests: nesting 25 ⊂ … ⊂ 1000 for seeds 0–2; the seed-0, N = 25 indices and their hash `65104831155be79e` pinned; a loaded subset equals a plain load of a file holding only those samples (tensors and normalisers), and differs from the fit on the whole split; old `data_cfg.json` (no subset) and `costs.json` (no `optimisation_steps`) still load; a wrong hash is rejected before any file is read. Without a subset the loader is unchanged (`poisson_output_is_pinned`, synthetic fields; not compared on the real stage 2 files). Smoke run on flex (not a result): stage 3, s = 64, N = 25, subset seed 0, seed 42, 500 epochs, 2.4 s/epoch, 1185 s; `data_cfg.json` records seed 0, population 1000 and hash `65104831155be79e`, `costs.json` 1000 optimisation steps. `predict_poisson` evaluates it unchanged: error 4 2.64e-1 mean (final train L² 1.26e-2), error 2 2.45e-5 as in Phase 5, error 3 within 1.1e-7 of the last test L². `poisson_resolution` rejects subset runs | T1 |
| Matrix cost, timed and stated before the full runs | Timed on the cuda VM (NVIDIA A100, the GPU otherwise free), stage 3, s = 64, p = 0, subset seed 0, seed 42, about 60 s per N, partial runs deleted. Steady s/epoch (epoch 0, the warm-up, excluded): N = 25: 0.15–0.16; 50: 0.183–0.188; 100: 0.24–0.25; 250: 0.494–0.510; 500: 0.867–0.900. They fit 0.117 s + 1.53 ms × N per epoch: a fixed cost, mostly the 200-sample test pass every epoch, plus about 31 ms per training batch of 20; the fit predicts 1.65 s at N = 1000 against Phase 5's measured 1.69. Extrapolated to 500 epochs: 78 / 93 / 123 / 251 / 442 s, 986 s per (subset seed, seed) pair, about 2.5 h of training for the 45 runs, about 3.5–4 h wall time with warm-up and per-run overhead (about 1.5 min per run in Phase 5 T5). Approved by the author 2026-10-29 | T2 |
| Coarse and fine-grid error per N, mean [min–max] over seeds; spread by subset seed and by initialisation seed | T2, coarse error as the final-epoch test relative L² on the learning grid, mean [min–max] over the 9 runs: N = 25: 2.340e-1 [1.995e-1–2.838e-1]; 50: 1.495e-1 [1.329e-1–1.660e-1]; 100: 7.618e-2 [7.008e-2–8.265e-2]; 250: 3.623e-2 [3.462e-2–3.900e-2]; 500: 1.969e-2 [1.909e-2–2.058e-2]; 1000: T3 (Phase 5: error 4 1.080e-2 over seeds 42–46). Mean by subset seed 0 / 1 / 2: N = 25: 2.55e-1 / 2.09e-1 / 2.38e-1; 50: 1.51e-1 / 1.49e-1 / 1.48e-1; 100: 7.50e-2 / 7.79e-2 / 7.56e-2; 250: 3.62e-2 / 3.52e-2 / 3.74e-2; 500: 2.02e-2 / 1.93e-2 / 1.96e-2. By seed 42 / 43 / 44: N = 25: 2.17e-1 / 2.32e-1 / 2.52e-1; 50: 1.40e-1 / 1.48e-1 / 1.61e-1; 100: 7.25e-2 / 7.47e-2 / 8.15e-2; 250: 3.52e-2 / 3.72e-2 / 3.63e-2; 500: 1.97e-2 / 1.97e-2 / 1.96e-2. No run diverged or stands apart from its N (all within 1.25× of the N's median); the mean best-epoch test L² is within 3 % of the mean final one at every N. Final train L² 1.0e-2 to 1.3e-2 at N = 25 and 50, 3.1e-3 to 4.0e-3 at N = 500: the gap to the test error closes with N. Fine-grid error (error 4): T3 |
| Training time and optimisation steps per N | T2, cuda (A100), 500 epochs, from each run's `costs.json` and `metrics.csv` (median s/epoch after epoch 0). Steps 500 × ⌈N / 20⌉: 1000 / 1500 / 2500 / 6500 / 12500 (25,000 at N = 1000). On an unshared GPU: N = 25: 0.158–0.165 s/epoch, 87–91 s training; 50: 0.184–0.191, 101–105 s; 100: 0.249–0.259, 133–137 s; 250: 0.496–0.521, 258–268 s; 500: 0.867–0.886, 441–452 s; the timing fit's predictions (0.155 / 0.194 / 0.270 / 0.500 / 0.882 s/epoch) hold. Ten runs shared the GPU with another process for part of their run, seen as the whole-device peak memory rising from the matrix's own 938 MiB to 1.8–14.4 GB: N = 100 subset seed 1 seed 44 (memory only) and subset seed 2, all seeds (0.43 s/epoch, 200–225 s); N = 250 subset seed 0, all seeds (0.52–0.83 s/epoch, 272–419 s); N = 500 subset seed 0, all seeds, and subset seed 1 seed 42 (0.89–0.98 s/epoch, 470–629 s). The other process is unidentified (the VM may be shared). Their errors are in line with their N, but their times and peak memory are not costs of the run; the unshared ranges above are. Training time of the 45 runs: 10,157 s (2.8 h) against 2.5 h estimated. Cost per fall in error between successive N: T3 |
| The smallest N meeting D7 (mean error 4 ≤ 2e-2), with the spread | | T3 |
| Cost per additional reduction in error between successive N | | T3 |
| H7 against the results | | T3 |

## Exit checklist

- [ ] T1–T3 merged.
- [x] The learning resolution and transfer method are fixed from Phase 5.
- [ ] Every training subset uses the same held-out test set.
- [ ] Results are repeated across seeds.
- [ ] Error and cost are plotted against training-set size.
- [ ] The minimum practically useful dataset size is identified.
- [x] D6 and D7 recorded in design §12 under decision 30, before any run.
- [ ] Every row of Results filled in.
- [ ] Any result that contradicts the design recorded in design §12, dated.
- [ ] The "Current phase" bullet in `CLAUDE.md` is updated for what comes next.

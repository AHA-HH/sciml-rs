# Phase 5: grid-decoupled operator learning

As of 2026-10-26; planned, not started (decision 30, amended 2026-10-26 with the Phase 5
planning decisions). Design: `docs/design/2d-chebyshev-poisson-fno.md` (§10 Phase 5, §13).
`docs/CONVENTIONS.md` wins on any conflict.

Phase 5 asks how coarse the forcing representation and the FNO's learning grid can be
while a fine-grid Poisson solve provides the reference. Today the grids are coupled: the
129² Chebyshev solve feeds a 128² uniform FNO grid, s = n − 1. This phase decouples the
solver grid G_s, the learning grid G_l and the evaluation grid G_e (design §13.3). It
reports three errors:
- the coarse error on the learning grid;
- the fine-grid error after prolongation;
- the transfer-only error, the floor no model can beat.

It also fixes the spatial baseline that Phases 6 and 7 use: learning s, restriction and
prolongation, padding, modes and training settings.

It absorbs old Phase 6 T1–T2 (decision 24): the baseline's padding, and several seeds with
error 4 against ‖u‖. The model is not redesigned in this phase (design §13.7). One
convention changes: CONVENTIONS §12 lets s differ from n − 1 and defines the restriction,
with `CONVENTION_VERSION` 1 → 2 (T2).

## Read first

- Design §3.3 (forcing family, K, the tail table), §5.3 (stages), §6.2 (transfers), §7
  (the five errors), §13 (the whole programme; §13.3 is this phase, §13.6 the
  confounders, §13.15 the open decisions), §12 decisions 5, 8, 16, 17, 22, 23, 24 and 30.
- `docs/CONVENTIONS.md` §2 (grid channels), §3 (padding), §4 (FFT, Bluestein), §5 (mode
  limits, m ≤ s/2), §9 (versioning), §10 (normalisers), §12 (Chebyshev grids and
  transfers).
- `docs/phase4/README.md` Results (stage 3 dataset, the 129² run, the resolution study and
  the padding check).
- `CLAUDE.md`: Working agreement, Checks, Burn.

## Inputs from Phase 4 and 4s

| Item | Value | Source |
| --- | --- | --- |
| Stage 3 dataset | n = 129, K = 64; 1000 train / 200 test; `datasets/poisson/poisson_n129_K64_{train,test}.{npz,json}` (main checkout) | Phase 4 T3 |
| Its label check (exact sine series), max | train 9.17e-13, test 8.97e-13 | Phase 4 T3 |
| Error 2 at s = n − 1 = 128, mean / max (test) | 2.53e-6 / 9.50e-6 | Phase 4 T3 |
| Stage 4 dataset (reference check) | n = 257, K = 64, the same seeds; label check max 1.6e-12 | Phase 4 T4 |
| The coupled 129² baseline | `runs/poisson_fno_n129_p4_1792037830`: cuda, p = 4, seed 42, test relative L² 1.079e-2, about 112 s/epoch | Phase 4 T3 |
| Padding check (65² models, mean error 3 at 65 / 129 / 257) | p = 0 1.138e-2 / 1.378e-2 / 1.608e-2 (1.4×); p = 4 1.069e-2 / 4.666e-2 / 7.546e-2 (7.1×) | decision 23 |
| Padding cost (cuda, 65²) | 1.7 s/epoch at p = 0, 32.8 at p = 4 (Bluestein) | decision 24 |
| Seeds | every Phase 4 result comes from one seed (42) | decision 24 |
| Code in place | `chebyshev::transfer` (`cheb_to_uniform`, `uniform_to_cheb`, `apply`, `Transfers`); `PoissonConfig::s()` = n − 1; `RunDir`; `poisson_errors`; `poisson_resolution`; `TrainingConfig::seed` (default 42) | Phase 4s |

## Facts verified at `beb8cd7`

- **s = n − 1 is hard-wired in three places:** `Transfers::new(n)`
  (`chebyshev/transfer.rs`), `PoissonConfig::s()` (`data/loaders/poisson/mod.rs`) and the
  `s + 1 == n` assert in `poisson_errors` (`metrics/poisson.rs`).
- **CONVENTIONS §12 fixes s = n − 1** and defines only T_cu and T_uc. A change to §12
  bumps `CONVENTION_VERSION` (§9) and lands before the code that relies on it. The loaders
  never check a sidecar's `convention_version`, so the stage 3 and 4 files (version 1)
  stay readable after the bump.
- **The three Phase 5 errors are errors 2–4 with the run's R and P:** error 2 is
  P R u_f − u_f (transfer-only), error 3 the coarse error, error 4 the fine-grid error.
  Generalising `poisson_errors` to take the run's transfers gives all three.
- **There is no ungated Chebyshev-coefficient transform.** The only one (`unit_coeffs`,
  `chebyshev/differentiation.rs`) is FFTW-backed, behind `datagen`. A filtered R needs a
  closed-form dense nodal-to-coefficient (DCT-I) matrix, built in f64 like T_cu.
- **The 129 CGL nodes are the even-indexed 257 CGL nodes**
  (−cos(π·2j/256) = −cos(πj/128)), so the reference check compares stage 4 at those nodes
  with stage 3 directly, with no interpolation.
- **Cost gaps:** `EpochMetrics` has no time field (the per-epoch `Duration` already reaches
  the `EpochObserver`); the crate has no parameter count (Burn's `Module::num_params`, to be
  confirmed in the fork); `train_poisson` has no seed argument.

## Planning decisions

Recorded in design §12, decision 30 (amendment of 2026-10-26). Chosen by the author
(AHA-HH), 2026-10-26.

- **D1, restriction R:** chosen by measured transfer-only error in T1, among filtered
  Chebyshev restriction (recommended), cell average and unfiltered T_cu. P stays
  Floater–Hormann d = 2 unless T1 shows it dominates at small s. Recorded when T1 merges.
- **D2, modes:** m = min(12, s/2) per axis: 8, 12, 12, 12 at s = 16, 32, 64, 128. **No
  mode-control arm.** Bandwidth and grid size are therefore confounded below s = 32 (only
  s = 16 changes m); this is a stated limitation (design §13.14), not a measured effect.
- **D3, padding:** p = 0. No CONVENTIONS §3 change. The existing p = 4 129² run is the
  padding control.
- **D4, seeds:** 3 per configuration (42, 43, 44). T6 raises an arm to 5 where the spread
  between seeds is comparable to the differences being compared.
- **D8, evaluation grids:** 129 CGL and each learning grid; 257 only in T1's reference
  check.
- **Compute:** the T5 matrix trains on the cuda A100 VM, as Phase 4 T2 and T3 did, so
  every cost compares on one device. Evaluation runs on flex. Each matrix's cost is timed
  from its first epochs and stated before its full runs start.
- **T1 is a spike** (`spikes/transfer_floor/`), as Phase 0 T3 chose d: the candidates that
  lose never enter `src/`. T2 ports the winner into the library.
- **The CONVENTIONS §12 amendment opens T2**, in the same PR as the code, with
  `CONVENTION_VERSION = 2` (§9).
- **Baseline rule, fixed before any T5 run:** the Phase 7 learning grid is the smallest s
  whose mean fine-grid error over seeds is within 10% of the s = 128 main-arm mean. T7
  applies it; it is not re-chosen after the results.

D5–D7 belong to Phases 6 and 7.

## Tasks

| Task | Brief | Delivers | Depends on |
| --- | --- | --- | --- |
| T1 | [T1-transfer-floor.md](T1-transfer-floor.md) | Spike: the 257 → 129 reference check; transfer-only error per candidate R and P at s ∈ {16, 32, 64, 128}; filtered-out energy of f; fixes D1 | – |
| T2 | [T2-restriction.md](T2-restriction.md) | CONVENTIONS §12 amendment (version 2); `Transfers::with_s` for any learning grid s (R = T_cu, D1), with oracle tests | T1 |
| T3 | [T3-loader-s.md](T3-loader-s.md) | Learning s in `PoissonConfig`; the loaders use them; `train_poisson` takes s; s = n − 1 stays the default and reproduces today | T2 |
| T4 | [T4-run-metadata.md](T4-run-metadata.md) | Per-epoch time in `metrics.csv`, `costs.json`, the extended `data_cfg.json`, parameter count, a seed argument | T3 |
| T5 | [T5-matrix.md](T5-matrix.md) | The training matrix on cuda: main arm (12 runs) and aliasing control (3 runs) | T4 |
| T6 | [T6-evaluate.md](T6-evaluate.md) | The three errors (and errors 2–5) on every run; error 4 against ‖u‖; comparison table and accuracy/cost plots | T5 |
| T7 | [T7-baseline.md](T7-baseline.md) | The Phase 7 baseline by the rule above; phase close-out | T6 |

Task numbers are execution order (design §10). One task per branch and PR:
`phase5/T<k>-<name>`. Each task is worked from its brief alone, plus the files the brief
lists under "Read first".

## Module layout

- `spikes/transfer_floor/` (T1): a standalone package, a non-default workspace member like
  `spikes/metal_diag`, never built by CI. It path-depends on `sciml-rs` for the loaders and
  transfers.
- `chebyshev/transfer.rs` (T2): `Transfers::with_s` for an independent (n, s), with
  R = T_cu and d = 2 fixed by CONVENTIONS §12 (D1 chose the unfiltered restriction, so no
  coefficient matrix or restriction enum is added). Ungated, ndarray, f64.
- `data/loaders/poisson/` (T3): `PoissonConfig` gains the learning size;
  `uniform.rs` builds `Transfers` from it.
- `runs/` (T4): `epoch_time_s` in `metrics.csv` and `costs.json`; `RunDir` stays the only
  owner of the run-file names.
- `metrics/poisson.rs` (T6): `poisson_errors` takes the run's `Transfers`.
- `examples/train/poisson.rs` (T3, T4: s and seed arguments), `examples/predict/poisson.rs`
  (T4, T6). The task that adds a command or argument updates the Commands section of
  `CLAUDE.md`.

## Results

Filled in as tasks merge. Phase 5 is done when every row has a value.

| Item | Value | Source |
| --- | --- | --- |
| Reference check: stage 4 at the 129 nodes against stage 3, max relative CC-L² | u 7.19e-13, f 0 (bit-identical); passes | T1 |
| Transfer-only error per R and s (mean / max), P's own error per d, filtered-out energy of f per s; D1 chosen | D1: unfiltered T_cu, d = 2. Transfer-only (= P's own error) at s = 16 / 32 / 64 / 128: 3.03e-3 / 3.47e-4 / 2.45e-5 / 2.53e-6 (mean). Filtered-out energy of f beyond s − 2: 4.2e-2 / 9.0e-3 / 1.2e-3 / 0. Aliasing row skipped (needs `datagen`). Full tables in `spikes/transfer_floor/REPORT.md` | T1 |
| CONVENTIONS §12 amended, version 2; R and P tests; wall time of the new tests | Done: §12 lets s differ from n − 1 (s ≥ 3), R = T_cu, P = T_uc d = 2; 5 new tests in `chebyshev::transfer`, 0.02 s together (debug) | T2 |
| Loader at s = n − 1 reproduces today's tensors | Yes, by construction: with no s the loader builds `Transfers::with_s(n, n − 1)`, which is `Transfers::new(n)`; `poisson_output_is_pinned` unchanged and passing (not compared on the real stage 2 files). A short stage 3 run at s = 32 (p = 0, modes 12) starts on flex: 11.3 s/epoch, test L² 0.286 / 0.199 / 0.108 after epochs 0–2 (stopped; not a result) | T3 |
| Metadata and `costs.json` written by a short run; old runs still open | Done: a 2-epoch stage 2 run on flex (p = 4, seed 42) writes `metrics.csv` with `epoch_time_s` (424 s, 417 s), `data_cfg.json` with n, s, K, d and both split files with their sidecar commits, and `costs.json` (load 0.14 s, training 841 s, warm inference 31.5 s from `predict_poisson`, 32.8 MB / 6.6 MB host-to-device, 2,368,001 parameters, peak GPU memory "unavailable"). `predict_poisson` on a copy of `poisson_fno_p4_1791922918` prints its five errors and writes an `errors.json` identical to the old one | T4 |
| Matrix cost, timed and stated before the full runs | Timed on the cuda VM (NVIDIA A100, one GPU of two; the other was in use by another job), p = 0, stage 3, seed 42, 4 epochs per s, partial runs deleted. Steady s/epoch (mean of epochs 1–3) / epoch 0 (warm-up) / peak GPU memory (whole device, CUDA context included): s = 16: 1.645 / 20.9 s / 490 MiB; s = 32: 1.706 / 25.0 s / 620 MiB; s = 64: 1.700 / 9.8 s / 970 MiB; s = 128: 2.444 / 15.2 s / 2058 MiB. Extrapolated to 500 epochs (epoch 0 + 499 steady): 842 / 876 / 858 / 1235 s, 3811 s per seed, about 3.2 h for the 12 main-arm runs (about 3.5 h with loading and the final save and evaluation, not timed). Below s = 128 the epoch time barely depends on s; s = 64 matches Phase 4's 65² p = 0 cuda run (1.7 s/epoch). Approved by the author 2026-10-28 | T5 |
| Main arm: coarse / fine-grid / transfer-only error per s, mean ± spread over seeds | T5, coarse only: final test relative L² on the learning grid (`metrics.csv`, against R u; not comparable across s, see error 4), mean [min–max] over seeds 42, 43, 44: s = 16: 3.690e-2 [3.605e-2–3.767e-2]; s = 32: 1.327e-2 [1.282e-2–1.357e-2]; s = 64: 1.060e-2 [1.040e-2–1.095e-2]; s = 128: 1.047e-2 [1.040e-2–1.056e-2]. Final train relative L² 2.7e-3 to 3.6e-3 on every run. All 12 runs 500 epochs, none diverged. T6 (`predict_poisson` on flex; full table in `docs/phase5/results.md`), mean [min–max] over seeds 42–44 at s = 16 and 32 and over seeds 42–46 at s = 64 and 128 (D4). Error 4, fine-grid on the 129 CGL grid: s = 16: 3.730e-2 [3.646e-2–3.807e-2]; s = 32: 1.329e-2 [1.285e-2–1.360e-2]; s = 64: 1.080e-2 [1.041e-2–1.127e-2]; s = 128: 1.065e-2 [1.040e-2–1.107e-2]. Error 2, the transfer-only floor (the same for every seed): 3.030e-3 / 3.468e-4 / 2.455e-5 / 2.531e-6, matching T1; at s = 128 it matches Phase 4 T3 (2.53e-6 / 9.50e-6 mean / max). Error 3 equals the final test L² of each run (cross-check passes on all 16), and error 4 is within 1.1% of error 3 everywhere, so the model's own error dominates: the floor is 8% of error 4 at s = 16 and under 3% from s = 32 on. D4: with seeds 42–44, the spread over seeds at s = 64 (5.4e-4) exceeded the gap to s = 128 (1.4e-4), so s = 64 and 128 were raised to 5 seeds (45 and 46, cuda, same commit, all 500 epochs; approved by the author 2026-10-29). With 5 seeds the gap is 1.6e-4 (1.5%) against spreads of 8.6e-4 (s = 64) and 6.6e-4 (s = 128): s = 64 and s = 128 are not distinguishable by seed. Seeds 45 and 46 are both above the 3-seed mean at both s, raising each mean by about 2% | T5, T6 |
| Main arm costs per s: time per epoch, total training time, inference time, peak GPU memory, host-to-device bytes, parameter count, backend | T5 (cuda, NVIDIA A100, one run at a time on one GPU; `costs.json`), mean [min–max] over seeds. s/epoch (epochs 1–499) / training time / peak GPU memory (whole device, CUDA context included) / host-to-device bytes train, test / parameters: s = 16: 1.679 [1.667–1.691] / 847 s [842–854] / 490 MiB / 2.05 MB, 0.41 MB / 1,057,281 (modes [8, 8]); s = 32: 1.686 [1.662–1.710] / 851 s [839–863] / 682 MiB / 8.19 MB, 1.64 MB / 2,368,001; s = 64: 1.687 [1.654–1.710] / 851 s [835–862] / 970 MiB / 32.8 MB, 6.55 MB / 2,368,001; s = 128: 2.448 [2.445–2.453] / 1232 s [1230–1234] / 2250 MiB / 131 MB, 26.2 MB / 2,368,001. Matrix wall time about 3.2 h, against the 3.2 h estimate; the 4 D4 runs about 70 min. Peak GPU memory is from seeds 42–44: the D4 runs read 1128 MiB (s = 64) and 2408 / 3316 MiB (s = 128), which the sampler's whole-device reading cannot separate from another job on the same GPU. Warm inference time on the 200 test samples (flex, on the author's Mac): 0.2 / 0.8 / 1.7 / 5.6 s at s = 16 / 32 / 64 / 128 | T5, T6 |
| Aliasing control (s = 32, unfiltered T_cu) against the main arm at s = 32 | Not run: D1 chose unfiltered T_cu, so this arm is the main arm (T5 deliverable 3) | T5 |
| Padding control (`poisson_fno_n129_p4_1792037830`) against the main arm at s = 128 | Re-evaluated on flex; reproduces Phase 4 T3 (errors 2–5, cross-checks, verdict 0.009). Error 4 1.079e-2 / 3.34e-2 (mean / max) against the p = 0 main arm's 1.065e-2 [1.040e-2–1.107e-2] (5 seeds): p = 4 is 1.3% worse than the p = 0 mean, above four of the five p = 0 seeds and inside their range, so the control shows no gain from padding, consistent with D3 | T6 |
| Error 4 against ‖u‖ over the test set | corr(ln error 4, ln ‖u‖) over the 200 test samples: −0.66, −0.68, −0.68, −0.68, −0.68 for the s = 128 p = 0 seeds and −0.68 for the padding control; −0.86 / −0.74 / −0.68 / −0.68 (mean over seeds) at s = 16 / 32 / 64 / 128. High error 4 comes largely from small ‖u‖, as decision 22 suspected (sample 84 at Phase 4 T3); the scatter plot is attached to the PR | T6 |
| Dataset generation time (solver cost, reported separately) | n = 129: 1.1 s train, 0.2 s test; n = 257: 5.2 s train, 1.1 s test | T1 (sidecars) |
| Phase 7 baseline and its justification | | T7 |

## Exit checklist

- [ ] T1–T7 merged.
- [x] CONVENTIONS §12 amended and `CONVENTION_VERSION = 2`.
- [x] The solver and learning grids are independently configurable.
- [x] The same forcing examples are represented at several learning resolutions.
- [ ] Restriction and prolongation errors are measured separately.
- [ ] Coarse-grid and fine-grid FNO errors are both reported.
- [ ] Accuracy/cost trade-offs are reproducible across seeds.
- [ ] The baseline resolution for Phase 7 is justified.
- [ ] D1 recorded in design §12 under decision 30 (D2–D4 and D8 are recorded at planning).
- [ ] Every row of Results filled in.
- [ ] Any result that contradicts the design is recorded in design §12, dated.
- [ ] The "Current phase" bullet in `CLAUDE.md` is updated for what comes next.

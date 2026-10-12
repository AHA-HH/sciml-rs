# Phase 3: training and evaluation, local

As of 2026-10-09. Design: `docs/design/2d-chebyshev-poisson-fno.md` (signed off
2026-10-05, amended 2026-10-08 and 2026-10-09; decisions in its §12). `docs/CONVENTIONS.md`
wins on any conflict.

Phase 3 trains the FNO on Chebyshev data for the first time and measures how good it is:
- closed-form Clenshaw–Curtis weights and a relative CC-L² error that build without
  `chebyshev`, so evaluation needs no FFTW (decision 15);
- a `train_poisson` example and trainer, the padding p chosen by a full-length sweep, and
  the first run at 65² on metal and flex (the sweep beyond p = 0 and the flex run moved
  to Phase 4, design §12, decision 17);
- a `predict_poisson` example that reports the five errors of design §7 on the Chebyshev
  grid, including the boundary error, and states whether the boundary-condition ablation
  is warranted.

Nothing in this phase changes the model, CONVENTIONS, solver A, the transfers or the
existing behaviour of the loader. Training and evaluation are ungated; only generating the
datasets needs `chebyshev`.

## Read first

- Design §3.4 (norms), §5.2–5.3 (loader, stages), §7 (training, the five errors, the
  boundary condition), §10 (Phase 3), §11 (rows for the closed-form Clenshaw–Curtis
  weights and the pipeline), §12 decisions 2, 10, 13, 14, 15 and 16.
- `docs/CONVENTIONS.md` §10 (normalisation, the `y_test` asymmetry), §11 (`LpLoss`) and
  §12 (grids, transfers, norms).
- `docs/phase2/README.md` Results (the stage 2 dataset, error 2, the loader).
- `CLAUDE.md`: Working agreement, Checks, Burn (the autodiff device note).

## Inputs from Phases 1 and 2

| Item | Value | Source |
| --- | --- | --- |
| Stage 2 dataset | n = 65, s = 64, K = 32; 1000 train (seeds 0–999), 200 test (seeds 1 000 000–1 000 199); `datasets/poisson/poisson_n65_K32_{train,test}.{npz,json}`, git-ignored, made by `cargo run --release --features chebyshev --example generate_poisson` | Phase 2 T2 |
| Label check, stage 2 | max 2.2e-9 / mean 4.0e-10 (train), 1.5e-9 / 3.8e-10 (test), enforced at 1e-8 | Phase 2 T2 |
| Error 2, stage 2 test split | max 7.8e-5, mean 2.0e-5 (train: 1.04e-4 / 2.1e-5); decision 14 | Phase 2 T2 |
| Loader | `data::loaders::poisson::{PoissonConfig, load_poisson_uniform, PoissonNormalizers}`; inputs `[N, 64, 64, 1]`, targets `[N, 64, 64]`, `y_test` raw; 0.10 s release load | Phase 2 T3 |
| Transfers | `chebyshev::{cheb_to_uniform, uniform_to_cheb, apply, uniform_nodes}`, ungated; FH d = 2 | Phase 2 T1 |
| Error 1 (solver) | A floor ≤ 1e-12 from n = 33 on manufactured solutions; A against C at order 1.993–1.999 | Phase 1 T2, T3 |
| Darcy hyperparameters, the template | modes [12, 12], width 32, 4 layers, 500 epochs, batch 20, lr 1e-3, weight decay 1e-4, min_lr 1e-5, padding 9 at s = 85 (`examples/train/darcy.rs`) | today's code |
| Expected model error | Li et al. Darcy 1.08e-2 at 85² (design §7) | design §7 |
| Test budget | the chebyshev suite took 108.8 s in debug at the end of Phase 2 | Phase 2 T3 |

## Planning decisions

Recorded in design §12, decision 16.

- **Three tasks, not two.** §10 had the ungated Clenshaw–Curtis weights inside the
  evaluation task. They are library code that needs no dataset or run, so they become T1,
  which can run in parallel with training (T2). Evaluation is T3.
- **Pipeline threshold: test relative L² ≤ 2e-2 at 65²**, on flex and on metal (flex
  moved to Phase 4, decision 17). It is
  fixed now, before T2 runs. If it is missed, the result is recorded in design §12 before
  anything is retuned.
- **Padding by a full-length sweep:** p ∈ {0, 4, 8, 16}, the Darcy hyperparameters and
  500 epochs each. The lowest final test relative L² wins, and values within 2% of each
  other go to the smaller p. Phase 3 ran p = 0 only; p = 4, 8, 16 moved to Phase 4
  (decision 17, amended 2026-10-12).
- **Error 1 is quoted, not recomputed:** `predict_poisson` is ungated, and solvers A and C
  need `chebyshev`. Error 2 is recomputed in evaluation and cross-checked against the
  sidecar.
- **Boundary threshold:** the hard-constraint ablation of §7 is warranted if the mean
  relative boundary RMS (error 5) is at least 10% of the mean error 4.

## Tasks

| Task | Brief | Delivers | Depends on |
| --- | --- | --- | --- |
| T1 | [T1-cc-weights.md](T1-cc-weights.md) | `chebyshev::{clencurt, cc_rel_l2_error}`, ungated, tested against the RLST-backed weights | Phase 2 |
| T2 | [T2-train-poisson.md](T2-train-poisson.md) | `training::trainers::poisson`, the `train_poisson` example, the padding sweep, the 65² run on metal and flex (sweep beyond p = 0 and flex: Phase 4, decision 17) | Phase 2 (parallel with T1) |
| T3 | [T3-predict-poisson.md](T3-predict-poisson.md) | `load_poisson_cheb`, `metrics::poisson` (the five errors), the `predict_poisson` example, the boundary verdict | T1, T2 |

Task numbers are execution order (design §10); T1 and T2 can run in parallel. One task per
branch and PR: `phase3/T<k>-<name>`. Each task is worked from its brief alone, plus the
files the brief lists under "Read first".

The datasets and `runs/` are git-ignored and live in each worktree. A task that needs the
stage 2 dataset generates it in its worktree or copies `datasets/poisson/` in, and records
its numbers in Results, since run directories are not committed.

## Module layout

- `src/neural_operators/chebyshev/clencurt.rs` (T1): ungated, registered and re-exported
  next to `transfer` in `chebyshev/mod.rs`.
- `src/neural_operators/training/trainers/poisson.rs` (T2), registered in
  `trainers/mod.rs`.
- `examples/train/poisson.rs` (T2, `train_poisson`) and `examples/predict/poisson.rs`
  (T3, `predict_poisson`), registered in `Cargo.toml` with no required features.
- `load_poisson_cheb` (T3): a new public function in `data/loaders/poisson.rs`.
- `src/neural_operators/metrics/poisson.rs` (T3): ungated, ndarray, f64, registered in
  `metrics/mod.rs`.
- The task that adds a module adds its line to the Layout section of `CLAUDE.md`; T2 and
  T3 add their commands to its Commands section.

## Results

Filled in as tasks merge. Phase 3 is done when every row has a value.

| Item | Value | Source |
| --- | --- | --- |
| `clencurt` against the RLST-backed `clenshaw_curtis` (max abs difference, per n), and the exact-integral tolerance | max 1.1e-16 over n = 2..=129 (worst at n = 7, 11, 13); n = 2: 0, 3: 5.6e-17, 9: 5.6e-17, 33: 2.1e-17, 65: 1.4e-17, 129: 1.0e-17; asserted ≤ 1e-14. Exact integrals of x^k, k ≤ n − 1, n ∈ {2, 3, 9, 33, 65}: within 1e-14 | T1 |
| Padding sweep: final test relative L² for p = 0, 4, 8, 16 (backend, wall time per run) | p = 0: 1.16e-2 (metal, 1763 s). p = 4, 8, 16 moved to Phase 4 (design §12, decision 17, amended 2026-10-12): a p = 8 probe ran at about 101 s/epoch on metal, about 40 h for the three on metal | T2 |
| Chosen p, and the full run at 65²: final train / test relative L², time per epoch, on metal and on flex | p = 0 for Phase 3, revisited by the Phase 4 sweep (decision 17). Metal: 3.47e-3 / 1.16e-2, 3.5 s/epoch (first epoch 8.5 s), run `runs/poisson_fno_p0_1791563533`. That run and the dataset were lost with T2's worktree (both git-ignored), so T3 regenerated the dataset (identical label check and error 2) and re-ran p = 0 on metal with unchanged code: train / test 3.45e-3 / 1.116e-2, 3.6 s/epoch (first epoch 30.6 s, while other builds were running), run `runs/poisson_fno_p0_1791628852`, which T3 evaluates. Flex: moved to Phase 4 (decision 17) | T2 |
| Pipeline threshold (test relative L² ≤ 2e-2) met on metal / on flex | Metal: met (1.16e-2). Flex: moved to Phase 4 (decision 17) | T2 |
| The five errors on the stage 2 test split, mean / max: 1 (quoted), 2, 3, 4, 5 (max ratio and relative RMS) | Run `runs/poisson_fno_p0_1791628852` (metal, 200 test samples). 1: quoted, A floor ≤ 1e-12, A vs C order 1.993–1.999. 2: 2.03e-5 / 7.81e-5. 3: 1.116e-2 / 3.67e-2. 4: 1.117e-2 / 3.67e-2. 5: max ratio 7.19e-5 / 1.60e-4, relative RMS 8.84e-5 / 1.62e-4 (∮ û² ds integrates each edge separately, so each corner carries the end weight of both its edges: the reading the brief now states) | T3 |
| Error 2 recomputed against the sidecar, and error 3 against the run's final test_l2 | Error 2: relative difference 3.3e-16 (mean) and 1.2e-15 (max) from the test sidecar (tolerance 1e-12). Error 3: mean 1.115918e-2 against `test_l2` 1.115918e-2 at epoch 499, relative difference 1.0e-7 (tolerance 1e-5) | T3 |
| Boundary verdict: relative boundary RMS / error 4, and whether the ablation is warranted | 8.84e-5 / 1.117e-2 = 0.008, below the 0.10 threshold: the hard-constraint ablation is not warranted | T3 |
| Wall time of the new tests under debug `cargo test` and `cargo test --features chebyshev` | T1: 9 ungated tests 0.00 s; 11 with `chebyshev` 0.46 s (macOS). T2 `train_poisson_smoke`: 0.11 s on flex (8.9 s with `--features metal`). T3: 9 `metrics::poisson` tests 0.06 s and 4 `cheb_loader` tests 0.01 s; with `chebyshev` 0.04 s and 0.01 s; `error3_matches_lploss` 0.89 s with `--features metal` | T1–T3 |

## Exit checklist

- [x] T1, T2 and T3 merged.
- [x] The design §11 row for the closed-form Clenshaw–Curtis weights passes under
      `cargo test --features chebyshev`, and the ungated tests pass under `cargo test`,
      locally on macOS and in both CI jobs.
- [x] The §11 pipeline row met: test relative L² ≤ 2e-2 at 65², on metal. The flex run
      and the padding sweep beyond p = 0 moved to Phase 4 (design §12, decision 17).
- [x] The five errors of §7 reported for the stage 2 run and recorded above.
- [x] The boundary verdict stated. If the ablation is warranted, it is planned then as a
      new Phase 3 task.
- [x] Every row of Results filled in.
- [x] Any result that contradicts the design is recorded in the design's §12 under
      "Recorded decisions", dated, before Phase 4 is planned.
- [x] The "Current phase" bullet in `CLAUDE.md` points to `docs/phase4/README.md`.
      Ticked when Phase 4 is planned.

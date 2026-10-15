# Phase 4: HPC

As of 2026-10-12. Design: `docs/design/2d-chebyshev-poisson-fno.md` (signed off
2026-10-05, amended 2026-10-08, 2026-10-09, 2026-10-10, 2026-10-12, 2026-10-13, 2026-10-14 and
2026-10-15; decisions in its §12). `docs/CONVENTIONS.md` wins on any conflict.

Phase 4 finishes what Phase 3 deferred, then scales up:
- a `poisson_plot` example that shows one test sample's forcing f, solution u, prediction
  û and error map on the Chebyshev grid (agreed 2026-10-12 as the first task);
- the padding sweep over p ∈ {4, 8, 16} at 65² and the full-length flex run at the chosen
  p, carried over from Phase 3 (design §12, decision 17);
- production runs at 129² on GPU (cuda, on HPC);
- the resolution study of design §7, in two parts on the same seeds.

Nothing in this phase changes the model, CONVENTIONS, solver A, the transfers, the
dataset format or the five errors. Training and evaluation stay ungated; only generating
datasets needs `chebyshev`.

## Read first

- Design §5.3 (stages, the K = 32 evaluation sets, storage), §7 (training, the five
  errors, the resolution study), §9 (backends), §10 (Phase 4), §11 (the pipeline row),
  §12 decisions 16, 17 and 18.
- `docs/CONVENTIONS.md` §3 (padding), §4 (FFT, Bluestein), §10 and §12.
- `docs/phase3/README.md` Results (the p = 0 run, the five errors, the boundary verdict).
- `CLAUDE.md`: Working agreement, Checks, Burn (backends, the autodiff device note).

## Inputs from Phase 3

| Item | Value | Source |
| --- | --- | --- |
| Stage 2 dataset | n = 65, s = 64, K = 32; 1000 train / 200 test; `datasets/poisson/poisson_n65_K32_{train,test}.{npz,json}`, git-ignored; present in the main checkout | Phase 2 T2 |
| p = 0 run at 65², metal | train / test relative L² 3.45e-3 / 1.116e-2, 3.6 s/epoch; run `runs/poisson_fno_p0_1791628852`, since lost with T3's worktree | Phase 3 T2, T3 |
| Five errors of that run, mean / max | 2: 2.03e-5 / 7.81e-5; 3: 1.116e-2 / 3.67e-2; 4: 1.117e-2 / 3.67e-2; 5: max ratio 7.19e-5 / 1.60e-4, relative RMS 8.84e-5 / 1.62e-4 | Phase 3 T3 |
| Boundary verdict | 0.008 against the 0.10 threshold: ablation not warranted | Phase 3 T3 |
| Padding cost | p = 8: about 101 s/epoch on metal, 520 s/epoch on flex (Bluestein at m = 256, chirp rebuilt per call); about 40 h for p ∈ {4, 8, 16} on metal | decision 17 |
| Decision 16 sweep rule | full length (500 epochs, Darcy hyperparameters); lowest final test relative L² wins; values within 2% go to the smaller p | decision 16 |
| Pipeline threshold | test relative L² ≤ 2e-2 at 65²; met on metal, flex outstanding | decisions 16, 17 |
| Examples | `train_poisson [-- <padding>]`, `predict_poisson -- <run_dir>`, both hard-wired to stage 2 (`N = 65`, the n65_K32 files); `darcy_plot`, the plot template | today's code |

## Planning decisions

Recorded in design §12, decision 18.

- **Four tasks, in this order:** T1 `poisson_plot`; T2 the padding sweep and the flex run
  at 65²; T3 production runs at 129² on cuda; T4 the resolution study. §10's T1 and T2
  become T3 and T4.
- **`poisson_plot` plots on the CGL nodes at their coordinates**, not on the node index,
  so the clustering at the edges shows. f and u come from the raw split; û is
  T_uc(prediction), as in error 4.
- **`poisson_plot` draws cells, not points (2026-10-13, decision 19).** Each node is a
  filled cell reaching halfway to its neighbours, which removes the white lattice between
  the square points. Flat cells still show steps on smooth fields; whether to draw a
  smooth display instead is a supplementary Phase 5 task (design §10), not Phase 4 work.
- **Runs cited in Results live in the main checkout's `runs/`.** Two runs were lost with
  their worktrees in Phase 3. From now on `train_poisson` runs used in Results are moved
  (or written) to the main checkout's `runs/`, and the datasets are read from there, by copy
  or symlink into the worktree. Nothing about this is committed.
- **The sweep runs on metal, locally,** before anything runs on cuda: about 40 h, on
  mains power. cuda comes in with T3. The flex run uses the chosen p; if that is
  p > 0, its cost (about 72 h at 520 s/epoch) is stated before it starts, and a fix to
  Bluestein in Burn becomes its own task first if the author asks for one (decision 17).
- **Stages are chosen by argument, not by editing constants.** T3 lets `train_poisson`
  and `predict_poisson` select the stage (n = 65, 129, 257) from a short table of file
  names; the stage 2 defaults and the padding argument keep working unchanged.
- **`poisson_plot` follows the run's stage; `predict_poisson` names the worst sample
  (2026-10-15, decision 22).** A small follow-up to T3, outside the briefs, as #29 was to
  T1. `poisson_plot` takes n from the run's `data_cfg.json`, from the same stage table as
  `predict_poisson`, and writes `prediction_<sample>.png`, so plots of several samples
  sit side by side (T1 wrote `prediction.png`). `predict_poisson` prints the index of the
  sample with the max error 4 and records it in `errors.json` as `error4_worst_sample`.
- **The resolution study evaluates without retraining** a 65² model on sets 3e and 4e
  (K = 32) and a 129² model on stage 4 (K = 64), and reports errors 3 and 4 at each
  resolution. The pointwise normalisers are refit on the same training fields
  (same seeds and K) generated at the new n, so K = 32 train splits at 129 and 257 are
  generated for that alone; the weights are never refit.

## Tasks

| Task | Brief | Delivers | Depends on |
| --- | --- | --- | --- |
| T1 | [T1-poisson-plot.md](T1-poisson-plot.md) | `load_poisson_cheb_field`, the `poisson_plot` example, a regenerated p = 0 run at 65² | Phase 3 |
| T2 | [T2-padding-sweep.md](T2-padding-sweep.md) | the sweep p ∈ {4, 8, 16} at 65², the chosen `PADDING`, the full-length flex run, `predict_poisson` on the chosen run | T1 (the run at p = 0) |
| T3 | [T3-runs-129.md](T3-runs-129.md) | stage selection in `train_poisson` / `predict_poisson`, the stage 3 dataset, production runs at 129² on cuda | T2 (the chosen p) |
| T4 | [T4-resolution-study.md](T4-resolution-study.md) | sets 3e, 4e and stage 4, K = 32 train splits at 129 and 257 for the normalisers, a `poisson_resolution` example, both parts of the study | T3 |

Task numbers are execution order (design §10). One task per branch and PR:
`phase4/T<k>-<name>`. Each task is worked from its brief alone, plus the files the brief
lists under "Read first".

## Module layout

- `load_poisson_cheb_field` (T1): a new public function in `data/loaders/poisson.rs`;
  `load_poisson_cheb` delegates to it with `"u"`.
- `examples/predict/poisson_plot.rs` (T1, `poisson_plot`) and
  `examples/predict/poisson_resolution.rs` (T4, `poisson_resolution`), registered in
  `Cargo.toml` with no required features.
- `poisson_plot` keeps its own copy of `cgl_nodes` because the library function is
  `pub(crate)`; the formula is fixed by CONVENTIONS §12.
- `examples/train/poisson.rs` and `examples/predict/poisson.rs` gain a stage argument (T3).
- `examples/predict/poisson_plot.rs` reads the stage from the run, and
  `examples/predict/poisson.rs` reports the worst sample (decision 22).
- `examples/generate/poisson.rs`: the stage 3, 4, 3e and 4e lines are uncommented as T3
  and T4 need them.
- The task that adds an example adds its command to the Commands section of `CLAUDE.md`.

## Results

Filled in as tasks merge. Phase 4 is done when every row has a value.

| Item | Value | Source |
| --- | --- | --- |
| `poisson_plot`: run plotted, sample, output file; the regenerated p = 0 run's train / test relative L² against Phase 3's 3.45e-3 / 1.116e-2 | `runs/poisson_fno_p0_1791811912` (main checkout; metal, 65², unchanged code), final train / test relative L² 3.450e-3 / 1.116e-2 against 3.45e-3 / 1.116e-2. Plotted on metal for samples 0 (error 4 = 5.744e-3) and 17 (2.932e-2), written to the run's `prediction.png` (sample 0 is the one left there); û resembles u, the error is about 1e-3 and near 0 on ∂Ω, and the nodes cluster at the edges. Sample 17's error is mostly one-signed (û below u in the interior) | T1 |
| Padding sweep: final test relative L² for p = 0, 4, 8, 16 (backend, s/epoch, wall time) | On cuda (VM, NVIDIA A100 80GB PCIe), not on metal as planned (decision 18); 500 epochs, Darcy hyperparameters, seed 42. Final train / test relative L²: p = 0 3.397e-3 / 1.138e-2 (`runs/poisson_fno_p0_1791906496`); p = 4 2.802e-3 / 1.069e-2 (`runs/poisson_fno_p4_1791922918`); p = 8 2.742e-3 / 1.066e-2 (`runs/poisson_fno_p8_1791938538`); p = 16 2.517e-3 / 1.077e-2 (`runs/poisson_fno_p16_1791955593`); all in the main checkout. Wall time / s/epoch: p = 0 868 s / 1.7; p = 4 16417 s (4.6 h) / 32.8; p = 8 15617 s (4.3 h) / 31.2; p = 16 17053 s (4.7 h) / 34.1. Any p > 0 is about 20× slower than p = 0 (Bluestein, decision 17). Against T1's metal p = 0 run (1.116e-2), the cuda p = 0 run differs by 2.0% | T2 |
| Chosen p, by the decision 16 rule; `PADDING` changed or not | p = 4. p = 8 is lowest (1.066e-2); p = 4 is 0.23% above it and p = 16 0.99% above, both within 2%, so the smaller p wins; p = 0 is 6.8% above. p = 4 beats T1's p = 0 run by 4.2%, more than the 3.9% spread between the Phase 3 and T1 p = 0 runs (1.16e-2 against 1.116e-2), so no §12 entry is needed for the margin. `PADDING` changed from 0 to 4 in `examples/train/poisson.rs`; "provisional" dropped | T2 |
| Flex run at the chosen p: final train / test relative L², s/epoch; threshold 2e-2 met on flex | Deferred to a supplementary Phase 5 task (design §10, decision 21): about 72 h on flex at p = 4. The threshold is met on metal at p = 0 and on cuda at p = 4; the p = 4 model evaluates identically on flex | T2 |
| Five errors and boundary verdict on the chosen run (if p ≠ 0) | `runs/poisson_fno_p4_1791922918` (the cuda p = 4 run, decision 20), evaluated on flex; mean / max: 2: 2.03e-5 / 7.81e-5; 3: 1.069e-2 / 2.97e-2; 4: 1.069e-2 / 2.97e-2; 5: max ratio 7.81e-5 / 1.86e-4, relative RMS 9.74e-5 / 1.84e-4. Cross-checks pass (error 2 against the sidecar, error 3 against `metrics.csv`). Boundary verdict 0.009 against the 0.10 threshold: ablation not warranted. Against Phase 3's p = 0 run, error 4 falls from 1.117e-2 to 1.069e-2 and its max from 3.67e-2 to 2.97e-2; error 5 rises slightly (relative RMS 8.84e-5 to 9.74e-5) | T2 |
| Stage 3 dataset: label check and error 2, train / test | n = 129, s = 128, K = 64; 1000 train / 200 test; `datasets/poisson/poisson_n129_K64_{train,test}.{npz,json}` (main checkout), generated at `3f5145b`. Label check (enforced, tolerance 1e-8), mean / max: train 4.74e-13 / 9.17e-13, test 4.71e-13 / 8.97e-13. Error 2, mean / max: train 2.57e-6 / 1.28e-5, test 2.53e-6 / 9.50e-6, against 2.03e-5 / 7.81e-5 at 65² and Phase 0 T3's 4.4e-6 (max) at 129² (design §7); still three orders below error 4 | T3 |
| 129² runs on cuda: train / test relative L², s/epoch, GPU model, the five errors | One run, `runs/poisson_fno_n129_p4_1792037830` (main checkout): cuda, p = 4, seed 42, 500 epochs, Darcy hyperparameters, modes [12, 12]; no second seed. NVIDIA A100; wall time about 56000 s (15.6 h), about 112 s/epoch, against 32.8 s/epoch for the 65² p = 4 run (about 3.4× for 4× the grid points). Final train / test relative L² 3.123e-3 / 1.079e-2, against 2.802e-3 / 1.069e-2 for the 65² p = 4 run: test 0.9% higher, so near the 65² error and no §12 entry is needed (T3 acceptance). Five errors, evaluated on flex, mean / max: 2: 2.53e-6 / 9.50e-6; 3: 1.079e-2 / 3.34e-2; 4: 1.079e-2 / 3.34e-2; 5: max ratio 7.66e-5 / 1.88e-4, relative RMS 9.49e-5 / 1.77e-4. Cross-checks pass (error 2 against the sidecar, rel diff 5.4e-16; error 3 against `metrics.csv`, 4.6e-6). Boundary verdict 0.009 against the 0.10 threshold: ablation not warranted. Max error 4 rises from 2.97e-2 at 65². The worst sample is 84 (decision 22). `poisson_plot` on flex for samples 0 (error 4 = 8.427e-3), 17 (1.985e-2) and 84 (3.344e-2), written to the run's `prediction_<sample>.png`: û resembles u in each, the largest pointwise error is about 2e-3 in all three and near 0 on ∂Ω, and samples 17 and 84 are mostly one-signed (û below u in the interior). Sample 84's u is the smallest of the three (about −0.035 to 0.065), so its error 4 is high because ‖u‖ is small, not because its absolute error is larger; a plot of error 4 against ‖u‖ over the test set would test this. The GPU machine has no gnuplot, so `plots.png` was redrawn locally from `metrics.csv` | T3 |
| Resolution study part 1: errors 3 and 4 of the 65² model at 65, 129, 257 (K = 32) | `runs/poisson_fno_p4_1791922918` (T2's chosen run, p = 4), weights unchanged, normalisers refit on the K = 32 train split at each n (decision 18); `poisson_resolution` on flex, 200 test samples on the stage 2 test seeds, 9.8 min. Error 3 / error 4, mean / max: n = 65: 1.069e-2 / 2.97e-2, 1.069e-2 / 2.97e-2 (cross-check against `metrics.csv` rel diff 2.8e-6); n = 129 (3e): 4.666e-2 / 1.522e-1, 4.666e-2 / 1.522e-1 (4.4×); n = 257 (4e): 7.546e-2 / 2.455e-1, 7.546e-2 / 2.455e-1 (7.1×). The error grows with n rather than staying flat; error 2 stays ≤ 2.0e-5, so the growth is the model's. New datasets (generated at `e5ea6cb` plus this task's generator edit), label check (enforced) max and error 2 mean / max: n129_K32 train 9.1e-13, 2.5e-6 / 1.2e-5; test (3e) 9.0e-13, 2.4e-6 / 9.2e-6; n257_K32 train 1.6e-12, 2.7e-7 / 1.4e-6; test (4e) 1.4e-12, 2.7e-7 / 1.0e-6 | T4 |
| Resolution study part 2: errors 3 and 4 of the 129² model at 129, 257 (K = 64) | `runs/poisson_fno_n129_p4_1792037830` (T3's run, p = 4), weights unchanged, normalisers refit on the stage 4 train split at 257; flex, 200 test samples on the same seeds, 11.0 min. Error 3 / error 4, mean / max: n = 129: 1.079e-2 / 3.34e-2, 1.079e-2 / 3.34e-2 (cross-check rel diff 4.6e-6); n = 257 (stage 4): 2.593e-2 / 1.050e-1, 2.594e-2 / 1.050e-1 (2.4×). Doubling n costs less here than from 65 to 129 in part 1 (4.4×), but the error still grows. Stage 4 dataset, label check (enforced) max and error 2 mean / max: train 1.6e-12, 2.8e-7 / 1.4e-6; test 1.4e-12, 2.8e-7 / 1.0e-6 | T4 |
| Padding check (added to T4): the resolution study of part 1 on each 65² sweep run, p ∈ {0, 4, 8, 16} | T2's sweep runs (main checkout `runs/`, cuda-trained, seed 42, K = 32), weights unchanged, normalisers refit as in part 1; `poisson_resolution` on flex, 200 test samples. Mean error 3 at n = 65 / 129 / 257: p = 0 (`poisson_fno_p0_1791906496`) 1.138e-2 / 1.378e-2 / 1.608e-2 (1.4×); p = 4 (`poisson_fno_p4_1791922918`) 1.069e-2 / 4.666e-2 / 7.546e-2 (7.1×); p = 8 (`poisson_fno_p8_1791938538`) 1.066e-2 / 7.718e-2 / 1.354e-1 (12.7×); p = 16 (`poisson_fno_p16_1791955593`) 1.077e-2 / 1.409e-1 / 2.355e-1 (21.9×). Max error 3 at 257: 5.20e-2, 2.45e-1, 3.48e-1, 6.10e-1. Error 4 within 5e-6 of error 3 throughout. Cross-check at n = 65 against each run's `metrics.csv` passes (rel diff ≤ 3.2e-6). The growth rises steadily with p, so the fixed grid-point padding is the main cause (decision 23); p = 0 is the worst at 65 but the best at 129 and 257. Run with the `poisson_resolution` of the uncommitted `refactor/poisson-examples` tree; its p = 4 rows reproduce part 1 to every printed digit | T4 |
| Wall time of the new tests under debug `cargo test` | T1: `cheb_field_loader_reads_f`, `cheb_field_loader_rejects_unknown_field` under 0.01 s each (reported 0.00 s). T2–T4 added no tests | T1–T4 |

## Exit checklist

- [ ] T1, T2, T3 and T4 merged.
- [ ] `poisson_plot` writes its figure for a stage 2 run, and the run is recorded above.
- [ ] The padding sweep reported, p chosen by the decision 16 rule, and the §11 pipeline
      row met on metal and cuda (the flex run deferred to Phase 5, decision 21).
- [ ] cuda runs reported at 129², with the five errors.
- [ ] Both parts of the resolution study reported on the same seeds (§7).
- [ ] Every row of Results filled in.
- [ ] Any result that contradicts the design is recorded in the design's §12 under
      "Recorded decisions", dated, before Phase 5 is planned or the project closes.
- [ ] The "Current phase" bullet in `CLAUDE.md` is updated for what comes next.

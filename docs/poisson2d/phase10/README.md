# Phase 10: Gauss–Newton optimisation

As of 2026-11-24: **done.** T1 (the generic core, #98); T2 (the optimiser, #102); T3 (trainer integration, #104; its cuda f64 check blocked upstream, decision 31's amendment of 2026-11-24); T4 (validation gate, #105; go); T5 (the matrix, #106); T6 (conditioning, #107); the VM runs, in the order of [runs.md](runs.md), and the follow-ups of #118. Results: [results.md](results.md) (H10a) and [conditioning.md](conditioning.md) (H10b). More GN seeds at μ = 0 and a long Adam run are post-Phase 10 work. Plan and briefs T1–T6 written 2026-11-09;
decisions G1–G8 chosen by the author the same day (below). It can start now: G8, amended
2026-11-09, has it wait neither for Phase 9 nor for the θ-map (now an optional phase).
Design: `docs/design/2d-chebyshev-poisson-fno.md` (§10 Phase 10, §13.17, decision
31). `docs/CONVENTIONS.md` wins on any conflict.

Phase 10 adds a second-order optimiser: matrix-free, damped Gauss–Newton (Levenberg–
Marquardt with truncated conjugate gradients), written against Burn's `Module` so that
it serves any differentiable neural operator, not only the FNO. The first validation
case is the Phase 5 baseline: the 2D Poisson FNO at n = 129, s = 64, p = 0, N = 1000.

It is research, like Phases 5–7: errors and costs are compared on one device of one kind
(one A100), against the same baseline, seeds and test set. It does not block the
completion of Phases 5–7 (design §13.16). **The Adam path is unchanged**: every existing
run, test and result stays valid, and the baseline is retrained only to measure its
cost on the Phase 10 commit (G6).

## Read first

- Design §13.17 (this phase: the derivation, the residual, the conditioning study),
  §13.13 (H10a, H10b), decisions 30 and 31.
- `docs/poisson2d/phase7/README.md` and `docs/poisson2d/phase7/results.md` (the N = 1000 baseline and the
  nested subsets T4 uses); `docs/poisson2d/phase6/results.md` (the POD spectra T6 compares with);
  `docs/poisson2d/phase9/README.md` (checkpoints, the VM).
- `CLAUDE.md`: Working agreement, Checks, Burn, Commands.

## Inputs from earlier phases

| Item | Value | Source |
| --- | --- | --- |
| Baseline run | stage 3 (n = 129, K = 64), s = 64, p = 0, modes [12, 12], width 32, 4 layers, batch 20, 500 epochs, Adam 1e-3 / wd 1e-4, cosine to 1e-5, N = 1000, 200 test samples | Phase 5 T7, Phase 7 |
| Baseline error 4 (fine grid), 3 seeds | 1.061e-2 [1.041e-2–1.095e-2]; final train L² 3.185e-3 [2.862e-3–3.513e-3] | Phase 7 results |
| Baseline cost (cuda, A100) | 1.69 s/epoch, about 856 s per run, 25,000 steps; about 31 ms per batch of 20, flat in s up to 64 (overhead-bound) | Phase 5 T5, Phase 7 T2, Phase 9 "Closed at planning" |
| Smallest-experiment comparator | Phase 7's N = 100 runs (subset 0, seed 42 among them); mean error 4 at N = 100: 7.619e-2 [7.009e-2–8.266e-2] | Phase 7 results |
| Parameter count | 2,368,001, of which 2,359,296 (99.6 %) spectral weights (4 layers × 2 corners × re/im × 32·32·12·12) | `costs.json`, `FNOConfig::init` |
| Checkpoint | `checkpoint.bin`: weights, optimiser and scheduler records, next epoch, resumes, training time | Phase 9 T1 |
| `costs.json` | `peak_gpu_memory`: nothing in the crate measures it; on cuda it is filled in from `nvidia-smi` sampled beside the run (938 MiB on the Phase 7 baseline, 970 MiB in Phase 5), else `"unavailable"` | `runs/costs.rs`, Phase 5 T4, Phase 7 |

## Facts verified at `bd83393`

- **Burn autodiff is first-order reverse mode only.** `Device::autodiff` documents "only
  first-order autodiff is supported" (`burn-tensor/src/device.rs:581`). There is no
  forward mode (no tangent rules, no dual tensors) and no differentiable backward, so a
  Jacobian–vector product (JVP) cannot come from Burn: neither directly nor by the
  double-VJP trick. A vector–Jacobian product (VJP) is one `backward()`.
- **`backward()` consumes the graph.** Its doc (`burn-tensor/src/tensor/api/autodiff.rs`,
  above `backward`) asks for a fresh forward pass for each backward; parameter leaves are
  reusable. Every VJP inside CG therefore costs a forward and a backward.
- **Any model's parameters can be read and written generically.** `ModuleVisitor`
  (`visit_float(&Param<Tensor<D>>)`) and `ModuleMapper` (`burn-core/src/module/base.rs`)
  walk every float parameter of any `Module`; `GradientsParams::from_grads(grads,
  &module)` and `get::<D>(ParamId)` (`burn-optim/src/optim/grads.rs`) return the gradient
  per parameter. Parameters are real: the FNO stores complex spectral weights as
  separate `weights_re` / `weights_im` tensors (`layers/spectral_convolution.rs`).
- **The loss is the summed, unsquared relative L².** `forward_loss`
  (`training/step.rs`): forward, `flatten_pair`, decode with the y normaliser,
  `LpLoss::new(2, Sum).rel` = Σᵢ ‖ûᵢ − uᵢ‖₂ / ‖uᵢ‖₂ over the batch, equal weights on the
  64² uniform grid, targets in physical units (CONVENTIONS §10–§11).
- **Everything in `training/` is typed to `FNO<R>`** (`forward_loss`, `train_epoch`,
  `training_loop`, `Checkpoint<R>`); the optimiser is Burn's `ModuleOptimizer` (Adam with
  weight decay) with a `ModuleLrScheduler` (cosine), stepped once per batch.
- **There is no JVP, CG, Krylov or preconditioner code on tensors.** The only CG is
  `pde/poisson/sparse.rs`, a test-only f64 ndarray oracle. `burn-optim` ships an L-BFGS
  (`optim/lbfgs.rs`), unused here.
- **An explicit Jacobian is infeasible.** In f32: one sample (4,096 outputs × 2,368,001
  parameters) is 38.8 GB; a batch of 20 is 776 GB; the full train split 38.8 TB. Even
  the output-space Gram J Jᵀ of one batch is 26.8 GB. One parameter vector is 9.5 MB.

## The method in brief

Design §13.17 has the derivation. For a batch of B samples, the residual is the
per-sample relative error (G1, option A):

  rᵢ(θ) = (ûᵢ(θ) − uᵢ) / ‖uᵢ‖, Φ(θ) = (1/2B) Σᵢ ‖rᵢ(θ)‖², J = ∂r/∂θ.

The gradient is Jᵀr; the Gauss–Newton curvature is G = JᵀJ. Each step solves
(G + (λ + μ)I) p = −(Jᵀr + μθ) by truncated, preconditioned CG, with weight decay μ as a
fixed Tikhonov term (G5) and λ adapted from the ratio ρ of actual to predicted
reduction. G is never formed: G v = Jᵀ(J v) costs one JVP (central finite differences,
G2) and one VJP (backward with the residual seeded by J v).

## Planning decisions

Chosen by the author (AHA-HH), 2026-11-09, and recorded in design §12, decision 31.
G1–G5 and G8 as recommended at planning; G6 and G7 amended by the author (A100s); G8
amended by the author later the same day (below).

- **G1, residual: (A), the squared per-sample relative error**, mean over the batch.
  Training still reports the baseline loss L = Σᵢ ‖ûᵢ − uᵢ‖/‖uᵢ‖ in `metrics.csv`.
  Alternatives kept for later: (C) iteratively reweighted (wᵢ = 1/eᵢ, frozen per step,
  whose gradient equals ∇L exactly) as a follow-up arm; (B), the exact generalised
  Gauss–Newton of L, rejected (no curvature along the residual direction).
- **G2, JVP: central finite differences first**, generic for any `Module`. An exact
  tangent pass for the FNO (an override of `LeastSquares::jvp`) is inserted as a task
  after T4 only if T4 shows finite-difference noise limiting CG. Forward mode in the Burn
  fork is out of scope.
- **G3, batches: two arms, 20 (the baseline's) and 200**; full batch at N = 100 in T4.
  Gradient, curvature and ρ always on the same batch.
- **G4, stopping: equal wall-clock on the A100**: the mean training time of the Adam
  reruns (G6). Updates, epochs, forward-equivalent passes and peak memory are reported
  beside it, and error against each.
- **G5, weight decay: kept**, μ = 1e-4 as a Tikhonov term, for parity with Adam's.
- **G6, baseline arm: Adam rerun on the A100, three seeds (42, 43, 44)**, on the Phase 10
  commit, unchanged settings. It gives the time budget (G4) and peak memory, and its
  error 4 must fall within Phase 7's spread (a regression check of the unchanged Adam
  path), widened by ±2 % to 1.020e-2–1.117e-2 (amended 2026-11-11 by the author, decision 31:
  the spread is over the same seeds, two on its edges, and cuda round-off differs across
  commits). Phase 7's runs, also on an A100, stay as context.
- **G7, hardware: cuda on one A100 (the VM) for every Phase 10 run**; flex for tests; metal
  optional for development. Whether f64 works end to end on cuda (FFTs included) is
  checked in T3; if it does, f64 finite differences are an option for T4.
- **G8, order: amended 2026-11-09 by the author: Phase 10 starts now** and waits for
  nothing. It needs one A100 per run, not NCCL, so the VM's NCCL fix (Phase 9 T2) does
  not gate it, and the θ-map became an optional phase that nothing waits for. G7 still
  puts its runs (T4–T6) on the VM. (At planning, superseded: Phase 10 started after
  Phase 9, once the NCCL fix and the θ-map, then Phase 9 T5–T6, were done.)
- **Order, amended 2026-11-11 by the author:** task code ahead of its runs. T5's code (the Hybrid arm's Adam-then-GN switch, `TrainingConfig::adam_epochs`; `OPTIMIZER=hybrid` and `GN_PRECONDITIONER` in `train_poisson`; `poisson_matrix --by-optimizer`) is written before T4's go. Every VM run comes at the end of the phase, in task order (T3's acceptance, T4's gate, T5's matrix, T6's measurements); T5's runs still wait for T4's go, and a gate failure on JVP noise still inserts the exact-JVP task before them (G2). Decided by the author (AHA-HH), 2026-11-11.
  Recorded in design §12, decision 31.

## Tasks

| Task | Brief | Delivers | Depends on |
| --- | --- | --- | --- |
| T1 | [T1-gn-core.md](T1-gn-core.md) | The generic core: parameter vectors, VJP, finite-difference JVP, adjoint check, CG with Lanczos estimates, for any Burn `Module` | – (G8, amended) |
| T2 | [T2-gauss-newton.md](T2-gauss-newton.md) | `GaussNewton`: Levenberg–Marquardt damping, ρ, warm start, Hutchinson-diagonal preconditioner, per-step diagnostics | T1 |
| T3 | [T3-trainer.md](T3-trainer.md) | Trainer integration: optimiser choice in `TrainingConfig`, the FNO residual adapter, `gn_steps.csv`, forward-equivalent passes in `costs.json`, GN state in checkpoints, `OPTIMIZER` in `train_poisson`; the CONVENTIONS §11 note | T2 |
| T4 | [T4-validation-gate.md](T4-validation-gate.md) | The smallest viable experiment: full-batch GN at N = 100 against Phase 7's Adam run at equal wall-clock; go / no-go for T5 | T3; Phase 7's runs; the VM |
| T5 | [T5-matrix.md](T5-matrix.md) | The N = 1000 matrix (Adam reruns and four GN arms, 3 seeds) and `docs/poisson2d/phase10/results.md` | T3 for its code; T4 (go) for its runs |
| T6 | [T6-conditioning.md](T6-conditioning.md) | PDE against neural-operator conditioning: curvature spectra across s, the single-sample tangent kernel in the sine basis, in `docs/poisson2d/phase10/conditioning.md` | T3 for its code; T4 (go) for its runs, beside or after T5 |

**Runs:** [runs.md](runs.md), the run sheet: every VM run of T3–T6 in order, with its
commands and decision points.

Task numbers are execution order (design §10); T6 needs only T4 and may run beside T5.
T5's brief is a draft for its runs, refined when T4's go is recorded; T6's was refined
2026-11-11. One task per branch and PR:
`phase10/T<k>-<name>`. Each task is worked from its brief alone, plus its "Read first".

## Experiment matrix (T5)

Stage 3, s = 64, p = 0, N = 1000, seeds 42, 43, 44, one A100, equal wall-clock (G4).

| Arm | Optimiser | Batch | Preconditioner |
| --- | --- | --- | --- |
| Adam | Adam, baseline settings (G6) | 20 | – |
| GN20 | Gauss–Newton, residual (A) | 20 | none |
| GN20-D | as GN20 | 20 | Hutchinson diagonal |
| GN200 | as GN20 | 200 | the better of GN20, GN20-D at T4 |
| Hybrid | Adam for half the budget, then GN | 20 | as GN200 |

15 runs. Follow-up arms, not in the minimal matrix: residual (C), the fine-grid residual
(via T_uc and Clenshaw–Curtis weights), the exact FNO JVP, L-BFGS, a PDE-induced output
norm (§13.17). Martens' backtracking and ρ rule (against T2's cheaper rule) is an
optional item in `docs/poisson2d/optional/README.md` ("Other optional items"), to try if T4 shows λ
oscillating or GN leaning on backtracking.

## Diagnostics

Per GN step, in `gn_steps.csv`: Φ before and after, the baseline loss L, ‖g‖, ‖p‖,
predicted and actual reduction, ρ, λ, accepted, CG iterations and relative residual,
the Lanczos Ritz values θ_max and θ_min of the (preconditioned) damped curvature,
the adjoint error (every k steps), passes and wall time. Per epoch, as now:
`metrics.csv` (train L², test L², epoch time). Per run: `errors.json` (`predict_poisson`)
and `costs.json` (training time, steps, forward-equivalent passes, peak memory).

## Risks

- Finite-difference JVP noise, worst in f32: CG loses conjugacy and ρ becomes noisy.
  Checked by the adjoint error; the fallbacks are f64 on cuda (G7) and the exact FNO JVP
  (G2).
- Cost: a GN step with k CG iterations is about 3 + 5k forward-equivalents, so about 18×
  an Adam step at k = 10. At batch 20 the A100 is overhead-bound (31 ms per batch, flat
  in s), so a pass on a batch of 200 should cost far less than ten passes on 20: GN200
  may be the efficient arm.
- G is rank-deficient on a mini-batch (rank ≤ 81,920 of 2.37M at B = 20) and
  ill-conditioned (low POD rank of u, Phase 6): damping must hold.
- ReLU makes J defined only almost everywhere; finite differences straddle kinks.
- Generalisation: Adam already ends with train L² about 3× below test. A faster optimiser
  may lower train L² without lowering error 4; both are reported.

## Module layout

- `optim/gauss_newton/` (new, T1–T2): the generic core. Imports nothing from `models/`
  or `training/`.
- `training/` (T3): `TrainingConfig.optimizer`, the FNO adapter, `train_epoch_gn`, the
  branch in `training_loop`. The Adam path is not edited. T5: `TrainingConfig.adam_epochs`
  (the Hybrid arm's Adam epochs before Gauss–Newton).
- `runs/` (T3): `gn_steps.csv`, the checkpoint's GN section, the new `costs.json` fields.
- `optim/gauss_newton/` (T6): `lanczos`, every Ritz value with full reorthogonalisation.
  `pde/poisson/collocation` (T6, `datagen`): `interior_d2_eigenvalues`.
- `examples/predict/poisson_conditioning.rs` (T6): the conditioning measurements.
- `tracking/` (T3): `gn_steps.csv` among the exported artifacts.
- `examples/train/poisson.rs` (T3), `examples/predict/poisson_matrix.rs` (T5): optimiser
  selection, grouping by optimiser.
- `docs/CONVENTIONS.md` §11 (T3): the GN residual, noted in the PR (no version bump; §11
  does not bump it).
- No change to the model, the data, the transfers, the metrics or the Adam settings.

## Results

Filled in as tasks merge. Phase 10 is done when every row has a value.

| Item | Value | Source |
| --- | --- | --- |
| Generic core, adjoint and explicit-Jacobian tests | `optim/gauss_newton/` (`ParamVec`, `params` / `with_params`, `LeastSquares`, `vjp`, finite-difference `jvp`, `adjoint_error`, `cg` with Lanczos Ritz values), generic over `Module`. Tiny 2D FNO (P = 591, m = 32 outputs, deterministic weights), f32 on flex, against an explicit Jacobian built from VJPs: `vjp` 4.7e-7, finite-difference `jvp` 1.3e-4 (max-relative) at the default step ∛u = 4.9e-3; worst adjoint error over 4 (u, v) pairs 4.6e-6; a sign-flipped J v gives about 2. Step sweep on that model: the error has the expected V shape, minimum at h ≈ 5e-3 to 1e-2. On a randomly initialised FNO the ReLU kinks dominated instead (error about 2e-2 at any h from 1e-2 to 1e-3), the G2 risk T4 measures. **f64 on flex not usable**: flex's f64 `rfft` uses f32 twiddles (1.9e-7 error on a length-16 rfft), so the f64 FNO is f32-accurate (vjp 8.0e-8, jvp 1.0e-3); kept as `#[ignore]`d repro and oracle in `tests/gauss_newton_f64.rs`; tests stay f32 | T1 |
| Gauss–Newton on synthetic problems (linear one-step, nonlinear fit, tiny FNO) | `GaussNewton` (`GaussNewtonConfig`, `GnStep`, `GaussNewtonState`, `Preconditioner::{None, HutchinsonDiagonal}`, `hutchinson_diagonal`), f32 on flex. **Linear least squares** through a `Linear(3, 2)`, 12 residuals: one step with λ = 1e-10 reaches the host-f64 normal-equations solution within 1e-4, ρ within 1e-3 of 1. **Nonlinear fit** (a 1–8–1 tanh MLP, 16 points of 0.8 sin 1.5x): from Φ = 7.5 to Φ ≤ 1e-3 in **13 GN steps** (551 forward and 188 backward passes) against **12,458 steps** of gradient descent at its best tested rate η = 0.01 (12,458 forward and backward each); η = 0.003 and 0.02 do not reach it in 30,000, η ≥ 0.1 diverges. ρ ∈ [0.89, 0.96] while Φ is between 1e-3 and 1e-5; below that, f32 rounding of Φ makes ρ noisy (0.5–0.9). **Tiny 2D FNO** (P = 591; T1's synthetic batch of the same shapes, not `tiny_dataset_2d`, which is crate-private to `training/`): Φ 7.0 → 5.9e-3 in 6 steps, all accepted, ρ 0.91–1.24; a forced rejection (sign-flipped J v, negative curvature) leaves the weights bit for bit and raises λ; so does a rejection with CG candidates that all raise Φ (a linear fit with J v under-estimated 4×). Hutchinson's diagonal (2,000 probes) within 5σ of diag(JᵀJ); a step after saving and loading the state equals one without | T2 |
| Adam path unchanged; GN smoke and resume on flex | `TrainingConfig.optimizer` (`None` = Adam; a pre-T3 `train_cfg.json` loads as Adam and trains as the same settings given in code), `FnoResidual` (½‖r‖² equals (1/2B) Σᵢ eᵢ² of `LpLoss` to 1e-5), `train_epoch_gn` (one step per loader batch: GN200 is `GN_BATCH_SIZE` = 200, no second batching), `gn_steps.csv`, `forward_passes` / `backward_passes` in `costs.json`, the GN state as `checkpoint.bin`'s optional fourth section. Every Phase 9 trainer, run-dir and checkpoint test passes unedited; the two `RunCosts` test literals gained the two new fields. Flex: a GN run of `train_poisson_smoke`'s fixture (3 epochs, 6 steps), killed in epoch 1 and resumed, matches the uninterrupted run bit for bit (`metrics.csv`, weights, `gn_steps.csv` but wall times). Metal: trainer and run-dir tests pass. **VM (one A100, 2026-11-11, commit `9dbeb91`):** **cuda f64 fails**: `f64_oracle_and_adjoint_on_cuda` gives NaN for the vjp, the finite-difference jvp and the adjoint error alike (P = 591), so the fault is in the f64 forward or backward on cuda, not the finite differences; f64 is no fallback for T4, and T4's optional f64 run needs no code. **Cause** (a throwaway probe on the VM's second GPU, 2026-11-11): basic f64 ops are correct, every f64 FFT is not (`rfft`, `cfft`, `irfft`: values such as 1e275 and 1e-69, and stale memory). cubek-fft at the pinned rev `333c406` launches every FFT kernel as `launch::<f32>` whatever the dtype, into buffers sized for f64, with f32 twiddles; Burn's `burn-cubecl` wrapper passes the dtype correctly. Fixed upstream in cubek `2f74bf1b` (#677, "launch the kernels at the input's element type"), in cubek 0.3.0 and later, so in Burn 0.22.0 and later; flex's f64 twiddle table is still f32 on Burn main. **On Burn 0.23.0-pre.2** (#119) the f64 FFTs are fixed and every f64 matmul panics in cubek-matmul's autotune instead ([upstream/cubek/f64-matmul-declined-on-cuda](../../../upstream/cubek/f64-matmul-declined-on-cuda/README.md)); flex's f64 FFT keeps f32 twiddles ([upstream/burn/flex-f64-fft-f32-twiddles](../../../upstream/burn/flex-f64-fft-f32-twiddles/README.md)). **Recorded as blocked upstream, and f64 is not needed** for this work (decision 31, amendment of 2026-11-24). **Short GN run** (stage 3, s = 64, N = 100, subset 0, seed 42; a 60 s budget for the brief's 2 epochs; `poisson_fno_n129_s64_p0_N100_sub0_gn_1794395782`): 60.5 s, 4 epochs, 20 steps, 2,148 forward and 716 backward passes, peak 2,941 MiB (`nvidia-smi`); error 4 1.046e-1, test L² 0.402 → 0.105; 20 / 20 steps accepted, ρ 0.33–1.36, λ 1 → 2.3e-3; CG converged in 4–16 iterations on the first steps, at its cap of 50 (`MaxIters`, relative residual 0.008–0.14) from the fourth step of epoch 2; adjoint error 1.3e-6 at step 0, 1.17e-3 at step 10. Error 3 matches `metrics.csv` (relative difference 4.3e-9, as `predict_poisson` reports it). **Adam check:** T5's seed-42 Adam rerun (`poisson_fno_n129_s64_p0_1794397601`) against Phase 7's `poisson_fno_n129_s64_p0_1793188648`: train and test L² equal to every printed digit at epochs 0–4, 99 and 499, and error 4 1.0952e-2 in both: the Adam path is unchanged, bit for bit on cuda | T3 |
| Validation gate (N = 100): adjoint error, ρ, error 4 against Adam, go / no-go | **Go, preconditioner `<P>` = hutchinson** (the author, 2026-11-11; gate thresholds as proposed). GN-D passes all three criteria, GN fails the adjoint median (1.33e-3); at equal time both lose to Adam (error 4 7.68e-2 and 8.35e-2 against 7.13e-2). Table and paragraph below ("T4: validation gate") | T4 |
| N = 1000 matrix: error 4 against wall-clock, updates and passes per arm | **Adam wins at G5's μ = 1e-4**: error 4 1.061e-2 against 2.16e-2 for the best GN arm (GN200-D), 3.65e-2 (GN20), 4.10e-2 (GN20-D), 3.73e-2 (Hybrid-D), all at about 900 s. A GN update costs about 150 forward and 50 backward passes (about 100 Adam updates), and passes run at the same rate for both, so wall-clock follows passes. **Follow-ups (#118): weight decay is the limiter.** At μ = 0, GN200-D at equal time gives 9.06e-3 [6.55e-3–1.200e-2], below Adam's mean but one seed of three above Adam's range; with 6.3× the time, CG cap 150 reaches 3.03e-3 (one seed). Adam at μ = 0 overfits (1.539e-2). H10a partly confirmed: fewer updates, 5–7× more costly per update than assumed, equal-time question open at μ = 0. [results.md](results.md) | T5 |
| Conditioning: curvature spectra across s, tangent kernel against 1/μ_kl, H10b | **H10b confirmed for convergence**: θ_max changes by at most 1.6 % from s = 32 to 128 (Adam, GN200-D and T4's GN-D runs) while the finite-difference Laplacian's κ grows 16.8×. The tangent kernel follows the sine modes (log correlation 0.91–0.95) but decays like (1/μ_kl)^p, p ≈ 2.1–3.4 (2.3–2.7 at the higher modes), of the order of u's POD spectrum, not the inverse Laplacian's. GN's minima are sharper (θ_max 119× Adam's). Chebyshev κ not measured (no `datagen` on the VM). [conditioning.md](conditioning.md) | T6 |

### T4: validation gate

VM, one A100, commit `9dbeb91`, 2026-11-11. Stage 3, s = 64, p = 0, N = 100 (subset 0),
seed 42; GN full batch (`GN_BATCH_SIZE` = 100), budget 132.5 s (the Adam comparator's
`training_time_s`). Updates are optimiser steps; passes are forward / backward passes of
the FNO on a batch (`costs.json`).

| Run | Error 4 | Final train L² | Updates | Passes (fwd / bwd) | Wall time | Peak memory |
| --- | --- | --- | --- | --- | --- | --- |
| Adam, Phase 7 (`poisson_fno_n129_s64_p0_N100_sub0_1793291675`) | 7.133e-2 | 6.73e-3 | 2,500 | 2,500 / 2,500 (one each per step; not recorded before T3) | 132.5 s | 938 MiB |
| GN, none (`..._N100_sub0_gn_1794396092`) | 8.345e-2 | 6.37e-2 | 26 | 3,111 / 1,031 | 135.8 s | 3,901 MiB |
| GN-D, Hutchinson (`..._N100_sub0_gn_1794396230`) | 7.677e-2 | 5.70e-2 | 25 | 3,020 / 1,020 | 133.9 s | 2,493 MiB |

| Run | Adjoint error (median; samples) | Accepted | ρ in (¼, 2) | CG iterations (median; range; at cap 50) | CG final rel. residual (median; max) | λ | Ritz θ_min; θ_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GN | **1.33e-3**; 2.4e-3, 1.3e-3, 1.4e-5 | 100 % | 81 % (0.01–1.42) | 46.5; 5–50; 13 of 26 | 1.6e-2; 0.11 | 1 → 3.9e-2 | 5.2e-2–1.0; 7.9–1.9e3 |
| GN-D | 8.3e-4; 2.4e-3, 7.1e-5, 8.3e-4 | 100 % | 88 % (0.03–1.42) | 43; 8–50; 11 of 25 | 9.6e-3; 0.15 | 1 → 1.7e-2 | 0.15–0.82; 5.4–4.6e3 |

GN is healthy as an optimiser: every step accepted, ρ near 1, Φ 0.505 → 0.0068 in 25
full-batch steps (GN-D), and per update far ahead of Adam (test L² 7.7e-2 after 25 updates;
Adam needs 695 for that). It loses at equal time because an update costs about 120
forward and 40 backward passes (about 3.2 forward and 1.1 backward per CG iteration), and
CG needs more iterations as λ falls (the curvature's Ritz spread reaches about 3e4): Adam
reaches GN-D's final test L² at epoch 138, 42.5 s. The adjoint errors sit at the 1e-3
threshold on three samples per run; ρ ≈ 1 shows the finite-difference JVP does not yet
spoil the model of Φ. `gn_steps.csv` keeps only CG's final residual, so "falls in its first
iterations" is read as: it fell at least 6.8× on every step, never stagnating. Not
investigated: both GN runs have adjoint error 2.4e-3 at step 0, against 1.3e-6 for T3's
batch-20 run from the same weights; and GN-D's peak memory is below GN's. Error 4 of the
Adam comparator is its Phase 7 `errors.json` (evaluated on flex); the GN runs' on cuda.
Plots (test L² and Φ against updates and seconds): `runs/phase10_t4/t4_curves.png`, from
`runs/phase10_t4/plots.gp` (git-ignored, beside the runs).

## Exit checklist

- [x] T1–T6 merged, or a task dropped with the reason recorded in design §12: all six merged (#98, #102, #104–#107); T3's cuda f64 check recorded as blocked upstream (decision 31, amendment of 2026-11-24).
- [x] G1–G8 chosen by the author and recorded in design §12 (2026-11-09).
- [x] The Adam reruns' error 4 within Phase 7's spread ±2 %, 1.020e-2–1.117e-2 (G6): 1.0952e-2, 1.0409e-2, 1.0464e-2 (seeds 42–44, 2026-11-11).
- [x] H10a and H10b answered: H10a in `docs/poisson2d/phase10/results.md`, H10b in `docs/poisson2d/phase10/conditioning.md` (2026-11-24).
- [x] Every row of Results filled in.
- [x] The "Current phase" bullet in `CLAUDE.md` updated for what comes next.

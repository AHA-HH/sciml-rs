# CO1.1 The reference recipe

Started 2026-11-25 on `optimisers/co1-1`. Roadmap: "CO1.1 The reference recipe.
`OptimizerConfig::AdamW` (Burn's `AdamWConfig`, ε 1e-8), `TrainingConfig.lr_schedule`
(`None` = today's cosine; a step variant halving every k epochs), an `FNOConfig` GELU option
and a squared relative-L² option for the Adam path (both off by default), `OPTIMIZER=adamw`,
`WEIGHT_DECAY` and the subsample rate as variables of `train_burgers` with n_train = 1,000
(1,024 today), and `costs.json` for Burgers runs; s = 128 and the 1,000/100 split checked
against Lu et al. 2022 and PETScML's scripts; Burn's `StepLrSchedulerConfig` reused for the
step schedule." Proposal: §10; §1.4, §2, §3.1 (O1), §6, §9 (O1); decisions D10, D11.

## Goal
Make PETScML's reference recipe (A1, and A0 at weight decay 0) runnable with
`train_burgers`, every new option off by default, so that CO1.2 and CO1.3 only add arms
and runs.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)
"the O1 tests of proposal §9 pass (the parameter count 549,569, AdamW at weight decay 0 equal
to Adam at 0 under the same schedule, the step schedule's halvings); A1 on seed 42 on flex
or metal against 0.0014; constraint 1 shown (`train_burgers` and `train_darcy` with nothing
set unchanged bit for bit)."

Proposal §9 (O1): "the Burgers model's parameter count is 549,569; … the AdamW arm with
weight decay 0 equals Adam with weight decay 0 under the same schedule to f32 round-off; the
GELU and squared-loss options off reproduce today's outputs bit for bit; the step schedule
halves at the stated epochs". (`FnoResidual` on a rank-3 batch and the Gauss–Newton smoke
run are CO1.2's.) §3.1, D10: A1's mean within a factor 2 of 0.0014 reproduces PETScML;
CO1.1 runs seed 42 only, so it is an early reading, not the verdict (CO1.3's five seeds).

## Decision taken at planning (the author, 2026-11-25)
Constraint 1 keeps `train_burgers` with nothing set as today (n_train 1,024, rate 32, s =
256, ReLU, unsquared loss, Adam with cosine). The O1 recipe is switched on by **one preset,
`RECIPE=petscml`**: n_train 1,000, subsample rate 64 (s = 128), GELU, the squared loss.
`OPTIMIZER=adamw` gives AdamW with the step schedule halving every 100 epochs (D11);
`OPTIMIZER=adam` (or unset) keeps our coupled Adam with the cosine schedule (D11's extra
cells). `WEIGHT_DECAY` and `SUBSAMPLE` override. The roadmap's "n_train = 1,000" is read as
the recipe's value, not the new default.

## The source check (2026-11-25, plain HTTP)
PETScML arXiv v1 (https://arxiv.org/html/2403.12188, §4.1.1); Lu et al. 2022 (arXiv
2111.05512, §5.1, Table S1) and their code (`lu-group/deeponet-fno` @ `8096dda`,
`src/burgers/fourier_1d.py`); Li et al.'s `fourier_1d.py` (a mirror, the original is gone
from the official repository). PETScML's repository (gitlab.com/stefanozampini/petscml, HEAD
`127a5fdf`, every branch) has **no Burgers or FNO script**; the PASC '24 version and the
Zenodo data tarball could not be read.

| | PETScML text | Lu et al. code | Li et al. code | O1 |
| --- | --- | --- | --- | --- |
| s | 128 of 8192, "following [44]" | rate `2**13 // 128` = 64 | rate 8 (s = 1024) | rate 64, s = 128 |
| train | 1,000 | first 1,000 | first 1,000 | first 1,000 (our loader) |
| test | 100 | **last 200** | last 100 | last 100 (our loader takes the last `n_test`) |
| epochs | not stated (Fig. 1 ends near 500) | 500 | 500 | 500 |
| schedule | lr 1e-3 halved every 100 epochs | step 100, γ 0.5 | step 50, γ 0.5 | step 100, γ 0.5 |
| optimiser | "ADAMW", wd 1e-4, batch 20 | Adam, coupled L2 1e-4 | Adam, coupled L2 1e-4 | AdamW (D11), coupled Adam as extra cells |
| ε | not stated | 1e-8 | 1e-8 | 1e-8 |
| activation | GELU (DM-Haiku, JAX) | ReLU | GELU (PyTorch, erf form) | GELU (see the open question) |
| loss | mean squared relative L² (eq. 6) | MSE | summed relative L² | squared option |
| precision | f64 "unless otherwise stated" | f32 | f32 | f32 (D10) |

So s = 128 and the 1,000/100 split are confirmed against PETScML's text, Lu et al. for s
and Li et al. for the 100 test samples; PETScML mixes the two codes. Epochs and ε are not
in PETScML's text; 500 and 1e-8 follow both codes.

## Read first
- `training/config.rs` (`OptimizerConfig`, `TrainingConfig`), `training/trainer.rs`
  (`build_components`, `train_epoch`, `run_loop`), `training/step.rs` (`forward_loss`),
  `training/data_parallel.rs` (its own backward on the summed L²).
- `models/fno/mod.rs` (`forward`'s two `relu`s), the `FNOConfig` struct literals.
- Burn: `burn-optim-0.23.0-pre.2/src/optim/adamw.rs` (decoupled decay `θ ← θ(1 − lr·wd)`),
  `lr_scheduler/step.rs` (`StepLrScheduler`: `lr₀ γ^⌊i/k⌋`, stepped per batch),
  `burn-tensor` `activation::gelu` (erf form) and `gelu_approximate` (tanh form).
- `examples/train/burgers.rs`, `examples/train/poisson.rs` (`env_var`, `costs.json`).

## Changes
- `training/config.rs`: `OptimizerConfig::AdamW` (lr, weight decay from `TrainingConfig`,
  ε 1e-8); `LrSchedule::Step { every_epochs }` and `TrainingConfig.lr_schedule:
  Option<LrSchedule>` (`None` = cosine); `TrainingConfig.loss: Option<TrainLoss>` with
  `RelL2` (today's, the `None` default) and `SquaredRelL2` (PETScML eq. 6: the batch mean
  of eᵢ²). Options, so saved configs load unchanged.
- `training/trainer.rs`: in `build_components`, AdamW when asked (else the Adam
  construction untouched); the step schedule (`StepLrSchedulerConfig`, γ = 0.5, step size
  k × steps per epoch) when asked (else the cosine untouched); `train_epoch` takes the loss
  choice and back-propagates the squared mean when asked; the metrics stay the unsquared
  relative L². The squared loss with replicas panics (data-parallel is out of scope for O1).
- `models/fno/mod.rs`: `FNOConfig.activation: Option<Activation>` (`Relu`, the `None`
  default; `Gelu`), used at both places `forward` applies ReLU today. Every `FNOConfig { … }`
  literal gains `activation: None` (a compile-only edit in tests and examples).
- `examples/train/burgers.rs`: `RECIPE`, `OPTIMIZER`, `WEIGHT_DECAY`, `SUBSAMPLE`;
  `costs.json` (load and training time, steps and passes as `train_poisson`'s Adam rule).
- Conventions: none changed.

## Tests that prove it
- Parameter count: the Burgers model (`modes [16]`, width 64, 4 layers) has 549,569
  parameters (`num_params`), against PETScML's stated count; GELU adds none.
- AdamW at weight decay 0 against Adam at weight decay 0, same schedule and seed, a few
  steps on the tiny 1D fixture: equal weights to f32 round-off (and with weight decay > 0
  they differ, so the test sees the variant).
- Step schedule: the learning rates of a run with `Step { every_epochs: 2 }` on a fixture of
  known steps per epoch halve exactly at the epoch boundaries (bit for bit `lr₀ · 0.5^j`).
- Squared loss: on a fixed batch, the backward objective equals the mean of the squared
  per-sample relative L² (closed form against `LpLoss` `Reduction::None`), and training with
  it changes the weights while `train_l2` stays the unsquared mean.
- GELU: `forward` with `Gelu` equals a hand-written forward using `gelu` on a small model;
  with `None` the existing reference tests (`models/fno/tests/reference.rs`) pass unedited
  but for the field.
- Old configs: a `train_cfg.json` and a `model_cfg.json` without the new fields load as
  cosine, `RelL2` and ReLU.
- Constraint 1: the existing suite (unedited but for `activation: None` in literals);
  `spikes/linalg`'s `fingerprint` (a seeded forward, Adam and hybrid Gauss–Newton runs) equal
  on `origin/main` and this branch; `train_burgers` and `train_darcy` with nothing set: their
  first epochs' `metrics.csv` rows bit for bit equal before and after (both run with a
  temporary epoch cap, see Checks).

## Checks
- `/verify` (fmt, the clippy lines, `cargo test`, `cargo doc`, datagen clippy and tests;
  `--features metal` tests and `gpu_parity`, since the model's forward is touched).
- Fingerprint: `cargo run -p linalg-spike --release --bin fingerprint` on `origin/main` and
  this branch, equal.
- `train_burgers` and `train_darcy` with nothing set, on `origin/main` and on this branch,
  each stopped after its first epochs (a temporary local edit of the epoch literal on both
  sides, never committed): equal `metrics.csv` rows.
- A Mac smoke run of `RECIPE=petscml OPTIMIZER=adamw` (a few epochs) to check the wiring and
  measure the epoch time.

## VM runs
None. The roadmap asks for A1 on seed 42 on flex or metal (the Mac): 500 epochs of
`RECIPE=petscml OPTIMIZER=adamw WEIGHT_DECAY=1e-4 cargo run --release --features metal
--example train_burgers`. Its duration is estimated from the smoke run and the author's go
is asked before it starts.

## Constraints and risks
- Constraint 1 (§1.4): every new field is an `Option` whose `None` takes today's code path
  unchanged (the Adam construction, the cosine, the unsquared backward, ReLU). The 2D and 3D
  objectives are untouched.
- `training/` is serial (the author's order): CM0.6 waits for this PR's merge.
- GELU form: Burn's `gelu` is the erf form (PyTorch's default); JAX's `nn.gelu` defaults to
  the tanh approximation. The form PETScML used is checked against their scripts.
- `costs.json` for Burgers is a new file; no existing output changes.

## Pending runs
None. A1 on seed 42 ran on the Mac (metal) at the author's go, 2026-11-25:
`docs/optimisers/results/CO1.1.md`.

## Open questions
1. GELU's form. PETScML's network is DM-Haiku, whose `jax.nn.gelu` defaults to the tanh
   approximation; Li et al.'s PyTorch `F.gelu` is the erf form. Their script is not public.
   CO1.1 implements the erf form (Burn's `gelu`); the two differ by under 1e-3 in value.
   For sign-off: keep erf, or switch to the tanh form (Burn's `gelu_approximate`).
   Answered 2026-11-25 by the author: keep erf.
2. PETScML's "AdamW" is attributed to Li et al., whose code is Adam with a coupled L2 term.
   D11 already runs both (AdamW cells A0, A1 and our coupled Adam cells), so nothing changes;
   recorded for CO1.3's discussion of the reference.

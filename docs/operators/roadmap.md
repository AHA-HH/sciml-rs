# Neural operator library roadmap

The tasks of `docs/design/neural-operator-library.md` (draft 0.1), Section 10, in the order
they are taken. The proposal is the full record: where a task's *Done when* and the
proposal differ, the proposal wins. Progress is logged in `docs/operators/progress.md`;
each task's plan is `docs/operators/tasks/<ID>.md`, written when it starts; each
milestone's gate report is `docs/operators/gates/<milestone>.md`; the feature matrix is
`docs/operators/feature-matrix.md`.

Status: **M0 runnable.** Draft 0.1 written 2026-11-11; D0 signed off 2026-11-24 (draft 0.1
as the plan of record); CM0.1 and CM0.2 done 2026-11-24; CM0.3 done 2026-11-24 (D4 signed
off): once it is merged the optimisers tasks that change `training/` may start. CM0.4 done
2026-11-25 (the registry). D3 signed off 2026-11-25, and the CM0.3 test-import question
answered (moving imports is not editing a test). CM0.5 done 2026-11-25 (the parity harness);
CM1.1 done 2026-11-25 (legacy spectral convolution parity, 1D to 3D). CM1.6 split 2026-11-25
into CM1.6a–c; CM1.6a done 2026-11-25 (finite differences, H1, Hdiv, quantile, MSE). CM1.3
split; CM1.3a done 2026-11-25 (the block parts). D5 signed off 2026-11-25. CM1.6b done
2026-11-25 (the aggregators). CM1.2 done 2026-11-25 (the centred layer, separable, modes
below the stored maximum). CM1.3b done 2026-11-25 (the FNO's block options; FNO outputs
unchanged bit for bit). CM2.5 done 2026-11-26 (the equation losses, PINO; independent of
D6). CM1.4 done 2026-11-26 (embeddings, lifting and projection ratios; FNO outputs
unchanged bit for bit). CM1.6c done 2026-11-26 (Fourier and point-cloud derivatives). CM1.5
split 2026-11-26 into CM1.5a–d; CM1.5a done 2026-11-26 (2.0.0's whole `FNO` within D3; FNO
outputs unchanged bit for bit). CM3.6 split 2026-11-27 into CM3.6a–b; CM3.6a done 2026-11-27
(the Tensor GaLore projector for real tensors, not wired into training). Next CM0.6, CM1.5b,
CM1.5c, CM1.5d, CM3.6b.

## How to read this file

- `- [ ]` is open, `- [x]` done. The tick is part of the task's own pull request.
- A task is **runnable** when it is open, has no branch or open PR, and everything in its
  *Needs* is done: tasks ticked on `main`, decisions signed off below. Every task also
  needs **D0**.
- `/next-task` (`.claude/skills/next-task`) takes the first runnable task in the order of
  this file unless the author names one; `/milestone-gate` closes a milestone by writing
  its gate report and changing its `**Gate:**` line to `**Gate (closed <YYYY-MM-DD>):**`.
- A task too big for one reviewable PR is split here into lettered sub-tasks (CM1.3a,
  CM1.3b) before it starts.
- Task IDs are `CM<milestone>.<n>`; milestones are M0–M4. In prose elsewhere, name the
  objective with the number ("library CM1.2").
- *Machines* names what a task's runs need besides the Mac; *Sign-off* marks a task whose
  PR ends with questions for the author.
- **Constraint 1** applies to every task: every existing FNO output stays bit for bit
  identical (proposal §1.4).
- **Every operator task** follows the recipe of proposal §6.4 and ends by setting its
  feature-matrix row.

## Decisions

The proposal's Section 12. Recorded when the author takes them:
`signed off <YYYY-MM-DD>: <the answer as given>`.

| ID | Decision | Proposal's recommendation | Prepared by | Needed by | Status |
| --- | --- | --- | --- | --- | --- |
| D0 | Draft 0.1 of the proposal as the plan of record | after review | – | every task | signed off 2026-11-24: draft 0.1 as the plan of record |
| D1 | Reference version | `neuraloperator` 2.0.0 | – | CM0.1 | signed off 2026-11-11: "its neuraloperator and the major release 2.0.0" |
| D2 | Where the Python side lives | `tools/parity/`, outside CI, lock file; fixtures in `tests/fixtures/parity/` | – | CM0.2 | signed off 2026-11-24: as recommended (`tools/parity/`, outside CI, lock file; fixtures in `tests/fixtures/parity/`) |
| D3 | Output parity tolerance | rel. max error ≤ 1e-5 (f32), ≤ 1e-12 (f64), CPU | CM0.5 | CM1.1 | signed off 2026-11-25: as recommended (relative max error ≤ 1e-5 in f32, ≤ 1e-12 in f64, on the CPU); amended 2026-11-25 by the author (CM0.5, #148): the error is max |a − e| / max |e| over the tensor; f32 parity on the CPU suffices for M1 while Flex's f64 `rfft` uses f32 twiddles (`upstream/burn/flex-f64-fft-f32-twiddles`), each f64 test `#[ignore]`d with its measured error recorded |
| D4 | Shape of the operator trait | trait with associated config and input type, generic | CM0.3 | CM0.3 | signed off 2026-11-24: option A as recommended (a generic trait with associated config and input types, static dispatch, no trait objects; `docs/operators/tasks/CM0.3.md`) |
| D5 | Spectral weight layout | corners internally; convert at fixture load | – | CM1.2 | signed off 2026-11-25: A, as recommended (corners internally, CONVENTIONS §5 unchanged; 2.0.0's centred weights converted at fixture load, FFT norm and mode counts included). A centred layout may come later only as an opt-in option, off by default, in its own task, if a need (PyTorch checkpoint interchange) or a measured gain appears |
| D6 | Order of M2 and M3 | at the M1 gate; default M2 first | M1 gate | CM2.1, CM3.1 | open |
| D7 | Training-parity hardware | one A100 on the VM | – | CM1.8 | signed off 2026-11-27: as recommended, one A100 on the VM |
| D8 | Fixture size | tiny models, under about 1 MB per fixture | CM0.2 | CM0.2 | signed off 2026-11-24: as recommended (tiny models, P ≲ 10⁴, under about 1 MB per fixture) |

## M0: Foundations

**Gate:** the stack is model-agnostic with the FNO unchanged; a fixture round trip works;
the feature matrix is verified.

- [x] **CM0.1 Feature matrix at 2.0.0.** Fill `feature-matrix.md` from the tag's source
  (every public model, layer, loss, training feature and dataset, with its options and
  defaults). *Done when:* every row has a status and a source path, checked by an agent
  independent of the writer. *Needs:* D1. Done 2026-11-24: plan `docs/operators/tasks/CM0.1.md`.
- [x] **CM0.2 Pinned Python environment and fixture generator.** `tools/parity/`:
  `neuraloperator` 2.0.0 with torch, tensorly, tltorch and torch_harmonics pinned in a
  lock file; a generator writing weights, input and output of a named configuration to
  `.npz`. *Done when:* a 1D and a 2D FNO fixture written, sizes within D8. *Needs:* D2,
  D8. Plan `docs/operators/tasks/CM0.2.md`; environment and generator `tools/parity/`
  (README); fixtures `tests/fixtures/parity/fno{1d,2d}_{f32,f64}`. Done 2026-11-24.
- [x] **CM0.3 The operator trait and a generic stack.** Proposal §6.1: trainer, evaluate,
  `predict_host`, checkpoints, the epoch observer, the run directory's weights and the
  Gauss–Newton residual generic over the trait; `FNO<R>` implements it. *Done when:* every
  existing test passes unedited, and a fixed FNO forward and one Phase 5 epoch are bit for
  bit unchanged. *Needs:* D4. *Sign-off:* D4. Done 2026-11-24: plan
  `docs/operators/tasks/CM0.3.md`; `models/operator.rs`; bit identity
  `tests/fno_bit_identity.rs`, compared against `main` by `/verify`.
- [x] **CM0.4 The registry.** Proposal §6.2: `kind` in `model_cfg.json`, absent meaning
  the FNO. *Done when:* every existing run directory loads unchanged; a model builds by
  name. *Needs:* CM0.3. Also the MLflow export's
  model config (`tracking/builder` reads `model_cfg.json` as `FNOConfig`; CM0.3 left it).
  Done 2026-11-25: plan `docs/operators/tasks/CM0.4.md`; `models/registry.rs`.
- [x] **CM0.5 The parity harness.** A test helper that loads a fixture by parameter-name
  map into a module and compares outputs (and gradients) within D3. *Done when:* the
  CM0.2 1D fixture passes through a hand-written map. *Needs:* CM0.2, D3.
  *Sign-off:* D3. Done 2026-11-25: plan `docs/operators/tasks/CM0.5.md`;
  `tests/parity_common/` (harness, and `fno1d`: 2.0.0's 1D FNO mirrored in test code),
  `tests/parity_fno1d.rs` (f32, 3.5e-7), `tests/parity_fno1d_f64.rs` (`#[ignore]`d: Flex's
  f64 `rfft`, 1.2e-8). No fixture carries gradients yet; `assert_parity` takes any tensor.
  Follow-ups: `NpzFileReader::names()` so the harness opens a fixture once (CM0.6); read
  the sidecar `.json` (dtype, `complex_params`) once a JSON dev-dependency is agreed, and
  build the grid as 2.0.0 does (f32, then cast) before a non-power-of-two f64 fixture
  (CM1.1); a shared `flex_f64` helper with `tests/gauss_newton_f64.rs` and a generated map
  per block (CM1.5).
- [ ] **CM0.6 Tidy after the generic stack.** The CM0.3 code-quality review's findings
  (`docs/operators/progress.md`, CM0.3): one name per concept for the FNO wrappers
  (`train_fno`, `build_training_components`, `load_fno`, `checkpoint`, `Checkpoint`) and
  their generic twins (keep, deprecate or retire); `FnoResidual` renamed for any operator;
  the `#[cfg(test)]` imports that keep the old test modules' `use super::*` compiling
  moved into those modules; `training/trainer.rs`'s test module split out. *Done when:*
  the same bit-identity check as CM0.3 passes against `main`. *Needs:* CM0.3, and the
  author's answer on whether moving a test module's imports edits a test (CM0.3 PR):
  answered 2026-11-25, it does not; moving the `#[cfg(test)]` imports into their test
  modules is allowed.

## M1: The FNO family

**Gate:** 2.0.0's `FNO` with its defaults and every option passes output parity; H1 and
the aggregators exist; one training-parity run on the 2D Poisson baseline.

- [x] **CM1.1 Legacy spectral convolution parity.** 2.0.0's corner-layout layer against
  `SpectralConv`, 1D to 3D. *Needs:* CM0.5. Done 2026-11-25: plan
  `docs/operators/tasks/CM1.1.md`; fixtures `legacy_spectral_conv{1,2,3}d_{f32,f64}`;
  `tests/parity_legacy_spectral_conv.rs` (f32, ≤ 4.7e-7), `_f64.rs` (`#[ignore]`d: 2.0.0's
  legacy spectrum is `cfloat`, so its f64 reference is f32-accurate, ≤ 2.9e-7). CM0.5's
  sidecar and grid follow-ups did not arise (no complex parameters in the sidecar, no grid
  in a bare layer): they pass to CM1.2 and CM1.5.
- [x] **CM1.2 Centred layout, separable, `max_n_modes`.** The weight converter of proposal
  §6.3; separable weights; mode truncation below the stored maximum. *Needs:* CM1.1, D5.
  Done 2026-11-25: plan `docs/operators/tasks/CM1.2.md`; `SpectralConv::new_separable` and
  `with_modes`; the converter `centred_to_corners` at fixture load
  (`tests/parity_common/centred_conv.rs`); fixtures `spectral_conv*_{f32,f64}`;
  `tests/parity_spectral_conv.rs` (f32, ≤ 4.0e-7), `_f64.rs` (`#[ignore]`d, ≤ 2.8e-7). Not
  representable in corners: a grid of odd extent on a non-last axis (2.0.0 shifts its output
  spectrum by one bin there), and the cases the plan lists. For CM1.5: wire `separable` and
  `max_n_modes` into `FNOConfig` (off by default), and run FNO parity on even grids only.
  For CM1.8: an odd centred mode count's zero row is a trainable weight, so training parity
  at odd counts needs it masked (or even counts only). Code-quality follow-ups for CM1.5: one
  shared `#[path]` file for the biased test layer of
  `legacy_conv.rs` and `centred_conv.rs`; converted arrays passed to `load_into` so
  `Fixture`'s fields are private again; `layers/spectral_convolution.rs` (now over 1,000
  lines with its tests) split as `optim/soap/` is.
- **CM1.3 Block parts.** Channel MLP; linear, identity and soft-gating skips;
  instance, group and AdaIN norms; pre-activation; the activation choice. *Needs:* CM1.1.
  Split 2026-11-25 (parts in `layers/`, then the FNO's block options):
  - [x] **CM1.3a The parts.** `ChannelMlp` (2.0.0's `ChannelMLP`), the skips of
    `skip_connection` (identity, linear, soft-gating) and `AdaIn` in `layers/`; the
    activation choice as Burn's `ActivationConfig`; instance and group norm as Burn's
    `InstanceNorm` and `GroupNorm`. *Done when:* each part (Burn's norms and activations
    included) matches a 2.0.0 fixture of its own within D3, in f32 and f64, across 1D to 3D;
    nothing under `models/` or `training/` changes. *Needs:* CM1.1. Done 2026-11-25: plan
    `docs/operators/tasks/CM1.3a.md`; `tests/parity_block_parts{,_f64}.rs` (f32 ≤ 1.9e-7,
    f64 ≤ 7.1e-16).
  - [x] **CM1.3b The FNO's block options.** `FNOConfig` gains the activation, the FNO skip,
    the channel MLP (expansion, skip), the norm and pre-activation, each absent meaning
    today's block. *Done when:* the CM0.3 bit-identity check passes against `main`; a block
    built from the options matches 2.0.0's `FNOBlocks` (with the legacy convolution) within
    D3, post- and pre-activation, for each norm; the FNO's own pointwise matmul
    (`FNO::pointwise`, `conv1x1_cf`) replaced by `layers::pointwise::conv1x1` (CM1.3a review).
    *Needs:* CM1.3a. Done 2026-11-25: plan `docs/operators/tasks/CM1.3b.md`;
    `FNO::fourier_layers`, `layers::norm::{NormKind, Norm}`; `tests/parity_fno_blocks.rs`
    (nine `FNOBlocks` fixtures, f32 only as the legacy layer computes in `cfloat`, ≤ 2.3e-6);
    the CM0.3 dumps identical to `main`.
- [x] **CM1.4 Lifting, projection and embeddings.** Channel-MLP lifting and projection with
  their ratios; `GridEmbeddingND`, `SinusoidalEmbedding`, `RotaryEmbedding2D`. *Needs:*
  CM1.3a (the channel MLP). Done 2026-11-26: plan `docs/operators/tasks/CM1.4.md`;
  `layers::embedding`; `FNOConfig::{lifting_channel_ratio, projection_channel_ratio,
  positional_embedding}`; `tests/parity_embeddings.rs` (eight fixtures, `FNO` with
  `n_layers=0` for the lifting and projection, f32 ≤ 7.8e-7), `_f64.rs` (seven `#[ignore]`d:
  2.0.0's grid and frequencies are f32); the CM0.3 dumps identical to `main`.
- **CM1.5 Full FNO parity.** 2.0.0's `FNO` at its defaults and option sweeps, on
  Darcy- and Burgers-shaped inputs; resampling (`resolution_scaling_factor`) and domain
  padding. *Needs:* CM1.2, CM1.3b, CM1.4. Also from CM1.3b: the spectral convolution's bias
  inside the block (2.0.0 adds it before the norm; CM1.3b's fixtures turn it off), and the
  block options' `FNOConfig` struct literals in the examples and tests (five `None` lines
  each) replaced by `FNOConfig::new(..).with_*` where that keeps them readable. From CM1.4's
  code-quality review: one coordinate-channel builder for `FNO::grid_cl` and
  `GridEmbeddingNd::grid` (each keeps its own copy of the single-input `cat` workaround;
  the dumps and `gpu_parity` decide), and the parity helpers shared (`Fixture::open_as`,
  a D3 check that prints, the two permutes) across `parity_common/{block_parts,fno_blocks,
  embeddings}.rs`; from its plan review: the f64 fixtures of 2.0.0's grid and frequencies
  generated under `torch.set_default_dtype(torch.float64)`, so their seven f64 tests run
  (CM1.4's are `#[ignore]`d, f32-accurate), if CM1.5's own f64 FNO fixtures need it.
  Split 2026-11-26 (over 1,000 lines across the FNO, two new layers and the parity tests):
  - [x] **CM1.5a Whole-FNO parity, defaults and block sweeps.** 2.0.0's `FNO` end to end
    (grid, lifting, centred convolutions with their bias, blocks, projection) through one
    parameter map; `FNOConfig` gains the convolution's bias, `separable`, `max_n_modes` and
    the tanh stabiliser, each off by default; with it CM1.4's shared parity helpers (F4).
    *Done when:* `fno1d` and `fno2d` (2.0.0's defaults, Burgers- and Darcy-shaped) and the
    sweep fixtures (norms, pre-activation, skips, ratios, activations, no grid, separable,
    `max_n_modes`, an odd mode count, 3D, the stabiliser) pass through `FNO` within D3 in
    f32; the f64 tests `#[ignore]`d with their errors recorded; the FNO's bit-identity
    dumps equal `main`'s. *Needs:* CM1.2, CM1.3b, CM1.4. Done 2026-11-26: plan
    `docs/operators/tasks/CM1.5a.md`; `FNOConfig::{conv_bias, separable, max_modes,
    tanh_stabilizer}`; `tests/parity_fno.rs` (eight fixtures, f32 ≤ 1.8e-6), `_f64.rs`
    (two `#[ignore]`d, ≤ 1.2e-8: Flex's f32 twiddles, 2.0.0's f32 grid); shared map pieces
    `tests/parity_common/fno_map.rs`. The f64 fixtures with f64 constants (CM1.4's plan
    review) are not needed while Flex's f64 `rfft` uses f32 twiddles.
  - [ ] **CM1.5b Domain padding and resampling.** 2.0.0's `DomainPadding` (symmetric, a
    fraction of the resolution, unpadded at the scaled size) as an FNO option beside
    CONVENTIONS §3's one-sided `padding`; `resample` (linear in 1D, bicubic in 2D, Fourier
    for three or more axes) and the convolution's output size (`resolution_scaling_factor`
    per layer, `forward`'s `output_shape`). *Done when:* fixtures with padding and with
    scaling factors, 1D–3D, pass through `FNO` within D3 in f32; the dumps equal `main`'s.
    *Needs:* CM1.5a.
  - [ ] **CM1.5c Batch norm.** `NormKind::Batch` as 2.0.0's `batch_norm` (eval mode: the
    running statistics), and 2.0.0's `fno_skip=None` if the author wants it (CM1.5a PR).
    *Done when:* an `FNO` fixture with `norm="batch_norm"` passes within D3 in f32; the
    dumps equal `main`'s. *Needs:* CM1.5a.
  - [ ] **CM1.5d Tidy after full parity.** The CM1.3b, CM1.4 and CM1.2 follow-ups above
    not taken by CM1.5a: the `FNOConfig` literals, one coordinate-channel builder (the
    dumps and `gpu_parity` decide), `layers/spectral_convolution.rs` split, one shared
    biased test layer, converted arrays passed to `load_into` with `Fixture`'s fields
    private. From CM1.5a's code-quality review: CM0.5's test-only `Fno1dRef`
    (`tests/parity_common/fno1d.rs`, `tests/parity_fno1d*.rs`) retired, its harness
    self-checks ported to the library `FNO` (F4); one rule for `FNOConfig`'s boolean options
    (`Option<bool>` with `None` false, or `#[config(default = false)]`; F7, the author's
    choice at the M1 gate); `parity_common::fno::output`'s map hook replaced by a load step
    the control can corrupt (F6). *Done when:* the dumps equal `main`'s and `gpu_parity` passes on Metal.
    *Needs:* CM1.5a.
- **CM1.6 Losses.** H1, Hdiv, quantile, MSE; `WeightedSumLoss`,
  `FieldwiseAggregatorLoss`, SoftAdapt, Relobralo; the grid derivatives of
  `differentiation`. *Done when:* each against fixtures. *Needs:* CM0.5. Split 2026-11-25
  (over 1,000 lines across three loss families):
  - [x] **CM1.6a Finite differences and the data losses.** `FiniteDiff` (first and second
    derivatives, periodic or one-sided boundaries, 1D–3D; gradient, Laplacian,
    divergence, curl), `H1Loss`, `HdivLoss`, `PointwiseQuantileLoss`, `MSELoss`; the
    relative `LpLoss` against 2.0.0's. *Done when:* each against 2.0.0 fixtures within D3
    (f32; f64 too, no FFT involved; `LpLoss::rel` within 1e-8 in f64, as 2.0.0 adds
    eps = 1e-8 to its denominator and the crate's, fixed by CONVENTIONS §11 and constraint 1,
    does not). *Needs:* CM0.5. Done 2026-11-25: plan
    `docs/operators/tasks/CM1.6a.md`; `losses/{differentiation,sobolev,pointwise}.rs`; parity
    `tests/parity_losses.rs` (f32, ≤ 1.7e-7), `tests/parity_losses_f64.rs` (≤ 6e-16);
    fixtures from `tools/parity/generate_losses.py`.
  - [x] **CM1.6b Aggregators.** `WeightedSumLoss`, `FieldwiseAggregatorLoss`, `SoftAdapt`,
    `Relobralo` (its Bernoulli draw taken from a seeded source). *Done when:* each against
    a fixture of a sequence of steps (the weights and the aggregated loss per step).
    *Needs:* CM0.5. Also CM1.6a's code-quality follow-ups: `Reduction::apply` for the three
    copies of the reduction match, and `sqrt_zero_safe` shared with `lp_norm` (both in
    `data_losses.rs`, so with the bit-identity check); in the parity harness one
    `flex_f64` in `parity_common` (four copies with CM1.3a's), `Fixture::tensor` public (the copy in
    `parity_losses_common`), and with them the two `#[allow(dead_code)]` on the loss
    binaries' harness modules gone (coordinate `parity_common` with CM1.1). Done
    2026-11-25: plan `docs/operators/tasks/CM1.6b.md`; `losses/meta_losses.rs`; parity per
    step in `tests/parity_losses{,_f64}.rs` (≤ 2.0e-7 f32, ≤ 2.0e-16 f64); the follow-ups
    done (`tests/flex_f64/` for the four f64 binaries), except the `#[allow(dead_code)]` on
    `mod parity_common` in the two loss binaries, which use no parameter map (kept, see
    the plan).
  - [x] **CM1.6c Fourier and non-uniform derivatives.** `FourierDiff` without Fourier
    continuation (that option with CM2.4), `non_uniform_fd`. *Done when:* each against
    fixtures. *Needs:* CM0.5. Done 2026-11-26: plan `docs/operators/tasks/CM1.6c.md`;
    `losses/differentiation.rs` (`FourierDiff`; the vector operators on `GridDiff`),
    `losses/non_uniform_fd.rs`; parity `tests/parity_losses.rs` (f32 ≤ 6.3e-7),
    `tests/parity_losses_f64.rs` (`non_uniform_fd` ≤ 1.2e-15; Fourier `#[ignore]`d at 4.0e-7,
    Flex's f32 twiddles; the ridge case within 1e-9, signed off 2026-11-26); fixtures
    `derivatives_{f32,f64}` from `tools/parity/generate_losses.py`.
- [ ] **CM1.7 Training features.** Incremental modes and resolution, AdamW as 2.0.0's,
  mixed precision, multigrid patching. *Needs:* CM0.3, CM1.5.
- [ ] **CM1.8 Training parity on the 2D Poisson baseline.** The protocol of proposal §5,
  2.0.0 configured to this crate's FNO and at its own defaults. *Needs:* CM1.5, D7.
  *Machines:* the VM.

## M2: New building blocks

The order of M2 and M3 is set at the M1 gate (D6).

- [ ] **CM2.1 TFNO.** Tucker, CP and tensor-train factorised spectral weights, factorised
  and reconstructed contractions, with gradient parity. *Needs:* M1. As the second
  registered kind: its `ModelConfig` and `Model` variants, the MLflow export's arm
  (`tracking/builder`), and, if a run must load by name, `Model<R>` as a Burn `Module`
  (CM0.4 review F4).
- [ ] **CM2.2 UNO.** Resolution-changing blocks on `resample`. *Needs:* M1.
- [ ] **CM2.3 LocalNO.** Differential and DISCO convolutions, `local_no_block`. *Needs:* M1.
- [ ] **CM2.4 Fourier continuation.** `FCLegendre`, `FCGram`. *Needs:* M1. With it,
  `FourierDiff`'s `use_fc`, and the two `FourierDiff` options CM1.6c left out:
  `low_pass_filter_ratio` and the mixed `derivative` / `compute_multiple_derivatives`.
- [x] **CM2.5 Equation losses (PINO).** `equation_losses` on CM1.6's derivatives; the
  crate's Chebyshev differentiation as a second path. *Needs:* CM1.6a (2.0.0's
  `equation_losses` imports only `FiniteDiff`). Done 2026-11-26: plan
  `docs/operators/tasks/CM2.5.md`; `losses/equation_losses.rs` (`BurgersEqnLoss`,
  `ic_loss`, the nonlinear Poisson losses on a point list and on a grid), `GridDiff` and
  `MatrixDiff` in `losses/differentiation.rs` (`MatrixDiff::chebyshev` behind `datagen`);
  parity `tests/parity_losses{,_f64}.rs` (≤ 6.9e-8 f32, ≤ 5.3e-16 f64). 2.0.0's
  `autograd` method (derivatives with respect to point-cloud queries) moves to CM3.4.
- [ ] **CM2.6 UQNO.** *Needs:* M1, CM1.6a (quantile loss).
- [ ] **CM2.7 Divergence-free spectral projection.** *Needs:* M1.

## M3: The costly operators

- [ ] **CM3.1 Spike: spherical harmonic transforms.** Legendre quadrature and the
  transform in Rust against torch_harmonics. *Done when:* a spike report with accuracy
  and cost. *Needs:* M1.
- [ ] **CM3.2 SFNO.** *Needs:* CM3.1.
- [ ] **CM3.3 Point-cloud inputs.** Neighbour search, segment reductions, the GNO block,
  the trait's input type (R4). *Needs:* M1.
- [ ] **CM3.4 GINO and FNOGNO.** *Needs:* CM3.3. With them 2.0.0's
  `PoissonInteriorLoss` `method="autograd"`: the output's first and second derivatives
  with respect to the query points, for `losses::equation_losses::PoissonEqnLoss::points`
  (CM2.5 takes them as arguments; Burn's autodiff is first order: `.grad()` returns a
  tensor with no autodiff association, so a derivative cannot be differentiated again).
  Routes, agreed with the author 2026-11-26 to be chosen when CM3.4 starts: first, finite
  differences in the query points (1 + 2D forward passes, one backward; O(h²), about 1e-4
  relative in f32, an estimate); if too inaccurate, the layers' derivatives with respect to
  the query points written by hand (exact). Finite differences would miss D3 (1e-5)
  against 2.0.0's autograd values, so that parity test needs a tolerance signed off by the
  author.
- [ ] **CM3.5 CODANO.** Attention kernel integral, CODA layer. *Needs:* M1.
- **CM3.6 GaLore.** `tensor_galore_projector`. *Needs:* CM0.3. Split 2026-11-27:
  - [x] **CM3.6a Real tensors.** The projector, tensorly's HOOI Tucker factors and rank
    rules. *Done when:* factors, `project` and `project_back` against fixtures. *Needs:*
    CM0.3. Done 2026-11-27: `docs/operators/tasks/CM3.6a.md`.
  - [ ] **CM3.6b Complex tensors.** 2.0.0's complex Tucker (complex SVD, phase sign rule)
    for the spectral weights. *Done when:* against complex fixtures. *Needs:* CM3.6a.

## M4: Beyond 2.0.0

Operators from the author's reading, one task each, under the recipe of proposal §6.4.
Candidates (not yet chosen): Spectral Neural Operator (Chebyshev basis), Convolutional
Neural Operator, Wavelet Neural Operator, Laplace Neural Operator, Neural Spectral
Methods.

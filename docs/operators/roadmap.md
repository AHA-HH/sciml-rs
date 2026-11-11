# Neural operator library roadmap

The tasks of `docs/design/neural-operator-library.md` (draft 0.1), Section 10, in the order
they are taken. The proposal is the full record: where a task's *Done when* and the
proposal differ, the proposal wins. Progress is logged in `docs/operators/progress.md`;
each task's plan is `docs/operators/tasks/<ID>.md`, written when it starts; each
milestone's gate report is `docs/operators/gates/<milestone>.md`; the feature matrix is
`docs/operators/feature-matrix.md`.

Status: **planning.** Draft 0.1 written 2026-11-11; D0 open, so no task is runnable.

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
| D0 | Draft 0.1 of the proposal as the plan of record | after review | – | every task | open |
| D1 | Reference version | `neuraloperator` 2.0.0 | – | CM0.1 | signed off 2026-11-11: "its neuraloperator and the major release 2.0.0" |
| D2 | Where the Python side lives | `tools/parity/`, outside CI, lock file; fixtures in `tests/fixtures/parity/` | – | CM0.2 | open |
| D3 | Output parity tolerance | rel. max error ≤ 1e-5 (f32), ≤ 1e-12 (f64), CPU | CM0.5 | CM1.1 | open |
| D4 | Shape of the operator trait | trait with associated config and input type, generic | CM0.3 | CM0.3 | open |
| D5 | Spectral weight layout | corners internally; convert at fixture load | – | CM1.2 | open |
| D6 | Order of M2 and M3 | at the M1 gate; default M2 first | M1 gate | CM2.1, CM3.1 | open |
| D7 | Training-parity hardware | one A100 on the VM | – | CM1.8 | open |
| D8 | Fixture size | tiny models, under about 1 MB per fixture | CM0.2 | CM0.2 | open |

## M0: Foundations

**Gate:** the stack is model-agnostic with the FNO unchanged; a fixture round trip works;
the feature matrix is verified.

- [ ] **CM0.1 Feature matrix at 2.0.0.** Fill `feature-matrix.md` from the tag's source
  (every public model, layer, loss, training feature and dataset, with its options and
  defaults). *Done when:* every row has a status and a source path, checked by an agent
  independent of the writer. *Needs:* D1.
- [ ] **CM0.2 Pinned Python environment and fixture generator.** `tools/parity/`:
  `neuraloperator` 2.0.0 with torch, tensorly, tltorch and torch_harmonics pinned in a
  lock file; a generator writing weights, input and output of a named configuration to
  `.npz`. *Done when:* a 1D and a 2D FNO fixture written, sizes within D8. *Needs:* D2,
  D8.
- [ ] **CM0.3 The operator trait and a generic stack.** Proposal §6.1: trainer, evaluate,
  `predict_host`, checkpoints, the epoch observer, the run directory's weights and the
  Gauss–Newton residual generic over the trait; `FNO<R>` implements it. *Done when:* every
  existing test passes unedited, and a fixed FNO forward and one Phase 5 epoch are bit for
  bit unchanged. *Needs:* D4. *Sign-off:* D4.
- [ ] **CM0.4 The registry.** Proposal §6.2: `kind` in `model_cfg.json`, absent meaning
  the FNO. *Done when:* every existing run directory loads unchanged; a model builds by
  name. *Needs:* CM0.3.
- [ ] **CM0.5 The parity harness.** A test helper that loads a fixture by parameter-name
  map into a module and compares outputs (and gradients) within D3. *Done when:* the
  CM0.2 1D fixture passes through a hand-written map. *Needs:* CM0.2, D3.
  *Sign-off:* D3.

## M1: The FNO family

**Gate:** 2.0.0's `FNO` with its defaults and every option passes output parity; H1 and
the aggregators exist; one training-parity run on the 2D Poisson baseline.

- [ ] **CM1.1 Legacy spectral convolution parity.** 2.0.0's corner-layout layer against
  `SpectralConv`, 1D to 3D. *Needs:* CM0.5.
- [ ] **CM1.2 Centred layout, separable, `max_n_modes`.** The weight converter of proposal
  §6.3; separable weights; mode truncation below the stored maximum. *Needs:* CM1.1, D5.
- [ ] **CM1.3 Block parts.** Channel MLP; linear, identity and soft-gating skips;
  instance, group and AdaIN norms; pre-activation; the activation choice. *Needs:* CM1.1.
- [ ] **CM1.4 Lifting, projection and embeddings.** Channel-MLP lifting and projection with
  their ratios; `GridEmbeddingND`, `SinusoidalEmbedding`, `RotaryEmbedding2D`. *Needs:*
  CM1.3.
- [ ] **CM1.5 Full FNO parity.** 2.0.0's `FNO` at its defaults and option sweeps, on
  Darcy- and Burgers-shaped inputs; resampling (`resolution_scaling_factor`) and domain
  padding. *Needs:* CM1.2, CM1.4.
- [ ] **CM1.6 Losses.** H1, Hdiv, quantile, MSE; `WeightedSumLoss`,
  `FieldwiseAggregatorLoss`, SoftAdapt, Relobralo; the grid derivatives of
  `differentiation`. *Done when:* each against fixtures. *Needs:* CM0.5.
- [ ] **CM1.7 Training features.** Incremental modes and resolution, AdamW as 2.0.0's,
  mixed precision, multigrid patching. *Needs:* CM0.3, CM1.5.
- [ ] **CM1.8 Training parity on the 2D Poisson baseline.** The protocol of proposal §5,
  2.0.0 configured to this crate's FNO and at its own defaults. *Needs:* CM1.5, D7.
  *Machines:* the VM.

## M2: New building blocks

The order of M2 and M3 is set at the M1 gate (D6).

- [ ] **CM2.1 TFNO.** Tucker, CP and tensor-train factorised spectral weights, factorised
  and reconstructed contractions, with gradient parity. *Needs:* M1.
- [ ] **CM2.2 UNO.** Resolution-changing blocks on `resample`. *Needs:* M1.
- [ ] **CM2.3 LocalNO.** Differential and DISCO convolutions, `local_no_block`. *Needs:* M1.
- [ ] **CM2.4 Fourier continuation.** `FCLegendre`, `FCGram`. *Needs:* M1.
- [ ] **CM2.5 Equation losses (PINO).** `equation_losses` on CM1.6's derivatives; the
  crate's Chebyshev differentiation as a second path. *Needs:* CM1.6.
- [ ] **CM2.6 UQNO.** *Needs:* M1, CM1.6 (quantile loss).
- [ ] **CM2.7 Divergence-free spectral projection.** *Needs:* M1.

## M3: The costly operators

- [ ] **CM3.1 Spike: spherical harmonic transforms.** Legendre quadrature and the
  transform in Rust against torch_harmonics. *Done when:* a spike report with accuracy
  and cost. *Needs:* M1.
- [ ] **CM3.2 SFNO.** *Needs:* CM3.1.
- [ ] **CM3.3 Point-cloud inputs.** Neighbour search, segment reductions, the GNO block,
  the trait's input type (R4). *Needs:* M1.
- [ ] **CM3.4 GINO and FNOGNO.** *Needs:* CM3.3.
- [ ] **CM3.5 CODANO.** Attention kernel integral, CODA layer. *Needs:* M1.
- [ ] **CM3.6 GaLore.** `tensor_galore_projector`. *Needs:* CM0.3.

## M4: Beyond 2.0.0

Operators from the author's reading, one task each, under the recipe of proposal §6.4.
Candidates (not yet chosen): Spectral Neural Operator (Chebyshev basis), Convolutional
Neural Operator, Wavelet Neural Operator, Laplace Neural Operator, Neural Spectral
Methods.

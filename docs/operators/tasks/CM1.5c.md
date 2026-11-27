# CM1.5c Batch norm, and no FNO skip

Started 2026-11-27 on `operators/cm1-5c`. Roadmap: CM1.5c (and the author's answers of
2026-11-27 to CM1.5a's PR questions 1 and 2). Proposal: §10 (M1, CM1.5); §1.4 and §2.

## Goal

2.0.0's `norm="batch_norm"` and `fno_skip=None` in the crate's `FNO`, each within D3 of
2.0.0 in f32, with every existing FNO output unchanged bit for bit.

## Acceptance

Roadmap *Done when*: "an `FNO` fixture with `norm="batch_norm"` passes within D3 in f32; the
dumps equal `main`'s." The author's answer to CM1.5a's question 1 (2026-11-27): 2.0.0's
`fno_skip=None`, a block with no FNO skip at all, is taken here; `FNOConfig::fno_skip: None`
keeps meaning today's 1×1 convolution with bias, so old configs load unchanged. A fixture
with `fno_skip=None` passes within D3 in f32 as well.

## Read first

- `src/neural_operators/layers/{norm,skip}.rs`, `models/fno/mod.rs` (`init`, `post_block`,
  `pre_block`, `fno_skip`).
- Burn's `BatchNorm` (`burn-nn-0.23.0-pre.2/src/modules/norm/batch.rs`): any rank, the
  running statistics in inference (a device without autodiff, or `training` off).
- 2.0.0's `layers/normalization_layers.py::BatchNorm` (torch's `BatchNorm{1,2,3}d`, eps
  1e-5, momentum 0.1, affine; for four or more spatial axes flattened to 1D, the same map
  per channel) and `layers/fno_block.py` (`fno_skips is None`: `x = x_fno`).
- `tools/parity/generate.py`, `tests/parity_common/{fno,fno_map}.rs`, `tests/parity_fno.rs`.

## Changes

- `layers::norm`: `NormKind::Batch` → `Norm::Batch(BatchNorm)` (Burn's, eps 1e-5, momentum
  0.1, `γ = 1`, `β = 0`, mean 0, variance 1, as torch). Its forward is Burn's: the batch
  statistics on an autodiff device (training), the running ones otherwise (eval, 2.0.0's
  `.eval()`).
- `layers::skip`: `SkipKind::Disabled` (2.0.0's `skip_type=None`, as
  `PositionalEmbedding::Disabled`); `SkipKind::init` returns `Option<Skip<R>>`, `None` for
  `Disabled`. The FNO's skip lists skip it; `FNOConfig::fno_skip: Some(SkipKind::Disabled)`
  is a block with no FNO skip, and the convolution `w` is not built.
- `models::fno`: `fno_skip` returns `Option<Tensor>`; the two blocks add it only when
  present. With `fno_skip: None` the same operations run in the same order as on `main`.
- No new `FNOConfig` field (the author's M1-gate rule on boolean options is recorded on
  CM1.5d's line and binds no field here). No convention change, no dependency.

## Tests that prove it

- Two fixtures from `generate.py` (f32 only, as CM1.5a's sweeps; the running mean and
  variance redrawn, the variance positive, so the map is tested and not torch's initial
  statistics): `fno2d_batch_norm` (2.0.0's defaults, `norm="batch_norm"`) and
  `fno1d_pre_no_skip` (`fno_skip=None`, `norm="batch_norm"`, pre-activation).
  `generate.py` leaves torch's integer `num_batches_tracked` out of the fixture (it does not
  enter the eval forward) and its reload check ignores only that key.
- `tests/parity_fno.rs`: both within D3 (1e-5) through `FNO` in f32; the map gains the batch
  norm's `γ`, `β`, running mean and variance.
- A control: the batch norm's running mean and variance swapped at load fails D3.
- Unit tests: `NormKind::Batch` builds torch's initial values and, on a plain device, uses
  the running statistics (a closed form per channel); `SkipKind::Disabled` builds nothing;
  an `FNO` with `fno_skip: Some(Disabled)` has no `w` and no skip parameters.
- Constraint 1: `tests/fno_bit_identity.rs` dumps on `main` and the branch, compared.

## Checks

`/verify` (fmt, clippy ×4, `cargo test`, `cargo test --features datagen`, `cargo doc`, the
bit-identity dumps against `main`), `uv run python generate.py --check` (every fixture, old
and new, reproduced bit for bit), and `cargo test --features metal --test gpu_parity`.
`CARGO_TARGET_DIR=~/sciml-rs-target/cm1-5c`.

## VM runs

None.

## Constraints and risks

- Constraint 1: no field added to `FNO` or `FNOConfig`; `Norm` and `SkipKind` gain a
  variant each, and with `fno_skip: None` the forward is the same tensor operations in the
  same order.
- Training with batch norm: Burn updates the running variance with the biased batch
  variance, torch with the unbiased one; eval parity (this task) does not see it. Noted in
  the rustdoc, a follow-up if a trained-model parity needs it.
- 2D objective: untouched (no change to training, loaders or the default FNO).
- Parallel lanes: CM1.5b also edits `FNOConfig` and the parity tests; conflicts are merged
  from `origin/main`. `training/` is not touched.

## Pending runs

None.

## Open questions

None.

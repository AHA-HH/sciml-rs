# CM1.5d The FNO's config, padding and grid

Started 2026-11-27 on `operators/cm1-5d`. Roadmap: CM1.5d (split 2026-11-27 from "CM1.5d
Tidy after full parity" into CM1.5d–f). Proposal: §10 (M1); §1.4, §2 (constraint 1).

## Goal
Tidy `models/fno` after full parity: the boolean-option rule written down and tested, the
`FNOConfig` struct literals gone, one coordinate-channel builder, one padding model.

## Acceptance (roadmap *Done when*)
The dumps equal `main`'s and `gpu_parity` passes on Metal.

Items of the line, each with its check:
1. F7: `FNOConfig`'s boolean options `Option<bool>`, `None` meaning `false` (already so for
   `preactivation`, `conv_bias`, `separable`, `tanh_stabilizer`; no `#[config(default =
   false)]` in `src/`); the rule stated in `FNOConfig`'s rustdoc; a test loading an old
   `model_cfg.json`.
2. `FNOConfig { .. }` literals replaced by `FNOConfig::new(..).with_*`, outside `training/`
   (`training/fixtures.rs` keeps its literal: the task stays out of `training/`).
3. One coordinate-channel builder for `FNO::grid_cl` and `GridEmbeddingNd::grid`.
4. R2-1 (with R2-3): one per-axis `(lo, hi)` padding, one pad, one crop of
   `round(lo · r)`, `round(hi · r)` with `r` the product of the axis's scaling factors;
   CONVENTIONS §3's `padding` is `(0, p)`, `domain_padding` `(round(f·s), round(f·s))`, both
   together their sum. The asserts "domain_padding and padding are exclusive" and "padding
   with resolution_scaling_factor" go. Kept: domain padding with scaling needs a one-layer
   FNO (#183 decision 2), and `forward_with_output_shape` still rejects either padding (no
   crop is defined for a shape the layers did not compute, as 2.0.0).

## Read first
`src/neural_operators/models/fno/mod.rs` (`init`, `forward_full`, `grid_cl`),
`layers/embedding.rs` (`GridEmbeddingNd::grid`), `models/fno/tests/{padding,resolution}.rs`,
`tests/fno_bit_identity.rs`, CONVENTIONS §2, §3; the CM1.5b r2 review (R2-1, R2-3).

## Changes
- `models/fno/mod.rs`: the rule in `FNOConfig`'s doc; `FNO` keeps `padding` and
  `domain_padding` fields (no checkpoint change: neither is a parameter); `pad_spatial` and
  `crop_spatial` replaced by one `(lo, hi)` computation, one `pad`, one `slice`.
- `layers/embedding.rs`: `pub(crate) fn coordinate_channels` (expand each axis's line,
  concatenate along the channel axis, the single-channel workaround once); `grid_cl` and
  `GridEmbeddingNd::grid` give it their axes and lines. Same ops in the same order.
- Literals: `models/{registry,predict}.rs`, `runs/mod.rs`, `tracking/builder.rs`,
  `models/fno/{mod,tests/*}.rs`, `tests/parity_common/*.rs`, `tests/fno_bit_identity.rs`,
  `examples/train/*.rs`, `spikes/fno3d_cost`.
- No convention change, no dependency change.

## Tests that prove it
- New `models/fno/tests/config.rs`: a Phase 5 `model_cfg.json` (verbatim, as written before
  CM1.3b) loads; every newer option is `None`; its model forwards bit for bit as
  `FNOConfig::new(..)` with the same weights.
- `models/fno/tests/padding.rs`: `padding` with `domain_padding` equals a model padded by
  hand; `padding` with `resolution_scaling_factor` builds and gives the scaled extents
  (formerly a panic test, now the behaviour).
- Constraint 1: `fno_bit_identity` dumps `cmp` against `main` (the trusted path).
- `parity_fno` (f32, Flex and Metal), `gpu_parity` on Metal, the existing grid tests
  (`models/fno/tests/grid.rs`) and `parity_embeddings`, unchanged errors.

## Checks
`/verify` (fmt, clippy ×4, `cargo test`, `cargo test --features datagen`, `cargo doc`,
the bit-identity dumps), `cargo test --features metal --test parity_fno --test gpu_parity
--test parity_embeddings`. `CARGO_TARGET_DIR=~/sciml-rs-target/CM1.5d`.

## VM runs
None.

## Constraints and risks
Constraint 1: the default FNO's forward and training epochs bit for bit; the grid and pad
refactors keep the same tensor ops. Parallel lanes: CM0.7 (registry), CO3.3 (`training/`,
`examples/train`) may conflict on literal lines; merge `origin/main`, keep both.

## Pending runs
None.

## Open questions
None.

# CM1.3b The FNO's block options

Started 2026-11-25 on `operators/cm1-3b`. Roadmap: "CM1.3b The FNO's block options.
`FNOConfig` gains the activation, the FNO skip, the channel MLP (expansion, skip), the norm
and pre-activation, each absent meaning today's block. *Needs:* CM1.3a." Proposal: §10;
§1.4, §2, §3, §4.4, §5. D3 (as amended 2026-11-25).

## Goal

`FNOConfig` can build each Fourier layer as 2.0.0's `FNOBlocks` does, and with every new
option absent the FNO is today's model, bit for bit.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)

- Roadmap: "the CM0.3 bit-identity check passes against `main`; a block built from the
  options matches 2.0.0's `FNOBlocks` (with the legacy convolution) within D3, post- and
  pre-activation, for each norm; the FNO's own pointwise matmul (`FNO::pointwise`,
  `conv1x1_cf`) replaced by `layers::pointwise::conv1x1` (CM1.3a review)."
- Proposal §1.4, constraint 1: every existing FNO output bit for bit identical; "a part made
  configurable defaults to today's behaviour". §5: output parity within D3 (relative max
  error max |a − e| / max |e| ≤ 1e-5 in f32 on the CPU; f64 tests that involve an FFT are
  `#[ignore]`d with the error recorded).

## Read first

- 2.0.0 `neuralop/layers/fno_block.py` (`FNOBlocks.__init__`,
  `forward_with_postactivation`, `forward_with_preactivation`).
- `src/neural_operators/models/fno/mod.rs`; `layers/{pointwise,channel_mlp,skip,norm}.rs`.
- `tests/fno_bit_identity.rs` (CM0.3), `tests/parity_common/{mod,legacy_conv}.rs`.

What 2.0.0's block does, per layer index `i` (`last` = the last layer):

| Post-activation (default) | Pre-activation |
| --- | --- |
| `skip = fno_skip(x)`, `mlp_skip = mlp_skip(x)` | `x = act(x)`; `x = norm[2i](x)` |
| `x = norm[2i](conv(x)) + skip` | skips of `x`; `x = conv(x) + skip` |
| `if !last: x = act(x)` | `if !last: x = act(x)` |
| `x = mlp(x) + mlp_skip` (with an MLP) | `x = norm[2i+1](x)` |
| `x = norm[2i+1](x)`; `if !last: x = act(x)` | `x = mlp(x) + mlp_skip` (with an MLP) |

Today's block is the post-activation column with ReLU, no MLP, no norm, and as FNO skip a
pointwise convolution *with* bias (2.0.0's `"linear"` skip has none). Without an MLP or norm,
2.0.0 applies the activation twice in a row; ReLU is idempotent, so the second is skipped for
ReLU only (today's ops unchanged), and applied for any other activation (2.0.0's output).

## Changes

- `layers/pointwise.rs`: the FNO's `pointwise` matmul moves here as the shared core;
  `conv1x1` (CM1.3a) and a new `linear_cf` (a `Linear` along the channel axis) call it;
  `pub(crate) mod pointwise`. `FNO::pointwise`, `linear_cf` and `conv1x1_cf` go.
- `layers/norm.rs`: `NormKind` (config: `Instance`, `Group`, `AdaIn { embed_dim }`) and
  `Norm` (module enum: Burn's `InstanceNorm` without affine, Burn's `GroupNorm` with one
  group, the crate's `AdaIn`), `Norm::forward(x, embedding)`; panics without an embedding
  for AdaIN.
- `models/fno/mod.rs`: `FNOConfig` gains, each an `Option` so saved configs load as `None`:
  `activation: Option<ActivationConfig>` (`None`: ReLU), `fno_skip: Option<SkipKind>`
  (`None`: today's pointwise convolution with bias, `w`), `channel_mlp:
  Option<BlockMlpConfig>` (`None`: none; `BlockMlpConfig { expansion = 0.5, skip =
  Some(SoftGating) }`, 2.0.0's defaults, the hidden width rounded half to even as Python's
  `round`, its MLP 2.0.0's two-layer GELU `ChannelMLP`),
  `norm: Option<NormKind>`, `preactivation: Option<bool>`. `FNO` gains fields after the
  existing ones, empty or constant when the options are absent, so checkpoint paths and
  parameter order are unchanged: `fno_skips`, `channel_mlp`, `channel_mlp_skips`, `norms` (two per
  layer), `activation`, `preactivation`, `repeat_activation`. New `FNO::fourier_layers(x, embedding)` (the
  layers alone, channels-first, 2.0.0's `FNOBlocks` over every index) and
  `FNO::forward_with_embedding` (AdaIN); `forward` calls `fourier_layers(x, None)`.
  The projection's ReLU (`fc1`) is unchanged: lifting and projection are CM1.4's.
- Tests: `tests/parity_common/fno_blocks.rs` (cases and maps), `tests/parity_fno_blocks.rs`
  (f32 only: no f64 fixtures, see Results); `load_into_part` in the
  harness, a `load_into` restricted by a `PathFilter` to the block's paths (fixtures carry no
  lifting or projection), the completeness check unchanged for those paths.
- `tools/parity/generate.py`: `FNOBlocks` fixtures, the legacy convolution with
  `bias=False` (the crate's `SpectralConv` has no bias; CM1.2 keeps it so); parameters whose
  2.0.0 init is constant are redrawn (soft gating, group norm), now only those.
- Docs: root `CLAUDE.md` Layout, `tools/parity/README.md`, feature matrix, roadmap, progress.
- Conventions: none changed.

## Tests that prove it

- Constraint 1: `tests/fno_bit_identity.rs` (CM0.3) on `main` and on the branch, dumps
  compared (forward, Phase 5 epoch, Gauss–Newton epoch); the whole existing suite unedited.
- Parity (f32, D3 1e-5), 2.0.0 `FNOBlocks`, two layers, 4 channels, legacy convolution
  without bias, against `FNO::fourier_layers` with the matching options:

  | Fixture | Options | Input |
  | --- | --- | --- |
  | `fno_blocks_post1d` | defaults (GELU, linear skip, MLP 0.5 with soft gating), no norm | (2, 4, 20) |
  | `fno_blocks_pre1d` | the same, pre-activation | (2, 4, 20) |
  | `fno_blocks_instance_post2d`, `_pre2d` | instance norm | (2, 4, 12, 10) |
  | `fno_blocks_group_post3d` | group norm, soft-gating FNO skip, linear MLP skip, expansion 1 | (2, 4, 8, 7, 6) |
  | `fno_blocks_group_pre3d` | group norm, identity skips | (2, 4, 8, 7, 6) |
  | `fno_blocks_ada_in_post2d`, `_pre2d` | AdaIN, embed_dim 2, one seeded embedding; one layer, 2 channels (D8) | (2, 2, 12, 10) |
  | `fno_blocks_plain1d` | no MLP, no norm, tanh, soft-gating skip (2.0.0's repeated activation) | (2, 4, 20) |

- Unit tests: ReLU's skipped repeat is the same output as applying it (a no-MLP, no-norm
  FNO); the MLP's hidden width rounds half to even (width 5 → 2); `forward` panics for an AdaIN FNO without an embedding; `linear_cf` against
  `Linear::forward`; old JSON without the new fields loads as `None` (existing test).

## Checks

`/verify` (fmt, clippy ×4, `cargo test`, `cargo test --features datagen`, `cargo doc`, the
bit-identity dumps against `main`); `uv run python generate.py --check`. The forward's op
sequence with the options absent is today's, so `--features metal` `gpu_parity` is run as
well (`fno` is on the replay path).

## VM runs

None.

## Constraints and risks

- Constraint 1: the options-absent path keeps today's ops in today's order (conv, skip, add,
  ReLU); new fields hold no parameters then; the pointwise core keeps `pointwise`'s ops.
- The legacy convolution's bias is left out of the block fixtures: a conv bias before the
  norm is a full-FNO matter (CM1.5's line gains it).
- 2.0.0's `fno_skip=None` / no channel-MLP skip `None`: only the latter is an option here
  (`BlockMlpConfig::skip = None`); the FNO skip always exists (2.0.0's default "linear").
  Stabilizer, batch norm, dropout and complex data stay out (feature matrix).
- Effect on the 2D objective: none (outputs unchanged).

## Pending runs

None.

## Results

Relative max error (Flex, M2 Pro, f32): `post1d` 7.4e-8, `pre1d` 1.5e-7, `instance_post2d`
4.1e-7, `instance_pre2d` 2.3e-7, `group_post3d` 2.0e-7, `group_pre3d` 1.2e-6,
`ada_in_post2d` 2.3e-6, `ada_in_pre2d` 2.0e-7, `plain1d` 2.3e-7 (D3: 1e-5).

Departures from the plan as first written:
- No f64 fixtures or f64 test binary: the legacy layer computes its spectrum in `cfloat` and
  returns f32 (without a bias nothing promotes it), so an f64 reference would be
  f32-accurate, and torch refuses a group norm with f64 parameters on its f32 output
  (`F32_ONLY` in `generate.py`). D3 as amended needs f32 parity only.
- AdaIN's MLP is 512 wide, so two layers of 4 channels put the fixture at 25,460 parameters,
  over D8's 10⁴: its fixtures have one layer, 2 channels and `embed_dim` 2 (7,285). The
  between-layer activation and layer 1's norms are covered by the other fixtures.
- 2.0.0's `FNOBlocks` passes the new `SpectralConv`'s keywords to `conv_module`; the
  generator's `legacy_conv` drops them (all at their defaults).

## Open questions

None.

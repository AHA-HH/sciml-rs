# CM1.3a The parts

Started 2026-11-25 on `operators/cm1-3a`. Roadmap: "CM1.3a The parts. `ChannelMlp`
(2.0.0's `ChannelMLP`), the skips of `skip_connection` (identity, linear, soft-gating) and
`AdaIn` in `layers/`; the activation choice as Burn's `ActivationConfig`; instance and group
norm as Burn's `InstanceNorm` and `GroupNorm`. *Needs:* CM1.1." Split from CM1.3 on this
branch (CM1.3b, the FNO's block options, follows). Proposal: §10; §1.4, §2, §3, §4.4, §5,
§6.4. D3 (as amended 2026-11-25), D8.

## Goal

The block parts of 2.0.0's `FNOBlocks` exist as rank-generic layers in `layers/`, each shown
equal to 2.0.0's own within D3.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)

- Roadmap: "each part (Burn's norms and activations included) matches a 2.0.0 fixture of its
  own within D3, in f32 and f64, across 1D to 3D; nothing under `models/` or `training/`
  changes."
- Proposal §4.4: `layers/` gains `channel_mlp`, `skip`, `norm`. §5 and D3 as amended: relative
  max error max |a − e| / max |e| ≤ 1e-5 (f32), ≤ 1e-12 (f64), on the CPU. These parts run
  no FFT, so the f64 bound applies as signed off; a part that misses it is `#[ignore]`d with
  its error recorded, as D3 as amended allows, and named in the PR.
- Constraint 1 (§1.4): every existing FNO output bit for bit identical. Only new modules and
  two `pub mod` lines change under `src/`.

## Read first

- 2.0.0 (`tools/parity/.venv/.../neuralop/layers/`): `channel_mlp.py` (`ChannelMLP`),
  `skip_connections.py` (`skip_connection`, `SoftGating`, `Flattened1dConv`),
  `normalization_layers.py` (`InstanceNorm`, `AdaIN`), `fno_block.py` (how `FNOBlocks` builds
  them: group norm is `nn.GroupNorm(num_groups=1, num_channels=C)`).
- Burn 0.23.0-pre.2: `burn-nn` `activation/activation_wrapper.rs` (`ActivationConfig`,
  `Activation`), `modules/norm/{instance,group}.rs`, `modules/linear.rs` (`LinearLayout`).
- `tests/parity_common/mod.rs` (CM0.5's harness), `tools/parity/generate.py`.

What the source says, and how it maps:

| 2.0.0 | Here |
| --- | --- |
| `ChannelMLP(in, out=None, hidden=None, n_layers=2, non_linearity=F.gelu, dropout=0)`: `Conv1d(·, ·, 1)` layers with bias on the flattened spatial axes, the activation between layers, none after the last | `ChannelMlpConfig` / `ChannelMlp`, `fcs: Vec<Conv1d>` (2.0.0's names and shapes), the activation an `ActivationConfig` (default GELU, the erf form as `F.gelu`) |
| `skip_connection(in, out, n_dim, bias=False, skip_type)`: `"identity"` → `nn.Identity`, `"linear"` → `Flattened1dConv` (a `Conv1d(in, out, 1, bias=False)` on flattened axes), `"soft-gating"` → `SoftGating` (weight `(1, C, 1, …)`, ones; `in == out` or it raises) | `SkipKind` (config) and `Skip<R>` (module enum); the linear skip a bias-free `Conv1d`; soft gating a `[1, C, 1, …]` parameter of rank `R` |
| `InstanceNorm()`: `F.instance_norm`, no affine, eps 1e-5, input statistics | Burn's `InstanceNorm` with `affine = false`: tested, not wrapped |
| `nn.GroupNorm(1, C)`: affine weight and bias of `[C]`, eps 1e-5 | Burn's `GroupNorm` (`gamma`, `beta`): tested, not wrapped |
| `AdaIN(embed_dim, in_channels, eps=1e-5)`: `Linear(embed, 512) → GELU → Linear(512, 2C)` of an embedding set beforehand; split into weight and bias; `group_norm` with `C` groups | `AdaInConfig` / `AdaIn`; the two `Linear`s in Burn's `[in, out]` layout, the map transposing torch's `[out, in]` (Burn's `LinearLayout::Col` keeps `[in, out]` in memory and transposes only on save, so it does not help); the embedding an argument of `forward`, not state |

Left out, named in the feature matrix: `ChannelMLP`'s dropout (2.0.0's default 0; a no-op in
eval), `LinearChannelMLP` (for GNO models, M3), soft gating's bias (no 2.0.0 caller sets it),
batch norm (not in the roadmap line; stateful), AdaIN's custom `mlp` argument. The selection
of a norm and pre-activation belong to the block (CM1.3b).

## Changes

- `src/neural_operators/layers/channel_mlp.rs` (new): `ChannelMlpConfig`, `ChannelMlp`
  (`forward<const R>`: `[B, C_in, s..]` → `[B, C_out, s..]`), and `pub(crate) conv1x1`, a
  kernel-1 `Conv1d` as one batched matmul (as `FNO::conv1x1_cf`; the FNO is not changed here,
  CM1.3b may share it).
- `src/neural_operators/layers/skip.rs` (new): `SkipKind`, `Skip<R>` (`forward`), panics on
  `in ≠ out` for identity and soft gating.
- `src/neural_operators/layers/norm.rs` (new): `AdaInConfig`, `AdaIn`
  (`forward(x: [B, C, s..], embedding: [embed_dim])`).
- `src/neural_operators/layers/mod.rs`: three `pub mod` lines.
- `tools/parity/generate.py`: the fixtures below; parameters redrawn from a seeded normal for
  parts whose 2.0.0 init is constant (soft gating's ones, group norm's ones and zeros), so
  the fixture tests the map; an `AdaIN` subclass that keeps its embedding as a buffer so the
  fixture carries it; the sidecar names a function argument by its `__name__`.
- `tools/parity/README.md`: the new fixtures' rows.
- Tests: `tests/parity_common/block_parts.rs` (the cases, maps, and AdaIN with its embedding
  as a test-only parameter, as CM1.1's test-only bias), included by `#[path]` from
  `tests/parity_block_parts.rs` (f32) and `tests/parity_block_parts_f64.rs` (its own binary:
  Flex's process-wide f64). The harness (`tests/parity_common/mod.rs`) gains
  `Take::Transposed` for torch's `Linear` weights, and a scoped `allow(dead_code)` on `Take`
  (each binary constructs only some variants).
- Docs: root `CLAUDE.md` "Layout" (the three modules), feature matrix rows, roadmap,
  progress.
- Public API: the types above. Conventions: none changed (channels-first inside, as the FNO;
  CONVENTIONS §1).

## Tests that prove it

Fixtures (2.0.0, seeded, f32 and f64 each), every one with in ≠ out where the part allows it
and odd extents:

| Fixture | 2.0.0 part | Input |
| --- | --- | --- |
| `channel_mlp1d` | `ChannelMLP(4, hidden=2)` (GELU, out = in) | (2, 4, 7) |
| `channel_mlp3d` | `ChannelMLP(3, out=5, hidden=4, n_layers=3, non_linearity=torch.tanh)` | (2, 3, 5, 4, 3) |
| `linear_skip3d` | `skip_connection(3, 5, n_dim=3, skip_type="linear")` | (2, 3, 5, 4, 3) |
| `soft_gating2d` | `skip_connection(4, 4, n_dim=2, skip_type="soft-gating")`, weight redrawn | (2, 4, 6, 5) |
| `instance_norm1d` | `InstanceNorm()` | (2, 4, 9) |
| `group_norm3d` | `nn.GroupNorm(1, 4)`, affine redrawn | (2, 4, 5, 4, 3) |
| `ada_in2d` | `AdaIN(embed_dim=3, in_channels=4)`, embedding seeded | (2, 4, 6, 5) |

Each against its fixture with `assert_parity` (D3: 1e-5 f32, 1e-12 f64), with Burn's
`ActivationConfig::Gelu` and `Tanh` exercised through the channel MLPs (ReLU is exact and
needs no fixture). The harness's `load_into` checks the load is complete both ways. Unit
tests in each module: shapes, the identity skip returns its input unchanged, the panics
(soft gating and identity with `in ≠ out`, AdaIN's embedding length).

## Checks

`/verify` (fmt, clippy default, `--examples`, `--all-targets`, `--features datagen
--all-targets`, `cargo test`, `cargo test --features datagen`, `cargo doc --no-deps`);
`uv run python generate.py --check` (the new fixtures reproduced, the old ten unchanged).
Not backend-sensitive (no FFT, no fused-cat pattern): `--features metal` not needed.

## VM runs

None.

## Constraints and risks

- Constraint 1: nothing an existing FNO runs is edited; the bit-identity dumps are needed only
  if `/verify` finds a protected path touched.
- Burn's `Module` derive on an enum adds the variant to snapshot paths, maybe; the maps follow
  what `ModuleSnapshot::collect` names.
- Burn's f64 `erf` on Flex may be less accurate than 1e-12 relative; if so, the GELU fixtures'
  f64 tests are `#[ignore]`d with the error recorded (D3 as amended).
- Effect on the 2D objective: none.

## Pending runs

None.

## Results

Relative max error (Flex, M2 Pro), f32 / f64: `channel_mlp1d` 1.078e-7 / 0; `channel_mlp3d`
7.484e-8 / 1.859e-16; `linear_skip3d` 0 / 0; `soft_gating2d` 0 / 0; `instance_norm1d`
8.773e-8 / 1.634e-16; `group_norm3d` 1.141e-7 / 4.249e-16; `ada_in2d` 1.881e-7 / 7.009e-16.
Every f64 test runs (none `#[ignore]`d).

## Open questions

None.

# CM1.4 Lifting, projection and embeddings

Started 2026-11-26 on `operators/cm1-4`. Roadmap: "CM1.4 Lifting, projection and
embeddings. Channel-MLP lifting and projection with their ratios; `GridEmbeddingND`,
`SinusoidalEmbedding`, `RotaryEmbedding2D`. *Needs:* CM1.3a (the channel MLP)." Proposal:
§10; §1.4, §2, §3, §4.4, §5. D3 (as amended 2026-11-25).

## Goal

`FNOConfig` can build 2.0.0's lifting, projection and grid embedding, and the library has
2.0.0's three embedding layers; with every new option absent the FNO is today's model, bit
for bit.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)

- The roadmap line has no *Done when*; the M1 gate ("2.0.0's `FNO` with its defaults and
  every option passes output parity") and proposal §5 give it: each part matches 2.0.0
  within D3 (relative max error max |a − e| / max |e| ≤ 1e-5 in f32 on the CPU; f64 tests
  `#[ignore]`d with the error recorded where an f32 step limits them).
- Proposal §1.4, constraint 1: every existing FNO output bit for bit identical; "a part made
  configurable defaults to today's behaviour". Checked by the CM0.3 dumps
  (`tests/fno_bit_identity.rs`) on `main` and the branch, and the existing suite unedited
  (struct literals completed with `None` only).

## Read first

- 2.0.0 `neuralop/layers/embeddings.py` (all of it), `neuralop/models/fno.py`
  (`__init__`: the embedding, `lifting_channels = int(ratio · hidden)`, the lifting and
  projection `ChannelMLP`s; `forward`).
- `src/neural_operators/models/fno/mod.rs`, `layers/channel_mlp.rs`,
  `tests/parity_common/{mod,fno_blocks}.rs`, `tools/parity/generate.py`, CONVENTIONS §2.

What 2.0.0 does:

| Part | 2.0.0 | the crate today |
| --- | --- | --- |
| grid | `GridEmbeddingND`: per axis `linspace(a, b, s + 1)[:-1]` (j/s on [0, 1) by default), appended after the data in axis order; built in f32 and cast | j/(s − 1) on [0, 1], reverse axis order (CONVENTIONS §2) |
| no grid | `positional_embedding=None` | – |
| lifting | `ChannelMLP` (C + D) → L → W, the FNO's activation between, L = int(r · W), r = 2; L = 0: one layer (C + D) → W | `fc0`, one linear layer (C + D) → W |
| projection | `ChannelMLP` W → P → out, activation between, P = int(r · W), r = 2 | `fc1` W → 128, activation, `fc2` 128 → out |

`SinusoidalEmbedding` and `RotaryEmbedding2D` are not FNO options in 2.0.0 (GNO, GINO and
CODANO use them): they are standalone layers here, wired into a model by the task that adds
that model.

## Changes

- New `src/neural_operators/layers/embedding.rs`:
  - `GridEmbeddingNdConfig { boundaries: Vec<(f64, f64)> }` / `GridEmbeddingNd`: `forward`
    on channels-first `[B, C, s..]` → `[B, C + D, s..]`; `grid` builds the channels for a
    given shape and channel axis (the FNO builds them channels-last, CONVENTIONS §2's
    Metal note).
  - `SinusoidalEmbeddingConfig { in_channels, num_frequencies, kind }`,
    `SinusoidalKind::{Transformer { max_positions }, Nerf}`: `[B, N, C]` → `[B, N, 2LC]`,
    sin and cos interleaved per frequency, as 2.0.0.
  - `RotaryEmbedding2dConfig { dim, min_freq, scale }`: `forward` `[B, N]` → `[B, N, d]`;
    `rotate_half`, `apply_rotary_pos_emb`, `apply_2d_rotary_pos_emb`.
- `FNOConfig` gains, each `None` today's model and an `Option` so saved configs load:
  - `lifting_channel_ratio: Option<f64>`: L = trunc(r · W); L > 0 makes `fc0` (C + D) → L
    and a second layer L → W after the activation; L = 0 keeps one layer.
  - `projection_channel_ratio: Option<f64>`: `fc1`'s width trunc(r · W) instead of 128
    (the projection is already linear, activation, linear, as 2.0.0's two-layer MLP).
  - `positional_embedding: Option<PositionalEmbedding>`, `PositionalEmbedding::{GridNd(
    GridEmbeddingNdConfig), Disabled}`; `None` is CONVENTIONS §2's grid.
  - New FNO fields after the existing ones (checkpoint paths unchanged): `lifting`
    (`Vec<Linear>`, empty by default) and the embedding (not a parameter).
- `FNO::forward_with_embedding`: the grid by the option; the data-channel check subtracts
  the embedding's channel count.
- Test harness: `Take::Transposed` also takes a pointwise convolution's weight
  `[out, in, 1]` (2.0.0's `ChannelMLP` into the FNO's `Linear`s).
- `generate.py`: fixtures (below). Struct literals of `FNOConfig` in `src/`, `tests/`,
  `examples/` gain three `None` lines.
- Conventions: none changed. The `GridNd` option is an opt-in alternative to §2's grid; a
  sentence in §2 naming it is a question for sign-off in the PR.

## Tests that prove it

Fixtures from 2.0.0 (`tools/parity/generate.py`), f32 and f64, D3:
- `fno_lift_project1d`: `FNO(n_modes=(8,), in 2, out 1, hidden 6, n_layers=0)`, 2.0.0's
  defaults (grid, ratios 2, GELU).
- `fno_lift_project2d_linear`: lifting ratio 0 (one layer), projection ratio 1.5 with
  hidden 5 (int(7.5) = 7, truncation), tanh.
- `fno_lift_project3d_no_grid`: `positional_embedding=None`, three channels in, two out.
- `fno_lift_project2d_boundaries`: a `GridEmbeddingND` instance with boundaries
  [[0, 2], [−1, 1]].
- `grid_embedding2d`: the layer alone, boundaries [[0, 2], [−1, 1]].
- `sinusoidal_transformer`, `sinusoidal_nerf`: `[2, 10, 3]`, 4 frequencies.
- `rotary2d`: input `[2, 10, 2 + 2d]`, coordinates in channels 0 and 1, features the rest;
  output `apply_2d_rotary_pos_emb(t, rope(x), rope(y))`; d = 8.

`tests/parity_embeddings.rs` (f32) and `_f64.rs` (f64; `#[ignore]`d where 2.0.0's f32 grid
or frequencies limit the error, with it recorded). Unit tests: the trunc widths, `Disabled`
expects C channels, default options build no new parameters, `rotate_half` by hand.
Constraint 1: `fno_bit_identity` dumps on `main` and the branch, `cmp` equal.

## Checks

`/verify` (fmt, clippy ×4, `cargo test`, `cargo test --features datagen`, `cargo doc`, the
bit-identity dumps), `uv run python generate.py --check`, `--features metal` for the FNO
tests and `gpu_parity` (the forward changed). `CARGO_TARGET_DIR=~/sciml-rs-target/cm1-4`.

## VM runs

None.

## Constraints and risks

- Constraint 1: `forward`'s default path keeps `grid_cl`, `fc0`, `fc1` (128), `fc2`
  unchanged; new fields come last. Risk: the data-channel check's arithmetic; covered by
  the dumps.
- 2.0.0 builds its grid in f32 whatever the dtype, so f64 grid fixtures carry f32 values.
- 2D objective: unaffected (no default changes).

## Pending runs

None.

## Open questions

None yet.

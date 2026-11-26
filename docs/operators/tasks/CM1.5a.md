# CM1.5a Whole-FNO parity: defaults and block sweeps

Started 2026-11-26 on `operators/cm1-5a`. Roadmap: "CM1.5a Whole-FNO parity, defaults and
block sweeps" (the split of CM1.5, this branch). Proposal: §10; §1.4, §2, §5, §6.3. D3 (as
amended 2026-11-25), D5, D8.

## Goal

2.0.0's `FNO`, end to end (grid embedding, lifting, centred spectral convolutions with
their bias, the blocks, projection), loads into the crate's `FNO` through one parameter map
and gives 2.0.0's output within D3, at 2.0.0's defaults and over a sweep of its block
options; every new option off by default, so today's FNO is unchanged bit for bit.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)

- *Done when:* "`fno1d` and `fno2d` (2.0.0's defaults, Burgers- and Darcy-shaped) and the
  sweep fixtures (norms, pre-activation, skips, ratios, activations, no grid, separable,
  `max_n_modes`, an odd mode count, 3D, the stabiliser) pass through `FNO` within D3 in f32;
  the f64 tests `#[ignore]`d with their errors recorded; the FNO's bit-identity dumps equal
  `main`'s."
- Proposal §5: output parity within D3 (max |a − e| / max |e| ≤ 1e-5 in f32 on the CPU).
- Proposal §1.4, constraint 1: every existing FNO output bit for bit identical; "a part
  made configurable defaults to today's behaviour".
- M1 gate (roadmap): "2.0.0's `FNO` with its defaults and every option passes output
  parity". This sub-task covers every option but padding and resampling (CM1.5b), batch
  norm (CM1.5c) and those listed under "Not represented" below.

## Read first

- 2.0.0 `neuralop/models/fno.py` (`__init__`, `forward`), `layers/fno_block.py`
  (`forward_with_postactivation`, `forward_with_preactivation`: the stabiliser's place),
  `layers/spectral_convolution.py` (bias `(out, 1, …)`, `init_std · randn`; `n_modes`'
  last axis `n // 2 + 1`; `max_n_modes`, separable).
- `src/neural_operators/models/fno/mod.rs`; `layers/spectral_convolution.rs`
  (`new_separable`, `with_modes`).
- `tests/parity_common/{mod,fno_blocks,embeddings,centred_conv}.rs`: the three maps this
  task joins, and `centred_to_corners`.
- `tools/parity/generate.py` (`CONFIGS`, `REDRAW`, `F32_ONLY`).

## Changes

- `models/fno/mod.rs`, `FNOConfig` gains, each `None` today's model and an `Option` so
  saved configs load:
  - `conv_bias: Option<bool>`: the spectral convolution's bias (2.0.0's `SpectralConv`
    `bias=True`), `[W]` per layer, added to the convolution's output before the norm, drawn
    as 2.0.0's (normal, std √(2 / (W + W))). Stored in a new `FNO::conv_biases:
    Vec<Param<Tensor<1>>>`, after every existing field (checkpoint paths and order of
    today's parameters unchanged, as CM1.3b did); not in `SpectralConv`, whose record the
    existing runs hold.
  - `separable: Option<bool>`: `SpectralConv::new_separable` in every layer.
  - `max_modes: Option<Vec<usize>>`: the modes stored per corner (2.0.0's `max_n_modes`),
    with `modes` the modes used (`SpectralConv::with_modes`); `None` is `modes`.
  - `tanh_stabilizer: Option<bool>`: 2.0.0's `stabilizer="tanh"`, `tanh` on the
    convolution's input only (after the skips are taken), post- and pre-activation.
- `tests/parity_common/`: F4 of CM1.4's code-quality review, the shared helpers into
  `mod.rs` (`Fixture::open_as(stem, precision)`, `check_d3` that asserts and prints, the
  two permutes); new `fno.rs`, the whole-FNO map (lifting and projection from
  `embeddings.rs`, block parts from `fno_blocks.rs`, convolution weights through
  `centred_to_corners` per layer, the bias), each sub-map a function the others call, no
  copies. 2.0.0's `n_modes` and `max_n_modes` translate to corners as CM1.2's `Case`
  (`K − K/2` on a non-last axis, `K // 2 + 1` on the last).
- `tests/parity_fno.rs` (f32) and `tests/parity_fno_f64.rs` (`#[ignore]`d), one test per
  fixture; `tests/parity_fno1d.rs` stays (the harness's own checks on CM0.5's reference
  model).
- `tools/parity/generate.py`: six sweep fixtures, f32 only (as `fno_blocks_*`, the
  constant inits redrawn), each within D8:
  - `fno3d`: `n_modes=(4, 4, 4)`, 2 → 1 channels, hidden 4, 2 layers; input (2, 2, 8, 6, 5).
  - `fno2d_instance_pre`: `n_modes=(6, 4)`, hidden 6, `norm="instance_norm"`,
    `preactivation=True`, tanh; input (2, 1, 16, 12).
  - `fno1d_group_skips`: `n_modes=(10,)`, 2 → 1, hidden 6, `norm="group_norm"`,
    `fno_skip="soft-gating"`, `channel_mlp_skip="linear"`, `channel_mlp_expansion=1.0`,
    `lifting_channel_ratio=0`, `projection_channel_ratio=1.5`, `positional_embedding=None`,
    SiLU; input (2, 2, 40).
  - `fno2d_separable`: `n_modes=(6, 6)`, `max_n_modes=(8, 8)`, `separable=True`, hidden 6;
    input (2, 1, 16, 16).
  - `fno2d_odd_stabilizer`: `n_modes=(7, 6)` (an odd centred count: the zero row),
    `fno_skip="identity"`, `channel_mlp_skip=None`, `stabilizer="tanh"`; input (2, 1, 16, 14).
  - `fno1d_pre_no_mlp`: `n_modes=(8,)`, `use_channel_mlp=False`, `preactivation=True`,
    `stabilizer="tanh"`, ReLU; input (2, 1, 32).
  The existing `fno1d` and `fno2d` (f32, f64) are 2.0.0's defaults; no fixture changes.
- Conventions: none. `docs/operators/feature-matrix.md`: the `FNO`, `FNOBlocks` and
  centred `SpectralConv` rows.

### Not represented (said so in the PR and the matrix)

- An odd grid extent on a non-last axis (CM1.2: 2.0.0's output spectrum moves one bin).
- `fno_skip=None` (no FNO skip): `FNOConfig::fno_skip: None` already means the original
  1×1 convolution with bias; adding "none" is an API choice, asked in the PR.
- `complex_data` (complex arithmetic, proposal §2), `fno_block_precision` half and mixed
  (CM1.7), factorised weights (CM2.1), `channel_mlp_dropout` (identity in eval, so parity
  says nothing), AdaIN through `FNO` (2.0.0's `FNO` passes no `ada_in_features`; the block
  is covered by CM1.3b).

## Tests that prove it

- Parity (trusted path: 2.0.0's output in the fixture): `fno1d_f32`, `fno2d_f32` and the
  six sweeps within D3 (1e-5); `fno1d_f64`, `fno2d_f64` `#[ignore]`d with the measured
  error (Flex's f64 `rfft` uses f32 twiddles, 2.0.0's grid and spectral weights are f32).
- A control: one converted corner shifted (the high corner given the low one's rows)
  fails D3, so the map is not passing vacuously.
- Unit tests in `models/fno/tests/`: each new option absent builds no new parameter (the
  record's parameter count unchanged); `conv_bias` adds `n_layers · W`; `max_modes` below
  `modes` panics; `separable` with `hidden_channels` W stores `W` weights per mode.
- Constraint 1: `tests/fno_bit_identity.rs` dumps on `main` and the branch, byte-identical
  (`forward`, `phase5_epoch`, `gauss_newton_epoch`); the existing suite passes with test
  code edited only to complete `FNOConfig` literals with `None`.

## Checks

`/verify` (fmt, clippy default, `--examples`, `--all-targets`, `--features datagen
--all-targets`, `cargo test`, `cargo test --features datagen`, `cargo doc --no-deps`, the
bit-identity dumps against `main`), `uv run python generate.py --check` in `tools/parity`
(the committed fixtures, old and new, reproduced bit for bit), and `cargo test --features
metal --test gpu_parity` (the forward changes order of nothing when the options are off,
but the FNO is backend-sensitive). `CARGO_TARGET_DIR=~/sciml-rs-target/cm1-5`.

## VM runs

None.

## Constraints and risks

- Constraint 1: the new fields are appended after every existing one and are empty or
  `false` when absent, so no existing parameter's path, order or value changes; the forward
  takes the same operations in the same order when every option is absent (the bias add and
  the stabiliser are skipped, not applied as zeros or identities).
- 2D objective: untouched (no change to training, loaders or the default FNO).
- Risk: 2.0.0 adds the convolution bias inside the padded domain; with padding that is
  CM1.5b's to check, not this sub-task's (no fixture here pads).

## Pending runs

None.

## Open questions

- `fno_skip=None` (see "Not represented").

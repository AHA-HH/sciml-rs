# Parity fixtures from `neuraloperator` 2.0.0

The Python side of the library objective (`docs/design/neural-operator-library.md`,
decision D2): a locked environment with `neuraloperator` 2.0.0 and a generator that writes
output-parity fixtures to `tests/fixtures/parity/`. It stays outside CI; the Rust tests read
the committed fixtures and need no Python.

## Setup

[uv](https://docs.astral.sh/uv/) installs Python 3.12 and the locked packages into
`tools/parity/.venv/`:

```sh
cd tools/parity
uv sync --locked
uv run python generate.py            # write every fixture
uv run python generate.py --check    # regenerate in memory, compare bit for bit; exit 1 on a difference
```

## Pins

`pyproject.toml` pins `neuraloperator==2.0.0`, `torch==2.9.0`, `tensorly==0.9.0`,
`tensorly-torch==0.5.0`, `torch-harmonics==0.7.4` and `numpy==2.3.4`; `uv.lock` pins
everything else, with hashes. The rule: the tag's own pin where it gives one
(`torch_harmonics==0.7.4`, its `doc` extra; 2.0.0 does not declare torch at all),
otherwise the last release before the tag's commit date (`b307030d`, 2025-10-22). The
PyPI wheel's Python code is the tag's (`diff -r` against a clone of tag `2.0.0`; only data
files, absent from the wheel, differ).

## Fixtures

`tests/fixtures/parity/<config>_<dtype>.npz`, one per entry of `CONFIGS` in
`generate.py` and dtype (`f32`, `f64`), with a `.json` sidecar.

- `.npz` keys: `input` and `output` (`(batch, channels, *grid)`), and every tensor of the
  model's `state_dict` as `param.<2.0.0 name>`. Every array is `f32` or `f64`, as
  `NpzFileReader` reads. A complex tensor is stored as `torch.view_as_real`: a trailing
  axis of 2, (real, imaginary); the sidecar's `complex_params` lists them. The non-tensor
  `_metadata` entry is left out.
- The model is built and the input drawn in f32 (seeds in the sidecar); the f64 fixture
  casts both, because 2.0.0 creates spectral weights as `cfloat` whatever the default
  dtype. So the f32 and f64 fixtures of a configuration share weights and input.
- The output is the forward on the CPU in eval mode under `torch.no_grad`, with
  `torch.use_deterministic_algorithms(True)`.
- After writing, the generator reloads each file into a fresh model and checks the output
  is reproduced bit for bit, and it asserts the D8 bounds (P ≤ 10⁴ real parameters, file
  ≤ 1 MB).

| Fixture | Model | P (real) | Size |
| --- | --- | --- | --- |
| `fno1d_{f32,f64}` | `FNO(n_modes=(8,), in=1, out=1, hidden=8)`, 2.0.0's defaults otherwise; input (2, 1, 64) | 3,529 | 28 kB, 43 kB |
| `fno2d_{f32,f64}` | `FNO(n_modes=(4, 4), in=1, out=1, hidden=8)`, 2.0.0's defaults otherwise; input (2, 1, 32, 32) | 7,129 | 58 kB, 103 kB |
| `legacy_spectral_conv1d_{f32,f64}` | legacy `SpectralConv(3, 4, n_modes=(12,))` (`layers/legacy_spectral_convolution.py`), defaults otherwise (bias on); input (2, 3, 50) | 148 | 4 kB, 8 kB |
| `legacy_spectral_conv2d_{f32,f64}` | legacy `SpectralConv(3, 4, n_modes=(8, 6))`; input (2, 3, 24, 21) | 580 | 32 kB, 62 kB |
| `legacy_spectral_conv3d_{f32,f64}` | legacy `SpectralConv(2, 3, n_modes=(6, 4, 4))`; input (2, 2, 12, 10, 9) | 579 | 47 kB, 93 kB |
| `spectral_conv2d_{f32,f64}` | centred `SpectralConv(3, 4, n_modes=(8, 6))` (`layers/spectral_convolution.py`), defaults otherwise (bias, `fft_norm="forward"`); input (2, 3, 24, 21) | 772 | 32 kB, 64 kB |
| `spectral_conv2d_backward_{f32,f64}` | as `spectral_conv2d`, `fft_norm="backward"` | 772 | 32 kB, 64 kB |
| `spectral_conv2d_odd_{f32,f64}` | centred `SpectralConv(3, 4, n_modes=(7, 6))`; input (2, 3, 24, 21) | 676 | 32 kB, 63 kB |
| `spectral_conv3d_{f32,f64}` | centred `SpectralConv(2, 3, n_modes=(6, 5, 4))`; input (2, 2, 12, 10, 9) | 1,083 | 49 kB, 96 kB |
| `spectral_conv2d_separable_{f32,f64}` | centred `SpectralConv(4, 4, n_modes=(8, 6), separable=True)`; input (2, 4, 24, 21) | 260 | 34 kB, 68 kB |
| `spectral_conv2d_max_modes_{f32,f64}` | centred `SpectralConv(3, 4, n_modes=(6, 6), max_n_modes=(10, 8))`; input (2, 3, 24, 21) | 1,924 | 37 kB, 73 kB |
| `spectral_conv2d_odd_extent_{f32,f64}` | as `spectral_conv2d` on input (2, 3, 23, 21): an odd extent, which corners cannot represent | 772 | 31 kB, 61 kB |
| `channel_mlp1d_{f32,f64}` | `ChannelMLP(4, hidden_channels=2)`; input (2, 4, 7) | 22 | 2 kB, 3 kB |
| `channel_mlp3d_{f32,f64}` | `ChannelMLP(3, out_channels=5, hidden_channels=4, n_layers=3, non_linearity=torch.tanh)`; input (2, 3, 5, 4, 3) | 61 | 6 kB, 10 kB |
| `linear_skip3d_{f32,f64}` | `skip_connection(3, 5, n_dim=3, skip_type="linear")`; input (2, 3, 5, 4, 3) | 15 | 5 kB, 9 kB |
| `soft_gating2d_{f32,f64}` | `skip_connection(4, 4, n_dim=2, skip_type="soft-gating")`, weight redrawn; input (2, 4, 6, 5) | 4 | 3 kB, 5 kB |
| `instance_norm1d_{f32,f64}` | `InstanceNorm()`; input (2, 4, 9) | 0 | 1 kB, 2 kB |
| `group_norm3d_{f32,f64}` | `nn.GroupNorm(1, 4)`, affine redrawn; input (2, 4, 5, 4, 3) | 8 | 5 kB, 9 kB |
| `ada_in2d_{f32,f64}` | `AdaIN(embed_dim=3, in_channels=4)`, its embedding a seeded buffer (`embedding`); input (2, 4, 6, 5) | 6,152 | 28 kB, 55 kB |
| `fno_blocks_post1d_f32`, `fno_blocks_pre1d_f32` | `FNOBlocks(4, 4, n_modes=(8,), n_layers=2)`, post- and pre-activation; input (2, 4, 20) | 340 | 8 kB |
| `fno_blocks_instance_{post,pre}2d_f32` | the same, `n_modes=(6, 4)`, `norm="instance_norm"`; input (2, 4, 12, 10) | 852 | 17 kB |
| `fno_blocks_group_post3d_f32` | `n_modes=(4, 4, 4)`, `norm="group_norm"`, `fno_skip="soft-gating"`, `channel_mlp_skip="linear"`, `channel_mlp_expansion=1.0`; input (2, 4, 8, 7, 6) | 2,200 | 39 kB |
| `fno_blocks_group_pre3d_f32` | `norm="group_norm"`, both skips `"identity"`, pre-activation; input (2, 4, 8, 7, 6) | 2,124 | 38 kB |
| `fno_blocks_ada_in_{post,pre}2d_f32` | `FNOBlocks(2, 2, n_modes=(6, 4), n_layers=1, norm="ada_in", ada_in_features=2)`, one seeded embedding buffer; input (2, 2, 12, 10) | 7,285 | 39 kB |
| `fno_blocks_plain1d_f32` | `use_channel_mlp=False`, `fno_skip="soft-gating"` (GELU twice per layer); input (2, 4, 20) | 264 | 4 kB |

The legacy layer's `forward` builds its spectrum as `cfloat`, so its f64 fixtures are only
f32-accurate (library CM1.1). tltorch stores its weights as real tensors with the trailing
(re, im) axis, so their sidecars list no `complex_params`.

Library CM1.3a: parts whose 2.0.0 initialisation is a constant (soft gating's ones, group
norm's ones and zeros; `REDRAW`) have their parameters redrawn from the weight seed's normal,
so their fixtures test the map. A function argument (`non_linearity`) is named in the sidecar
by its `__name__`.

Library CM1.3b: `FNOBlocks` runs as `FNOBlocksRun`, every layer index in turn as 2.0.0's
`FNO.forward` does, with the legacy convolution (`legacy_conv`: `bias=False`, as the crate's
`SpectralConv` has no bias; the new layer's keywords `FNOBlocks` passes are dropped at their
defaults) and, for AdaIN, one embedding set on every norm. Only parameters whose 2.0.0 init
is a constant are redrawn (soft gating, group norm), and the block fixtures are f32 only
(`F32_ONLY`): the legacy layer computes in `cfloat` and returns f32, so an f64 reference
would be f32-accurate. AdaIN's 512-wide MLP keeps its fixtures to one layer and 2 channels
(D8).

Torch's CPU kernels can differ in the last bits between machines, so `--check` is a
reproducibility check on one machine; the committed files are the reference, and the
sidecar names the platform that wrote them.

The centred layer's `forward` also builds its spectrum as `cfloat` (library CM1.2), and on an
axis of odd extent it applies `fftshift`, not `ifftshift`, before the inverse, moving its
output spectrum by one bin (`spectral_conv2d_odd_extent`).

## Loss fixtures (CM1.6a)

`generate_losses.py` (same `--check`) writes `tests/fixtures/parity/losses{1,2,3}d_{f32,f64}`:
a seeded pair `x`, `y` (2, 2, *grid) and a vector field `v` (2, D, *grid) on the grids 32,
16 × 12 and 8 × 6 × 5, drawn in f32 and cast for f64, and 2.0.0's outputs on them:
`FiniteDiff` (h = 0.1, 0.2, 0.3) as `fd.d<axis><order>.<bc>` and
`fd.{laplacian,gradient,divergence,curl}.<bc>`; `H1Loss` and `HdivLoss` (measure 1, 2, 0.5) as
`{h1,hdiv}.{rel,abs}.<bc>.sum` and `h1.rel.per.mean`; `PointwiseQuantileLoss(alpha=0.1)` as
`quantile.{sum,mean}`; `MSELoss` as `mse`; `LpLoss(d, p=2).rel` as `lp.rel`. `<bc>` is `per`
(all periodic), `non` (none) or `mix` (x one-sided; D ≥ 2). Scalars are stored with shape
(1,). The sidecar lists the settings; the Rust side is `tests/parity_losses*.rs`.

`meta_losses_{f32,f64}` (CM1.6b), from the same script: `WeightedSumLoss([H1Loss(d=1),
MSELoss()])` on a pair (2, 2, 32) with default and (0.3, 0.7) weights (`wsum.*`);
`FieldwiseAggregatorLoss` on (16, 3), `u` channel 0 with `MSELoss`, `p` channel 2 with
`PointwiseQuantileLoss(0.1)` (`fieldwise`); and `SoftAdapt` / `Relobralo` over 6 steps of 3
seeded losses (`table`), with and without weights and Relobralo also at α = 0.9, β = 0.5,
τ = 2: per step the aggregate, the weights and the gradient with respect to each loss
(`<name>.{loss,weights,grad}`), and Relobralo's recorded Bernoulli draws (`<name>.rho`). The
f64 fixture runs with torch's default dtype f64, so 2.0.0's buffers are f64 too.

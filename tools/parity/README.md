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

The legacy layer's `forward` builds its spectrum as `cfloat`, so its f64 fixtures are only
f32-accurate (library CM1.1). tltorch stores its weights as real tensors with the trailing
(re, im) axis, so their sidecars list no `complex_params`.

Torch's CPU kernels can differ in the last bits between machines, so `--check` is a
reproducibility check on one machine; the committed files are the reference, and the
sidecar names the platform that wrote them.

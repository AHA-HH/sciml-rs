# CM1.1 Legacy spectral convolution parity

Started 2026-11-25 on `operators/cm1-1`. Roadmap: "CM1.1 Legacy spectral convolution
parity. 2.0.0's corner-layout layer against `SpectralConv`, 1D to 3D. *Needs:* CM0.5."
Proposal: §10; §1.4, §2, §5, §6.3. D3 (as amended 2026-11-25), D8.

## Goal

Show that the crate's `SpectralConv<R>` reproduces 2.0.0's legacy `SpectralConv`
(`layers/legacy_spectral_convolution.py`) in 1D, 2D and 3D, loaded with the same weights,
within D3.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)

- Roadmap: "2.0.0's corner-layout layer against `SpectralConv`, 1D to 3D." (no separate
  *Done when*).
- Proposal §6.3: "The legacy layer maps corner to corner, so it is the first test (CM1.1)."
- Proposal §5 and D3 as amended 2026-11-25: relative max error max |a − e| / max |e| ≤ 1e-5
  in f32 on the CPU; each f64 test `#[ignore]`d with its measured error recorded.
- Constraint 1 (§1.4): every existing FNO output bit for bit identical.

## Read first

- 2.0.0 `neuralop/layers/legacy_spectral_convolution.py` (in `tools/parity/.venv`):
  `SpectralConv.__init__` and `forward`.
- `src/neural_operators/layers/spectral_convolution.rs` (`corner_ranges`, `forward`).
- `tests/parity_common/mod.rs` (CM0.5's harness), `tests/parity_fno1d{,_f64}.rs`.
- `tools/parity/generate.py`, `README.md`.

What the source says, and how it maps:

| 2.0.0 legacy | this crate |
| --- | --- |
| `fft_norm="backward"` (default) | Burn's unnormalised forward, 1/n inverse (CONVENTIONS §4): the same, no rescaling |
| weight `(in, out, *[m // 2 for m in n_modes])`, last axis included | `[in, out, modes..]` with `modes = n_modes // 2` per axis |
| `weight.<i>.tensor`: real, trailing (re, im) axis (tltorch `ComplexDense`) | `weights_re.<c>`, `weights_im.<c>` via `Take::Re` / `Take::Im` |
| corners in `itertools.product` order: axis 0 most significant | bit `j` of the corner index is axis `j` (CONVENTIONS §5) |
| `bias` `(n_layers, out, 1, …)`, added after the inverse (default `bias=True`) | none in `SpectralConv`: a test-only `Param` added after `forward` |

The corner orders agree in 1D and 2D and differ in 3D: 2.0.0's `weight.1` (low axis 0, high
axis 1) is the crate's corner 2, and `weight.2` its corner 1.

2.0.0's legacy `forward` allocates `out_fft` as `cfloat` whatever the input's dtype, so an
f64 forward is f32-accurate (measured: 7.8e-8 against a numpy f64 evaluation of the same
layer, 2D). Its f64 fixtures cannot show D3's 1e-12 whatever Burn does.

## Changes

No change under `src/`; no dependency; no convention.

- `tools/parity/generate.py`: three entries in `CONFIGS`, 2.0.0's legacy `SpectralConv` at
  its defaults (bias on) with in ≠ out and non-power-of-two, odd and even extents (Bluestein
  paths):
  - `legacy_spectral_conv1d`: in 3, out 4, `n_modes=(12,)`, input (2, 3, 50);
  - `legacy_spectral_conv2d`: in 3, out 4, `n_modes=(8, 6)`, input (2, 3, 24, 21);
  - `legacy_spectral_conv3d`: in 2, out 3, `n_modes=(6, 4, 4)`, input (2, 2, 12, 10, 9).
  Fixtures `tests/fixtures/parity/legacy_spectral_conv{1,2,3}d_{f32,f64}`; the existing four
  regenerated bit for bit (checked by `git status`). README table updated.
- `tests/parity_legacy_spectral_conv.rs`: `LegacyConv<R>` (a `SpectralConv<R>` and a bias
  `Param`), its three hand-written maps, and the f32 parity tests.
- `tests/parity_legacy_spectral_conv_f64.rs`: the f64 tests, `#[ignore]`d (own binary:
  Flex's process-wide f64 dtype).
- Feature matrix row of the legacy `SpectralConv`: stays **partial** (dense corner layer at
  parity; bias, `incremental_n_modes`, `joint_factorization`, factorisations, separable,
  `resolution_scaling_factor` missing), its note naming the tests.

## Tests that prove it

- `legacy_spectral_conv{1,2,3}d_f32_matches_2_0_0`: each f32 fixture through its map,
  output within D3's f32 bound (1e-5), on Flex. The acceptance tests.
- `legacy_3d_corner_order_matters`: the 3D fixture loaded with the corners in 2.0.0's
  order unpermuted fails the f32 bound, so the 3D test checks the permutation.
- `legacy_spectral_conv{1,2,3}d_f64_matches_2_0_0`: against 1e-12, `#[ignore]`d (2.0.0's
  `cfloat` `out_fft`, and Flex's f64 `rfft` twiddles); errors recorded below.

## Checks

`/verify` (fmt, clippy default / `--examples` / `--all-targets` / `--features datagen
--all-targets`, `cargo test`, `cargo test --features datagen`, `cargo doc --no-deps`);
`cargo test --test parity_legacy_spectral_conv_f64 -- --ignored` for the f64 numbers;
`uv run python generate.py --check`. No `src/` change, so the bit-identity dumps are not
rerun.

## VM runs

None.

## Constraints and risks

- Constraint 1: nothing under `src/` changes.
- The bias wrapper is test code; the library's layer with bias is CM1.2/CM1.3's.
- Not covered: the `SpectralConv1d/2d/3d` subclasses (same weights and corners, written
  "for reference only"; the 2D and 3D ones cast to f32), `incremental_n_modes`,
  `joint_factorization`, factorised weights, `resolution_scaling_factor`.
- CM0.5's follow-ups for CM1.1: the sidecar's `complex_params` is empty for this layer
  (tltorch stores the weights real), so reading the sidecar adds nothing here; the 2.0.0
  grid is not used by a bare layer. Both move on to the first task that needs them.

## Pending runs

None.

## Results

(filled after the run)

## Open questions

None.

# CM1.5b Domain padding and resampling

Started 2026-11-27 on `operators/cm1-5b`. Roadmap: "CM1.5b Domain padding and resampling"
(the split of CM1.5). Proposal: §10; §1.4, §2, §5, §6.3. D3 (as amended 2026-11-25), D5, D8.

## Goal

2.0.0's `DomainPadding`, its per-layer `resolution_scaling_factor` and `forward`'s
`output_shape` as `FNO` options, with 2.0.0's `resample` for the skips, each off by
default, so that 2.0.0 fixtures with them pass through the crate's `FNO` and today's FNO is
unchanged bit for bit.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)

- *Done when:* "fixtures with padding and with scaling factors, 1D–3D, pass through `FNO`
  within D3 in f32; the dumps equal `main`'s."
- Proposal §5: output parity within D3 (max |a − e| / max |e| ≤ 1e-5 in f32 on the CPU).
- Proposal §1.4, constraint 1: every existing FNO output bit for bit identical; a part made
  configurable defaults to today's behaviour.

## Read first

- 2.0.0 `neuralop/layers/padding.py` (`DomainPadding`), `layers/resample.py` (`resample`),
  `layers/spectral_convolution.py` (`transform`, `forward`'s `irfftn(s=…)`),
  `layers/fno_block.py` (the skips through `convs[i].transform`), `models/fno.py`
  (`forward`'s `output_shape`, `DomainPadding` built with the FNO's scaling factor),
  `utils.py` (`validate_scaling_factor`).
- `src/neural_operators/models/fno/mod.rs`, `layers/spectral_convolution.rs`
  (`forward`, `ifft_ctensor`); Burn's `interpolate` (bilinear and bicubic, `a = -0.75`,
  `AlignCorners`), `burn_signal::irfft` with `n`.
- `tests/parity_fno.rs`, `tests/parity_common/fno.rs`, `tools/parity/generate.py`.

## 2.0.0's behaviour, as read

- `DomainPadding`: per axis `p = round(f · s)` (Python's `round`, half to even) zeros on
  **both** sides (`F.pad`), after the lifting; the crop after the last layer removes
  `round(p · r)` per side, `r` the FNO's scaling factor as given (a scalar), so 2.0.0 runs
  padding with scaling only when one layer scales (else its unpad lookup raises
  `KeyError`), and not at all with a per-layer list (it zips the list with the axes).
- `SpectralConv` with a scaling factor `r` or an `output_shape`: the spectrum is built at
  the input's size, then `irfftn(s=out)`, which trims or zero-pads each axis **at its end**
  in the standard layout (torch's `n`), `fft_norm="forward"`.
- The skips go through `transform`: `resample` to the layer's output shape; linear
  (`align_corners=True`) in 1D, bicubic (`align_corners=True`) in 2D, a Fourier resize for
  three or more axes (`rfftn` forward norm, the `min` modes per axis copied, low `m // 2`
  and high `⌈m / 2⌉` on the non-last axes).
- `forward(x, output_shape=t)` with a tuple sets the last layer's output shape.

## Changes

- `layers/resample.rs` (new): `resample(x, out)`, channels-first `[B, C, s..]` to
  `[B, C, out..]`, 2.0.0's three methods; the identity when `out` is the input's size.
- `layers/spectral_convolution.rs`: `SpectralConv::forward_to(x, out)`, the output at
  `out`'s spatial size (2.0.0's spectrum trim or zero-pad at the end of each axis, the
  amplitude kept as with `fft_norm="forward"`); `forward(x)` calls it with the input's size
  and takes the same operations as today; `fft_ctensor`/`ifft_ctensor` `pub(crate)` for
  `resample`.
- `models/fno/mod.rs`: `FNOConfig::domain_padding: Option<Vec<f64>>` (a fraction per
  spatial axis, symmetric), `FNOConfig::resolution_scaling_factor: Option<Vec<Vec<f64>>>`
  (per layer, per axis); `FNO::forward_with_output_shape(x, out)`. `None` is today's model.
  With both options the crop removes `round(p · r)` per side, `r` the one scaled layer's
  factor (2.0.0's rule; more than one scaled layer panics, as 2.0.0 raises). `init` panics on
  both paddings set, on CONVENTIONS §3's `padding` with scaling (its crop restores the
  unscaled extents; code-quality review F1), or on a wrong length;
  `forward_with_output_shape` panics with either padding (2.0.0 raises `KeyError` unless the
  shape is the padded one's).
- No new boolean option (the author's rule of 2026-11-27 needs none here). Conventions:
  none changed; CONVENTIONS §3's `padding` stays as it is, beside the new option.

## Tests that prove it

- Parity (`tests/parity_fno.rs`, f32, D3, Flex): new fixtures from `generate.py` (f32 only,
  in `SWEEPS`): `fno1d_padding`, `fno2d_padding` (a fraction per axis), `fno3d_padding`,
  `fno1d_scaling` (factors 1.5, 1, 0.5, 1), `fno2d_scaling` (per axis, up and down),
  `fno3d_scaling` (the Fourier resample), `fno2d_padding_scaling` (one layer, the only
  combination 2.0.0 runs), `fno2d_output_shape`.
- Controls: the padding fixture with padding left off fails D3; the scaling fixture with
  the spectrum resized symmetrically instead of at the end would fail (checked once by
  hand, recorded, not a test).
- Unit tests: `resample` is the identity at the same size, exact on linear data in 1D,
  keeps a band-limited signal in 3D; `forward_to` at the input size equals `forward` bit
  for bit; the new `init` and forward panics.
- Constraint 1: `tests/fno_bit_identity.rs` dumps on `main` and the branch, `cmp`.

## Checks

`/verify` (fmt, clippy ×4, `cargo test`, `cargo test --features datagen`, `cargo doc`,
the bit-identity dumps against `main`); `uv run python generate.py --check` in
`tools/parity`. `CARGO_TARGET_DIR=~/sciml-rs-target/cm1-5b` for every cargo command.

## VM runs

None.

## Constraints and risks

- Constraint 1: the new paths run only with an option set; `SpectralConv::forward`'s
  default path takes the same operations. Shown by the dumps.
- Torch's C2R inverse ignores the imaginary part of the DC and Nyquist bins; if Burn's
  `irfft` treats a resized half-spectrum differently, the scaling fixtures show it.
- The parallel CM1.5c adds `FNOConfig` fields too: conflicts in struct literals are
  mechanical, resolved by merging `origin/main`.

## Pending runs

None.

## Open questions

1. CONVENTIONS §4 says `irfft`'s treatment of a nonzero imaginary part in the DC and
   Nyquist bins is not fixed by a test, and that a design relying on it adds one and records
   it there. `forward_to` and `resample` rely on it when a resize breaks Hermitian symmetry
   (`fno3d_scaling`'s 6 → 3 axis); the fixture shows Flex and Metal agree with torch. Add a
   §4 verification entry naming `fno3d_scaling_matches_2_0_0` (no convention changes, no
   version bump)? Asked in the PR, not edited here.

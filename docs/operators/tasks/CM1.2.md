# CM1.2 Centred layout, separable, `max_n_modes`

Started 2026-11-25 on `operators/cm1-2`. Roadmap: "CM1.2 Centred layout, separable,
`max_n_modes`. The weight converter of proposal §6.3; separable weights; mode truncation
below the stored maximum. *Needs:* CM1.1, D5." Proposal: §10; §1.4, §2, §5, §6.3. D3 (as
amended 2026-11-25), D5 (signed off 2026-11-25 by the author, recorded in #153: A, corners
internally, CONVENTIONS §5 unchanged; 2.0.0's centred weights converted at fixture load, the
FFT normalisation and the mode counts included; an odd centred mode count tested
explicitly), D8.

## Goal

Load 2.0.0's centred `SpectralConv` (`layers/spectral_convolution.py`) into the crate's
corner-layout `SpectralConv<R>`, dense and separable, at its stored maximum and truncated
below it, and match its output within D3.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)

- Roadmap: "The weight converter of proposal §6.3; separable weights; mode truncation below
  the stored maximum." (no separate *Done when*).
- Proposal §6.3: "A fixture loader maps 2.0.0's parameter names and shapes onto this crate's
  modules: the centred spectral tensor to the 2^(D−1) corners (and `n_modes` counted over
  both signs to modes per corner), complex tensors to real and imaginary pairs."
- D5: corners internally; conversion at fixture load, FFT normalisation and mode counts
  included; an odd centred mode count tested; any 2.0.0 configuration corners cannot
  represent said plainly in the PR.
- D3 as amended: max |a − e| / max |e| ≤ 1e-5 in f32 on the CPU; f64 tests `#[ignore]`d
  with the error recorded.
- Constraint 1 (§1.4): every existing FNO output bit for bit identical; every new option
  off by default.

## Read first

- 2.0.0 `neuralop/layers/spectral_convolution.py`: `SpectralConv.__init__`, the `n_modes`
  setter, `forward`; `_contract_dense`, `_contract_dense_separable`.
- `src/neural_operators/layers/spectral_convolution.rs` (`corner_ranges`, `forward`).
- `tests/parity_common/mod.rs` (CM0.5), `tests/parity_common/legacy_conv.rs` (CM1.1).

What the source says, and how it maps:

| 2.0.0 centred | this crate |
| --- | --- |
| weight `(in, out, *max_n_modes)`, `max_n_modes` defaulting to `n_modes` after the setter (last axis `n // 2 + 1`); an explicit `max_n_modes` is stored as given | `[in, out, modes..]` per corner; the stored maximum is the weights' shape |
| non-last axes `fftshift`ed: weight index `j` of a used extent `K` is frequency `j − K // 2`, so `K // 2` negative and `K − K // 2` non-negative frequencies | low corner frequencies `0..m`, high corner `−m..−1`, the same `m` |
| last axis (rfft): index `k` is frequency `k` | the same |
| `fft_norm="forward"` (1/N forward, 1 inverse) | `backward` (1, 1/N); the layer is one forward and one inverse transform at the same N, so the two products are equal and the weights need no rescaling |
| `separable=True`: weight `(C, *modes)`, `x * weight` per channel | new: weights `[1, C, modes..]`, an element-wise complex product |
| `n_modes` below `max_n_modes`: the centred sub-block, `start // 2` trimmed in front | new `with_modes`: low `0..m'`, high the last `m'` of the stored high corner, last axis `0..m'` |
| bias `(out, 1, …)` after the inverse | none in `SpectralConv`: a test-only bias as in CM1.1 |

The converter, per non-last axis of extent `K`: `m = ⌈K/2⌉` per corner; the low corner is
indices `K//2 .. K`, the high corner indices `K//2 − m .. K//2`, which for odd `K` starts one
before 0: that row is zero (frequency −m, which 2.0.0 does not keep, contributes nothing).

## Changes

- `src/neural_operators/layers/spectral_convolution.rs`:
  - `separable: bool` (a `Module` constant, not a tensor: Burnpack files are unchanged);
    `SpectralConv::new_separable(device, channels, modes, init)`, weights `[1, C, modes..]`;
    `forward` takes an element-wise complex product for it.
  - `with_modes(self, modes) -> Self`: the active modes, at most the stored ones per axis;
    `forward` slices each corner's weights with `corner_ranges` when they differ, and not at
    all when they agree (the default), so the existing path is untouched.
- Tests side (D5's "at fixture load"), `tests/parity_common/mod.rs`, additive:
  `centred_to_corners` (the pure conversion) and `Fixture::centred_to_corners(name, d)`,
  which replaces `param.<name>` by `param.<name>.corner<c>` in the fixture as loaded (an
  override map read by `Fixture::array`), so `load_into`'s completeness checks hold as they
  are.
- `tests/parity_common/centred_conv.rs`: the test-only bias wrapper, cases and maps.
- `tests/parity_spectral_conv.rs` (f32), `tests/parity_spectral_conv_f64.rs` (`#[ignore]`d).
- `tools/parity/generate.py`: 2.0.0's centred `SpectralConv` at its defaults (bias,
  `fft_norm="forward"`) except where named:
  - `spectral_conv2d`: in 3, out 4, `n_modes=(8, 6)`, input (2, 3, 24, 21);
  - `spectral_conv2d_backward`: the same with `fft_norm="backward"` (normalisation check);
  - `spectral_conv2d_odd`: `n_modes=(7, 6)`, input (2, 3, 24, 21) (odd count);
  - `spectral_conv3d`: in 2, out 3, `n_modes=(6, 5, 4)`, input (2, 2, 12, 10, 9) (even and
    odd counts, 3D corner order);
  - `spectral_conv2d_separable`: in = out = 4, `separable=True`, `n_modes=(8, 6)`, input
    (2, 4, 24, 21);
  - `spectral_conv2d_max_modes`: in 3, out 4, `n_modes=(6, 6)`, `max_n_modes=(10, 8)`, input
    (2, 3, 24, 21) (stored (10, 8), used (6, 4));
  - `spectral_conv2d_odd_extent`: `spectral_conv2d` on input (2, 3, 23, 21) (added during
    implementation, see Results).
  The ten existing fixtures regenerated bit for bit (`--check`). README table.
- `docs/operators/feature-matrix.md`: the centred `SpectralConv` and `BaseSpectralConv` rows.
- `FNOConfig` is not touched: wiring `separable` and `max_n_modes` into the FNO is CM1.5's
  (full FNO parity), with the block parts of CM1.3.

No dependency, no convention change (CONVENTIONS §5 stays the internal form, as D5).

## Tests that prove it

- Parity, against the fixtures (2.0.0 on the CPU), D3's f32 bound 1e-5:
  `spectral_conv2d{,_backward,_odd,_separable,_max_modes}_f32_matches_2_0_0`,
  `spectral_conv3d_f32_matches_2_0_0`. The backward fixture goes through the same converter
  as the forward one, so the FFT normalisation needs no conversion; the odd fixtures show the
  zero row; the max-modes one loads the stored (10, 8) and truncates with `with_modes`.
- `odd_extent_is_not_representable`: the odd-extent fixture fails D3 (added during
  implementation, see Results).
- `centred_layout_needs_conversion`: the 2D fixture with its centred weight split into
  halves unconverted fails D3 (the conversion is load-bearing).
- `centred_to_corners` on a hand-built array, even and odd `K`, 2D and 3D: exact values.
- Unit tests in `spectral_convolution.rs`: `with_modes` equals a layer built at the smaller
  modes with the matching sub-weights (bit for bit); `with_modes` at the stored modes is the
  untruncated output (bit for bit); `with_modes` panics above the stored maximum and on the
  wrong length; `with_modes` on separable weights (to 1e-6 relative: bit for bit on Flex, a few ulps apart on Metal, where the fused element-wise kernel rounds the strided slice differently); separable equals a dense layer with diagonal weights (to rounding, 1e-6).
- f64: the same six, `#[ignore]`d, errors recorded (2.0.0's centred `forward` also builds its
  spectrum as `cfloat`).
- Constraint 1: `tests/fno_bit_identity.rs` on `main` and the branch, dumps `cmp`ed
  (`/verify`), plus every existing test.

## Checks

`/verify` (fmt, clippy default / `--examples` / `--all-targets` / `--features datagen
--all-targets`, `cargo test`, `cargo test --features datagen`, `cargo doc --no-deps`, the
bit-identity dumps); `cargo test --test parity_spectral_conv_f64 -- --ignored`;
`uv run python generate.py --check`. `--features metal`: the forward path changes, so the
spectral-conv unit tests and `gpu_parity` under `metal` if it builds here.

## VM runs

None.

## Constraints and risks

- Constraint 1: the default path (dense, active = stored modes) runs the same tensor ops in
  the same order; the new fields are constants, so saved weights are unchanged.
- Not representable in corners, to state in the PR: any grid of odd extent on a non-last
  axis (found in implementation, see Results); an odd centred count equal to an odd
  extent (`K = n`: the corners would overlap at frequency ±m); an odd count *below* the
  stored maximum as a truncation (the high corner's extra row is a live weight; the realised
  slice converts instead); 2.0.0's even-maximum, odd-count trim (its weights shift by one
  frequency); `complex_data=True` (full FFT, 2^D corners); factorised weights (reconstruct to
  dense first, CM2.1); `resolution_scaling_factor` / `output_shape` (different N in and out,
  where the normalisations no longer cancel); half and mixed precision; 2.0.0's silent
  clamping of modes to a small grid (the crate panics). Under training, an odd count's zero
  row is a trainable weight, so training parity at odd counts would need a mask (CM1.8).

## Pending runs

None.

## Results

Flex, M2 Pro, `cargo test --test parity_spectral_conv -- --nocapture` and
`cargo test --test parity_spectral_conv_f64 -- --ignored --nocapture`:

| Fixture | f32 (D3 1e-5) | f64 (1e-12, `#[ignore]`d) |
| --- | --- | --- |
| `spectral_conv2d` | 3.560e-7 | 2.255e-7 |
| `spectral_conv2d_backward` | 3.204e-7 | 2.475e-7 |
| `spectral_conv2d_odd` | 3.466e-7 | 2.173e-7 |
| `spectral_conv3d` | 2.966e-7 | 2.749e-7 |
| `spectral_conv2d_separable` | 3.972e-7 | 2.732e-7 |
| `spectral_conv2d_max_modes` | 2.629e-7 | 1.500e-7 |
| `spectral_conv2d_odd_extent` | 1.771e0 (expected to fail) | – |

Negative control: the 2D centred weight cut into halves unconverted gives 1.029e0.

**Odd extents.** The first odd and 3D fixtures were on grids (2, 3, 23, 21) and
(2, 2, 12, 11, 9), with an odd extent on a shifted axis, and failed at 1.210 and 0.947. A
numpy evaluation of the layer from the fixture's weights
(`rfftn(norm="forward")`, `fftshift`, the centred contraction, then the second shift,
`irfftn`) reproduces the fixture to 7e-8 when the second shift is `fftshift`, as 2.0.0's
code has it, and gives 1.210 and 0.947, the crate's errors exactly, when it is `ifftshift`,
the inverse of the first. For even n the two are the same permutation; for odd n,
`fftshift` twice is a shift by n − 1, so 2.0.0's output frequency g carries input
frequency g + 1 on that axis. That map is not diagonal in frequency, so no corner weights
represent it. The odd-count and 3D fixtures moved to even extents (24; 12, 10), and
`spectral_conv2d_odd_extent` keeps the case as a test that fails D3. This binds CM1.5 (2.0.0's
FNO uses this layer, so FNO parity holds on even grids only) and CM1.8 (the Poisson
learning grids s = 64, 128, 256 are even).

The f64 errors are at the f32 level, as the reference is (2.0.0's `cfloat` spectrum), as in
CM1.1. `generate.py --check`: the ten earlier fixtures unchanged.

## Open questions

None.

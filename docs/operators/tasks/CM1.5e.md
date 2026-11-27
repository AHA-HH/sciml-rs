# CM1.5e The spectral layer and the FFTs

Started 2026-11-27 on `operators/cm1-5e`. Roadmap: CM1.5e (split 2026-11-27 from CM1.5d).
Proposal: §10 (M1); §1.4, §2 (constraint 1).

## Goal
Tidy the spectral layer: `spectral_convolution.rs` split into a folder with its tests in a
file of their own, and the N-D FFTs out of `SpectralConv` into `utils/fft.rs`, with one
inverse that resizes and scales.

## Acceptance (roadmap *Done when*)
The dumps equal `main`'s and `gpu_parity` passes on Metal.

Items of the line, each with its check:
1. CM1.2 / CM1.5b F6: `layers/spectral_convolution.rs` becomes
   `layers/spectral_convolution/{mod,tests}.rs`, as `optim/soap/` is; the module path
   `layers::spectral_convolution` unchanged (no caller edited).
2. CM1.5b F2: `fft_ctensor` and `ifft_ctensor` free functions in `utils/fft.rs`
   (`pub(crate)`, beside `icfft_full_spectrum`); `SpectralConv` and `resample` import them,
   `resample` no longer reaches into `SpectralConv`. Their tests (round trips, the host DFT
   at Bluestein sizes) move with them.
3. CM1.5b F3: `ifft_ctensor_to(re, im, in_dims, out_dims)`, the inverse to `out_dims` at
   2.0.0's forward-norm amplitude (`N_out / N_in`), called by `SpectralConv::forward_to` and
   `resample`'s Fourier resize; each keeps its own spectrum resize (end trim vs corners, two
   2.0.0 behaviours) and its same-size early return (the default path's ops unchanged).

## Read first
`layers/spectral_convolution.rs`, `layers/resample.rs`, `utils/fft.rs`,
`optim/soap/mod.rs` (the pattern), the CM1.5b review (F2, F3, F6), CONVENTIONS §4.

## Changes
- `layers/spectral_convolution.rs` → `layers/spectral_convolution/mod.rs` and `tests.rs`
  (the `#[cfg(test)] mod tests` body, verbatim but for the FFT tests that move).
- `utils/fft.rs`: `fft_ctensor`, `ifft_ctensor`, `ifft_ctensor_to`; their tests.
- `layers/resample.rs`: calls `utils::fft`; its scale and inverse through
  `ifft_ctensor_to`.
- CLAUDE.md "Layout": the new paths. No public API change, no convention or dependency
  change. Stays in `src/` (CM1.5f holds `tests/parity_common`; CO3.3 holds `training/`).

## Tests that prove it
- Constraint 1: `fno_bit_identity` dumps (`forward`, `phase5_epoch`, `gauss_newton_epoch`,
  `lbfgs_epoch`) `cmp` against `main` (the trusted path).
- The moved unit tests unchanged in body: the spectral layer's (PyTorch/NumPy references,
  mode limits, `with_modes`, separable), the FFT round trips and the host DFT at Bluestein
  sizes.
- New `ifft_ctensor_to` test: a constant field resized from 6 × 5 to 9 × 8 stays the same
  constant (the forward-norm amplitude), and to the input's own extents equals
  `ifft_ctensor`.
- `parity_fno` (the scaling and output-shape fixtures go through `forward_to` and
  `resample`), `parity_spectral_conv`, `parity_legacy_spectral_conv`, `resample`'s tests:
  unchanged errors; on Flex and Metal; `gpu_parity` on Metal.

## Checks
`/verify` (fmt, clippy ×4, `cargo test`, `cargo test --features datagen`, `cargo doc`, the
bit-identity dumps); `cargo test --features metal --test gpu_parity --test parity_fno` and
the spectral layer's and FFT unit tests with `--features metal`.
`CARGO_TARGET_DIR=~/sciml-rs-target/CM1.5e`.

## VM runs
None.

## Constraints and risks
Constraint 1: a pure move; the default `forward` path runs the same ops in the same order.
The resized paths (`forward_to` off the input size, `resample` in 3D) compute the same
inverse and the same `mul_scalar`. Parallel lane CM1.5f edits `tests/parity_common`; only the
docs (roadmap, progress, CLAUDE.md) may conflict.

## Pending runs
None.

## Open questions
None.

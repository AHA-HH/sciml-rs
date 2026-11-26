//! Library CM1.6a–c and CM2.5 in f64: `parity_losses_common` on the f64 fixtures (D3: 1e-12). Its own
//! test binary because it configures Flex's process-wide float dtype to f64. Only
//! `FourierDiff` goes through an FFT: its test is `#[ignore]`d (D3 as amended), as Flex's f64
//! `rfft` uses f32 twiddles.

#[cfg(feature = "flex")]
mod flex_f64;
// Each test binary uses part of the shared harness.
#[allow(dead_code)]
mod parity_common;
#[cfg(feature = "flex")]
mod parity_losses_common;

#[cfg(feature = "flex")]
use flex_f64::flex_f64;
#[cfg(feature = "flex")]
use parity_losses_common::{run, run_equation, run_fourier, run_meta, run_non_uniform};

#[cfg(feature = "flex")]
#[test]
fn losses1d_f64_match_2_0_0() {
    run::<1, 3, 4, 2>("losses1d_f64", &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
fn losses2d_f64_match_2_0_0() {
    run::<2, 4, 5, 3>("losses2d_f64", &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
fn losses3d_f64_match_2_0_0() {
    run::<3, 5, 6, 4>("losses3d_f64", &flex_f64());
}

/// CM1.6b: the aggregators against `meta_losses_f64`.
#[cfg(feature = "flex")]
#[test]
fn meta_losses_f64_match_2_0_0() {
    run_meta("meta_losses_f64", &flex_f64());
}

/// CM2.5: the equation losses against `equation_losses_f64`.
#[cfg(feature = "flex")]
#[test]
fn equation_losses_f64_match_2_0_0() {
    run_equation("equation_losses_f64", &flex_f64());
}

/// CM1.6c: `FourierDiff` against `derivatives_f64`. Ignored: Flex's f64 `rfft` uses f32
/// twiddles (`upstream/burn/flex-f64-fft-f32-twiddles`); measured 4.0e-7 at worst (1D 1.8e-7, 2D 4.0e-7, 3D 2.1e-7; f32 4.9e-7).
#[cfg(feature = "flex")]
#[test]
#[ignore = "Flex's f64 rfft uses f32 twiddles (D3 as amended)"]
fn fourier_diff_f64_match_2_0_0() {
    let device = flex_f64();
    run_fourier::<1, 3, 4, 2>("derivatives_f64", &device);
    run_fourier::<2, 4, 5, 3>("derivatives_f64", &device);
    run_fourier::<3, 5, 6, 4>("derivatives_f64", &device);
}

/// CM1.6c: `non_uniform_fd` against `derivatives_f64`.
#[cfg(feature = "flex")]
#[test]
fn non_uniform_fd_f64_match_2_0_0() {
    run_non_uniform("derivatives_f64", &flex_f64());
}

//! Library CM1.6a and CM1.6b in f64: `parity_losses_common` on the f64 fixtures (D3: 1e-12). Its own
//! test binary because it configures Flex's process-wide float dtype to f64. No FFT is
//! involved, so unlike the FNO's f64 parity these are not held back by Flex's f32 twiddles.

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
use parity_losses_common::{run, run_meta};

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

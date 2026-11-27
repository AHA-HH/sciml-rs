//! Library CM1.1 in f64: the `legacy_spectral_conv{1,2,3}d_f64` fixtures. Its own test
//! binary because it configures Flex's process-wide float dtype to f64.
//!
//! `#[ignore]`d under D3 as amended 2026-11-25, with the measured errors in
//! `docs/operators/tasks/CM1.1.md`. Two causes keep them above 1e-12: 2.0.0's legacy
//! `forward` allocates its spectrum as `cfloat`, so the reference itself is f32-accurate,
//! and Flex's f64 `rfft` uses f32 twiddles (`tests/gauss_newton_f64.rs`).

#[cfg(feature = "flex")]
#[path = "parity_common/biased_conv.rs"]
mod biased_conv;
#[cfg(feature = "flex")]
mod flex_f64;
#[cfg(feature = "flex")]
#[path = "parity_common/legacy_conv.rs"]
mod legacy_conv;
mod parity_common;

#[cfg(feature = "flex")]
use flex_f64::flex_f64;
#[cfg(feature = "flex")]
use legacy_conv::*;
#[cfg(feature = "flex")]
use parity_common::Precision::F64;

#[cfg(feature = "flex")]
#[test]
#[ignore = "2.0.0's legacy spectrum is cfloat, and flex's f64 rfft uses f32 twiddles"]
fn legacy_spectral_conv1d_f64_matches_2_0_0() {
    check::<3>(&CASE_1D, F64, &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
#[ignore = "2.0.0's legacy spectrum is cfloat, and flex's f64 rfft uses f32 twiddles"]
fn legacy_spectral_conv2d_f64_matches_2_0_0() {
    check::<4>(&CASE_2D, F64, &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
#[ignore = "2.0.0's legacy spectrum is cfloat, and flex's f64 rfft uses f32 twiddles"]
fn legacy_spectral_conv3d_f64_matches_2_0_0() {
    check::<5>(&CASE_3D, F64, &flex_f64());
}

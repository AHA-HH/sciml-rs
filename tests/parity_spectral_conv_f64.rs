//! Library CM1.2 in f64: the centred `spectral_conv*_f64` fixtures. Its own test binary
//! because it configures Flex's process-wide float dtype to f64.
//!
//! `#[ignore]`d under D3 as amended 2026-11-25, with the measured errors in
//! `docs/operators/tasks/CM1.2.md`. Two causes keep them above 1e-12: 2.0.0's centred
//! `forward` allocates its spectrum as `cfloat`, so the reference itself is f32-accurate,
//! and Flex's f64 `rfft` uses f32 twiddles (`tests/gauss_newton_f64.rs`).
//!
//! `cargo test --test parity_spectral_conv_f64 -- --ignored`

#[cfg(feature = "flex")]
#[path = "parity_common/biased_conv.rs"]
mod biased_conv;
#[cfg(feature = "flex")]
#[path = "parity_common/centred_conv.rs"]
mod centred_conv;
#[cfg(feature = "flex")]
mod flex_f64;
mod parity_common;

#[cfg(feature = "flex")]
use centred_conv::*;
#[cfg(feature = "flex")]
use flex_f64::flex_f64;
#[cfg(feature = "flex")]
use parity_common::Precision::F64;

#[cfg(feature = "flex")]
#[test]
#[ignore = "2.0.0's centred spectrum is cfloat, and flex's f64 rfft uses f32 twiddles"]
fn spectral_conv2d_f64_matches_2_0_0() {
    check::<4>(&CASE_2D, F64, &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
#[ignore = "2.0.0's centred spectrum is cfloat, and flex's f64 rfft uses f32 twiddles"]
fn spectral_conv2d_backward_f64_matches_2_0_0() {
    check::<4>(&CASE_2D_BACKWARD, F64, &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
#[ignore = "2.0.0's centred spectrum is cfloat, and flex's f64 rfft uses f32 twiddles"]
fn spectral_conv2d_odd_f64_matches_2_0_0() {
    check::<4>(&CASE_2D_ODD, F64, &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
#[ignore = "2.0.0's centred spectrum is cfloat, and flex's f64 rfft uses f32 twiddles"]
fn spectral_conv3d_f64_matches_2_0_0() {
    check::<5>(&CASE_3D, F64, &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
#[ignore = "2.0.0's centred spectrum is cfloat, and flex's f64 rfft uses f32 twiddles"]
fn spectral_conv2d_separable_f64_matches_2_0_0() {
    check::<4>(&CASE_2D_SEPARABLE, F64, &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
#[ignore = "2.0.0's centred spectrum is cfloat, and flex's f64 rfft uses f32 twiddles"]
fn spectral_conv2d_max_modes_f64_matches_2_0_0() {
    check::<4>(&CASE_2D_MAX_MODES, F64, &flex_f64());
}

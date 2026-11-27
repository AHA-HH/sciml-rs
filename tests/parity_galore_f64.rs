//! Library CM3.6a and CM3.6b in f64: `parity_galore_common` on the f64 fixtures, real and
//! complex (D3: 1e-12; the complex one is tensorly's `tucker` with 2.0.0's transforms, since
//! 2.0.0 fails on `cdouble`). Its own test binary because it configures Flex's process-wide
//! float dtype to f64. No FFT is involved, so the tests run.

#[cfg(feature = "flex")]
mod flex_f64;
// Each test binary uses part of the shared harness.
#[cfg(feature = "flex")]
#[allow(dead_code)]
mod parity_common;
#[cfg(feature = "flex")]
#[allow(dead_code)]
mod parity_galore_common;

#[cfg(feature = "flex")]
use burn::prelude::*;
#[cfg(feature = "flex")]
use flex_f64::flex_f64;
#[cfg(feature = "flex")]
use parity_galore_common::{run, run_up_to_phase};
#[cfg(feature = "flex")]
use sciml_rs::neural_operators::optim::galore::ComplexTensor;

#[cfg(feature = "flex")]
#[test]
fn matrix_f64_matches_2_0_0() {
    run::<2, Tensor<2>>("galore_f64", "matrix", &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
fn tensor4_f64_matches_2_0_0() {
    run::<4, Tensor<4>>("galore_f64", "tensor4", &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
fn fraction_f64_matches_2_0_0() {
    run::<3, Tensor<3>>("galore_f64", "fraction", &flex_f64());
}

/// Compared up to phase (the author, 2026-11-27): its sweeps' unfoldings are tall, so the
/// factors are defined only up to each column's phase, which a 1e-7 change of the gradient
/// moves by O(1); `tests/parity_galore.rs` gives the reason in full.
#[cfg(feature = "flex")]
#[test]
fn complex_matrix_f64_matches_tensorly_up_to_phase() {
    run_up_to_phase::<2>("galore_complex_f64", "matrix", &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
fn complex_tensor4_f64_matches_tensorly() {
    run::<4, ComplexTensor<4>>("galore_complex_f64", "tensor4", &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
fn complex_fraction_f64_matches_tensorly() {
    run::<3, ComplexTensor<3>>("galore_complex_f64", "fraction", &flex_f64());
}

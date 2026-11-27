//! Library CM3.6a and CM3.6b in f32: `parity_galore_common` on the f32 fixtures, real and
//! complex (D3: 1e-5). The f64 fixtures are `tests/parity_galore_f64.rs`.

// Each test binary uses part of the shared harness.
#[allow(dead_code)]
mod parity_common;
mod parity_galore_common;

use burn::prelude::*;
use parity_common::{Fixture, Take, rel_max_error, take};
use parity_galore_common::{config, run, run_up_to_phase};
use sciml_rs::neural_operators::optim::galore::{ComplexTensor, tucker_factors};

#[test]
fn matrix_f32_matches_2_0_0() {
    run::<2, Tensor<2>>("galore_f32", "matrix", &Device::default());
}

#[test]
fn tensor4_f32_matches_2_0_0() {
    run::<4, Tensor<4>>("galore_f32", "tensor4", &Device::default());
}

#[test]
fn fraction_f32_matches_2_0_0() {
    run::<3, Tensor<3>>("galore_f32", "fraction", &Device::default());
}

/// The comparison can fail: factors from the second gradient, not the first.
#[test]
fn factors_of_another_gradient_fail_parity() {
    let device = Device::default();
    let mut fx = Fixture::open("galore_f32");
    let mut projector = config("tensor4").init::<4, Tensor<4>>();
    projector.project(fx.tensor::<4>("tensor4.grad2", &device), 0);
    let f0 = projector.factors().unwrap()[0].clone();
    let err = rel_max_error(f0, &fx.array("tensor4.factor0"));
    assert!(err > 0.1, "error {err:.3e}");
}

/// Compared up to phase (the author, 2026-11-27): its sweeps' unfoldings are tall (12 × 3,
/// 7 × 3), so 2.0.0's factors are defined only up to each column's phase. LAPACK's phase
/// reference is then a column of the other factor's product, so the phases turn every
/// sweep, and a 1e-7 change of the gradient moves them by O(1), in complex128 too. Their
/// subspaces and the back-projection are well defined, and are what `run_up_to_phase`
/// compares.
#[test]
fn complex_matrix_f32_matches_2_0_0_up_to_phase() {
    run_up_to_phase::<2>("galore_complex_f32", "matrix", &Device::default());
}

#[test]
fn complex_tensor4_f32_matches_2_0_0() {
    run::<4, ComplexTensor<4>>("galore_complex_f32", "tensor4", &Device::default());
}

#[test]
fn complex_fraction_f32_matches_2_0_0() {
    run::<3, ComplexTensor<3>>("galore_complex_f32", "fraction", &Device::default());
}

/// The complex comparison can fail: factors from the second gradient, not the first.
#[test]
fn complex_factors_of_another_gradient_fail_parity() {
    let device = Device::default();
    let mut fx = Fixture::open("galore_complex_f32");
    let (re, im) = fx.complex_tensor::<4>("tensor4.grad2", &device);
    let f0 = tucker_factors(ComplexTensor::new(re, im), [3, 2, 3, 2])[0].clone();
    let expected = fx.array("tensor4.factor0");
    let err = rel_max_error(f0.re, &take(expected.clone(), Take::Re))
        .max(rel_max_error(f0.im, &take(expected, Take::Im)));
    assert!(err > 0.1, "error {err:.3e}");
}

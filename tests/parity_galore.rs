//! Library CM3.6a in f32: `parity_galore_common` on the f32 fixture (D3: 1e-5). The f64
//! fixture is `tests/parity_galore_f64.rs`.

// Each test binary uses part of the shared harness.
#[allow(dead_code)]
mod parity_common;
mod parity_galore_common;

use burn::prelude::*;
use parity_common::{Fixture, rel_max_error};
use parity_galore_common::{config, run};

#[test]
fn matrix_f32_matches_2_0_0() {
    run::<2>("galore_f32", "matrix", &Device::default());
}

#[test]
fn tensor4_f32_matches_2_0_0() {
    run::<4>("galore_f32", "tensor4", &Device::default());
}

#[test]
fn fraction_f32_matches_2_0_0() {
    run::<3>("galore_f32", "fraction", &Device::default());
}

/// The comparison can fail: factors from the second gradient, not the first.
#[test]
fn factors_of_another_gradient_fail_parity() {
    let device = Device::default();
    let mut fx = Fixture::open("galore_f32");
    let mut projector = config("tensor4").init::<4>();
    projector.project(fx.tensor::<4>("tensor4.grad2", &device), 0);
    let f0 = projector.factors().unwrap()[0].clone();
    let err = rel_max_error(f0, &fx.array("tensor4.factor0"));
    assert!(err > 0.1, "error {err:.3e}");
}

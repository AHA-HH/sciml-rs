//! Library CM1.6a in f32: `parity_losses_common` on the f32 fixtures (D3: 1e-5). The f64
//! fixtures are `tests/parity_losses_f64.rs`.

// Each test binary uses part of the shared harness.
#[allow(dead_code)]
mod parity_common;
mod parity_losses_common;

use burn::prelude::*;
use parity_common::{Fixture, rel_max_error};
use parity_losses_common::{run, tensor};
use sciml_rs::neural_operators::losses::differentiation::FiniteDiff;

#[test]
fn losses1d_f32_match_2_0_0() {
    run::<1, 3, 4, 2>("losses1d_f32", &Device::default());
}

#[test]
fn losses2d_f32_match_2_0_0() {
    run::<2, 4, 5, 3>("losses2d_f32", &Device::default());
}

#[test]
fn losses3d_f32_match_2_0_0() {
    run::<3, 5, 6, 4>("losses3d_f32", &Device::default());
}

/// The comparison can fail: dx and dy swapped on the non-square 2D grid.
#[test]
fn swapped_axes_fail_parity() {
    let device = Device::default();
    let mut fx = Fixture::open("losses2d_f32");
    let x: Tensor<4> = tensor(&mut fx, "x", &device);
    // Equal steps, so only the axis differs.
    let fd = FiniteDiff {
        h: [0.1, 0.1],
        periodic: [true, true],
    };
    let err = rel_max_error(fd.partial(x, 1, 1), &fx.array("fd.dx1.per"));
    assert!(err > 1e-1, "a swapped axis went unnoticed: {err:.3e}");
}

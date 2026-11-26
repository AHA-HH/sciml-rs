//! Library CM1.6a–c and CM2.5 in f32: `parity_losses_common` on the f32 fixtures (D3: 1e-5). The f64
//! fixtures are `tests/parity_losses_f64.rs`.

// Each test binary uses part of the shared harness.
#[allow(dead_code)]
mod parity_common;
mod parity_losses_common;

use burn::prelude::*;
use parity_common::{Fixture, rel_max_error};
use parity_losses_common::{run, run_equation, run_fourier, run_meta, run_non_uniform};
use sciml_rs::neural_operators::losses::{
    differentiation::{FiniteDiff, FourierDiff, GridDiff},
    equation_losses::BurgersEqnLoss,
};

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
    let x: Tensor<4> = fx.tensor("x", &device);
    // Equal steps, so only the axis differs.
    let fd = FiniteDiff {
        h: [0.1, 0.1],
        periodic: [true, true],
    };
    let err = rel_max_error(fd.partial(x, 1, 1), &fx.array("fd.dx1.per"));
    assert!(err > 1e-1, "a swapped axis went unnoticed: {err:.3e}");
}

/// CM1.6b: the aggregators against `meta_losses_f32`.
#[test]
fn meta_losses_f32_match_2_0_0() {
    run_meta("meta_losses_f32", &Device::default());
}

/// CM2.5: the equation losses against `equation_losses_f32`.
#[test]
fn equation_losses_f32_match_2_0_0() {
    run_equation("equation_losses_f32", &Device::default());
}

/// The comparison can fail: the Burgers loss with t and x swapped on the non-square grid.
#[test]
fn burgers_swapped_axes_fail_parity() {
    let device = Device::default();
    let mut fx = Fixture::open("equation_losses_f32");
    let u: Tensor<4> = fx.tensor("u", &device);
    let out = BurgersEqnLoss::default().loss(u.swap_dims(2, 3));
    let err = rel_max_error(out, &fx.array("burgers.default"));
    assert!(err > 1e-1, "a swapped axis went unnoticed: {err:.3e}");
}

/// CM1.6c: `FourierDiff` in 1D–3D against `derivatives_f32`.
#[test]
fn fourier_diff_f32_match_2_0_0() {
    let device = Device::default();
    run_fourier::<1, 3, 4, 2>("derivatives_f32", &device);
    run_fourier::<2, 4, 5, 3>("derivatives_f32", &device);
    run_fourier::<3, 5, 6, 4>("derivatives_f32", &device);
}

/// The comparison can fail: the Fourier derivative along y against 2.0.0's dx on the 16 × 15
/// grid (equal lengths, so only the axis differs).
#[test]
fn fourier_swapped_axes_fail_parity() {
    let device = Device::default();
    let mut fx = Fixture::open("derivatives_f32");
    let u: Tensor<4> = fx.tensor("u2", &device);
    let fd = FourierDiff { l: [2.0, 2.0] };
    let err = rel_max_error(fd.partial(u, 1, 1), &fx.array("fourier2.dx1"));
    assert!(err > 1e-1, "a swapped axis went unnoticed: {err:.3e}");
}

/// CM1.6c: `non_uniform_fd` against `derivatives_f32`.
#[test]
fn non_uniform_fd_f32_match_2_0_0() {
    run_non_uniform("derivatives_f32", "derivatives_f64", &Device::default());
}

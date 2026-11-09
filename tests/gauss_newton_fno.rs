//! Phase 10 T1 on a tiny 2D FNO, in the default dtype (f32): the parameter round trip,
//! the explicit-Jacobian oracle for `vjp` and the finite-difference `jvp`, and the
//! adjoint check. The measured errors are printed (`-- --nocapture`) for the phase
//! README's Results row.

mod gauss_newton_common;

use burn::prelude::*;
use gauss_newton_common::*;
use sciml_rs::neural_operators::{
    models::fno::FNO,
    optim::gauss_newton::{LeastSquares, ParamVec, adjoint_error, jvp, params, with_params},
};

#[test]
fn params_round_trip_the_fno_bit_for_bit() {
    let device = Device::default();
    let model = tiny_fno(&device);
    let p = params(&model);
    let bits = |v: &ParamVec| -> Vec<u32> {
        v.to_flat()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    };
    let back = with_params(model.clone(), &p);
    assert_eq!(bits(&params(&back)), bits(&p));
    // the round-tripped model computes exactly what the original does
    let x = batch(&device).inputs;
    assert_eq!(
        back.forward(x.clone()).into_data(),
        model.forward(x).into_data()
    );
}

#[test]
fn vjp_and_finite_difference_jvp_match_the_explicit_jacobian() {
    let device = Device::default().autodiff();
    let model = tiny_fno(&device);
    let errors = oracle_errors(&model, &batch(&device));
    println!(
        "f32 tiny FNO: P = {}, m = {}; vjp vs Jᵀu {:.2e}, finite-difference jvp vs J v {:.2e}",
        errors.n_params,
        B * S * S,
        errors.vjp,
        errors.jvp
    );
    assert!(errors.vjp < 1e-5, "vjp error {}", errors.vjp);
    assert!(errors.jvp < 1e-3, "jvp error {}", errors.jvp);
}

/// `RelResidual` with a sign-flipped "exact" J v.
struct FlippedJvp;

impl LeastSquares<FNO<4>> for FlippedJvp {
    type Batch = Batch;
    fn residual(&self, model: &FNO<4>, batch: &Batch) -> Tensor<1> {
        RelResidual.residual(model, batch)
    }
    fn jvp(&self, model: &FNO<4>, batch: &Batch, v: &ParamVec) -> Option<Tensor<1>> {
        Some(jvp(&RelResidual, model, batch, v, None).neg())
    }
}

#[test]
fn adjoint_error_is_small_and_a_wrong_jvp_is_caught() {
    let device = Device::default().autodiff();
    let model = tiny_fno(&device);
    let data = batch(&device);
    let theta = params(&model);
    let mut worst: f64 = 0.0;
    for k in 0..4 {
        let u = from_host(&wave(B * S * S, 0.4 + k as f64), &device);
        let v = theta.from_flat_like(from_host(&wave(theta.numel(), 2.6 + k as f64), &device));
        worst = worst.max(adjoint_error(&RelResidual, &model, &data, &u, &v));
        // The error is normalised by ‖u‖ ‖J v‖, so a flip shows as 2 |cos ∠(u, J v)|: test
        // it with u along J v, where a flipped J v gives about 2.
        let along = jvp(&RelResidual, &model, &data, &v, None);
        let bad = adjoint_error(&FlippedJvp, &model, &data, &along, &v);
        assert!(bad > 1.5, "a sign-flipped J v must be caught, got {bad}");
    }
    println!("f32 tiny FNO: worst adjoint error over 4 (u, v) pairs {worst:.2e}");
    assert!(worst < 1e-3, "adjoint error {worst}");
}

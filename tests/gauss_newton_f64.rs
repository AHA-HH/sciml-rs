//! Phase 10 T1 in f64 on flex, `#[ignore]`d while a Burn bug remains: flex's f64 `rfft`
//! (`burn-flex/src/ops/fft.rs`, `rfft_f64`) multiplies by the shared twiddle table, which
//! is stored in f32 (`get_twiddles`), so f64 FFTs are only f32-accurate and the FNO's f64
//! forward is no more accurate than its f32 one. `-- --ignored` shows both tests failing:
//! the repro (a length-16 rfft against the exact DFT, 1.9e-7 where f64 gives ~1e-15) and
//! the oracle on the tiny FNO (vjp ~1e-7, finite-difference jvp ~1e-3 at the f64 step).
//! Until it is fixed the Gauss–Newton tests run in f32 (`tests/gauss_newton_fno.rs`).
//!
//! Its own test binary because a device's float dtype can be configured only once, before
//! any tensor is created on it.

#![cfg(feature = "flex")]

mod gauss_newton_common;

use std::sync::OnceLock;

use burn::{
    prelude::*,
    tensor::{DType, FloatDType, TensorData},
};
use gauss_newton_common::*;
use sciml_rs::neural_operators::optim::gauss_newton::{adjoint_error, default_rel_step, params};

/// Flex configured for f64, once per process.
fn flex_f64() -> Device {
    static DEVICE: OnceLock<Device> = OnceLock::new();
    DEVICE
        .get_or_init(|| {
            let mut device = Device::flex();
            device
                .configure(FloatDType::F64)
                .expect("flex accepts f64 as its default float dtype");
            device
        })
        .clone()
}

#[test]
#[ignore = "Burn bug: flex's f64 rfft uses f32 twiddles"]
fn repro_flex_rfft_f64_is_f64_accurate() {
    let device = flex_f64();
    let n = 16;
    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37 + 0.1).sin()).collect();
    let signal = Tensor::<2>::from_data(TensorData::new(x.clone(), [1, n]), &device);
    let (re, im) = burn::signal::rfft(signal, 1, None);
    assert_eq!(re.dtype(), DType::F64);
    let (re, im) = (to_host(re), to_host(im));
    let mut worst: f64 = 0.0;
    for k in 0..=n / 2 {
        let (mut exact_re, mut exact_im) = (0.0, 0.0);
        for (j, v) in x.iter().enumerate() {
            let angle = -2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64;
            exact_re += v * angle.cos();
            exact_im += v * angle.sin();
        }
        worst = worst
            .max((re[k] - exact_re).abs())
            .max((im[k] - exact_im).abs());
    }
    assert!(worst < 1e-12, "f64 rfft max error {worst:.2e}");
}

#[test]
#[ignore = "Burn bug: flex's f64 rfft uses f32 twiddles (see repro_flex_rfft_f64_is_f64_accurate)"]
fn f64_oracle_and_adjoint_on_flex() {
    let device = flex_f64().autodiff();
    assert_eq!(float_dtype(&device), DType::F64);
    let model = tiny_fno(&device);
    let data = batch(&device);
    assert_eq!(model.forward(data.inputs.clone()).dtype(), DType::F64);

    let errors = oracle_errors(&model, &data);
    let theta = params(&model);
    let u = from_host(&wave(B * S * S, 0.4), &device);
    let v = theta.from_flat_like(from_host(&wave(theta.numel(), 2.6), &device));
    let adjoint = adjoint_error(&RelResidual, &model, &data, &u, &v);
    println!(
        "f64 tiny FNO: P = {}, step {:.1e}; vjp vs Jᵀu {:.2e}, finite-difference jvp vs J v {:.2e}, adjoint error {:.2e}",
        errors.n_params,
        default_rel_step(DType::F64),
        errors.vjp,
        errors.jvp,
        adjoint
    );
    assert!(errors.vjp < 1e-12, "vjp error {}", errors.vjp);
    assert!(errors.jvp < 1e-8, "jvp error {}", errors.jvp);
    assert!(adjoint < 1e-8, "adjoint error {adjoint}");
}

//! Shared by `tests/gauss_newton_fno.rs` (f32) and `tests/gauss_newton_f64.rs`: a tiny 2D
//! FNO, the per-sample relative residual of Phase 10 (G1), and the explicit-Jacobian
//! oracle. These tests live outside `src/` because `optim/` imports nothing from
//! `models/` (Phase 10 T1).

use burn::{
    prelude::*,
    tensor::{DType, TensorData},
};
use sciml_rs::neural_operators::{
    models::fno::{FNO, FNOConfig},
    optim::gauss_newton::{LeastSquares, ParamVec, jvp, params, vjp, with_params},
};

/// Batch size and grid side of the tiny problem: inputs `[2, 4, 4, 1]`.
pub const B: usize = 2;
pub const S: usize = 4;

/// A 2D FNO small enough for an explicit Jacobian (591 parameters), with deterministic
/// weights 0.3 sin(0.37 i + 0.2) in place of the random initialisation, so every run
/// measures the same model.
pub fn tiny_fno(device: &Device) -> FNO<4> {
    let model = FNOConfig::new(vec![2, 2], 1, 1)
        .with_hidden_channels(2)
        .with_n_layers(1)
        .init::<4>(device);
    let p = params(&model);
    let values: Vec<f64> = wave(p.numel(), 0.2).iter().map(|x| 0.3 * x).collect();
    with_params(model, &p.from_flat_like(from_host(&values, device)))
}

/// Inputs `[B, S, S, 1]` and targets `[B, S·S]`.
pub struct Batch {
    pub inputs: Tensor<4>,
    pub targets: Tensor<2>,
}

/// The device's default float dtype.
pub fn float_dtype(device: &Device) -> DType {
    device.settings().float_dtype.into()
}

/// `values` as a `[n]` tensor in the device's default float dtype.
pub fn from_host(values: &[f64], device: &Device) -> Tensor<1> {
    let data = TensorData::new(values.to_vec(), [values.len()]).convert_dtype(float_dtype(device));
    Tensor::from_data(data, device)
}

/// A tensor's values on the host, in f64, whatever its dtype.
pub fn to_host<const D: usize>(t: Tensor<D>) -> Vec<f64> {
    t.into_data().convert::<f64>().try_to_vec::<f64>().unwrap()
}

/// sin(0.37 i + phase), i < n: deterministic, no global RNG.
pub fn wave(n: usize, phase: f64) -> Vec<f64> {
    (0..n).map(|i| (i as f64 * 0.37 + phase).sin()).collect()
}

/// The tiny problem's batch: deterministic inputs and targets 2 + sin(·), so every
/// per-sample norm is nonzero.
pub fn batch(device: &Device) -> Batch {
    let inputs = from_host(&wave(B * S * S, 0.3), device).reshape([B, S, S, 1]);
    let targets = from_host(
        &wave(B * S * S, 1.1)
            .into_iter()
            .map(|x| 2.0 + x)
            .collect::<Vec<_>>(),
        device,
    )
    .reshape([B, S * S]);
    Batch { inputs, targets }
}

/// rᵢ = (ûᵢ − uᵢ) / ‖uᵢ‖ per sample, flattened to `[B·S·S]` (Phase 10, G1).
pub struct RelResidual;

impl LeastSquares<FNO<4>> for RelResidual {
    type Batch = Batch;
    fn residual(&self, model: &FNO<4>, batch: &Batch) -> Tensor<1> {
        let out = model.forward(batch.inputs.clone()).reshape([B, S * S]);
        let norms = batch.targets.clone().powf_scalar(2.0).sum_dim(1).sqrt();
        ((out - batch.targets.clone()) / norms).reshape([B * S * S])
    }
}

/// J by rows, on the host: row i is Jᵀeᵢ, one VJP per output. `model` must be on an
/// autodiff device.
pub fn explicit_jacobian(model: &FNO<4>, batch: &Batch) -> Vec<Vec<f64>> {
    let m = B * S * S;
    let device = batch.inputs.device();
    (0..m)
        .map(|i| {
            let mut e = vec![0.0; m];
            e[i] = 1.0;
            to_host(vjp(&RelResidual, model, batch, &from_host(&e, &device)).to_flat())
        })
        .collect()
}

/// max |a − b| / max |b|.
pub fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    let scale = b.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
        / scale
}

/// The relative errors of `vjp` against Jᵀu and of the finite-difference `jvp` (default
/// step) against J v, for fixed u and v, from the explicit Jacobian.
pub struct OracleErrors {
    pub n_params: usize,
    pub vjp: f64,
    pub jvp: f64,
}

/// [`OracleErrors`] of `model` on `batch` (on an autodiff device), for u = sin(0.37 i + 0.7)
/// and v = sin(0.37 i + 1.9).
pub fn oracle_errors(model: &FNO<4>, batch: &Batch) -> OracleErrors {
    let device = batch.inputs.device();
    let j = explicit_jacobian(model, batch);
    let (m, p) = (j.len(), j[0].len());

    let u = wave(m, 0.7);
    let jtu_oracle: Vec<f64> = (0..p)
        .map(|k| (0..m).map(|i| j[i][k] * u[i]).sum())
        .collect();
    let jtu = to_host(vjp(&RelResidual, model, batch, &from_host(&u, &device)).to_flat());

    let v_flat = wave(p, 1.9);
    let v: ParamVec = params(model).from_flat_like(from_host(&v_flat, &device));
    let jv_oracle: Vec<f64> = (0..m)
        .map(|i| (0..p).map(|k| j[i][k] * v_flat[k]).sum())
        .collect();
    let jv = to_host(jvp(&RelResidual, model, batch, &v, None));

    OracleErrors {
        n_params: p,
        vjp: max_rel(&jtu, &jtu_oracle),
        jvp: max_rel(&jv, &jv_oracle),
    }
}

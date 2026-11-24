//! 3D CL0.1 (`docs/laplace3d/tasks/CL0.1.md`): time per epoch of `FNO<5>` (width 32,
//! 4 layers) on synthetic data, through the real training path (`build_training_components`,
//! `train_epoch`, `evaluate`). Peak memory is read outside the process (`REPORT.md`).
//!
//! Usage: `cargo run --release -p fno3d-cost-spike --bin fno3d_cost [--features metal|cuda]
//! -- <s> <modes> <batch> <n_train> <n_test>`; `BURN_DEVICE` picks the backend.

use std::time::Instant;

use burn::module::Module;
use burn::prelude::*;
use ndarray::{ArrayD, IxDyn};
use sciml_rs::neural_operators::data::dataset::OperatorDataset;
use sciml_rs::neural_operators::models::fno::FNOConfig;
use sciml_rs::neural_operators::training::config::TrainingConfig;
use sciml_rs::neural_operators::training::step::evaluate;
use sciml_rs::neural_operators::training::trainer::{build_training_components, train_epoch};

/// The 3D model of proposal §7: width 32, 4 layers, `modes` per axis, one data channel.
fn model_config(modes: usize) -> FNOConfig {
    FNOConfig::new(vec![modes; 3], 1, 1)
}

/// `n` samples of `[s, s, s, 1]` inputs and `[s, s, s]` targets, xorshift values in
/// [−1, 1): synthetic data, only the cost matters.
fn synthetic(n: usize, s: usize, seed: u64) -> OperatorDataset<f32> {
    let mut state = seed.max(1);
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    };
    let inputs = ArrayD::from_shape_simple_fn(IxDyn(&[n, s, s, s, 1]), &mut next);
    let targets = ArrayD::from_shape_simple_fn(IxDyn(&[n, s, s, s]), &mut next);
    OperatorDataset::new(inputs, targets)
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("arguments are integers"))
        .collect();
    let [s, modes, batch, n_train, n_test] = args[..] else {
        panic!("usage: fno3d_cost <s> <modes> <batch> <n_train> <n_test>");
    };
    let device = Device::default().autodiff();
    let train_cfg = TrainingConfig::new()
        .with_batch_size(batch)
        .with_test_batch_size(batch);
    let c = build_training_components::<5, 4, f32>(
        &model_config(modes),
        &train_cfg,
        synthetic(n_train, s, 1),
        synthetic(n_test, s, 2),
        &device,
    );
    let (mut model, mut optim, mut scheduler) = (c.model, c.optim, c.scheduler);
    println!(
        "device {device:?} s {s} modes {modes} batch {batch} n_train {n_train} n_test {n_test} params {}",
        model.num_params()
    );
    // epoch 0 warms the kernels and the allocator; epoch 1 is the one reported
    for epoch in 0..2 {
        let t = Instant::now();
        let (m, means) = train_epoch(model, &c.train_loader, &mut optim, &mut scheduler, None);
        model = m;
        let train_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let test_l2 = evaluate(&model.valid(), &c.test_loader, None);
        let test_s = t.elapsed().as_secs_f64();
        println!(
            "epoch {epoch} train_s {train_s:.3} test_s {test_s:.3} train_l2 {:.4} test_l2 {test_l2:.4} \
             epoch_s_at_1000_200 {:.1}",
            means.train_l2,
            train_s / n_train as f64 * 1000.0 + test_s / n_test as f64 * 200.0
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Proposal §7's parameter totals: 4 layers × 4 corners × 32² × m³ × 2 spectral weights,
    /// plus fc0 (4 → 32), the pointwise `w` (32 → 32), fc1 (32 → 128), fc2 (128 → 1).
    #[test]
    fn param_counts_match_section_7() {
        let device = Device::default();
        for (modes, want) in [(8, 16_785_953), (12, 56_631_841)] {
            let model = model_config(modes).init::<5>(&device);
            assert_eq!(model.num_params(), want, "modes {modes}");
        }
    }
}

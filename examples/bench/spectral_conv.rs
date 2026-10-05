//! Spectral convolution benchmark
//!
//! Times `SpectralConv` forward + backward on its own, and one full FNO
//! training step (forward, relative L2 loss, backward, Adam step), at the
//! settings of design document `docs/design/spectral-conv-perf.md` §1 and §8.3.
//! This is the benchmark that defines "faster" for the spectral-conv-perf
//! objective: phase 0 records its output as the baseline, later phases rerun it
//! unchanged.
//!
//! Run with: cargo run --release --example bench_spectral_conv
//! (CPU by default; add `--features metal` for the Apple GPU - see "Backends"
//! in the README). An optional first argument filters cases by substring, e.g.
//! `cargo run --release --example bench_spectral_conv -- darcy`.
//!
//! Inputs and targets are random: the values do not matter, and no dataset is
//! needed. Each case runs `WARMUP` untimed steps, then `TIMED` steps, each
//! bracketed by `device.sync()` so the clock includes all queued GPU work.
//! Prints one table to stdout and writes no files. Peak memory is not measured:
//! Burn has no portable API for it.

use burn::{
    optim::{AdamConfig, GradientsParams, decay::WeightDecayConfig},
    tensor::{Device, Distribution, Tensor},
};
use sciml_rs::neural_operators::{
    layers::spectral_convolution::SpectralConv,
    losses::data_losses::{LpLoss, Reduction},
    models::fno::FNOConfig,
    training::trainer::flatten_pair,
};
use std::time::Instant;

const BATCH: usize = 20;
const WARMUP: usize = 5;
const TIMED: usize = 20;
// Layer cases: (I = O, spatial extents, modes)
// 1D: (64, [256], [16]), (64, [1024], [16]);  2D: (32, [94, 94], [12, 12])
// FNO step cases: burgers s = 256 and s = 1024 (FNOConfig of examples/train/burgers.rs),
//                 darcy s = 85 (FNOConfig of examples/train/darcy.rs, padding Some(9))
const LEARNING_RATE: f64 = 1e-3;
// From TrainingConfig::with_weight_decay in examples/train/{burgers,darcy}.rs
const WEIGHT_DECAY: f32 = 1e-4;

/// One row of the output table.
struct Row {
    case: String,
    /// Timed step durations in milliseconds, unsorted.
    samples: Vec<f64>,
}

fn main() {
    let device = Device::default().autodiff();
    println!("device: {device:?}");

    // The device debug string on macOS reads `Cube(Wgpu(.. backend: Auto))`
    // and never says "Metal", so label by build feature instead.
    let backend = if cfg!(feature = "metal") {
        "Metal"
    } else {
        "flex"
    };

    let filter = std::env::args().nth(1).unwrap_or_default();
    let selected = |name: &str| name.contains(filter.as_str());

    let mut rows = Vec::new();

    for s in [256, 1024] {
        let case = format!("layer 1D I=O=64 s={s} modes=[16]");
        if selected(&case) {
            let samples = bench_layer::<3>(&device, 64, &[s], &[16]);
            rows.push(Row { case, samples });
        }
    }

    let case = "layer 2D I=O=32 s=94x94 modes=[12,12]".to_string();
    if selected(&case) {
        let samples = bench_layer::<4>(&device, 32, &[94, 94], &[12, 12]);
        rows.push(Row { case, samples });
    }

    for s in [256, 1024] {
        let case = format!("fno step burgers s={s}");
        if selected(&case) {
            let samples = bench_fno_step::<3, 2>(&device, burgers_config(), &[s]);
            rows.push(Row { case, samples });
        }
    }

    let case = "fno step darcy s=85 (padded 94x94)".to_string();
    if selected(&case) {
        let samples = bench_fno_step::<4, 3>(&device, darcy_config(), &[85, 85]);
        rows.push(Row { case, samples });
    }

    print_table(&rows, backend);
}

/// `FNOConfig` literal of `examples/train/burgers.rs:69-86`.
fn burgers_config() -> FNOConfig {
    FNOConfig {
        modes: vec![16],
        hidden_channels: 64,
        data_channels: 1,
        out_channels: 1,
        n_layers: 4,
        spectral_init: None,
        padding: None,
    }
}

/// `FNOConfig` literal of `examples/train/darcy.rs:57-71`.
fn darcy_config() -> FNOConfig {
    FNOConfig {
        modes: vec![12, 12],
        hidden_channels: 32,
        data_channels: 1,
        out_channels: 1,
        n_layers: 4,
        spectral_init: None,
        padding: Some(9),
    }
}

/// Runs `WARMUP` untimed steps, then `TIMED` timed steps, and returns the
/// timed durations in milliseconds.
///
/// Each timed step is bracketed by `device.sync()`, so it includes all work
/// the step queued on the device. Whatever the step returns (e.g. the
/// `Gradients`) is dropped only after the closing sync, so a fusing backend
/// cannot skip work whose result was discarded early.
fn time_steps<T>(device: &Device, mut step: impl FnMut() -> T) -> Vec<f64> {
    for _ in 0..WARMUP {
        let _ = step();
    }
    device.sync().expect("device sync failed after warm-up");

    (0..TIMED)
        .map(|_| {
            device.sync().expect("device sync failed before a step");
            let start = Instant::now();
            let out = step();
            device.sync().expect("device sync failed after a step");
            let elapsed = start.elapsed();
            drop(out);
            elapsed.as_secs_f64() * 1e3
        })
        .collect()
}

/// Times `SpectralConv` forward + backward with `channels` input and output
/// channels on an input of shape `[BATCH, channels, spatial..]`.
///
/// One step is `conv.forward(x).powf_scalar(2.0).sum().backward()`.
fn bench_layer<const R: usize>(
    device: &Device,
    channels: usize,
    spatial: &[usize],
    modes: &[usize],
) -> Vec<f64> {
    let conv = SpectralConv::<R>::new(device, channels, channels, modes);

    let shape: Vec<usize> = [BATCH, channels]
        .into_iter()
        .chain(spatial.iter().copied())
        .collect();
    let x = Tensor::<R>::random(shape, Distribution::Normal(0.0, 1.0), device).require_grad();

    time_steps(device, || {
        conv.forward(x.clone()).powf_scalar(2.0).sum().backward()
    })
}

/// Times one FNO training step on random inputs `[BATCH, spatial.., C]` and
/// targets `[BATCH, spatial..]`, threading the model through the steps as
/// `training::trainer::train_epoch` does.
///
/// One step is the body of `train_epoch` (`trainer.rs:172-202`) without its
/// metrics: forward, [`flatten_pair`], `LpLoss::rel`, backward,
/// `GradientsParams::from_grads`, `optim.step` at a fixed learning rate.
fn bench_fno_step<const R: usize, const RM1: usize>(
    device: &Device,
    cfg: FNOConfig,
    spatial: &[usize],
) -> Vec<f64> {
    let mut model = Some(cfg.init::<R>(device));

    let input_shape: Vec<usize> = std::iter::once(BATCH)
        .chain(spatial.iter().copied())
        .chain(std::iter::once(cfg.data_channels))
        .collect();
    let target_shape: Vec<usize> = std::iter::once(BATCH)
        .chain(spatial.iter().copied())
        .collect();
    let inputs = Tensor::<R>::random(input_shape, Distribution::Normal(0.0, 1.0), device);
    let targets = Tensor::<RM1>::random(target_shape, Distribution::Normal(0.0, 1.0), device);

    // As built in trainer.rs:306
    let loss_fn = LpLoss::new(R - 2, 2, Reduction::Sum);

    // As built in trainer.rs:274-277
    let mut optim = AdamConfig::new()
        .with_epsilon(1e-8)
        .with_weight_decay(Some(WeightDecayConfig::new(WEIGHT_DECAY)))
        .init();

    time_steps(device, || {
        let m = model.take().expect("model is put back after every step");

        let out = m.forward(inputs.clone());
        let (out, target) = flatten_pair::<R, RM1>(out, targets.clone());
        let loss = loss_fn.rel(out, target);

        let grads = GradientsParams::from_grads(loss.backward(), &m);
        model = Some(optim.step(LEARNING_RATE, m, grads));
    })
}

/// Nearest-rank quartiles `(q1, median, q3)` of `samples`: the `p`-quantile is
/// the `ceil(p·n)`-th smallest sample (1-based). With 20 samples these are
/// x(5), x(10) and x(15), so the "median" is the lower of the two middle values.
///
/// # Panics
/// If `samples` is empty.
fn quartiles(samples: &[f64]) -> (f64, f64, f64) {
    assert!(!samples.is_empty(), "no samples to summarise");
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let rank = |p: f64| ((p * n as f64).ceil() as usize).max(1) - 1;
    (sorted[rank(0.25)], sorted[rank(0.5)], sorted[rank(0.75)])
}

fn print_table(rows: &[Row], backend: &str) {
    let summaries: Vec<(&str, f64, String)> = rows
        .iter()
        .map(|row| {
            let (q1, median, q3) = quartiles(&row.samples);
            let iqr = format!("{:.3} ({q1:.3}–{q3:.3})", q3 - q1);
            (row.case.as_str(), median, iqr)
        })
        .collect();
    let case_w = summaries
        .iter()
        .map(|s| s.0.len())
        .max()
        .unwrap_or(0)
        .max(4);
    let iqr_header = "IQR ms (q1–q3)";
    let iqr_w = summaries
        .iter()
        .map(|s| s.2.chars().count())
        .max()
        .unwrap_or(0)
        .max(iqr_header.chars().count());

    println!(
        "{:<case_w$} | {:<7} | {:>9} | {:<iqr_w$} | steps",
        "case", "backend", "median ms", iqr_header
    );
    println!("{}", "-".repeat(case_w + iqr_w + 33));
    for ((case, median, iqr), row) in summaries.iter().zip(rows) {
        println!(
            "{case:<case_w$} | {backend:<7} | {median:>9.3} | {iqr:<iqr_w$} | {}",
            row.samples.len()
        );
    }
}

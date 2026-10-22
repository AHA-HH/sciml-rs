//! FNO on 2D Poisson on the Chebyshev grid (n nodes per axis, s = n − 1 on the model
//! grid); stage 2 (n = 65) by default
//!
//! Run with: cargo run --release --example train_poisson [-- [<padding>] [<n>]]
//! (CPU by default; add `--features metal` or `--features cuda` for a GPU -
//! see "Backends" in the README).
//!
//! An optional first argument overrides `PADDING`, so the padding sweep of
//! docs/phase3/T2-train-poisson.md needs no edits:
//! cargo run --release --example train_poisson -- 8
//! An optional second argument picks the stage by n, one of `loaders::poisson::STAGES`
//! (design §5.3):
//! cargo run --release --example train_poisson -- 4 129
//!
//! Structurally identical to `examples/train/darcy.rs`, with the Darcy model and
//! training hyperparameters (design §7). The loader transfers every sample from the
//! CGL grid to the uniform grid (CONVENTIONS §12); the targets are normalized, which
//! the trainer undoes before computing the loss. Needs the chosen stage's dataset: see
//! datasets/README.md.

use burn::tensor::Device;
use sciml_rs::neural_operators::{
    data::loaders::{
        base_dataset::DatasetConfig,
        poisson::{PoissonConfig, load_poisson_uniform, stage},
    },
    models::fno::FNOConfig,
    runs::RunDir,
    training::trainer::{TrainingConfig, train_fno},
};
use std::path::Path;

/// The stage used when no n is given: stage 2.
const DEFAULT_N: usize = 65;
const N_TRAIN: usize = 1000;
/// The whole test split of each stage.
const N_TEST: usize = 200;

const MODES: [usize; 2] = [12, 12];
const WIDTH: usize = 32;
const N_LAYERS: usize = 4;
const EPOCHS: usize = 500;
const BATCH_SIZE: usize = 20;
const LEARNING_RATE: f64 = 1e-3;
const WEIGHT_DECAY: f32 = 1e-4;
const MIN_LR: f64 = 1e-5;
/// Zero cells appended to each spatial axis (CONVENTIONS §3). Chosen by the full-length
/// sweep over {0, 4, 8, 16}: p = 8 scored lowest and p = 4 lies within 2% of it, so the
/// smaller p wins (design §12, decision 16; Phase 4 T2).
const PADDING: usize = 4;

fn main() {
    let padding = match std::env::args().nth(1) {
        Some(arg) => arg
            .parse::<usize>()
            .unwrap_or_else(|e| panic!("padding argument must be an integer, got {arg:?}: {e}")),
        None => PADDING,
    };
    let n = match std::env::args().nth(2) {
        Some(arg) => arg
            .parse::<usize>()
            .unwrap_or_else(|e| panic!("n argument must be an integer, got {arg:?}: {e}")),
        None => DEFAULT_N,
    };
    let set = stage(n).unwrap_or_else(|e| panic!("{e}"));
    let k = set.k;

    let device = Device::default().autodiff();
    println!("device: {device:?}, padding: {padding}, n: {n}, K: {k}");

    let dataset_cfg = PoissonConfig::new(
        DatasetConfig {
            n_train: N_TRAIN,
            n_test: N_TEST,
        },
        n,
    );

    let datasets = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/poisson");
    let train_path = datasets.join(set.npz("train"));
    let test_path = datasets.join(set.npz("test"));
    for p in [&train_path, &test_path] {
        assert!(
            p.exists(),
            "Poisson dataset not found at {}\nSee datasets/README.md",
            p.display()
        );
    }

    let (train_data, test_data, normalizers) =
        load_poisson_uniform::<f32>(&train_path, &test_path, &dataset_cfg)
            .unwrap_or_else(|e| panic!("could not load Poisson data: {e}"));
    println!(
        "poisson: {} train / {} test, n={} -> s={}",
        train_data.inputs().shape()[0],
        test_data.inputs().shape()[0],
        dataset_cfg.n,
        train_data.inputs().shape()[1]
    );

    let model_cfg = FNOConfig {
        modes: MODES.to_vec(),
        hidden_channels: WIDTH,
        data_channels: 1,
        out_channels: 1,
        n_layers: N_LAYERS,
        // None keeps the Li et al. U(0, 1/(I·O)) spectral init
        spectral_init: None,
        // Poisson is non-periodic: pad each axis with zero cells before the
        // spectral layers (cropped after). Some(0) means no padding
        padding: Some(padding),
    };

    let train_cfg = TrainingConfig::new()
        .with_epochs(EPOCHS)
        .with_batch_size(BATCH_SIZE)
        .with_learning_rate(LEARNING_RATE)
        .with_weight_decay(WEIGHT_DECAY)
        .with_min_lr(MIN_LR);

    // Stage 2 keeps its run name; other stages carry n in it
    let name = if n == DEFAULT_N {
        format!("poisson_fno_p{padding}")
    } else {
        format!("poisson_fno_n{n}_p{padding}")
    };
    // Writes configs, metrics, normalizers and weights to runs/<name>_<timestamp>/.
    // Created before training: train_fno appends each epoch's row of metrics.csv to it
    let run = RunDir::create(&Path::new(env!("CARGO_MANIFEST_DIR")).join("runs"), &name)
        .unwrap_or_else(|e| panic!("could not create the run directory: {e}"));

    // y_normalizer is fitted on y_train during loading. train_fno uses it
    // to decode both prediction and target during training, but only the
    // prediction during evaluation - y_test was never encoded
    let (model, metrics) = train_fno::<4, 3, _>(
        train_data,
        test_data,
        Some(&normalizers.y),
        &model_cfg,
        &train_cfg,
        &device,
        &run,
    );

    run.save_configs(&model_cfg, &train_cfg, &dataset_cfg)
        .expect("could not save configs");
    run.save_metrics(&metrics).expect("could not save metrics");
    run.save_normalizers(&normalizers)
        .expect("could not save normalizers");
    run.save_weights(&model)
        .unwrap_or_else(|e| panic!("could not save model weights: {e}"));

    println!("run written to {}", run.path().display());
}

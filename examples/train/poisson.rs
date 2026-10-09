//! FNO on 2D Poisson, stage 2 (n = 65 on the Chebyshev grid, s = 64 on the model grid)
//!
//! Run with: cargo run --release --example train_poisson
//! (CPU by default; add `--features metal` or `--features cuda` for a GPU -
//! see "Backends" in the README).
//!
//! An optional first argument overrides `PADDING`, so the padding sweep of
//! docs/phase3/T2-train-poisson.md needs no edits:
//! cargo run --release --example train_poisson -- 8
//!
//! Structurally identical to `examples/train/darcy.rs`, with the Darcy model and
//! training hyperparameters (design §7). The loader transfers every sample from the
//! CGL grid to the uniform grid (CONVENTIONS §12); the targets are normalized, which
//! the trainer undoes before computing the loss. Needs the stage 2 dataset: see
//! datasets/README.md.

use burn::{
    prelude::Config,
    store::{BurnpackStore, ModuleSnapshot},
    tensor::Device,
};
use sciml_rs::neural_operators::{
    data::loaders::{
        base_dataset::DatasetConfig,
        poisson::{PoissonConfig, load_poisson_uniform},
    },
    metrics::io::write_run_artifacts,
    models::fno::FNOConfig,
    training::{trainer::TrainingConfig, trainers::poisson::train_poisson},
};
use std::path::Path;

/// CGL nodes per axis in the dataset; the model grid is s = N − 1.
const N: usize = 65;
const N_TRAIN: usize = 1000;
/// The whole stage 2 test split.
const N_TEST: usize = 200;
const TRAIN_FILE: &str = "poisson_n65_K32_train.npz";
const TEST_FILE: &str = "poisson_n65_K32_test.npz";

const MODES: [usize; 2] = [12, 12];
const WIDTH: usize = 32;
const N_LAYERS: usize = 4;
const EPOCHS: usize = 500;
const BATCH_SIZE: usize = 20;
const LEARNING_RATE: f64 = 1e-3;
const WEIGHT_DECAY: f32 = 1e-4;
const MIN_LR: f64 = 1e-5;
/// Zero cells appended to each spatial axis (CONVENTIONS §3). Provisional: only p = 0
/// has been run at full length; the rest of the sweep over {0, 4, 8, 16} is deferred
/// (design §12, decisions 16 and 17).
const PADDING: usize = 0;

fn main() {
    let padding = match std::env::args().nth(1) {
        Some(arg) => arg
            .parse::<usize>()
            .unwrap_or_else(|e| panic!("padding argument must be an integer, got {arg:?}: {e}")),
        None => PADDING,
    };

    let device = Device::default().autodiff();
    println!("device: {device:?}, padding: {padding}");

    let dataset_cfg = PoissonConfig::new(
        DatasetConfig {
            n_train: N_TRAIN,
            n_test: N_TEST,
        },
        N,
    );

    let datasets = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/poisson");
    let train_path = datasets.join(TRAIN_FILE);
    let test_path = datasets.join(TEST_FILE);
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

    // y_normalizer is fitted on y_train during loading. train_poisson uses it
    // to decode both prediction and target during training, but only the
    // prediction during evaluation - y_test was never encoded
    let (model, metrics) = train_poisson(
        train_data,
        test_data,
        &normalizers.y,
        &model_cfg,
        &train_cfg,
        &device,
    );

    let dir = write_run_artifacts(
        &format!("poisson_fno_p{padding}"),
        &metrics,
        &model_cfg,
        &train_cfg,
        &dataset_cfg,
    )
    .unwrap_or_else(|e| panic!("could not write run artifacts: {e}"));

    // Write normalizers files to runs/poisson_fno_p<p>_<timestamp>/
    normalizers
        .x
        .to_record()
        .save(dir.join("x_normalizer.json"))
        .expect("save x normalizer");
    normalizers
        .y
        .to_record()
        .save(dir.join("y_normalizer.json"))
        .expect("save y normalizer");

    let mut store = BurnpackStore::from_file(dir.join("model_weights"));
    model
        .save_into(&mut store)
        .expect("could not save model weights");

    println!("run written to {}", dir.display());
}

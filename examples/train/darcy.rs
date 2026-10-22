//! FNO on 2D Darcy flow
//!
//! Run with: cargo run --release --example train_darcy
//! (CPU by default; add `--features metal` or `--features cuda` for a GPU -
//! see "Backends" in the README).
//!
//! Structurally identical to `examples/train/burgers.rs` - see that file for the
//! fully annotated version. The differences: `modes` has two entries instead of
//! one and the targets are normalized which the trainer has to undo before
//! computing the loss.

use burn::tensor::Device;
use sciml_rs::neural_operators::{
    data::loaders::{
        base_dataset::DatasetConfig,
        darcy::{DarcyConfig, load_darcy_uniform},
    },
    models::fno::FNOConfig,
    runs::RunDir,
    training::{config::TrainingConfig, trainer::train_fno},
};
use std::path::Path;

fn main() {
    let device = Device::default().autodiff();
    println!("device: {device:?}");

    let data_cfg = DatasetConfig {
        n_train: 1000,
        n_test: 100,
    };

    // subsample rate is 5 -> s = (421-1)/5 + 1 = 85 grid points per axis,
    // the paper's setting. Non-power-of-two sizes are fine: the FFT falls
    // back to Bluestein's algorithm.
    let dataset_cfg = DarcyConfig::new(data_cfg, 5);

    let datasets = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets");
    let train_path = datasets.join("piececonst_r421_N1024_smooth1.mat");
    let test_path = datasets.join("piececonst_r421_N1024_smooth2.mat");
    for p in [&train_path, &test_path] {
        assert!(
            p.exists(),
            "Darcy dataset not found at {}\nSee datasets/README.md",
            p.display()
        );
    }

    let (train_data, test_data, normalizers) =
        load_darcy_uniform::<f32>(&train_path, &test_path, &dataset_cfg)
            .unwrap_or_else(|e| panic!("could not load Darcy data: {e}"));
    println!(
        "darcy: {} train / {} test at s={}",
        train_data.inputs().shape()[0],
        test_data.inputs().shape()[0],
        train_data.inputs().shape()[1]
    );

    let model_cfg = FNOConfig {
        // Two entries so a 2D FNO model. Same code path as Burgers' vec![16]
        modes: vec![12, 12],
        hidden_channels: 32,
        data_channels: 1,
        out_channels: 1,
        n_layers: 4,
        // None keeps the Li et al. U(0, 1/(I·O)) spectral init; the zero-mean
        // alternatives are SpectralInit::Normal and SpectralInit::SymmetricUniform
        spectral_init: None,
        // Darcy is non-periodic, so pad the domain with 9 zero cells at the end
        // of each axis before the spectral layers (cropped after), as Li et al.'s
        // fourier_2d.py does. The padded 94 x 94 grid is not a power of two
        padding: Some(9),
    };

    let train_cfg = TrainingConfig::new()
        .with_epochs(500)
        .with_batch_size(20)
        .with_learning_rate(1e-3)
        .with_weight_decay(1e-4)
        .with_min_lr(1e-5);

    // Writes configs, metrics, normalizers and weights to runs/darcy_fno_<timestamp>/.
    // Created before training: train_fno appends each epoch's row of metrics.csv to it
    let run = RunDir::create(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("runs"),
        "darcy_fno",
    )
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
    run.save_plot(&metrics);
    run.save_normalizers(&normalizers)
        .expect("could not save normalizers");
    run.save_weights(&model)
        .unwrap_or_else(|e| panic!("could not save model weights: {e}"));

    println!("run written to {}", run.path().display());
}

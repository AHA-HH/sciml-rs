//! FNO on 1D Burgers, trained with Burn's `Learner`
//!
//! Same data, model and optimizer as `train_burgers`, but the training loop
//! is Burn's `SupervisedTraining`, which draws a live terminal dashboard (TUI)
//! with train/valid loss and learning rate.
//! Run with: cargo run --release --example train_burgers_learner
//! (run it in a real terminal - the TUI needs one; press `q` to stop early).
//!
//! The reported loss is the per-sample relative L2, comparable to `train_l2`
//! / `test_l2` printed by `train_burgers`.
//!
//! A showcase of Burn's `Learner`: `train_burgers` (the hand-written
//! `train_epoch` loop) is the canonical trainer. The run goes to
//! `runs/burgers_fno_learner_<unix_secs>/` with the same configs and weights as
//! `train_burgers`, so `predict_burgers` can evaluate it, plus Burn's own logs.
//! There is no `metrics.csv`: the Learner does not return per-epoch metrics.

use burn::{
    tensor::Device,
    train::{
        Learner, SupervisedTraining,
        metric::{LearningRateMetric, LossMetric},
    },
};
use sciml_rs::neural_operators::data::catalogue::BurgersFiles;
use sciml_rs::neural_operators::{
    data::loaders::{
        base_dataset::DatasetConfig,
        burgers::{BurgersConfig, load_burgers_uniform},
    },
    models::fno::FNOConfig,
    runs::RunDir,
    training::{config::TrainingConfig, trainer::build_training_components},
};
use std::path::Path;

fn main() {
    // Model and training tensors live on the autodiff device; the Learner
    // moves the validation loader to the inner device itself.
    let device = Device::default().autodiff();

    let data_cfg = DatasetConfig {
        n_train: 1024,
        n_test: 100,
    };
    // s = 2^13 / 32 = 256 grid points
    let dataset_cfg = BurgersConfig::new(data_cfg, 32);

    let data_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("datasets")
        .join(BurgersFiles::DATA);
    assert!(
        data_path.exists(),
        "Burgers dataset not found at {}\nSee datasets/README.md for the download link.",
        data_path.display()
    );
    let (train_data, test_data) = load_burgers_uniform::<f32>(&data_path, &dataset_cfg)
        .unwrap_or_else(|e| panic!("could not load Burgers data: {e}"));
    println!(
        "burgers: {} train / {} test at s={}",
        train_data.inputs().shape()[0],
        test_data.inputs().shape()[0],
        train_data.inputs().shape()[1]
    );

    let model_cfg = FNOConfig {
        modes: vec![16],
        hidden_channels: 64,
        data_channels: 1,
        out_channels: 1,
        n_layers: 4,
        // None keeps the Li et al. U(0, 1/(I·O)) spectral init; the zero-mean
        // alternatives are SpectralInit::Normal and SpectralInit::SymmetricUniform
        spectral_init: None,
        // No domain padding: Burgers is periodic, and Li et al.'s fourier_1d.py
        // does not pad either
        padding: None,
    };

    // 100 epochs, fewer than `train_burgers`' 500, so a TUI run finishes
    // quickly. Same metric as `train_burgers`, but after fewer epochs, so the
    // values differ.
    let train_cfg = TrainingConfig::new()
        .with_epochs(100)
        .with_batch_size(20)
        .with_learning_rate(1e-3)
        .with_weight_decay(1e-4)
        .with_min_lr(1e-5);

    // Same model, Adam optimizer, cosine schedule and loaders as the
    // hand-written loop.
    let c = build_training_components::<3, 2, _>(
        &model_cfg, &train_cfg, train_data, test_data, &device,
    );

    // The configs go in before training so a run stopped early is still described.
    let run = RunDir::create(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("runs"),
        "burgers_fno_learner",
    )
    .unwrap_or_else(|e| panic!("could not create the run directory: {e}"));
    run.save_configs(&model_cfg, &train_cfg, &dataset_cfg)
        .expect("could not save configs");

    let result = SupervisedTraining::new(run.path(), c.train_loader, c.test_loader)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .num_epochs(train_cfg.epochs)
        .summary()
        .launch(Learner::new(c.model, c.optim, c.scheduler));

    run.save_weights(&result.model)
        .unwrap_or_else(|e| panic!("could not save model weights: {e}"));

    println!("run written to {}", run.path().display());
}

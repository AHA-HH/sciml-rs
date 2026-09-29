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

use burn::{
    config::Config,
    store::{BurnpackStore, ModuleSnapshot},
    tensor::Device,
    train::{
        Learner, SupervisedTraining,
        metric::{LearningRateMetric, LossMetric},
    },
};
use sciml_rs::neural_operators::{
    data::loaders::{
        base_dataset::DatasetConfig,
        burgers::{BurgersConfig, load_burgers_uniform},
    },
    models::fno::FNOConfig,
    training::trainer::{TrainingConfig, build_training_components},
};
use std::path::Path;

const ARTIFACT_DIR: &str = "runs/burgers_fno_learner";

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

    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/burgers_data_R10.mat");
    assert!(
        data_path.exists(),
        "Burgers dataset not found at {}\nSee datasets/README.md for the download link.",
        data_path.display()
    );
    let (train_data, test_data) = load_burgers_uniform(&data_path, &dataset_cfg)
        .unwrap_or_else(|e| panic!("could not load Burgers data: {e}"));

    let model_cfg = FNOConfig {
        modes: vec![16],
        hidden_channels: 64,
        data_channels: 1,
        out_channels: 1,
        n_layers: 4,
    };

    // Fewer epochs than `train_burgers` so a TUI run finishes quickly.
    let train_cfg = TrainingConfig::new()
        .with_epochs(500)
        .with_batch_size(20)
        .with_learning_rate(1e-3)
        .with_weight_decay(1e-4)
        .with_min_lr(1e-5);

    // Same model, Adam optimizer, cosine schedule and loaders as the
    // hand-written loop.
    let c =
        build_training_components::<3, 2>(&model_cfg, &train_cfg, train_data, test_data, &device);

    std::fs::create_dir_all(ARTIFACT_DIR).expect("could not create artifact dir");

    let result = SupervisedTraining::new(ARTIFACT_DIR, c.train_loader, c.test_loader)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .num_epochs(train_cfg.epochs)
        .summary()
        .launch(Learner::new(c.model, c.optim, c.scheduler));

    let dir = Path::new(ARTIFACT_DIR);
    model_cfg
        .save(dir.join("model_cfg.json"))
        .expect("could not save model config");
    train_cfg
        .save(dir.join("train_cfg.json"))
        .expect("could not save training config");

    let mut store = BurnpackStore::from_file(dir.join("model_weights"));
    result
        .model
        .save_into(&mut store)
        .expect("could not save model weights");

    println!("run written to {}", dir.display());
}

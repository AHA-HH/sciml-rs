//! FNO experiment on 1D Burgers dataset

use burn::{
    store::{BurnpackStore, ModuleSnapshot},
    tensor::Device,
};
use sciml_rs::neural_operators::{
    data::loaders::{
        base_dataset::DatasetConfig,
        burgers::{BurgersConfig, load_burgers_uniform},
    },
    metrics::io::write_run_artifacts,
    models::fno::FNOConfig,
    training::{trainer::TrainingConfig, trainers::burgers::train_burgers},
};
use std::path::Path;

fn main() {
    let device = Device::default().autodiff();

    let data_cfg = DatasetConfig {
        n_train: 1024,
        n_test: 100,
    };
    let dataset_cfg = BurgersConfig::new(data_cfg.clone(), 32);

    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/burgers_data_R10.mat");
    assert!(
        data_path.exists(),
        "Burgers dataset not found at {}\nSee datasets/README.md for the download link.",
        data_path.display()
    );

    let (train_data, test_data) = load_burgers_uniform(&data_path, &dataset_cfg);

    let model_cfg = FNOConfig {
        modes: vec![16],
        width: 64,
        data_channels: 1,
        out_channels: 1,
        n_layers: 4,
    };

    let train_cfg = TrainingConfig::new()
        .with_epochs(500)
        .with_batch_size(20)
        .with_learning_rate(1e-3)
        .with_weight_decay(1e-4)
        .with_min_lr(1e-5);

    let (model, metrics) = train_burgers(
        train_data, test_data, &model_cfg, &train_cfg, &data_cfg, &device,
    );

    let dir = write_run_artifacts(
        "burgers_fno",
        &metrics,
        &model_cfg,
        &train_cfg,
        &dataset_cfg,
    );

    let mut store = BurnpackStore::from_file(dir.join("model_weights"));
    model
        .save_into(&mut store)
        .expect("could not save model weights");

    println!("run written to {}", dir.display());
}

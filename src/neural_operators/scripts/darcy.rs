//! FNO experiment on the 2D Darcy flow dataset.

use burn::tensor::Device;
use crate::neural_operators::{
    data::datasets::{
        base_dataset::DatasetConfig,
        darcy::{DarcyConfig, load_darcy_uniform},
    },
    metrics::io::write_run_artifacts,
    models::fno::FNOConfig,
    training::{trainer::TrainingConfig, trainers::darcy::train_darcy},
};

pub fn run_darcy() {
    let device = Device::default().autodiff();

    let data_cfg = DatasetConfig { n_train: 1000, n_test: 100 };
    let dataset_cfg = DarcyConfig::new(data_cfg.clone(), 28);

    let (train_data, test_data, y_normaliser) =
        load_darcy_uniform("<path>/piececonst_r421_N1024_smooth1.mat", "<path>/piececonst_r421_N1024_smooth2.mat", &dataset_cfg);

    let model_cfg = FNOConfig {
        modes: vec![4, 4],
        width: 32,
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

    let (_model, metrics) = train_darcy(
        train_data, test_data, &y_normaliser, &model_cfg, &train_cfg, &data_cfg, &device,
    );

    let dir = write_run_artifacts("darcy_fno", &metrics);
    println!("run written to {}", dir.display());
}
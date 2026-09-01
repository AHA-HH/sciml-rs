//! FNO experiment on the 2D Darcy flow dataset.

use burn::{
    store::{BurnpackStore, ModuleSnapshot},
    tensor::Device,
};
use sciml_rs::neural_operators::{
    data::loaders::{
        base_dataset::DatasetConfig,
        darcy::{DarcyConfig, load_darcy_uniform},
    },
    metrics::io::write_run_artifacts,
    models::fno::FNOConfig,
    training::{trainer::TrainingConfig, trainers::darcy::train_darcy},
};
use std::path::Path;

fn main() {
    let device = Device::default().autodiff();

    let data_cfg = DatasetConfig {
        n_train: 1000,
        n_test: 100,
    };
    let dataset_cfg = DarcyConfig::new(data_cfg.clone(), 28);

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

    let (train_data, test_data, y_normaliser) =
        load_darcy_uniform(&train_path, &test_path, &dataset_cfg);

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

    let (model, metrics) = train_darcy(
        train_data,
        test_data,
        &y_normaliser,
        &model_cfg,
        &train_cfg,
        &data_cfg,
        &device,
    );

    let dir = write_run_artifacts("darcy_fno", &metrics, &model_cfg, &train_cfg, &dataset_cfg);

    let mut store = BurnpackStore::from_file(dir.join("model_weights"));
    model
        .save_into(&mut store)
        .expect("could not save model weights");

    println!("run written to {}", dir.display());
}

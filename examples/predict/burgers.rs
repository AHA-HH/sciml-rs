//! Re-evaluates a saved Burgers run on the test split.
//!
//! Run with: cargo run --release --example predict_burgers -- runs/burgers_fno_<timestamp>
//!
//! Loads the architecture from model_cfg.json, the weights from
//! model_weights.bpk and reconstructs the model.

use std::path::PathBuf;

use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    store::{BurnpackStore, ModuleSnapshot},
    tensor::Device,
};
use sciml_rs::neural_operators::{
    data::{
        batcher::OperatorBatcher,
        loaders::{
            base_dataset::HasBaseConfig,
            burgers::{BurgersConfig, load_burgers_uniform},
        },
    },
    losses::data_losses::{LpLoss, Reduction},
    models::fno::FNOConfig,
    training::trainer::{eval_epoch, identity},
};
use std::path::Path;

fn main() {
    let dir = PathBuf::from(std::env::args().nth(1).expect("usage: predict <run_dir>"));
    // Inference only: no autodiff, so forward passes record no backward graph.
    let device = Device::default();

    // Architecture from the saved config, weights from the saved record
    let model_cfg = FNOConfig::load(dir.join("model_cfg.json")).expect("load model config");
    let mut model = model_cfg.init::<3>(&device);
    let mut store = BurnpackStore::from_file(dir.join("model_weights.bpk"));
    model.load_from(&mut store).expect("load model weights");

    // Same test split the run was evaluated on
    let dataset_cfg = BurgersConfig::load(dir.join("data_cfg.json")).expect("load dataset config");
    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/burgers_data_R10.mat");
    assert!(
        data_path.exists(),
        "Burgers dataset not found at {}\nSee datasets/README.md for the download link.",
        data_path.display()
    );

    let (_, test_data) = load_burgers_uniform::<f32>(&data_path, &dataset_cfg)
        .unwrap_or_else(|e| panic!("could not load Burgers data: {e}"));

    let test_loader = DataLoaderBuilder::new(OperatorBatcher::<3, 2>::new())
        .set_device(device.clone())
        .batch_size(20)
        .build(test_data);

    // Burgers has no normalizer, so predictions need no postprocessing - hence `identity`
    // Compare examples/predict/darcy.rs, which decodes
    let loss_fn = LpLoss::new(1, 2, Reduction::Sum);
    let l2 = eval_epoch::<3, 2>(&model, &test_loader, &loss_fn, &identity);

    println!(
        "loaded model test_l2: {:.10}",
        l2 / dataset_cfg.base().n_test as f32
    );
    println!(
        "(should match the last test_l2 in {}/metrics.csv)",
        dir.display()
    );
}

//! Loads a saved Burgers run and re-evaluates it on the test split.
//!
//! Run with: cargo run --release --example predict -- runs/burgers_fno_<stamp>

use std::path::PathBuf;

use burn::{config::Config, data::dataloader::DataLoaderBuilder, store::{BurnpackStore, ModuleSnapshot}, tensor::Device};
use sciml_rs::neural_operators::{
    data::{batcher::OperatorBatcher, loaders::{base_dataset::HasBaseConfig, burgers::{BurgersConfig, load_burgers_uniform}}},
    losses::data_losses::LpLoss,
    models::fno::FNOConfig,
    training::trainer::{eval_epoch, identity},
};

fn main() {
    let dir = PathBuf::from(std::env::args().nth(1).expect("usage: predict <run_dir>"));
    let device = Device::default().autodiff();

    // Architecture from the saved config, weights from the saved record.
    let model_cfg = FNOConfig::load(dir.join("model_config.json")).expect("load model config");
    let mut model = model_cfg.init::<3>(&device);
    let mut store = BurnpackStore::from_file(dir.join("model.bpk"));
    model.load_from(&mut store).expect("load model weights");

    // Same test split the run was evaluated on.
    let dataset_cfg = BurgersConfig::load(dir.join("dataset.json")).expect("load dataset config");
    // let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/burgers_data_R10.mat");
    let (_, test_data) = load_burgers_uniform("/Users/aneeshussain/Code/Datasets/burgers_data_R10.mat", &dataset_cfg);

    let test_loader = DataLoaderBuilder::new(OperatorBatcher::<3, 2>::new(device.clone()))
        .batch_size(20)
        .build(test_data);

    let loss_fn = LpLoss::new(1, 2, false, true);
    let l2 = eval_epoch::<3, 2>(&model, &test_loader, &loss_fn, &identity);

    println!("loaded model test_l2: {:.6}", l2 / dataset_cfg.base().n_test as f32);
    println!("(should match the last test_l2 in {}/metrics.csv)", dir.display());
}
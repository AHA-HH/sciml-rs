//! Re-evaluates a saved Burgers run on the test split.
//!
//! Run with: cargo run --release --example predict_burgers -- runs/burgers_fno_<timestamp>
//!
//! Loads the model and the dataset config from the run directory (`RunDir`).

use burn::{data::dataloader::DataLoaderBuilder, tensor::Device};
use sciml_rs::neural_operators::{
    data::{
        batcher::OperatorBatcher,
        loaders::burgers::{BurgersConfig, load_burgers_uniform},
    },
    runs::{ConfigFile, RunDir},
    training::trainer::{eval_epoch, identity},
};
use std::path::Path;

fn main() {
    let run = RunDir::open(std::env::args().nth(1).expect("usage: predict <run_dir>"))
        .unwrap_or_else(|e| panic!("open the run: {e}"));
    // Inference only: no autodiff, so forward passes record no backward graph.
    let device = Device::default();

    // Architecture from the saved config, weights from the saved record
    let model = run
        .load_fno::<3>(&device)
        .unwrap_or_else(|e| panic!("load the model: {e}"));

    // Same test split the run was evaluated on
    let dataset_cfg: BurgersConfig = run
        .config(ConfigFile::Data)
        .unwrap_or_else(|e| panic!("load dataset config: {e}"));
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
    let l2 = eval_epoch::<3, 2>(&model, &test_loader, &identity);

    println!(
        "loaded model test_l2: {:.10}",
        l2 / dataset_cfg.base.n_test as f32
    );
    // `train_burgers_learner` runs have no metrics.csv to compare against
    if run.metrics().is_ok() {
        println!(
            "(should match the last test_l2 in {}/metrics.csv)",
            run.path().display()
        );
    }
}

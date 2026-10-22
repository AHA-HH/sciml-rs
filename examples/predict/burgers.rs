//! Re-evaluates a saved Burgers run on the test split.
//!
//! Run with: cargo run --release --example predict_burgers -- runs/burgers_fno_<timestamp>
//!
//! Loads the model and the dataset config from the run directory (`RunDir`).

use burn::tensor::Device;
use sciml_rs::neural_operators::{
    data::{
        device_batcher::device_loader,
        loaders::burgers::{BurgersConfig, load_burgers_uniform},
    },
    runs::RunDir,
    training::step::evaluate,
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
        .data_config()
        .unwrap_or_else(|e| panic!("load dataset config: {e}"));
    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/burgers_data_R10.mat");
    assert!(
        data_path.exists(),
        "Burgers dataset not found at {}\nSee datasets/README.md for the download link.",
        data_path.display()
    );

    let (_, test_data) = load_burgers_uniform::<f32>(&data_path, &dataset_cfg)
        .unwrap_or_else(|e| panic!("could not load Burgers data: {e}"));

    let test_loader = device_loader::<3, 2, _>(&test_data, &device, 20, None);

    // Burgers has no normalizer, so predictions need no decoding - hence `None`
    // Compare examples/predict/darcy.rs, which decodes
    let l2 = evaluate::<3, 2>(&model, &test_loader, None);

    println!("loaded model test_l2: {l2:.10}");
    // `train_burgers_learner` runs have no metrics.csv to compare against
    if run.metrics().is_ok() {
        println!(
            "(should match the last test_l2 in {}/metrics.csv)",
            run.path().display()
        );
    }
}

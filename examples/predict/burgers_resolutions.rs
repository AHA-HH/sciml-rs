//! Burgers Resolution Sweep
//! Evaluates a saved Burgers model at resolutions it was never trained on.
//!
//! Run with: cargo run --release --example burgers_resolution -- runs/burgers_fno_<timestamp>
//!
//! The FNO learns a mapping between function spaces rather than between grids,
//! so test error should stay roughly flat as the discretisation changes. The
//! model here was trained at s=256 (subsample rate 32).

use std::path::Path;

use burn::{data::dataloader::DataLoaderBuilder, tensor::Device};
use sciml_rs::neural_operators::{
    data::{
        batcher::OperatorBatcher,
        loaders::{
            base_dataset::DatasetConfig,
            burgers::{BurgersConfig, load_burgers_uniform},
        },
    },
    runs::{ConfigFile, RunDir},
    training::trainer::{eval_epoch, identity},
};

fn main() {
    let run = RunDir::open(
        std::env::args()
            .nth(1)
            .expect("usage: burgers_resolution <run_dir>"),
    )
    .unwrap_or_else(|e| panic!("open the run: {e}"));
    // Inference only: no autodiff, so forward passes record no backward graph.
    let device = Device::default();

    // Architecture from the saved config, weights from the saved record.
    let model = run
        .load_fno::<3>(&device)
        .unwrap_or_else(|e| panic!("load the model: {e}"));

    let trained_cfg: BurgersConfig = run
        .config(ConfigFile::Data)
        .unwrap_or_else(|e| panic!("load dataset config: {e}"));
    let n_test = trained_cfg.base.n_test;
    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/burgers_data_R10.mat");
    assert!(
        data_path.exists(),
        "Burgers dataset not found at {}\nSee datasets/README.md",
        data_path.display()
    );

    // s = 8192 / rate. Each rate must divide 8192 so the subsampled grid
    // matches `BurgersConfig::s()` (the loader rejects others); the FFT itself
    // handles any s.
    println!("{:>6}  {:>10}", "s", "test_l2");
    for rate in [128, 64, 32, 16, 8, 4] {
        let cfg = BurgersConfig::new(
            DatasetConfig {
                n_train: trained_cfg.base.n_train,
                n_test,
            },
            rate,
        );

        let (_, test_data) = load_burgers_uniform::<f32>(&data_path, &cfg)
            .unwrap_or_else(|e| panic!("could not load Burgers data: {e}"));
        let test_loader = DataLoaderBuilder::new(OperatorBatcher::<3, 2>::new())
            .set_device(device.clone())
            .batch_size(20)
            .build(test_data);

        let l2 = eval_epoch::<3, 2>(&model, &test_loader, &identity);
        let marker = if rate == trained_cfg.subsample_rate {
            "  <- trained here"
        } else {
            ""
        };
        println!("{:>6}  {:>10.5}{}", cfg.s(), l2 / n_test as f32, marker);
    }
}

//! Burgers Resolution Sweep
//! Evaluates a saved Burgers model at resolutions it was never trained on.
//!
//! Run with: cargo run --release --example burgers_resolution -- runs/burgers_fno_<timestamp>
//!
//! The FNO learns a mapping between function spaces rather than between grids,
//! so test error should stay roughly flat as the discretisation changes. The
//! model here was trained at s=256 (subsample rate 32).

use std::path::{Path, PathBuf};

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
            base_dataset::{BaseDatasetConfig, DatasetConfig},
            burgers::{BurgersConfig, load_burgers_uniform},
        },
    },
    losses::data_losses::{LpLoss, Reduction},
    models::fno::FNOConfig,
    training::trainer::{eval_epoch, identity},
};

fn main() {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: burgers_resolution <run_dir>"),
    );
    // Inference only: no autodiff, so forward passes record no backward graph.
    let device = Device::default();

    // Architecture from the saved config, weights from the saved record.
    let model_cfg = FNOConfig::load(dir.join("model_cfg.json")).expect("load model config");
    let mut model = model_cfg.init::<3>(&device);
    let mut store = BurnpackStore::from_file(dir.join("model_weights.bpk"));
    model.load_from(&mut store).expect("load model weights");

    let trained_cfg = BurgersConfig::load(dir.join("data_cfg.json")).expect("load dataset config");
    let n_test = trained_cfg.n_test();
    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/burgers_data_R10.mat");
    assert!(
        data_path.exists(),
        "Burgers dataset not found at {}\nSee datasets/README.md",
        data_path.display()
    );

    let loss_fn = LpLoss::new(1, 2, Reduction::Sum);

    // s = 8192 / rate. Each rate must divide 8192 so the subsampled grid
    // matches `BurgersConfig::s()` (the loader rejects others); the FFT itself
    // handles any s.
    println!("{:>6}  {:>10}", "s", "test_l2");
    for rate in [128, 64, 32, 16, 8, 4] {
        let cfg = BurgersConfig::new(
            DatasetConfig {
                n_train: trained_cfg.n_train(),
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

        let l2 = eval_epoch::<3, 2>(&model, &test_loader, &loss_fn, &identity);
        let marker = if rate == trained_cfg.subsample_rate {
            "  <- trained here"
        } else {
            ""
        };
        println!("{:>6}  {:>10.5}{}", cfg.s(), l2 / n_test as f32, marker);
    }
}

//! Zero-shot super-resolution: a Burgers model trained at one grid resolution,
//! evaluated at finer ones it never saw.
//!
//! Run with:
//!   cargo run --release --example burgers_superresolution -- runs/burgers_fno_<timestamp>
//!
//! An FNO learns a mapping between function spaces, not between grids: the
//! spectral layers act on Fourier modes, and the mode count is fixed by the
//! config rather than by the discretisation. So the same weights apply at any
//! resolution, and test error should stay roughly flat as the grid refines.
//!
//! Burgers' raw grid is 8192 = 2^13, so every power-of-two subsample rate
//! gives a power-of-two resolution — which Burn's radix-2 FFT requires.

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
    losses::data_losses::LpLoss,
    models::fno::FNOConfig,
    training::trainer::{eval_epoch, identity},
};

fn main() {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: burgers_superresolution <run_dir>"),
    );
    let device = Device::default().autodiff();

    // Architecture from the saved config, weights from the saved record.
    let model_cfg = FNOConfig::load(dir.join("model_cfg.json")).expect("load model config");
    let mut model = model_cfg.init::<3>(&device);
    let mut store = BurnpackStore::from_file(dir.join("model_weights.bpk"));
    model.load_from(&mut store).expect("load model weights");

    let trained_cfg = BurgersConfig::load(dir.join("data_cfg.json")).expect("load dataset config");
    let (n_train, n_test) = (trained_cfg.n_train(), trained_cfg.n_test());
    let trained_s = trained_cfg.s();

    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/burgers_data_R10.mat");
    assert!(
        data_path.exists(),
        "Burgers dataset not found at {}\nSee datasets/README.md",
        data_path.display()
    );

    let loss_fn = LpLoss::new(1, 2, false, true);
    let mut results = Vec::new();

    // Training resolution first, then progressively finer grids.
    for rate in [trained_cfg.subsample_rate, 16, 8, 4, 2, 1] {
        let cfg = BurgersConfig::new(DatasetConfig { n_train, n_test }, rate);
        let s = cfg.s();
        if s < trained_s {
            continue; // super-resolution only: skip anything coarser
        }

        let (_, test_data) = load_burgers_uniform(&data_path, &cfg);
        let test_loader = DataLoaderBuilder::new(OperatorBatcher::<3, 2>::new(device.clone()))
            .batch_size(20)
            .build(test_data);

        let l2 = eval_epoch::<3, 2>(&model, &test_loader, &loss_fn, &identity) / n_test as f32;
        results.push((s, l2));
    }

    println!(
        "\ntrained at s = {trained_s}, modes = {:?}\n",
        model_cfg.modes
    );
    println!("{:>7}  {:>10}  {:>8}", "s", "test_l2", "vs train");

    let baseline = results[0].1;
    for (s, l2) in &results {
        let ratio = l2 / baseline;
        let marker = if *s == trained_s { " <- trained" } else { "" };
        println!("{s:>7}  {l2:>10.5}  {ratio:>7.2}x{marker}");
    }
}

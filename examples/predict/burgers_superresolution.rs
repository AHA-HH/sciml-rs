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
//! The model generates the coordinate channels for each input's resolution,
//! so only the data is resampled here.
//!
//! Burgers' raw grid is 8192 = 2^13, and each subsample rate r gives
//! s = 8192 / r. The rates must divide 8192 so the subsampled grid matches
//! `BurgersConfig::s()` (the loader rejects others); the FFT itself handles
//! any s.

use std::path::Path;

use burn::tensor::Device;
use sciml_rs::neural_operators::{
    data::{
        device_batcher::device_loader,
        loaders::{
            base_dataset::DatasetConfig,
            burgers::{BurgersConfig, load_burgers_uniform},
        },
    },
    models::fno::FNOConfig,
    runs::{ConfigFile, RunDir},
    training::trainer::{eval_epoch, identity},
};

fn main() {
    let run = RunDir::open(
        std::env::args()
            .nth(1)
            .expect("usage: burgers_superresolution <run_dir>"),
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
    let (n_train, n_test) = (trained_cfg.base.n_train, trained_cfg.base.n_test);
    let trained_s = trained_cfg.s();

    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/burgers_data_R10.mat");
    assert!(
        data_path.exists(),
        "Burgers dataset not found at {}\nSee datasets/README.md",
        data_path.display()
    );

    let mut results = Vec::new();

    // Training resolution first, then progressively finer grids.
    for rate in [trained_cfg.subsample_rate, 16, 8, 4, 2, 1] {
        let cfg = BurgersConfig::new(DatasetConfig { n_train, n_test }, rate);
        let s = cfg.s();
        if s < trained_s {
            continue; // super-resolution only: skip anything coarser
        }

        let (_, test_data) = load_burgers_uniform::<f32>(&data_path, &cfg)
            .unwrap_or_else(|e| panic!("could not load Burgers data: {e}"));
        let test_loader = device_loader::<3, 2, _>(&test_data, &device, 20, None);

        let l2 = eval_epoch::<3, 2>(&model, &test_loader, &identity) / n_test as f32;
        results.push((s, l2));
    }

    let model_cfg: FNOConfig = run
        .config(ConfigFile::Model)
        .unwrap_or_else(|e| panic!("load model config: {e}"));
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

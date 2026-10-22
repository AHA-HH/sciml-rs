//! Burgers resolution sweep: a saved Burgers model evaluated at grid resolutions it was
//! never trained on, coarser (sub-resolution) and finer (zero-shot super-resolution).
//!
//! Run with:
//!   cargo run --release --example burgers_resolution -- runs/burgers_fno_<timestamp> [<rate> ...]
//!
//! An FNO learns a mapping between function spaces, not between grids: the spectral
//! layers act on Fourier modes, and the mode count is fixed by the config rather than by
//! the discretisation. So the same weights apply at any resolution, and test error should
//! stay roughly flat as the grid changes. The model generates the coordinate channels for
//! each input's resolution, so only the data is resampled here.
//!
//! Burgers' raw grid is 8192 = 2^13, and each subsample rate r gives s = 8192 / r. The
//! rates must divide 8192 so the subsampled grid matches `BurgersConfig::s()` (the loader
//! rejects others); the FFT itself handles any s. Without rates, the sweep runs
//! `DEFAULT_RATES`, s = 64 to 8192. Each row also gives the error relative to the
//! trained resolution's.

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
    runs::RunDir,
    training::step::{evaluate, identity},
};

/// Subsample rates swept when none are given: s = 64 to 8192.
const DEFAULT_RATES: [usize; 8] = [128, 64, 32, 16, 8, 4, 2, 1];
const BATCH_SIZE: usize = 20;

fn main() {
    let usage = "usage: burgers_resolution <run_dir> [<rate> ...]";
    let mut args = std::env::args().skip(1);
    let run =
        RunDir::open(args.next().expect(usage)).unwrap_or_else(|e| panic!("open the run: {e}"));
    let rates: Vec<usize> = args
        .map(|a| {
            a.parse()
                .unwrap_or_else(|_| panic!("rate must be a positive integer, got {a:?}\n{usage}"))
        })
        .collect();
    let rates = if rates.is_empty() {
        DEFAULT_RATES.to_vec()
    } else {
        rates
    };
    // Inference only: no autodiff, so forward passes record no backward graph.
    let device = Device::default();

    // Architecture from the saved config, weights from the saved record.
    let model = run
        .load_fno::<3>(&device)
        .unwrap_or_else(|e| panic!("load the model: {e}"));
    let model_cfg: FNOConfig = run
        .model_config()
        .unwrap_or_else(|e| panic!("load model config: {e}"));
    let trained_cfg: BurgersConfig = run
        .data_config()
        .unwrap_or_else(|e| panic!("load dataset config: {e}"));
    let (n_train, n_test) = (trained_cfg.base.n_train, trained_cfg.base.n_test);

    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/burgers_data_R10.mat");
    assert!(
        data_path.exists(),
        "Burgers dataset not found at {}\nSee datasets/README.md",
        data_path.display()
    );

    // Mean test relative L² at one subsample rate
    let test_l2 = |rate: usize| {
        let cfg = BurgersConfig::new(DatasetConfig { n_train, n_test }, rate);
        let (_, test_data) = load_burgers_uniform::<f32>(&data_path, &cfg)
            .unwrap_or_else(|e| panic!("could not load Burgers data at rate {rate}: {e}"));
        let test_loader = device_loader::<3, 2, _>(&test_data, &device, BATCH_SIZE, None);
        (cfg.s(), evaluate::<3, 2>(&model, &test_loader, &identity))
    };

    let trained_rate = trained_cfg.subsample_rate;
    let (trained_s, baseline) = test_l2(trained_rate);
    println!(
        "trained at s = {trained_s}, modes = {:?}\n",
        model_cfg.modes
    );
    println!("{:>7}  {:>10}  {:>8}", "s", "test_l2", "vs train");
    for rate in rates {
        let (s, l2) = test_l2(rate);
        let marker = if rate == trained_rate {
            " <- trained"
        } else {
            ""
        };
        println!("{s:>7}  {l2:>10.5}  {:>7.2}x{marker}", l2 / baseline);
    }
}

//! FNO on 2D Poisson on the Chebyshev grid (n nodes per axis, s on the model grid, by
//! default s = n − 1); stage 2 (n = 65) by default
//!
//! Run with: cargo run --release --example train_poisson [-- [<padding>] [<n>] [<s>] [<seed>]]
//! (CPU by default; add `--features metal` or `--features cuda` for a GPU -
//! see "Backends" in the README).
//!
//! An optional first argument overrides `PADDING`, so the padding sweep of
//! docs/phase3/T2-train-poisson.md needs no edits:
//! cargo run --release --example train_poisson -- 8
//! An optional second argument picks the stage by n, one of `catalogue::STAGES`
//! (design §5.3):
//! cargo run --release --example train_poisson -- 4 129
//! An optional third argument sets the learning grid s (CONVENTIONS §12), from the same
//! dataset files; the modes follow it (design §12, decision 30, D2):
//! cargo run --release --example train_poisson -- 0 129 32
//! An optional fourth argument sets `TrainingConfig::seed` (default 42; weight
//! initialisation and the shuffle), named in the run when not 42:
//! cargo run --release --example train_poisson -- 0 129 32 7
//!
//! Besides the configs, metrics, normalizers and weights, the run records `costs.json`
//! (`RunCosts`, design §13.9): the dataset's generation time from its sidecars, the load,
//! transfer and training times, host-to-device bytes, the parameter count, backend,
//! device and commit. `data_cfg.json` records n, K, s, d and both split files with their
//! generating commits.
//!
//! Structurally identical to `examples/train/darcy.rs`, with the Darcy model and
//! training hyperparameters (design §7). The loader transfers every sample from the
//! CGL grid to the uniform grid (CONVENTIONS §12); the targets are normalized, which
//! the trainer undoes before computing the loss. Needs the chosen stage's dataset: see
//! datasets/README.md.

use burn::{config::Config, module::Module, tensor::Device};
use sciml_rs::neural_operators::{
    chebyshev::TRANSFER_DEGREE,
    data::{
        catalogue::stage,
        dataset::OperatorDataset,
        loaders::{
            base_dataset::DatasetConfig,
            poisson::{
                DatasetSource, PoissonConfig, SidecarRecord, SplitSource, load_poisson_uniform,
            },
        },
    },
    models::fno::FNOConfig,
    runs::{RunCosts, RunDir},
    training::{config::TrainingConfig, trainer::train_fno},
    utils::backend::{backend_features, git_commit},
};
use std::fmt::Display;
use std::path::Path;
use std::str::FromStr;
use std::time::Instant;

/// The stage used when no n is given: stage 2.
const DEFAULT_N: usize = 65;
const N_TRAIN: usize = 1000;
/// The whole test split of each stage.
const N_TEST: usize = 200;

/// Fourier modes per axis at large s; m = min(MAX_MODES, s/2) (design §12, decision 30, D2).
const MAX_MODES: usize = 12;
const WIDTH: usize = 32;
const N_LAYERS: usize = 4;
const EPOCHS: usize = 500;
const BATCH_SIZE: usize = 20;
const LEARNING_RATE: f64 = 1e-3;
const WEIGHT_DECAY: f32 = 1e-4;
const MIN_LR: f64 = 1e-5;
/// Zero cells appended to each spatial axis (CONVENTIONS §3). Chosen by the full-length
/// sweep over {0, 4, 8, 16}: p = 8 scored lowest and p = 4 lies within 2% of it, so the
/// smaller p wins (design §12, decision 16; Phase 4 T2).
const PADDING: usize = 4;

/// The `i`-th command-line argument parsed as `T`, or `default` when it is absent.
///
/// # Panics
/// If the argument does not parse, naming it `name`.
fn arg<T: FromStr>(i: usize, name: &str, default: T) -> T
where
    T::Err: Display,
{
    match std::env::args().nth(i) {
        Some(arg) => arg
            .parse()
            .unwrap_or_else(|e| panic!("{name} argument must be an integer, got {arg:?}: {e}")),
        None => default,
    }
}

fn main() {
    let padding = arg(1, "padding", PADDING);
    let n = arg(2, "n", DEFAULT_N);
    let set = stage(n).unwrap_or_else(|e| panic!("{e}"));
    let k = set.k;
    let s = arg(3, "s", n - 1);
    let default_seed = TrainingConfig::new().seed;
    let seed = arg(4, "seed", default_seed);
    let modes = MAX_MODES.min(s / 2);

    let device = Device::default().autodiff();
    println!(
        "device: {device:?}, padding: {padding}, n: {n}, s: {s}, modes: {modes}, K: {k}, \
         seed: {seed}"
    );

    let datasets = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/poisson");
    let train_path = datasets.join(set.npz("train"));
    let test_path = datasets.join(set.npz("test"));
    for p in [&train_path, &test_path] {
        assert!(
            p.exists(),
            "Poisson dataset not found at {}\nSee datasets/README.md",
            p.display()
        );
    }
    let sidecar = |split: &str| {
        let path = datasets.join(set.json(split));
        SidecarRecord::load(&path)
            .unwrap_or_else(|e| panic!("could not read the sidecar {}: {e}", path.display()))
    };
    let (train_sidecar, test_sidecar) = (sidecar("train"), sidecar("test"));
    let split = |split: &str, sidecar: &SidecarRecord| {
        SplitSource::new(set.npz(split), sidecar.git_commit.clone())
    };

    let dataset_cfg = PoissonConfig::new(
        DatasetConfig {
            n_train: N_TRAIN,
            n_test: N_TEST,
        },
        n,
    )
    .with_s(Some(s))
    .with_source(Some(DatasetSource::new(
        k,
        TRANSFER_DEGREE,
        split("train", &train_sidecar),
        split("test", &test_sidecar),
    )));

    let load_start = Instant::now();
    let (train_data, test_data, normalizers) =
        load_poisson_uniform::<f32>(&train_path, &test_path, &dataset_cfg)
            .unwrap_or_else(|e| panic!("could not load Poisson data: {e}"));
    let load_time_s = load_start.elapsed().as_secs_f64();
    // what device_loader uploads: inputs and targets as stored
    let bytes = |d: &OperatorDataset<f32>| {
        (d.inputs().len() + d.targets().len()) * std::mem::size_of::<f32>()
    };
    let (train_bytes, test_bytes) = (bytes(&train_data), bytes(&test_data));
    println!(
        "poisson: {} train / {} test, n={} -> s={}",
        train_data.inputs().shape()[0],
        test_data.inputs().shape()[0],
        dataset_cfg.n,
        train_data.inputs().shape()[1]
    );

    let model_cfg = FNOConfig {
        modes: vec![modes; 2],
        hidden_channels: WIDTH,
        data_channels: 1,
        out_channels: 1,
        n_layers: N_LAYERS,
        // None keeps the Li et al. U(0, 1/(I·O)) spectral init
        spectral_init: None,
        // Poisson is non-periodic: pad each axis with zero cells before the
        // spectral layers (cropped after). Some(0) means no padding
        padding: Some(padding),
    };

    let train_cfg = TrainingConfig::new()
        .with_epochs(EPOCHS)
        .with_batch_size(BATCH_SIZE)
        .with_learning_rate(LEARNING_RATE)
        .with_weight_decay(WEIGHT_DECAY)
        .with_min_lr(MIN_LR)
        .with_seed(seed);

    // Stage 2 at s = n − 1 keeps its run name; other stages carry n, other s carry n and s,
    // and a seed other than the default is appended
    let grid = match (n, s == n - 1) {
        (DEFAULT_N, true) => String::new(),
        (_, true) => format!("_n{n}"),
        (_, false) => format!("_n{n}_s{s}"),
    };
    let seed_tag = if seed == default_seed {
        String::new()
    } else {
        format!("_seed{seed}")
    };
    let name = format!("poisson_fno{grid}_p{padding}{seed_tag}");
    // Writes configs, metrics, normalizers and weights to runs/<name>_<timestamp>/.
    // Created before training: train_fno appends each epoch's row of metrics.csv to it
    let run = RunDir::create(&Path::new(env!("CARGO_MANIFEST_DIR")).join("runs"), &name)
        .unwrap_or_else(|e| panic!("could not create the run directory: {e}"));

    // y_normalizer is fitted on y_train during loading. train_fno uses it
    // to decode both prediction and target during training, but only the
    // prediction during evaluation - y_test was never encoded
    // The recorded training_time_s: it includes building the model, optimiser and loaders,
    // so it is slightly above the "total training time" train_fno prints from its own clock
    let train_start = Instant::now();
    let (model, metrics) = train_fno::<4, 3, _>(
        train_data,
        test_data,
        Some(&normalizers.y),
        &model_cfg,
        &train_cfg,
        &device,
        &run,
    );
    let training_time_s = train_start.elapsed().as_secs_f64();

    let costs = RunCosts {
        dataset_train_wall_time_s: Some(train_sidecar.wall_time_s),
        dataset_test_wall_time_s: Some(test_sidecar.wall_time_s),
        load_time_s,
        training_time_s,
        inference_time_s: None,
        host_to_device_bytes_train: train_bytes,
        host_to_device_bytes_test: test_bytes,
        // no backend reports it; never estimated (design §13.9)
        peak_gpu_memory: "unavailable".to_string(),
        backend_features: backend_features().into_iter().map(String::from).collect(),
        device: format!("{device:?}"),
        git_commit: git_commit(),
        num_params: model.num_params(),
    };

    run.save_configs(&model_cfg, &train_cfg, &dataset_cfg)
        .expect("could not save configs");
    run.save_plot(&metrics);
    run.save_normalizers(&normalizers)
        .expect("could not save normalizers");
    run.save_weights(&model)
        .unwrap_or_else(|e| panic!("could not save model weights: {e}"));
    run.save_costs(&costs).expect("could not save costs.json");

    println!("run written to {}", run.path().display());
}

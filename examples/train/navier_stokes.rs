//! FNO on 2D Navier-Stokes (vorticity), ν = 1e-3
//!
//! Run with: cargo run --release --example train_navier_stokes
//!
//! Structurally identical to `examples/train/darcy.rs`. The differences: the
//! model convolves over (x, y, t), so `modes` has three entries; the input is
//! the first `t_in` vorticity snapshots and the target the next `t_out`.
//!
//! Hyperparameters follow `fourier_3d.py` (zongyi-li/fourier_neural_operator).
//! Known differences from that script, none of which are changed here:
//! - no padding of the (non-periodic) time axis; the script pads t by 6
//! - ReLU between layers; the script uses GELU
//! - cosine LR decay per step; the script uses StepLR(step=100, gamma=0.5)
//! - the normalizer std is the population std; torch.std is the sample std

use burn::{
    prelude::Config,
    store::{BurnpackStore, ModuleSnapshot},
    tensor::Device,
};
use sciml_rs::neural_operators::{
    data::loaders::{
        base_dataset::DatasetConfig,
        navier_stokes::{NavierStokesConfig, load_navier_stokes_uniform},
    },
    metrics::io::write_run_artifacts,
    models::fno::FNOConfig,
    training::{trainer::TrainingConfig, trainers::navier_stokes::train_navier_stokes},
};
use std::path::Path;

fn main() {
    let device = Device::default().autodiff();

    let data_cfg = DatasetConfig {
        n_train: 1000,
        n_test: 200,
    };

    // First 10 snapshots in, next 40 out, on the full 64x64 grid.
    let dataset_cfg = NavierStokesConfig::new(data_cfg.clone())
        .with_t_in(10)
        .with_t_out(40)
        .with_subsample_rate(1);

    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("datasets")
        .join("ns_V1e-3_N1200_T50.mat");
    assert!(
        path.exists(),
        "Navier-Stokes dataset not found at {}\nSee datasets/README.md",
        path.display()
    );

    let (train_data, test_data, normalizers) = load_navier_stokes_uniform(&path, &dataset_cfg);

    let model_cfg = FNOConfig {
        // Three entries so a 3D FNO over (x, y, t). Corner-block sizes, the
        // same convention as `modes1..3` in fourier_3d.py.
        modes: vec![8, 8, 8],
        hidden_channels: 20,
        data_channels: dataset_cfg.t_in,
        out_channels: 1,
        n_layers: 4,
    };

    let train_cfg = TrainingConfig::new()
        .with_epochs(500)
        .with_batch_size(10)
        .with_test_batch_size(10)
        .with_learning_rate(1e-3)
        .with_weight_decay(1e-4)
        .with_min_lr(1e-5);

    // As in Darcy: y_normalizer decodes both sides during training and only
    // the prediction during evaluation - y_test was never encoded
    let (model, metrics) = train_navier_stokes(
        train_data,
        test_data,
        &normalizers.y,
        &model_cfg,
        &train_cfg,
        &data_cfg,
        &device,
    );

    let dir = write_run_artifacts(
        "navier_stokes_fno",
        &metrics,
        &model_cfg,
        &train_cfg,
        &dataset_cfg,
    );

    normalizers
        .x
        .to_record()
        .save(dir.join("x_normalizer.json"))
        .expect("save x normalizer");
    normalizers
        .y
        .to_record()
        .save(dir.join("y_normalizer.json"))
        .expect("save y normalizer");

    let mut store = BurnpackStore::from_file(dir.join("model_weights"));
    model
        .save_into(&mut store)
        .expect("could not save model weights");

    println!("run written to {}", dir.display());
}

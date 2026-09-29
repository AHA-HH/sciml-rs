//! FNO on 2D Darcy flow
//!
//! Run with: cargo run --release --example train_darcy
//! (CPU by default; add `--features metal` or `--features cuda` for a GPU -
//! see "Backends" in the README).
//!
//! Structurally identical to `examples/train/burgers.rs` - see that file for the
//! fully annotated version. The differences: `modes` has two entries instead of
//! one and the targets are normalized which the trainer has to undo before
//! computing the loss.

use burn::{
    prelude::Config,
    store::{BurnpackStore, ModuleSnapshot},
    tensor::Device,
};
use sciml_rs::neural_operators::{
    data::loaders::{
        base_dataset::DatasetConfig,
        darcy::{DarcyConfig, load_darcy_uniform},
    },
    metrics::io::write_run_artifacts,
    models::fno::FNOConfig,
    training::{trainer::TrainingConfig, trainers::darcy::train_darcy},
};
use std::path::Path;

fn main() {
    let device = Device::default().autodiff();
    println!("device: {device:?}");

    let data_cfg = DatasetConfig {
        n_train: 1000,
        n_test: 100,
    };

    // subsample rate is 5 -> s = (421-1)/5 + 1 = 85 grid points per axis,
    // the paper's setting. Non-power-of-two sizes are fine: the FFT falls
    // back to Bluestein's algorithm.
    let dataset_cfg = DarcyConfig::new(data_cfg, 5);

    let datasets = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets");
    let train_path = datasets.join("piececonst_r421_N1024_smooth1.mat");
    let test_path = datasets.join("piececonst_r421_N1024_smooth2.mat");
    for p in [&train_path, &test_path] {
        assert!(
            p.exists(),
            "Darcy dataset not found at {}\nSee datasets/README.md",
            p.display()
        );
    }

    let (train_data, test_data, normalizers) =
        load_darcy_uniform(&train_path, &test_path, &dataset_cfg)
            .unwrap_or_else(|e| panic!("could not load Darcy data: {e}"));

    let model_cfg = FNOConfig {
        // Two entries so a 2D FNO model. Same code path as Burgers' vec![16]
        modes: vec![12, 12],
        hidden_channels: 32,
        data_channels: 1,
        out_channels: 1,
        n_layers: 4,
    };

    let train_cfg = TrainingConfig::new()
        .with_epochs(500)
        .with_batch_size(20)
        .with_learning_rate(1e-3)
        .with_weight_decay(1e-4)
        .with_min_lr(1e-5);

    // y_normalizer is fitted on y_train during loading. train_darcy uses it
    // to decode both prediction and target during training, but only the
    // prediction during evaluation - y_test was never encoded
    let (model, metrics) = train_darcy(
        train_data,
        test_data,
        &normalizers.y,
        &model_cfg,
        &train_cfg,
        &device,
    );

    let dir = write_run_artifacts("darcy_fno", &metrics, &model_cfg, &train_cfg, &dataset_cfg)
        .unwrap_or_else(|e| panic!("could not write run artifacts: {e}"));

    // Write normalizers files to runs/darcy_fno_<timestamp>/
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

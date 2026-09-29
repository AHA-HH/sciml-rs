//! Re-evaluates a saved Darcy run on the test split.
//!
//! Run with: cargo run --release --example predict_darcy -- runs/darcy_fno_<timestamp>
//!
//! Loads the architecture from model_config.json, the weights from
//! model_weights.bpk, and the y normalizer from y_normalizer.json - all three
//! are needed, since weights alone can't reconstruct the model and predictions
//! come out in normalized units.

use std::path::{Path, PathBuf};

use burn::{
    data::dataloader::DataLoaderBuilder,
    prelude::*,
    store::{BurnpackStore, ModuleSnapshot},
    tensor::Device,
};
use sciml_rs::neural_operators::{
    data::{
        batcher::OperatorBatcher,
        loaders::{
            base_dataset::HasBaseConfig,
            darcy::{DarcyConfig, load_darcy_uniform},
        },
        transforms::normalizers::{
            NormalizerRecord, UnitGaussianNormalizer, decode_flat, normalizer_to_flat_tensors,
        },
    },
    losses::data_losses::{LpLoss, Reduction},
    models::fno::FNOConfig,
    training::trainer::eval_epoch,
};

fn main() {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: predict_darcy <run_dir>"),
    );
    // Inference only: no autodiff, so forward passes record no backward graph.
    let device = Device::default();

    // Architecture from config, weights from the burnpack record
    let model_cfg = FNOConfig::load(dir.join("model_cfg.json")).expect("load model config");
    let mut model = model_cfg.init::<4>(&device);
    let mut store = BurnpackStore::from_file(dir.join("model_weights.bpk"));
    model.load_from(&mut store).expect("load model weights");

    // Same test split the run was evaluated on
    let dataset_cfg = DarcyConfig::load(dir.join("data_cfg.json")).expect("load dataset config");
    let datasets = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets");
    let (_, test_data, _) = load_darcy_uniform(
        &datasets.join("piececonst_r421_N1024_smooth1.mat"),
        &datasets.join("piececonst_r421_N1024_smooth2.mat"),
        &dataset_cfg,
    )
    .unwrap_or_else(|e| panic!("could not load Darcy data: {e}"));

    let test_loader = DataLoaderBuilder::new(OperatorBatcher::<4, 3>::new(device.clone()))
        .batch_size(20)
        .build(test_data);

    // Predictions come out normalized; test targets were never encoded
    let y_norm = UnitGaussianNormalizer::from_record(
        &NormalizerRecord::load(dir.join("y_normalizer.json")).expect("load y normalizer"),
    );
    let (mean, std) = normalizer_to_flat_tensors(&y_norm, &device);
    let eps = y_norm.eps_val();
    let eval_post =
        move |out: Tensor<2>, target: Tensor<2>| (decode_flat(out, &mean, &std, eps), target);

    let loss_fn = LpLoss::new(2, 2, Reduction::Sum);
    let l2 = eval_epoch::<4, 3>(&model, &test_loader, &loss_fn, &eval_post);

    println!(
        "loaded model test_l2: {:.10}",
        l2 / dataset_cfg.base().n_test as f32
    );
    println!(
        "(should match the last test_l2 in {}/metrics.csv)",
        dir.display()
    );
}

//! Re-evaluates a saved Navier-Stokes FNO run on the test split.
//!
//! Run with: cargo run --release --example predict_navier_stokes -- runs/navier_stokes_fno_<timestamp>
//!
//! Loads the architecture from model_cfg.json, the weights from
//! model_weights.bpk, the dataset config from data_cfg.json and the y
//! normalizer from y_normalizer.json. The loader refits the normalizers on the
//! same training split, but decoding uses the saved one.

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
            navier_stokes::{NavierStokesConfig, load_navier_stokes_uniform},
        },
        transforms::normalizers::{
            NormalizerRecord, UnitGaussianNormalizer, decode_flat, normalizer_to_flat_tensors,
        },
    },
    losses::data_losses::LpLoss,
    models::fno::FNOConfig,
    training::trainer::eval_epoch,
};

fn main() {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: predict_navier_stokes <run_dir>"),
    );
    let device = Device::default().autodiff();

    // Architecture from config, weights from the burnpack record
    let model_cfg = FNOConfig::load(dir.join("model_cfg.json")).expect("load model config");
    let mut model = model_cfg.init::<5>(&device);
    let mut store = BurnpackStore::from_file(dir.join("model_weights.bpk"));
    model.load_from(&mut store).expect("load model weights");

    // Same test split the run was evaluated on
    let dataset_cfg =
        NavierStokesConfig::load(dir.join("data_cfg.json")).expect("load dataset config");
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("datasets")
        .join("ns_V1e-3_N1200_T50.mat");
    let (_, test_data, _) = load_navier_stokes_uniform(&path, &dataset_cfg);

    let test_loader = DataLoaderBuilder::new(OperatorBatcher::<5, 4>::new(device.clone()))
        .batch_size(10)
        .build(test_data);

    // Predictions come out normalized; test targets were never encoded
    let y_norm = UnitGaussianNormalizer::from_record(
        &NormalizerRecord::load(dir.join("y_normalizer.json")).expect("load y normalizer"),
    );
    let (mean, std) = normalizer_to_flat_tensors(&y_norm, &device);
    let eps = y_norm.eps_val();
    let eval_post =
        move |out: Tensor<2>, target: Tensor<2>| (decode_flat(out, &mean, &std, eps), target);

    let loss_fn = LpLoss::new(3, 2, false, true);
    let l2 = eval_epoch::<5, 4>(&model, &test_loader, &loss_fn, &eval_post);

    println!(
        "loaded model test_l2: {:.10}",
        l2 / dataset_cfg.base().n_test as f32
    );
    println!(
        "(should match the last test_l2 in {}/metrics.csv)",
        dir.display()
    );
}

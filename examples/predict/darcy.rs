//! Re-evaluates a saved Darcy run on the test split.
//!
//! Run with: cargo run --release --example predict_darcy -- runs/darcy_fno_<timestamp>
//!
//! Loads the model, the dataset config and the y normalizer from the run directory
//! (`RunDir`) - all are needed, since weights alone can't reconstruct the model and
//! predictions come out in normalized units.

use std::path::Path;

use burn::{prelude::*, tensor::Device};
use sciml_rs::neural_operators::{
    data::{
        device_batcher::device_loader,
        loaders::darcy::{DarcyConfig, load_darcy_uniform},
        transforms::normalizers::FlatDecoder,
    },
    runs::{ConfigFile, RunDir},
    training::trainer::eval_epoch,
};

fn main() {
    let run = RunDir::open(
        std::env::args()
            .nth(1)
            .expect("usage: predict_darcy <run_dir>"),
    )
    .unwrap_or_else(|e| panic!("open the run: {e}"));
    // Inference only: no autodiff, so forward passes record no backward graph.
    let device = Device::default();

    // Architecture from config, weights from the burnpack record
    let model = run
        .load_fno::<4>(&device)
        .unwrap_or_else(|e| panic!("load the model: {e}"));

    // Same test split the run was evaluated on
    let dataset_cfg: DarcyConfig = run
        .config(ConfigFile::Data)
        .unwrap_or_else(|e| panic!("load dataset config: {e}"));
    let datasets = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets");
    let (_, test_data, _) = load_darcy_uniform::<f32>(
        datasets.join("piececonst_r421_N1024_smooth1.mat"),
        datasets.join("piececonst_r421_N1024_smooth2.mat"),
        &dataset_cfg,
    )
    .unwrap_or_else(|e| panic!("could not load Darcy data: {e}"));

    let test_loader = device_loader::<4, 3, _>(&test_data, &device, 20, None);

    // Predictions come out normalized; test targets were never encoded
    let normalizers = run
        .normalizers()
        .unwrap_or_else(|e| panic!("load the normalizers: {e}"))
        .expect("a Darcy run has normalizers");
    let decoder = FlatDecoder::new(&normalizers.y, &device);
    let eval_post = move |out: Tensor<2>, target: Tensor<2>| (decoder.decode(out), target);

    let l2 = eval_epoch::<4, 3>(&model, &test_loader, &eval_post);

    println!(
        "loaded model test_l2: {:.10}",
        l2 / dataset_cfg.base.n_test as f32
    );
    println!(
        "(should match the last test_l2 in {}/metrics.csv)",
        run.path().display()
    );
}

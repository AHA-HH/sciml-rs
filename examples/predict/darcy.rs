//! Re-evaluates a saved Darcy run on the test split.
//!
//! Run with: cargo run --release --example predict_darcy -- runs/darcy_fno_<timestamp>
//!
//! Loads the model, the dataset config and both normalizers from the run directory
//! (`RunDir`) - all are needed, since weights alone can't reconstruct the model, the
//! inputs are encoded with the run's x normalizer and predictions come out in normalized
//! units. Only the test split is read (`load_darcy_test`).

use std::path::Path;

use burn::tensor::Device;
use sciml_rs::neural_operators::{
    data::{
        device_batcher::device_loader,
        loaders::darcy::{DarcyConfig, load_darcy_test},
        transforms::normalizers::FlatDecoder,
    },
    runs::RunDir,
    training::step::evaluate,
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
        .data_config()
        .unwrap_or_else(|e| panic!("load dataset config: {e}"));
    // The statistics the model was trained with: x encodes the inputs, y decodes
    let normalizers = run
        .normalizers()
        .unwrap_or_else(|e| panic!("load the normalizers: {e}"))
        .unwrap_or_else(|| {
            panic!(
                "{} has no normalizers; a Darcy run has them",
                run.path().display()
            )
        });
    let datasets = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets");
    let test_data = load_darcy_test::<f32>(
        datasets.join("piececonst_r421_N1024_smooth2.mat"),
        &dataset_cfg,
        &normalizers,
    )
    .unwrap_or_else(|e| panic!("could not load Darcy data: {e}"));

    let test_loader = device_loader::<4, 3, _>(&test_data, &device, 20, None);

    // Predictions come out normalized; targets are physical
    let decoder = FlatDecoder::new(&normalizers.y, &device);

    let l2 = evaluate::<4, 3>(&model, &test_loader, Some(&decoder));

    println!("loaded model test_l2: {l2:.10}");
    println!(
        "(should match the last test_l2 in {}/metrics.csv)",
        run.path().display()
    );
}

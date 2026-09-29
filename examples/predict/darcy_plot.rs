//! Plots one Darcy test sample: input coefficient field, ground truth, and
//! the model's prediction, side by side.
//!
//! Run with: cargo run --release --example darcy_plot -- runs/darcy_fno_<timestamp>

use std::path::{Path, PathBuf};

use burn::{
    data::dataset::Dataset,
    prelude::*,
    store::{BurnpackStore, ModuleSnapshot},
    tensor::Device,
};
use gnuplot::{AxesCommon, Caption, Figure};
use sciml_rs::neural_operators::{
    data::{
        loaders::darcy::{DarcyConfig, load_darcy_uniform},
        transforms::normalizers::{Normalizer, NormalizerRecord, UnitGaussianNormalizer},
    },
    models::fno::FNOConfig,
};

fn main() {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: darcy_plot <run_dir>"),
    );
    // Inference only: no autodiff, so forward passes record no backward graph.
    let device = Device::default();

    let model_cfg = FNOConfig::load(dir.join("model_cfg.json")).expect("load model config");
    let mut model = model_cfg.init::<4>(&device);
    let mut store = BurnpackStore::from_file(dir.join("model_weights.bpk"));
    model.load_from(&mut store).expect("load model weights");

    let dataset_cfg = DarcyConfig::load(dir.join("data_cfg.json")).expect("load dataset config");
    let datasets = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets");
    let (_, test_data, _) = load_darcy_uniform(
        &datasets.join("piececonst_r421_N1024_smooth1.mat"),
        &datasets.join("piececonst_r421_N1024_smooth2.mat"),
        &dataset_cfg,
    );

    let y_norm = UnitGaussianNormalizer::from_record(
        &NormalizerRecord::load(dir.join("y_normalizer.json")).expect("load y normalizer"),
    );

    // One sample. input is [s, s, 3]: coefficient field, then the two grid channels.
    let item = test_data.get(0).expect("test set is non-empty");
    let s = dataset_cfg.s();

    let coeff: Vec<f64> = item
        .input
        .slice(ndarray::s![.., .., 0])
        .iter()
        .copied()
        .collect();
    let truth: Vec<f64> = item.target.iter().copied().collect();

    // Forward one sample: [s, s, 3] -> [1, s, s, 3] -> [1, s, s, 1].
    let x_flat: Vec<f64> = item.input.iter().copied().collect();
    let x = Tensor::<4>::from_data(TensorData::new(x_flat, vec![1, s, s, 3]), &device);
    let out = model.forward(x).reshape([s, s]);

    // Predictions come out normalized; decode before plotting.
    let pred_encoded: Vec<f64> = out.into_data().iter::<f64>().collect();
    let pred = y_norm.decode(
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[s, s]), pred_encoded)
            .expect("prediction shape"),
    );
    let pred: Vec<f64> = pred.iter().copied().collect();

    let mut fig = Figure::new();
    fig.set_multiplot_layout(1, 3)
        .set_title("Darcy flow: input, ground truth, FNO prediction");

    for (data, title) in [
        (&coeff, "Input a(x,y)"),
        (&truth, "Ground truth u(x,y)"),
        (&pred, "FNO prediction"),
    ] {
        fig.axes2d()
            .set_title(title, &[])
            .image(data.iter().copied(), s, s, None, &[Caption("")]);
    }

    let path = dir.join("prediction.png");
    match fig.save_to_png(&path, 1800, 600) {
        Ok(()) => println!("plot written to {}", path.display()),
        Err(e) => eprintln!("plot failed ({e}) — is gnuplot installed?"),
    }
}

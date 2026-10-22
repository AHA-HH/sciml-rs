//! Plots one Darcy test sample: input coefficient field, ground truth, and
//! the model's prediction, side by side.
//!
//! Run with: cargo run --release --example darcy_plot -- runs/darcy_fno_<timestamp>

use std::path::Path;

use burn::tensor::Device;
use gnuplot::{AxesCommon, Caption, Figure};
use sciml_rs::neural_operators::{
    data::loaders::darcy::{DarcyConfig, load_darcy_uniform},
    models::predict::predict_host,
    runs::RunDir,
};

fn main() {
    let run = RunDir::open(
        std::env::args()
            .nth(1)
            .expect("usage: darcy_plot <run_dir>"),
    )
    .unwrap_or_else(|e| panic!("open the run: {e}"));
    // Inference only: no autodiff, so forward passes record no backward graph.
    let device = Device::default();

    let model = run
        .load_fno::<4>(&device)
        .unwrap_or_else(|e| panic!("load the model: {e}"));

    let dataset_cfg: DarcyConfig = run
        .data_config()
        .unwrap_or_else(|e| panic!("load dataset config: {e}"));
    let datasets = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets");
    // f64 host storage: the plot reads the sample back as f64. The model
    // still sees the same f32 values (see `HostFloat`).
    let (_, test_data, _) = load_darcy_uniform::<f64>(
        datasets.join("piececonst_r421_N1024_smooth1.mat"),
        datasets.join("piececonst_r421_N1024_smooth2.mat"),
        &dataset_cfg,
    )
    .unwrap_or_else(|e| panic!("could not load Darcy data: {e}"));

    let y_norm = run
        .normalizers()
        .unwrap_or_else(|e| panic!("load the normalizers: {e}"))
        .expect("a Darcy run has normalizers")
        .y;

    // One sample. input is [s, s, 1]: the coefficient field only; the model
    // appends the two grid channels itself.
    let input = test_data.inputs().index_axis(ndarray::Axis(0), 0);
    let target = test_data.targets().index_axis(ndarray::Axis(0), 0);
    let s = dataset_cfg.s();

    let coeff: Vec<f64> = input
        .slice(ndarray::s![.., .., 0])
        .iter()
        .copied()
        .collect();
    let truth: Vec<f64> = target.iter().copied().collect();

    // Forward one sample: [1, s, s, 1] in, normalized [1, s, s] out (the model sees the
    // same f32 values as before: the f64 host data rounds once to f32)
    let x = test_data
        .inputs()
        .slice_axis(ndarray::Axis(0), ndarray::Slice::from(0..1))
        .mapv(|v| v as f32);
    let pred = predict_host::<4>(&model, x.view(), 1, &device);

    // Predictions come out normalized; decode before plotting.
    let pred = y_norm.decode(pred.index_axis_move(ndarray::Axis(0), 0));
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

    let path = run.path().join("prediction.png");
    match fig.save_to_png(&path, 1800, 600) {
        Ok(()) => println!("plot written to {}", path.display()),
        Err(e) => eprintln!("plot failed ({e}) — is gnuplot installed?"),
    }
}

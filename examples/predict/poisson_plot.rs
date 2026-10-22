//! Plots one test sample of a saved Poisson run on the Chebyshev grid: the forcing f, the
//! reference solution u, the FNO prediction û = T_uc(prediction) and the error û − u
//! (design §7, error 4; §12, decision 18).
//!
//! Run with: cargo run --release --example poisson_plot -- runs/<run_dir> [<sample>], where
//! the run is runs/poisson_fno_p<p>_<timestamp> or runs/poisson_fno_n<n>_p<p>_<timestamp>
//! (sample indexes the test split, default 0; add `--features metal` for the GPU).
//!
//! Loads the model, the dataset config and the y normalizer from the run directory
//! (`RunDir`), as `predict_poisson` does, runs the one sample and decodes it to f64. f and u
//! are the raw split's CGL fields (`load_poisson_cheb_field`). Each panel draws every one
//! of the n² CGL nodes as a filled cell at its (x, y) coordinates, reaching halfway to its
//! neighbours, so the cells tile [−1, 1]² and the clustering towards the edges shows.
//! Writes prediction_<sample>.png into the run directory; if gnuplot is missing it warns
//! and exits 0, as `darcy_plot` does. Needs the run's stage dataset (n from data_cfg.json):
//! see datasets/README.md.

use std::f64::consts::PI;
use std::path::Path;

use burn::tensor::Device;
use gnuplot::{AutoOption::Fix, AxesCommon, Color, Figure, PaletteType, VariablePaletteColor};
use ndarray::{ArrayView2, Axis, s};
use sciml_rs::neural_operators::{
    chebyshev::{Transfers, cc_rel_l2_error, nodes_closed_form},
    data::loaders::poisson::{PoissonConfig, load_poisson_cheb_field, load_poisson_uniform, stage},
    models::run::predict_host,
    runs::{ConfigFile, RunDir},
};

/// Figure size in pixels.
const WIDTH: u32 = 1400;
const HEIGHT: u32 = 1300;
/// Blue–white–red, for the error panel's range symmetric about 0.
const DIVERGING: [(f32, f32, f32, f32); 3] = [
    (0.0, 0.23, 0.30, 0.75),
    (0.5, 1.0, 1.0, 1.0),
    (1.0, 0.71, 0.02, 0.15),
];

/// The cell of each of the `n` ascending CGL nodes, as (low, high) edges: node
/// x_j = −cos(jh), h = π/(n − 1) (CONVENTIONS §12), owns [−cos((j − ½)h), −cos((j + ½)h)],
/// the angle clamped to [0, π]. The cells meet without gaps and the outer edges are ±1.
fn cgl_cell_edges(n: usize) -> (Vec<f64>, Vec<f64>) {
    let h = PI / (n - 1) as f64;
    // −cos θ in the sine form, as `nodes_closed_form`, so the edges are antisymmetric
    let edge = |k: usize| ((k as f64 - 0.5) * h).clamp(0.0, PI) - PI / 2.0;
    let edges: Vec<f64> = (0..=n).map(|k| edge(k).sin()).collect();
    let (lo, hi) = (edges[..n].to_vec(), edges[1..].to_vec());
    debug_assert!(lo[0] == -1.0 && hi[n - 1] == 1.0);
    debug_assert!((1..n).all(|j| hi[j - 1] == lo[j]));
    (lo, hi)
}

/// (min, max) of the values of all `fields`.
fn range<'a>(fields: impl IntoIterator<Item = ArrayView2<'a, f64>>) -> (f64, f64) {
    fields
        .into_iter()
        .flat_map(|f| f.into_iter().copied())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(v), hi.max(v))
        })
}

fn main() {
    let mut args = std::env::args().skip(1);
    let run = RunDir::open(
        args.next()
            .expect("usage: poisson_plot <run_dir> [<sample>]"),
    )
    .unwrap_or_else(|e| panic!("open the run: {e}"));
    let dir = run.path();
    let sample: usize = args.next().map_or(0, |a| {
        a.parse().expect("sample must be a non-negative integer")
    });
    // Inference only: no autodiff, so forward passes record no backward graph.
    let device = Device::default();

    let datasets = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/poisson");
    let model = run
        .load_fno::<4>(&device)
        .unwrap_or_else(|e| panic!("load the model: {e}"));

    let dataset_cfg = run
        .config::<PoissonConfig>(ConfigFile::Data)
        .unwrap_or_else(|e| panic!("load dataset config: {e}"));
    let n = dataset_cfg.n;
    let set = stage(n).unwrap_or_else(|e| panic!("the run's stage: {e}"));
    let test_file = datasets.join(set.npz("test"));
    let n_test = dataset_cfg.base.n_test;
    assert!(
        sample < n_test,
        "sample {sample} is out of range: the run has {n_test} test samples"
    );
    let s = dataset_cfg.s();

    let (_, test_data, _) =
        load_poisson_uniform::<f32>(datasets.join(set.npz("train")), &test_file, &dataset_cfg)
            .unwrap_or_else(|e| panic!("could not load Poisson data: {e}"));
    let load_cheb = |field| {
        load_poisson_cheb_field(&test_file, field, n, sample + 1)
            .unwrap_or_else(|e| panic!("could not load the CGL field '{field}': {e}"))
            .index_axis_move(Axis(0), sample)
    };
    let f = load_cheb("f");
    let u = load_cheb("u");

    // Forward one sample: [1, s, s, 1] in, normalized [s, s] out
    let input = test_data.inputs().slice(s![sample..=sample, .., .., ..]);
    let pred = predict_host(&model, input.into_dyn(), 1, &device).index_axis_move(Axis(0), 0);
    assert_eq!(pred.dim(), (s, s), "prediction is [s, s]");
    // Decode to physical scale in f64, then transfer to the CGL grid as error 4 does
    let y_norm = run
        .normalizers()
        .unwrap_or_else(|e| panic!("load the normalizers: {e}"))
        .expect("a Poisson run has normalizers")
        .y;
    let pred = y_norm
        .decode(pred.into_dyn())
        .into_dimensionality::<ndarray::Ix2>()
        .expect("decoded prediction is [s, s]");
    let u_hat = Transfers::new(n).to_cheb(pred.view());
    let err = &u_hat - &u;
    let error4 = cc_rel_l2_error(u_hat.view(), u.view());
    println!("sample {sample}: error 4 (total, CC norm) = {error4:.4e}");

    // Node coordinates in 'ij' order: field[a, b] sits at (x_a, y_b)
    let nodes = nodes_closed_form(n).to_vec();
    let xs: Vec<f64> = nodes
        .iter()
        .flat_map(|&x| std::iter::repeat_n(x, n))
        .collect();
    let ys: Vec<f64> = (0..n).flat_map(|_| nodes.iter().copied()).collect();
    // Cell edges in the same order: x edges repeat within a row, y edges cycle
    let (lo, hi) = cgl_cell_edges(n);
    let per_row =
        |e: &[f64]| -> Vec<f64> { e.iter().flat_map(|&v| std::iter::repeat_n(v, n)).collect() };
    let cycled = |e: &[f64]| -> Vec<f64> { (0..n).flat_map(|_| e.iter().copied()).collect() };
    let (x_lo, x_hi) = (per_row(&lo), per_row(&hi));
    let (y_lo, y_hi) = (cycled(&lo), cycled(&hi));

    let (f_lo, f_hi) = range([f.view()]);
    let (u_lo, u_hi) = range([u.view(), u_hat.view()]);
    let e_max = err.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let run_name = dir
        .file_name()
        .map_or_else(|| dir.display().to_string(), |d| d.to_string_lossy().into());

    let mut fig = Figure::new();
    // plain text: run names and T_uc hold '_', which enhanced text reads as a subscript
    fig.set_enhanced_text(false);
    fig.set_multiplot_layout(2, 2).set_title(&format!(
        "{run_name}, test sample {sample}: error 4 = {error4:.3e}"
    ));
    for (field, title, (lo, hi), diverging) in [
        (&f, "forcing f", (f_lo, f_hi), false),
        (&u, "reference u", (u_lo, u_hi), false),
        (&u_hat, "prediction û = T_uc(FNO)", (u_lo, u_hi), false),
        (&err, "error û − u", (-e_max, e_max), true),
    ] {
        let axes = fig.axes2d();
        axes.set_title(title, &[])
            .set_x_range(Fix(-1.0), Fix(1.0))
            .set_y_range(Fix(-1.0), Fix(1.0))
            .set_aspect_ratio(Fix(1.0))
            .set_x_label("x", &[])
            .set_y_label("y", &[])
            .set_cb_range(Fix(lo), Fix(hi));
        if diverging {
            axes.set_palette(PaletteType::Custom(&DIVERGING));
        }
        // Solid fill, each border in its cell's own colour: no seams between cells
        axes.box_xy_error_low_high(
            &xs,
            &ys,
            &x_lo,
            &x_hi,
            &y_lo,
            &y_hi,
            &[Color(VariablePaletteColor(field.iter().copied().collect()))],
        );
    }

    let path = dir.join(format!("prediction_{sample}.png"));
    match fig.save_to_png(&path, WIDTH, HEIGHT) {
        Ok(()) => println!("plot written to {}", path.display()),
        Err(e) => eprintln!("plot failed ({e}) — is gnuplot installed?"),
    }
}

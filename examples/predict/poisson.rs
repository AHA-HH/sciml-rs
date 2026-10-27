//! Evaluates a saved Poisson run on its stage's test split: the five errors of design §7,
//! on the Chebyshev grid, and the boundary verdict (design §12, decision 16).
//!
//! Run with: cargo run --release --example predict_poisson -- runs/poisson_fno_p<p>_<timestamp>
//! (add `--features metal` for the GPU).
//!
//! Loads the model, both normalizers and the dataset config from the run directory
//! (`RunDir`), as `predict_darcy` does; the config's n picks the stage
//! (`catalogue::stage`). The test inputs come from `load_poisson_uniform_test`,
//! encoded with the run's x normalizer (the train split is not read), the raw CGL
//! solutions from `load_poisson_cheb_field`;
//! predictions are decoded with the y normalizer and copied to the host as f64. Error 1
//! needs the `datagen` feature and is quoted from Phase 1. `metrics::poisson::evaluate`
//! cross-checks error 2 against the test split's `.json` sidecar and the mean error 3
//! against the run's last test_l2, and gives the boundary verdict; this prints its report
//! and writes its record as errors.json into the run directory, with the index of the
//! sample with the max error 4, the one to plot with `poisson_plot`. The prediction runs
//! twice and the second, warm pass is timed; a run with `costs.json` (since Phase 5 T4)
//! gets that time as its `inference_time_s`. Exits 1 if a cross-check failed.
//! Needs the run's stage dataset: see datasets/README.md.

use std::path::Path;
use std::time::Instant;

use burn::{config::Config, tensor::Device};
use ndarray::Axis;
use sciml_rs::neural_operators::{
    data::{
        catalogue::stage,
        loaders::poisson::{
            PoissonConfig, PoissonField, SidecarRecord, load_poisson_cheb_field,
            load_poisson_uniform_test,
        },
    },
    metrics::poisson::{evaluate, poisson_errors},
    models::predict::predict_host,
    runs::RunDir,
    utils::backend::backend_features,
};

const BATCH_SIZE: usize = 20;

fn main() {
    let run = RunDir::open(
        std::env::args()
            .nth(1)
            .expect("usage: predict_poisson <run_dir>"),
    )
    .unwrap_or_else(|e| panic!("open the run: {e}"));
    let dir = run.path();
    // Inference only: no autodiff, so forward passes record no backward graph.
    let device = Device::default();

    let model = run
        .load_fno::<4>(&device)
        .unwrap_or_else(|e| panic!("load the model: {e}"));

    let dataset_cfg = run
        .data_config::<PoissonConfig>()
        .unwrap_or_else(|e| panic!("load dataset config: {e}"));
    assert!(
        dataset_cfg.s() == dataset_cfg.n - 1,
        "this run trained at s = {} on n = {}; predict_poisson evaluates only s = n - 1 runs until \
         phase 5 T6",
        dataset_cfg.s(),
        dataset_cfg.n
    );
    let n = dataset_cfg.n;
    let set = stage(n).unwrap_or_else(|e| panic!("the run's stage: {e}"));
    let n_test = dataset_cfg.base.n_test;

    // The statistics the model was trained with: x encodes the inputs, y decodes
    let normalizers = run
        .normalizers()
        .unwrap_or_else(|e| panic!("load the normalizers: {e}"))
        .unwrap_or_else(|| panic!("{}/y_normalizer.json not found", dir.display()));
    let datasets = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/poisson");
    let test_data = load_poisson_uniform_test::<f32>(
        datasets.join(set.npz("test")),
        &dataset_cfg,
        &normalizers,
    )
    .unwrap_or_else(|e| panic!("could not load Poisson data: {e}"));
    let u = load_poisson_cheb_field(datasets.join(set.npz("test")), PoissonField::U, n, n_test)
        .unwrap_or_else(|e| panic!("could not load the CGL solutions: {e}"));

    // Forward pass in batches, in sample order; outputs are normalized [N, s, s]. The
    // first pass warms the backend up; the second is timed
    predict_host(&model, test_data.inputs().view(), BATCH_SIZE, &device);
    let start = Instant::now();
    let pred = predict_host(&model, test_data.inputs().view(), BATCH_SIZE, &device);
    let inference_time_s = start.elapsed().as_secs_f64();
    println!("inference on {n_test} test samples (warm): {inference_time_s:.3} s");
    // Decode to physical scale in f64; the test targets were never encoded
    let pred = normalizers.y.decode(pred.into_dyn());
    let pred = pred
        .into_dimensionality::<ndarray::Ix3>()
        .expect("decoded predictions are [N, s, s]");
    assert_eq!(pred.len_of(Axis(0)), n_test);

    let errors = poisson_errors(pred.view(), u.view());
    let sidecar =
        SidecarRecord::load(datasets.join(set.json("test"))).expect("load the test sidecar");
    let metrics = run.metrics().expect("read metrics.csv");
    let eval = evaluate(
        &errors,
        n,
        &sidecar,
        metrics.last().expect("metrics.csv has epochs"),
        format!("{device:?}"),
        backend_features().into_iter().map(String::from).collect(),
        dir,
    );
    println!("\n{eval}");

    let out = dir.join("errors.json");
    eval.record.save(&out).expect("write errors.json");
    println!("wrote {}", out.display());

    match run.costs() {
        Ok(Some(mut costs)) => {
            costs.inference_time_s = Some(inference_time_s);
            run.save_costs(&costs).expect("write costs.json");
            println!("recorded the inference time in costs.json");
        }
        Ok(None) => {
            println!("no costs.json (a run before phase 5 T4): inference time not recorded")
        }
        Err(e) => panic!("read costs.json: {e}"),
    }

    if !eval.record.cross_checks_passed {
        eprintln!("a cross-check failed; see above");
        std::process::exit(1);
    }
}

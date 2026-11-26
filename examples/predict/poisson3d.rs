//! Evaluates a saved 3D Poisson run (`train_poisson3d`, 3D CL4.1) on its set's test split:
//! errors 2–5 on the Chebyshev grid, in 3D (proposal §3.5: error 5 over the six faces), and
//! the boundary verdict, at the run's learning grid s.
//!
//! Run with: cargo run --release --example predict_poisson3d -- runs/poisson3d_fno_n<n>_s<s>_p<p>_<timestamp>
//! (add `--features metal` for the GPU; evaluate on the backend the run trained on, or the
//! error 3 cross-check compares two backends' round-off).
//!
//! `predict_poisson` for 3D: loads the model (`FNO<5>`), both normalizers and the dataset
//! config from the run directory, whose n picks the 3D set (`catalogue::GENERATED_3D`);
//! the test inputs come from `load_poisson3d_uniform_test`, encoded with the run's x
//! normalizer, the raw CGL solutions from `load_poisson3d_cheb_field`; predictions are
//! decoded with the y normalizer and copied to the host as f64. `poisson_errors_nd` gives
//! the per-sample errors and `metrics::poisson::evaluate_3d` the report: error 1 quoted
//! from 3D CL2.1, error 2 cross-checked against the test sidecar (at s = n − 1 only), the
//! mean error 3 against the run's last test_l2. It writes the record as errors.json into
//! the run directory and the warm inference time into `costs.json`. Exits 1 if a
//! cross-check failed. Needs the run's 3D dataset: see datasets/README.md.

use std::path::Path;
use std::time::Instant;

use burn::{config::Config, tensor::Device};
use ndarray::Axis;
use sciml_rs::neural_operators::{
    chebyshev::Transfers,
    data::{
        catalogue::GENERATED_3D,
        loaders::poisson::{
            PoissonConfig, PoissonField, SidecarRecord, load_poisson3d_cheb_field,
            load_poisson3d_uniform_test,
        },
    },
    metrics::poisson::{ERRORS_JSON, evaluate_3d, poisson_errors_nd},
    models::predict::predict_host,
    runs::RunDir,
    utils::backend::backend_features,
};

const BATCH_SIZE: usize = 20;

fn main() {
    let run = RunDir::open(
        std::env::args()
            .nth(1)
            .expect("usage: predict_poisson3d <run_dir>"),
    )
    .unwrap_or_else(|e| panic!("open the run: {e}"));
    let dir = run.path();
    // Inference only: no autodiff, so forward passes record no backward graph.
    let device = Device::default();

    let model = run
        .load_fno::<5>(&device)
        .unwrap_or_else(|e| panic!("load the model: {e}"));
    let dataset_cfg = run
        .data_config::<PoissonConfig>()
        .unwrap_or_else(|e| panic!("load dataset config: {e}"));
    let n = dataset_cfg.n;
    let set = GENERATED_3D
        .into_iter()
        .find(|set| set.n == n)
        .unwrap_or_else(|| panic!("no 3D Poisson set with n = {n}"));
    let n_test = dataset_cfg.base.n_test;

    let normalizers = run
        .normalizers()
        .unwrap_or_else(|e| panic!("load the normalizers: {e}"))
        .unwrap_or_else(|| panic!("{}/y_normalizer.json not found", dir.display()));
    let datasets = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/poisson");
    let test_data = load_poisson3d_uniform_test::<f32>(
        datasets.join(set.npz("test")),
        &dataset_cfg,
        &normalizers,
    )
    .unwrap_or_else(|e| panic!("could not load the 3D Poisson data: {e}"));
    let u = load_poisson3d_cheb_field(datasets.join(set.npz("test")), PoissonField::U, n, n_test)
        .unwrap_or_else(|e| panic!("could not load the CGL solutions: {e}"));

    // Forward pass in batches, in sample order; outputs are normalized [N, s, s, s]. The
    // first pass warms the backend up; the second is timed
    predict_host(&model, test_data.inputs().view(), BATCH_SIZE, &device);
    let start = Instant::now();
    let pred = predict_host(&model, test_data.inputs().view(), BATCH_SIZE, &device);
    let inference_time_s = start.elapsed().as_secs_f64();
    println!("inference on {n_test} test samples (warm): {inference_time_s:.3} s");
    // Decode to physical scale in f64; the test targets were never encoded
    let pred = normalizers.y.decode(pred.into_dyn());
    assert_eq!(pred.len_of(Axis(0)), n_test);

    let transfers = Transfers::with_s(n, dataset_cfg.s());
    let errors = poisson_errors_nd(pred.view(), u.view(), &transfers);
    let sidecar =
        SidecarRecord::load(datasets.join(set.json("test"))).expect("load the test sidecar");
    let metrics = run.metrics().expect("read metrics.csv");
    let eval = evaluate_3d(
        &errors,
        &transfers,
        &sidecar,
        metrics.last().expect("metrics.csv has epochs"),
        format!("{device:?}"),
        backend_features().into_iter().map(String::from).collect(),
    );
    println!("\n{eval}");

    let out = dir.join(ERRORS_JSON);
    eval.record.save(&out).expect("write errors.json");
    println!("wrote {}", out.display());

    let mut costs = run
        .costs()
        .unwrap_or_else(|e| panic!("read costs.json: {e}"))
        .unwrap_or_else(|| panic!("{} has no costs.json", dir.display()));
    costs.inference_time_s = Some(inference_time_s);
    run.save_costs(&costs).expect("write costs.json");
    println!("recorded the inference time in costs.json");

    if !eval.record.cross_checks_passed {
        eprintln!("a cross-check failed; see above");
        std::process::exit(1);
    }
}

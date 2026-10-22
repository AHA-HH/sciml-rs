//! Evaluates a saved Poisson run on its stage's test split: the five errors of design §7,
//! on the Chebyshev grid, and the boundary verdict (design §12, decision 16).
//!
//! Run with: cargo run --release --example predict_poisson -- runs/poisson_fno_p<p>_<timestamp>
//! (add `--features metal` for the GPU).
//!
//! Loads the model, the y normalizer and the dataset config from the run directory
//! (`RunDir`), as `predict_darcy` does; the config's n picks the stage
//! (`loaders::poisson::stage`). The test inputs come
//! from `load_poisson_uniform`, the raw CGL solutions from `load_poisson_cheb`;
//! predictions are decoded with the y normalizer and copied to the host as f64. Error 1
//! needs the `chebyshev` feature and is quoted from Phase 1. Cross-checks error 2 against
//! the test split's `.json` sidecar and the mean error 3 against the run's last test_l2,
//! then writes errors.json into the run directory, with the index of the sample with the max
//! error 4, the one to plot with `poisson_plot`. Needs the run's stage dataset: see
//! datasets/README.md.

use std::path::Path;

use burn::{prelude::*, tensor::Device};
use ndarray::Axis;
use sciml_rs::neural_operators::{
    chebyshev::TRANSFER_DEGREE,
    data::loaders::poisson::{PoissonConfig, load_poisson_cheb, load_poisson_uniform, stage},
    metrics::{
        poisson::{check_error3, poisson_errors},
        stats::ErrorStats,
    },
    models::run::predict_host,
    runs::{ConfigFile, RunDir},
    utils::backend::backend_features,
};

use records::{PoissonErrorsRecord, Sidecar};

const BATCH_SIZE: usize = 20;
/// Error 2 against the sidecar, relative (both f64, same transfers and norm).
const ERROR2_TOL: f64 = 1e-12;
/// The ablation is warranted if mean relative boundary RMS ≥ this × mean error 4
/// (design §12, decision 16).
const BOUNDARY_THRESHOLD: f64 = 0.10;
/// Error 1 is not recomputed here (decision 16); Phase 1 Results.
const ERROR1_QUOTE: &str =
    "solver A floor <= 1e-12 from n = 33 (manufactured); A vs C order 1.993-1.999 (Phase 1 T2, T3)";

/// The JSON records this example reads and writes. Burn's `Config` derive writes
/// `name: name` in the code it generates, as in the library (`src/lib.rs`).
#[allow(clippy::redundant_field_names)]
mod records {
    use super::ErrorStats;
    use burn::config::Config;

    /// The sidecar's `error2` entry (datasets/README.md); other keys are ignored.
    #[derive(Config, Debug)]
    pub struct SidecarError2 {
        pub max: f64,
        pub mean: f64,
        pub d: usize,
        pub s: usize,
    }

    /// The fields of the test split's sidecar this example reads.
    #[derive(Config, Debug)]
    pub struct Sidecar {
        pub n: usize,
        pub count: usize,
        pub error2: SidecarError2,
    }

    /// What errors.json records.
    #[derive(Config, Debug)]
    pub struct PoissonErrorsRecord {
        /// Test samples evaluated.
        pub n_test: usize,
        /// CGL nodes per axis; the model grid is n − 1.
        pub n: usize,
        /// The device the forward pass ran on, as Burn prints it.
        pub device: String,
        /// The backend features this build enabled. Metal runs through wgpu, so `device`
        /// alone does not name it; together the two do.
        pub backend_features: Vec<String>,
        /// Error 1, quoted.
        pub error1: String,
        /// Error 2 (transfer), recomputed.
        pub error2: ErrorStats,
        /// Error 2 as the test sidecar records it.
        pub error2_sidecar: ErrorStats,
        /// Whether error 2 was compared with the sidecar: only when `n_test` equals the
        /// sidecar's sample count, since the sidecar covers its whole split.
        pub error2_compared: bool,
        /// Error 3 (model).
        pub error3: ErrorStats,
        /// The last test_l2 in the run's metrics.csv.
        pub error3_metrics_csv: f64,
        /// Error 4 (total).
        pub error4: ErrorStats,
        /// The test sample with the max error 4, an index into the test split.
        pub error4_worst_sample: usize,
        /// Error 5, max over ∂Ω of |û| / max over Ω of |u|.
        pub error5_max_ratio: ErrorStats,
        /// Error 5, relative boundary RMS.
        pub error5_rel_rms: ErrorStats,
        /// Mean error 5 relative RMS / mean error 4.
        pub boundary_to_total: f64,
        /// Whether `boundary_to_total` ≥ the threshold.
        pub ablation_warranted: bool,
        /// Whether both cross-checks passed; a skipped error 2 comparison counts as passed.
        pub cross_checks_passed: bool,
    }
}

fn rel_diff(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

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
        .config::<PoissonConfig>(ConfigFile::Data)
        .unwrap_or_else(|e| panic!("load dataset config: {e}"));
    let n = dataset_cfg.n;
    let set = stage(n).unwrap_or_else(|e| panic!("the run's stage: {e}"));
    let n_test = dataset_cfg.base.n_test;
    let s = dataset_cfg.s();

    let datasets = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/poisson");
    let (_, test_data, _) = load_poisson_uniform::<f32>(
        datasets.join(set.npz("train")),
        datasets.join(set.npz("test")),
        &dataset_cfg,
    )
    .unwrap_or_else(|e| panic!("could not load Poisson data: {e}"));
    let u = load_poisson_cheb(datasets.join(set.npz("test")), n, n_test)
        .unwrap_or_else(|e| panic!("could not load the CGL solutions: {e}"));

    // Forward pass in batches, in sample order; outputs are normalized [N, s, s]
    let pred = predict_host(&model, test_data.inputs().view(), BATCH_SIZE, &device);
    // Decode to physical scale in f64; the test targets were never encoded
    let y_norm = run
        .normalizers()
        .unwrap_or_else(|e| panic!("load the normalizers: {e}"))
        .expect("a Poisson run has normalizers")
        .y;
    let pred = y_norm.decode(pred.into_dyn());
    let pred = pred
        .into_dimensionality::<ndarray::Ix3>()
        .expect("decoded predictions are [N, s, s]");
    assert_eq!(pred.len_of(Axis(0)), n_test);

    let errors = poisson_errors(pred.view(), u.view());
    let e2 = ErrorStats::of(&errors.transfer);
    let e3 = ErrorStats::of(&errors.model);
    let e4 = ErrorStats::of(&errors.total);
    // the first sample holding the max, NaN included, as ErrorStats reports it
    let worst = errors
        .total
        .iter()
        .position(|&v| v == e4.max || (v.is_nan() && e4.max.is_nan()))
        .expect("error 4's max is one of its values");
    let e5_ratio = ErrorStats::of(&errors.boundary_max_ratio);
    let e5_rms = ErrorStats::of(&errors.boundary_rel_rms);

    println!(
        "\nerrors on the test split (n = {n}, s = {s}, {n_test} samples), device {device:?}, backend features {:?}",
        backend_features()
    );
    println!("{:<34} {:>12} {:>12}", "error", "mean", "max");
    println!("{:<34} {ERROR1_QUOTE}", "1 solver (quoted)");
    for (name, st) in [
        ("2 transfer (CC norm)", &e2),
        ("3 model (uniform, equal weights)", &e3),
        ("4 total (CC norm)", &e4),
        ("5 boundary max ratio", &e5_ratio),
        ("5 boundary relative RMS", &e5_rms),
    ] {
        println!("{name:<34} {:>12.4e} {:>12.4e}", st.mean, st.max);
    }
    println!(
        "worst sample (max error 4): {worst}; plot it with: poisson_plot -- {} {worst}",
        dir.display()
    );

    // Cross-check 1: error 2 against the sidecar the generator wrote
    let sidecar = Sidecar::load(datasets.join(format!("{}.json", set.stem("test"))))
        .expect("load the test sidecar");
    assert_eq!(
        (sidecar.n, sidecar.error2.s, sidecar.error2.d),
        (n, s, TRANSFER_DEGREE),
        "sidecar n, s, d differ from this evaluation"
    );
    let e2_sidecar = ErrorStats {
        mean: sidecar.error2.mean,
        max: sidecar.error2.max,
    };
    // the sidecar covers its whole split; compare only when all of it was evaluated, and
    // count a skipped comparison as passed
    let error2_compared = sidecar.count == n_test;
    let e2_ok = if error2_compared {
        let ok = rel_diff(e2.mean, e2_sidecar.mean) <= ERROR2_TOL
            && rel_diff(e2.max, e2_sidecar.max) <= ERROR2_TOL;
        println!(
            "\ncheck: error 2 vs sidecar mean {:.6e} / max {:.6e}: rel diff {:.1e} / {:.1e} (tol {ERROR2_TOL:e}) {}",
            e2_sidecar.mean,
            e2_sidecar.max,
            rel_diff(e2.mean, e2_sidecar.mean),
            rel_diff(e2.max, e2_sidecar.max),
            if ok { "ok" } else { "FAILED" }
        );
        ok
    } else {
        println!(
            "\ncheck: error 2 vs sidecar skipped: the sidecar covers {} samples but {n_test} were evaluated",
            sidecar.count
        );
        true
    };

    // Cross-check 2: mean error 3 against the run's last test_l2
    let e3_check = check_error3(&run.metrics().expect("read metrics.csv"), e3.mean)
        .expect("metrics.csv has epochs");
    println!("check: {e3_check}");

    let boundary_to_total = e5_rms.mean / e4.mean;
    let warranted = boundary_to_total >= BOUNDARY_THRESHOLD;
    println!(
        "\nboundary verdict: mean relative boundary RMS / mean error 4 = {:.4e} / {:.4e} = {boundary_to_total:.3} \
         ({} the {BOUNDARY_THRESHOLD} threshold): the hard-constraint ablation is {}",
        e5_rms.mean,
        e4.mean,
        if warranted { "at or above" } else { "below" },
        if warranted {
            "warranted"
        } else {
            "not warranted"
        }
    );

    let cross_checks_passed = e2_ok && e3_check.passed;
    let record = PoissonErrorsRecord {
        n_test,
        n,
        device: format!("{device:?}"),
        backend_features: backend_features().into_iter().map(String::from).collect(),
        error1: ERROR1_QUOTE.to_string(),
        error2: e2,
        error2_sidecar: e2_sidecar,
        error2_compared,
        error3: e3,
        error3_metrics_csv: e3_check.test_l2,
        error4: e4,
        error4_worst_sample: worst,
        error5_max_ratio: e5_ratio,
        error5_rel_rms: e5_rms,
        boundary_to_total,
        ablation_warranted: warranted,
        cross_checks_passed,
    };
    let out = dir.join("errors.json");
    record.save(&out).expect("write errors.json");
    println!("wrote {}", out.display());

    if !cross_checks_passed {
        eprintln!("a cross-check failed; see above");
        std::process::exit(1);
    }
}

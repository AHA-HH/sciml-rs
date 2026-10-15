//! Evaluates a saved Poisson run on its stage's test split: the five errors of design §7,
//! on the Chebyshev grid, and the boundary verdict (design §12, decision 16).
//!
//! Run with: cargo run --release --example predict_poisson -- runs/poisson_fno_p<p>_<timestamp>
//! (add `--features metal` for the GPU).
//!
//! Loads the model as `predict_darcy` does (model_cfg.json, model_weights.bpk,
//! y_normalizer.json) and the dataset config from data_cfg.json, whose n picks the stage
//! from `STAGES`. The test inputs come
//! from `load_poisson_uniform`, the raw CGL solutions from `load_poisson_cheb`;
//! predictions are decoded with the y normalizer and copied to the host as f64. Error 1
//! needs the `chebyshev` feature and is quoted from Phase 1. Cross-checks error 2 against
//! the test split's `.json` sidecar and the mean error 3 against the run's last test_l2,
//! then writes errors.json into the run directory, with the index of the sample with the max
//! error 4, the one to plot with `poisson_plot`. Needs the run's stage dataset: see
//! datasets/README.md.

use std::path::{Path, PathBuf};

use burn::{
    prelude::*,
    store::{BurnpackStore, ModuleSnapshot},
    tensor::Device,
};
use ndarray::{Array3, Axis, s};
use sciml_rs::neural_operators::{
    data::{
        loaders::{
            base_dataset::HasBaseConfig,
            poisson::{PoissonConfig, load_poisson_cheb, load_poisson_uniform},
        },
        transforms::normalizers::{Normalizer, NormalizerRecord, UnitGaussianNormalizer},
    },
    metrics::{
        io::read_metrics_csv,
        poisson::{ErrorStats, TRANSFER_DEGREE, poisson_errors},
    },
    models::fno::FNOConfig,
};

use records::{PoissonErrorsRecord, Sidecar};

/// The stages `train_poisson` trains on (design §5.3): (CGL nodes per axis n, train file,
/// test file, test sidecar). The run's data_cfg.json gives n.
const STAGES: &[(usize, &str, &str, &str)] = &[
    (
        65,
        "poisson_n65_K32_train.npz",
        "poisson_n65_K32_test.npz",
        "poisson_n65_K32_test.json",
    ),
    (
        129,
        "poisson_n129_K64_train.npz",
        "poisson_n129_K64_test.npz",
        "poisson_n129_K64_test.json",
    ),
    (
        257,
        "poisson_n257_K64_train.npz",
        "poisson_n257_K64_test.npz",
        "poisson_n257_K64_test.json",
    ),
];
const BATCH_SIZE: usize = 20;
/// Error 2 against the sidecar, relative (both f64, same transfers and norm).
const ERROR2_TOL: f64 = 1e-12;
/// Mean error 3 against the run's last test_l2, relative (the run's value is f32).
const ERROR3_TOL: f64 = 1e-5;
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

/// The backend cargo features compiled into this build, in `Device::default()`'s order.
fn backend_features() -> Vec<String> {
    [
        ("cuda", cfg!(feature = "cuda")),
        ("metal", cfg!(feature = "metal")),
        ("wgpu", cfg!(feature = "wgpu")),
        ("flex", cfg!(feature = "flex")),
    ]
    .into_iter()
    .filter(|&(_, on)| on)
    .map(|(name, _)| name.to_string())
    .collect()
}

fn main() {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: predict_poisson <run_dir>"),
    );
    // Inference only: no autodiff, so forward passes record no backward graph.
    let device = Device::default();

    let model_cfg = FNOConfig::load(dir.join("model_cfg.json")).expect("load model config");
    let mut model = model_cfg.init::<4>(&device);
    let mut store = BurnpackStore::from_file(dir.join("model_weights.bpk"));
    model.load_from(&mut store).expect("load model weights");

    let dataset_cfg = PoissonConfig::load(dir.join("data_cfg.json")).expect("load dataset config");
    let n = dataset_cfg.n;
    let &(_, train_file, test_file, test_sidecar) =
        STAGES.iter().find(|stage| stage.0 == n).unwrap_or_else(|| {
            let ns: Vec<usize> = STAGES.iter().map(|stage| stage.0).collect();
            panic!("the run's n = {n} is not a known stage; expected one of {ns:?}")
        });
    let n_test = dataset_cfg.base().n_test;
    let s = dataset_cfg.s();

    let datasets = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/poisson");
    let (_, test_data, _) = load_poisson_uniform::<f32>(
        datasets.join(train_file),
        datasets.join(test_file),
        &dataset_cfg,
    )
    .unwrap_or_else(|e| panic!("could not load Poisson data: {e}"));
    let u = load_poisson_cheb(datasets.join(test_file), n, n_test)
        .unwrap_or_else(|e| panic!("could not load the CGL solutions: {e}"));

    // Forward pass in batches, in sample order; outputs are normalized [B, s, s, 1]
    let inputs = test_data.inputs();
    let mut pred = Array3::<f64>::zeros((n_test, s, s));
    for start in (0..n_test).step_by(BATCH_SIZE) {
        let end = (start + BATCH_SIZE).min(n_test);
        let batch = inputs.slice(s![start..end, .., .., ..]);
        let data: Vec<f32> = batch.iter().copied().collect();
        let x = Tensor::<4>::from_data(TensorData::new(data, batch.shape().to_vec()), &device);
        let out = model
            .forward(x)
            .into_data()
            .try_to_vec::<f32>()
            .expect("f32 model output");
        let out = Array3::from_shape_vec((end - start, s, s), out)
            .expect("output is [B, s, s, 1]")
            .mapv(f64::from);
        pred.slice_mut(s![start..end, .., ..]).assign(&out);
    }
    // Decode to physical scale in f64; the test targets were never encoded
    let y_norm = UnitGaussianNormalizer::from_record(
        &NormalizerRecord::load(dir.join("y_normalizer.json")).expect("load y normalizer"),
    );
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
    let sidecar = Sidecar::load(datasets.join(test_sidecar)).expect("load the test sidecar");
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
    let metrics = read_metrics_csv(dir.join("metrics.csv")).expect("read metrics.csv");
    let last = metrics.last().expect("metrics.csv has no epochs");
    let test_l2 = f64::from(last.test_l2);
    let e3_ok = rel_diff(e3.mean, test_l2) <= ERROR3_TOL;
    println!(
        "check: mean error 3 vs metrics.csv test_l2 {test_l2:.6e} (epoch {}): rel diff {:.1e} (tol {ERROR3_TOL:e}) {}",
        last.epoch,
        rel_diff(e3.mean, test_l2),
        if e3_ok { "ok" } else { "FAILED" }
    );

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

    let cross_checks_passed = e2_ok && e3_ok;
    let record = PoissonErrorsRecord {
        n_test,
        n,
        device: format!("{device:?}"),
        backend_features: backend_features(),
        error1: ERROR1_QUOTE.to_string(),
        error2: e2,
        error2_sidecar: e2_sidecar,
        error2_compared,
        error3: e3,
        error3_metrics_csv: test_l2,
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

//! The errors of design §7 for a Poisson FNO, measured on the Chebyshev grid.
//!
//! Given decoded uniform-grid predictions and the raw CGL solutions u of the same samples,
//! [`poisson_errors`] computes per sample:
//! - error 2 (transfer): the round trip ‖T_uc(T_cu u) − u‖ / ‖u‖ in the Clenshaw–Curtis
//!   norm, the sidecar's `error2` recomputed;
//! - error 3 (model): the relative L² error of the prediction against T_cu u on the uniform
//!   grid, with equal weights, as `LpLoss::rel` computes it during training (CONVENTIONS §11);
//! - error 4 (total): ‖T_uc(prediction) − u‖ / ‖u‖ in the Clenshaw–Curtis norm;
//! - error 5 (boundary): with û = T_uc(prediction), the max ratio max_∂Ω |û| / max_Ω |u|
//!   and the relative boundary RMS √(∮ û² ds / 8) / √(∫∫ u² dA / 4).
//!
//! [`check_error3`] cross-checks the mean error 3 against the run's last test_l2. Each
//! error is summarised over the samples with [`ErrorStats`]. [`evaluate`] applies the
//! evaluation policy, the worst sample, both cross-checks and the boundary verdict, and
//! returns the [`PoissonErrorsRecord`] a run's `errors.json` holds.
//!
//! Error 1 (the solver) needs the `datagen` feature and is quoted, not recomputed (design
//! §12, decision 16). Everything here is ndarray and f64, built without `datagen`
//! (decision 15). Grids, transfers and norms follow CONVENTIONS §12.

use super::stats::ErrorStats;
use crate::neural_operators::chebyshev::{
    TRANSFER_DEGREE, Transfers, cc_norm_sq, cc_rel_l2_error_weighted, clencurt,
};
use crate::neural_operators::data::loaders::poisson::SidecarRecord;
use crate::neural_operators::runs::EpochMetrics;
use crate::neural_operators::utils::backend::backend_features;
use burn::config::Config;
use burn::serde::{Deserialize, Serialize};
use ndarray::{Array1, Array2, ArrayView2, ArrayView3, Axis};
use std::fmt;
use std::path::{Path, PathBuf};

/// Per-sample errors from [`poisson_errors`], one entry per sample, in sample order.
#[derive(Debug, Clone)]
pub struct PoissonErrors {
    /// Error 2: ‖T_uc(T_cu u) − u‖ / ‖u‖, Clenshaw–Curtis norm.
    pub transfer: Vec<f64>,
    /// Error 3: ‖prediction − T_cu u‖ / ‖T_cu u‖, equal weights on the uniform grid.
    pub model: Vec<f64>,
    /// Error 4: ‖T_uc(prediction) − u‖ / ‖u‖, Clenshaw–Curtis norm.
    pub total: Vec<f64>,
    /// Error 5, max ratio: max over ∂Ω of |û| / max over Ω of |u|, on the CGL nodes.
    pub boundary_max_ratio: Vec<f64>,
    /// Error 5, relative boundary RMS: √(∮ û² ds / 8) / √(∫∫ u² dA / 4).
    pub boundary_rel_rms: Vec<f64>,
}

/// The errors 2–5 of design §7 for each sample (see the module docs).
///
/// - `pred`: decoded predictions on the uniform grid, `[N, s, s]`, physical scale, stored
///   'ij' (CONVENTIONS §12).
/// - `u`: the raw solutions on the CGL grid, `[N, n, n]`, the reference.
///
/// s = n − 1, and T_uc has degree [`TRANSFER_DEGREE`]. T_cu and T_uc are built once per
/// call, as [`Transfers`]. The Clenshaw–Curtis norms use [`clencurt`]`(n)` weights: over Ω
/// as a tensor product, and along each of the four edges of ∂Ω, so each corner node
/// carries the end weights of both its edges; the boundary maximum visits it twice, which
/// does not change the maximum. A sample with u ≡ 0 gives NaN or infinity, as
/// [`cc_rel_l2_error_weighted`] does.
///
/// # Panics
/// If `pred` and `u` hold different numbers of samples, either is not square in its
/// spatial axes, s ≠ n − 1, or n < [`Transfers::MIN_N`] (T_uc of degree
/// [`TRANSFER_DEGREE`] needs s > d).
///
/// [`TRANSFER_DEGREE`]: crate::neural_operators::chebyshev::TRANSFER_DEGREE
pub fn poisson_errors(pred: ArrayView3<f64>, u: ArrayView3<f64>) -> PoissonErrors {
    let (np, s, s2) = pred.dim();
    let (nu, n, n2) = u.dim();
    assert_eq!(
        np, nu,
        "poisson_errors: pred has {np} samples but u has {nu}"
    );
    assert_eq!(
        s,
        s2,
        "poisson_errors: pred must be [N, s, s], got {:?}",
        pred.dim()
    );
    assert_eq!(
        n,
        n2,
        "poisson_errors: u must be [N, n, n], got {:?}",
        u.dim()
    );
    assert_eq!(
        s + 1,
        n,
        "poisson_errors: the uniform grid must have s = n - 1 = {} points, got {s}",
        n - 1
    );

    let transfers = Transfers::new(n);
    let w = clencurt(n);

    let mut errors = PoissonErrors {
        transfer: Vec::with_capacity(nu),
        model: Vec::with_capacity(nu),
        total: Vec::with_capacity(nu),
        boundary_max_ratio: Vec::with_capacity(nu),
        boundary_rel_rms: Vec::with_capacity(nu),
    };
    for (p, ui) in pred.axis_iter(Axis(0)).zip(u.axis_iter(Axis(0))) {
        let u_uni = transfers.to_uniform(ui);
        let round_trip = transfers.to_cheb(u_uni.view());
        let u_hat = transfers.to_cheb(p);

        errors.transfer.push(cc_rel_l2_error_weighted(
            round_trip.view(),
            ui,
            w.view(),
            w.view(),
        ));
        errors.model.push(rel_l2_equal_weights(p, u_uni.view()));
        errors.total.push(cc_rel_l2_error_weighted(
            u_hat.view(),
            ui,
            w.view(),
            w.view(),
        ));
        let (ratio, rms) = boundary_errors(&u_hat, ui, &w);
        errors.boundary_max_ratio.push(ratio);
        errors.boundary_rel_rms.push(rms);
    }
    errors
}

/// Tolerance of [`check_error3`], relative: the run's test_l2 is f32.
pub const ERROR3_TOL: f64 = 1e-5;

/// The cross-check of a mean error 3 against a run's last test_l2, from [`check_error3`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Error3Check {
    /// The last epoch in metrics.csv.
    pub epoch: usize,
    /// That epoch's test_l2, widened to f64.
    pub test_l2: f64,
    /// |mean error 3 − test_l2| / |test_l2|.
    pub rel_diff: f64,
    /// Whether `rel_diff` ≤ [`ERROR3_TOL`].
    pub passed: bool,
}

impl fmt::Display for Error3Check {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "mean error 3 vs metrics.csv test_l2 {:.6e} (epoch {}): rel diff {:.1e} (tol {ERROR3_TOL:e}) {}",
            self.test_l2,
            self.epoch,
            self.rel_diff,
            if self.passed { "ok" } else { "FAILED" }
        )
    }
}

/// Compares `mean_error3`, error 3 averaged over the test split, with `last.test_l2`, the
/// run's final epoch (`RunDir::metrics`). Error 3 is the training loss's relative L²
/// (CONVENTIONS §11), so on the run's own test split the two agree to f32 round-off.
pub fn check_error3(last: &EpochMetrics, mean_error3: f64) -> Error3Check {
    let test_l2 = f64::from(last.test_l2);
    let rel_diff = (mean_error3 - test_l2).abs() / test_l2.abs();
    Error3Check {
        epoch: last.epoch,
        test_l2,
        rel_diff,
        passed: rel_diff <= ERROR3_TOL,
    }
}

/// Tolerance of the error 2 cross-check in [`evaluate`], relative: the recomputed error 2
/// and the sidecar's are both f64, with the same transfers and norm.
pub const ERROR2_TOL: f64 = 1e-12;

/// The boundary verdict of [`evaluate`]: the hard-constraint ablation is warranted if the
/// mean relative boundary RMS ≥ this × the mean error 4 (design §12, decision 16).
pub const BOUNDARY_THRESHOLD: f64 = 0.10;

/// Error 1, quoted from Phase 1's Results: it is not recomputed (design §12, decision 16).
pub const ERROR1_QUOTE: &str =
    "solver A floor <= 1e-12 from n = 33 (manufactured); A vs C order 1.993-1.999 (Phase 1 T2, T3)";

/// What a run's `errors.json` records: [`evaluate`]'s result. Its [`Display`](fmt::Display)
/// is the report `predict_poisson` prints.
///
/// Serde derives with [`Config`] implemented by hand, not `#[derive(Config)]`, because
/// that derive implements `Display` as the JSON. The `#[serde(skip)]` fields are not in
/// `errors.json`; they complete the report and are empty in a loaded record.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "burn::serde")]
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
    /// Whether `boundary_to_total` ≥ [`BOUNDARY_THRESHOLD`].
    pub ablation_warranted: bool,
    /// Whether both cross-checks passed; a skipped error 2 comparison counts as passed.
    pub cross_checks_passed: bool,
    /// The run directory, for the report's `poisson_plot` command.
    #[serde(skip)]
    pub run_dir: PathBuf,
    /// The sidecar's sample count, for the report when error 2 was not compared.
    #[serde(skip)]
    pub sidecar_count: usize,
    /// The error 3 cross-check, for the report.
    #[serde(skip)]
    pub error3_check: Option<Error3Check>,
}

impl Config for PoissonErrorsRecord {}

/// Evaluates the per-sample `errors` of a run's test split, as `predict_poisson` reports
/// them: the five errors, the worst sample, both cross-checks and the boundary verdict.
///
/// - `n`: the CGL nodes per axis of the evaluated split.
/// - `sidecar`: the test split's sidecar. Error 2 is compared with it, to
///   [`ERROR2_TOL`], only when it covers exactly the evaluated samples.
/// - `last`: the run's last epoch; the mean error 3 is checked with [`check_error3`].
/// - `device`: the device the forward pass ran on, as Burn prints it.
/// - `run_dir`: the run directory, only for the report.
///
/// The worst sample is the first holding the max error 4, NaN included, as
/// [`ErrorStats`] reports it. The ablation is warranted if the mean relative boundary RMS
/// / mean error 4 ≥ [`BOUNDARY_THRESHOLD`].
///
/// # Panics
/// If the sidecar's n, s or transfer degree differ from `n`, n − 1 and
/// [`TRANSFER_DEGREE`], or `errors` holds no samples.
pub fn evaluate(
    errors: &PoissonErrors,
    n: usize,
    sidecar: &SidecarRecord,
    last: &EpochMetrics,
    device: String,
    run_dir: &Path,
) -> PoissonErrorsRecord {
    let n_test = errors.total.len();
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

    // Cross-check 1: error 2 against the sidecar the generator wrote
    assert_eq!(
        (sidecar.n, sidecar.error2.s, sidecar.error2.d),
        (n, n - 1, TRANSFER_DEGREE),
        "sidecar n, s, d differ from this evaluation"
    );
    let e2_sidecar = ErrorStats {
        mean: sidecar.error2.mean,
        max: sidecar.error2.max,
    };
    // the sidecar covers its whole split; compare only when all of it was evaluated, and
    // count a skipped comparison as passed
    let error2_compared = sidecar.count == n_test;
    let e2_ok = !error2_compared || error2_matches(&e2, &e2_sidecar);

    // Cross-check 2: mean error 3 against the run's last test_l2
    let e3_check = check_error3(last, e3.mean);

    let boundary_to_total = e5_rms.mean / e4.mean;
    PoissonErrorsRecord {
        n_test,
        n,
        device,
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
        ablation_warranted: boundary_to_total >= BOUNDARY_THRESHOLD,
        cross_checks_passed: e2_ok && e3_check.passed,
        run_dir: run_dir.to_path_buf(),
        sidecar_count: sidecar.count,
        error3_check: Some(e3_check),
    }
}

fn rel_diff(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

/// Whether both the mean and the max of error 2 agree with the sidecar's to [`ERROR2_TOL`].
fn error2_matches(e2: &ErrorStats, sidecar: &ErrorStats) -> bool {
    rel_diff(e2.mean, sidecar.mean) <= ERROR2_TOL && rel_diff(e2.max, sidecar.max) <= ERROR2_TOL
}

impl fmt::Display for PoissonErrorsRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (n, n_test) = (self.n, self.n_test);
        writeln!(
            f,
            "errors on the test split (n = {n}, s = {}, {n_test} samples), device {}, backend features {:?}",
            n - 1,
            self.device,
            self.backend_features
        )?;
        writeln!(f, "{:<34} {:>12} {:>12}", "error", "mean", "max")?;
        writeln!(f, "{:<34} {}", "1 solver (quoted)", self.error1)?;
        for (name, st) in [
            ("2 transfer (CC norm)", &self.error2),
            ("3 model (uniform, equal weights)", &self.error3),
            ("4 total (CC norm)", &self.error4),
            ("5 boundary max ratio", &self.error5_max_ratio),
            ("5 boundary relative RMS", &self.error5_rel_rms),
        ] {
            writeln!(f, "{name:<34} {:>12.4e} {:>12.4e}", st.mean, st.max)?;
        }
        let worst = self.error4_worst_sample;
        writeln!(
            f,
            "worst sample (max error 4): {worst}; plot it with: poisson_plot -- {} {worst}",
            self.run_dir.display()
        )?;

        let (e2, e2_sidecar) = (&self.error2, &self.error2_sidecar);
        if self.error2_compared {
            writeln!(
                f,
                "\ncheck: error 2 vs sidecar mean {:.6e} / max {:.6e}: rel diff {:.1e} / {:.1e} (tol {ERROR2_TOL:e}) {}",
                e2_sidecar.mean,
                e2_sidecar.max,
                rel_diff(e2.mean, e2_sidecar.mean),
                rel_diff(e2.max, e2_sidecar.max),
                if error2_matches(e2, e2_sidecar) {
                    "ok"
                } else {
                    "FAILED"
                }
            )?;
        } else {
            writeln!(
                f,
                "\ncheck: error 2 vs sidecar skipped: the sidecar covers {} samples but {n_test} were evaluated",
                self.sidecar_count
            )?;
        }
        if let Some(check) = &self.error3_check {
            writeln!(f, "check: {check}")?;
        }

        let warranted = self.ablation_warranted;
        write!(
            f,
            "\nboundary verdict: mean relative boundary RMS / mean error 4 = {:.4e} / {:.4e} = {:.3} \
             ({} the {BOUNDARY_THRESHOLD} threshold): the hard-constraint ablation is {}",
            self.error5_rel_rms.mean,
            self.error4.mean,
            self.boundary_to_total,
            if warranted { "at or above" } else { "below" },
            if warranted {
                "warranted"
            } else {
                "not warranted"
            }
        )
    }
}

/// ‖a − b‖₂ / ‖b‖₂ over all entries with equal weights (`LpLoss::rel` with p = 2).
fn rel_l2_equal_weights(a: ArrayView2<f64>, b: ArrayView2<f64>) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum();
    (num / den).sqrt()
}

/// Error 5 for one sample: `(max ratio, relative boundary RMS)` of `u_hat: [n, n]` against
/// `u: [n, n]`, both on the CGL grid, with Clenshaw–Curtis weights `w: [n]`.
fn boundary_errors(u_hat: &Array2<f64>, u: ArrayView2<f64>, w: &Array1<f64>) -> (f64, f64) {
    let last = u_hat.nrows() - 1;
    let edges = [
        u_hat.row(0),
        u_hat.row(last),
        u_hat.column(0),
        u_hat.column(last),
    ];
    let max_edge = edges
        .iter()
        .flat_map(|e| e.iter())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    let max_u = u.iter().fold(0.0_f64, |m, v| m.max(v.abs()));

    // ∮ û² ds as four edge integrals of length 2 each (the weights sum to 2 per edge)
    let line: f64 = edges
        .iter()
        .map(|e| e.iter().zip(w).map(|(v, wi)| wi * v * v).sum::<f64>())
        .sum();
    let area = cc_norm_sq(u, w.view(), w.view());
    (max_edge / max_u, (line / 8.0).sqrt() / (area / 4.0).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::chebyshev::nodes;
    use crate::neural_operators::losses::data_losses::{LpLoss, Reduction};
    use burn::prelude::*;
    use ndarray::Array3;

    /// `count` smooth fields on the n × n CGL grid that vanish on ∂Ω; sample k is scaled
    /// and shifted so the samples differ.
    fn vanishing_u(count: usize, n: usize) -> Array3<f64> {
        let x = nodes(n);
        Array3::from_shape_fn((count, n, n), |(k, i, j)| {
            let (a, b) = (x[i], x[j]);
            (1.0 + k as f64) * (1.0 - a * a) * (1.0 - b * b) * (a + 0.3 * k as f64).exp() * b.cos()
        })
    }

    /// Deterministic noise in [−1, 1), a hash of the index.
    fn noise(k: usize, i: usize, j: usize) -> f64 {
        let t = (k as f64 * 3.17 + i as f64 * 12.9898 + j as f64 * 78.233).sin() * 43758.5453;
        2.0 * (t - t.floor()) - 1.0
    }

    /// T_cu applied to every sample of `u: [N, n, n]`, giving `[N, n − 1, n − 1]`.
    fn to_uniform(u: &Array3<f64>) -> Array3<f64> {
        let (count, n, _) = u.dim();
        let t = Transfers::new(n);
        let mut out = Array3::zeros((count, n - 1, n - 1));
        for (mut dst, src) in out.axis_iter_mut(Axis(0)).zip(u.axis_iter(Axis(0))) {
            dst.assign(&t.to_uniform(src));
        }
        out
    }

    #[test]
    fn exact_prediction_has_only_transfer_error() {
        let n = 17;
        let u = vanishing_u(3, n);
        let pred = to_uniform(&u);
        let e = poisson_errors(pred.view(), u.view());
        for k in 0..3 {
            assert!(e.model[k] <= 1e-14, "e3[{k}] = {:e}", e.model[k]);
            assert!(
                (e.total[k] - e.transfer[k]).abs() <= 1e-14,
                "e4 {:e} vs e2 {:e}",
                e.total[k],
                e.transfer[k]
            );
            assert!(
                e.transfer[k] > 0.0 && e.transfer[k] < 1e-2,
                "e2 = {:e}",
                e.transfer[k]
            );
            assert!(
                e.boundary_max_ratio[k] <= 1e-14,
                "{:e}",
                e.boundary_max_ratio[k]
            );
            assert!(
                e.boundary_rel_rms[k] <= 1e-14,
                "{:e}",
                e.boundary_rel_rms[k]
            );
        }
    }

    #[test]
    fn boundary_offset_is_measured() {
        // u_k = (1 + k)(1 − x²)(1 − y²): max |u_k| = 1 + k at the origin (a CGL node for
        // odd n), and ∫∫ u_k² dA = (1 + k)² (16/15)², which Clenshaw–Curtis integrates
        // exactly (degree 4 per axis ≤ n − 1)
        let n = 17;
        let c = 0.037;
        let x = nodes(n);
        let u = Array3::from_shape_fn((2, n, n), |(k, i, j)| {
            (1.0 + k as f64) * (1.0 - x[i] * x[i]) * (1.0 - x[j] * x[j])
        });
        let pred = to_uniform(&u) + c;
        let e = poisson_errors(pred.view(), u.view());
        for k in 0..2 {
            let scale = 1.0 + k as f64;
            // û = c on ∂Ω: T_uc reproduces constants and both grids share the endpoints.
            // ∮ û² ds = 8c² with each corner weighted by both its edges, so the RMS is c
            let ratio = c / scale;
            let rms = c / (scale * 16.0 / 15.0 / 2.0);
            assert!(
                (e.boundary_max_ratio[k] - ratio).abs() <= 1e-12 * ratio,
                "ratio {} vs {ratio}",
                e.boundary_max_ratio[k]
            );
            assert!(
                (e.boundary_rel_rms[k] - rms).abs() <= 1e-12 * rms,
                "rms {} vs {rms}",
                e.boundary_rel_rms[k]
            );
        }
    }

    #[test]
    fn error3_matches_lploss() {
        let (count, n) = (4, 17);
        let s = n - 1;
        let u = vanishing_u(count, n);
        let target = to_uniform(&u);
        let pred = Array3::from_shape_fn((count, s, s), |(k, i, j)| {
            target[[k, i, j]] + 0.05 * noise(k, i, j)
        });
        let e = poisson_errors(pred.view(), u.view());
        let mean = e.model.iter().sum::<f64>() / count as f64;

        let device: Device = Default::default();
        let flat = |a: &Array3<f64>| {
            let data: Vec<f32> = a.iter().map(|&v| v as f32).collect();
            Tensor::<2>::from_data(TensorData::new(data, vec![count, s * s]), &device)
        };
        let loss = LpLoss::new(2, 2, Reduction::Sum).rel(flat(&pred), flat(&target));
        let lp = f64::from(loss.into_scalar::<f32>()) / count as f64;
        assert!((mean - lp).abs() <= 1e-6, "e3 mean {mean} vs LpLoss {lp}");
    }

    #[test]
    #[should_panic(expected = "pred has 2 samples but u has 3")]
    fn sample_count_mismatch_panics() {
        let u = vanishing_u(3, 9);
        let pred = Array3::<f64>::zeros((2, 8, 8));
        poisson_errors(pred.view(), u.view());
    }

    #[test]
    #[should_panic(expected = "s = n - 1 = 8 points, got 9")]
    fn wrong_uniform_size_panics() {
        let u = vanishing_u(1, 9);
        let pred = Array3::<f64>::zeros((1, 9, 9));
        poisson_errors(pred.view(), u.view());
    }

    #[test]
    #[should_panic(expected = "pred must be [N, s, s]")]
    fn non_square_pred_panics() {
        let u = vanishing_u(1, 9);
        let pred = Array3::<f64>::zeros((1, 8, 7));
        poisson_errors(pred.view(), u.view());
    }

    #[test]
    #[should_panic(expected = "u must be [N, n, n]")]
    fn non_square_u_panics() {
        let u = Array3::<f64>::zeros((1, 9, 8));
        let pred = Array3::<f64>::zeros((1, 8, 8));
        poisson_errors(pred.view(), u.view());
    }

    #[test]
    #[should_panic(expected = "n must be >= 4")]
    fn too_small_n_panics() {
        let u = Array3::<f64>::zeros((1, 3, 3));
        let pred = Array3::<f64>::zeros((1, 2, 2));
        poisson_errors(pred.view(), u.view());
    }

    /// One metrics row per epoch with the given `test_l2`.
    fn metrics(test_l2: &[f32]) -> Vec<EpochMetrics> {
        test_l2
            .iter()
            .enumerate()
            .map(|(epoch, &test_l2)| EpochMetrics {
                epoch,
                train_mse: 0.0,
                train_l2: 0.0,
                test_l2,
                current_lr: 1e-3,
            })
            .collect()
    }

    /// Synthetic per-sample errors: error 4 is `total`, error 2 `transfer`, error 3 0.25
    /// (the test_l2 of [`last_epoch`]) and the relative boundary RMS `rms`.
    fn synthetic(total: &[f64], transfer: f64, rms: f64) -> PoissonErrors {
        let k = total.len();
        PoissonErrors {
            transfer: vec![transfer; k],
            model: vec![0.25; k],
            total: total.to_vec(),
            boundary_max_ratio: vec![0.0; k],
            boundary_rel_rms: vec![rms; k],
        }
    }

    fn last_epoch() -> EpochMetrics {
        metrics(&[0.5, 0.25]).pop().unwrap()
    }

    /// The fixture sidecar (n = 65), claiming `count` samples with error 2 `transfer`.
    fn sidecar(count: usize, transfer: f64) -> SidecarRecord {
        let mut sidecar = SidecarRecord::load_binary(
            include_str!("../../../tests/fixtures/poisson_n65_K32_test.json").as_bytes(),
        )
        .unwrap();
        sidecar.count = count;
        (sidecar.error2.mean, sidecar.error2.max) = (transfer, transfer);
        sidecar
    }

    fn eval(errors: &PoissonErrors, sidecar: &SidecarRecord) -> PoissonErrorsRecord {
        evaluate(
            errors,
            65,
            sidecar,
            &last_epoch(),
            "Cpu".into(),
            Path::new("runs/x"),
        )
    }

    #[test]
    fn evaluate_picks_the_first_worst_sample() {
        let e = synthetic(&[0.1, 0.3, 0.2, 0.3], 1e-5, 0.0);
        let record = eval(&e, &sidecar(4, 1e-5));
        assert_eq!(record.error4_worst_sample, 1);
        assert!(record.cross_checks_passed, "{record}");
        assert!(
            record
                .to_string()
                .contains("worst sample (max error 4): 1;")
        );

        // a NaN is the max, as ErrorStats reports it
        let e = synthetic(&[0.1, 0.3, f64::NAN, 0.2], 1e-5, 0.0);
        assert_eq!(eval(&e, &sidecar(4, 1e-5)).error4_worst_sample, 2);
    }

    #[test]
    fn evaluate_boundary_verdict_at_the_threshold() {
        // the ratio is the mean RMS over the mean error 4, here 0.5
        let total = [0.4, 0.6];
        let below = eval(&synthetic(&total, 1e-5, 0.0499), &sidecar(2, 1e-5));
        assert!(below.boundary_to_total < BOUNDARY_THRESHOLD);
        assert!(!below.ablation_warranted);
        assert!(below.to_string().ends_with("ablation is not warranted"));
        let above = eval(&synthetic(&total, 1e-5, 0.0501), &sidecar(2, 1e-5));
        assert!(above.ablation_warranted);
        assert!(above.to_string().ends_with("ablation is warranted"));
    }

    #[test]
    fn evaluate_flags_a_cross_check_mismatch() {
        let e = synthetic(&[0.1, 0.2], 1e-5, 0.0);
        let ok = eval(&e, &sidecar(2, 1e-5));
        assert!(ok.error2_compared && ok.cross_checks_passed, "{ok}");

        // error 2 off the sidecar by 1e-9, relative
        let bad = eval(&e, &sidecar(2, 1e-5 * (1.0 + 1e-9)));
        assert!(!bad.cross_checks_passed);
        assert!(bad.to_string().contains("(tol 1e-12) FAILED"), "{bad}");

        // the same mismatch is skipped when the sidecar covers other samples
        let skipped = eval(&e, &sidecar(200, 1e-5 * (1.0 + 1e-9)));
        assert!(!skipped.error2_compared && skipped.cross_checks_passed);
        assert!(
            skipped
                .to_string()
                .contains("covers 200 samples but 2 were evaluated")
        );

        // error 3 off the run's last test_l2
        let mut e3_off = e.clone();
        e3_off.model = vec![0.3; 2];
        let bad = eval(&e3_off, &sidecar(2, 1e-5));
        assert!(!bad.cross_checks_passed);
        assert_eq!(bad.error3_check.map(|c| c.passed), Some(false));
    }

    #[test]
    #[should_panic(expected = "sidecar n, s, d differ")]
    fn evaluate_rejects_another_grid() {
        let e = synthetic(&[0.1], 1e-5, 0.0);
        evaluate(
            &e,
            33,
            &sidecar(1, 1e-5),
            &last_epoch(),
            "Cpu".into(),
            Path::new("x"),
        );
    }

    #[test]
    fn error3_check_uses_the_last_epoch() {
        let rows = metrics(&[0.5, 0.25]);
        let last = rows.last().unwrap();
        let check = check_error3(last, 0.25 * (1.0 + 5e-6));
        assert_eq!((check.epoch, check.test_l2), (1, 0.25));
        assert!(check.passed, "{check}");
        let check = check_error3(last, 0.25 * (1.0 + 2e-5));
        assert!(!check.passed, "{check}");
        assert!(check.to_string().ends_with("FAILED"));
    }
}

//! The errors of design §7 for a Poisson FNO, measured on the Chebyshev grid.
//!
//! Given decoded predictions on the run's uniform learning grid of s points, the raw CGL
//! solutions u of the same samples and the run's [`Transfers`] (R = T_cu, P = T_uc, any s;
//! CONVENTIONS §12), [`poisson_errors`] computes per sample (design §13.3 names in
//! brackets):
//! - error 2 (transfer; the transfer-only floor): the round trip ‖P R u − u‖ / ‖u‖ in the
//!   Clenshaw–Curtis norm; at s = n − 1 the sidecar's `error2` recomputed;
//! - error 3 (model; the coarse error): the relative L² error of the prediction against R u
//!   on the learning grid, with equal weights, as `LpLoss::rel` computes it during training
//!   (CONVENTIONS §11);
//! - error 4 (total; the fine-grid error): ‖P(prediction) − u‖ / ‖u‖ in the
//!   Clenshaw–Curtis norm;
//! - error 5 (boundary): with û = T_uc(prediction), the max ratio max_∂Ω |û| / max_Ω |u|
//!   and the relative boundary RMS √(∮ û² ds / 8) / √(∫∫ u² dA / 4).
//!
//! [`poisson_errors_nd`] computes the same errors for 2D or 3D fields, 2D through
//! [`poisson_errors`] itself (`_nd` here means 2D or 3D, checked at run time, unlike
//! `chebyshev`'s any rank); in 3D error 5 runs over the six faces, √(∯ û² dS / 24) /
//! √(∭ u² dV / 8) (CONVENTIONS §12).
//!
//! [`check_error3`] cross-checks the mean error 3 against the run's last test_l2. Each
//! error is summarised over the samples with [`ErrorStats`]. [`evaluate`] applies the
//! evaluation policy, the worst sample, both cross-checks and the boundary verdict, and
//! returns a [`PoissonEvaluation`]: the [`PoissonErrorsRecord`] a run's `errors.json`
//! holds, with the report `predict_poisson` prints; [`evaluate_3d`] the same for a 3D run,
//! the report `predict_poisson3d` prints (3D CL4.1).
//!
//! Error 1 (the solver) needs the `datagen` feature and is quoted, not recomputed (design
//! §12, decision 16). Everything here is ndarray and f64, built without `datagen`
//! (decision 15). Grids, transfers and norms follow CONVENTIONS §12.

// Burn's `#[derive(Config)]` writes `field: field` in the `new` it generates, in an
// impl block outside the struct's attributes (burn-derive `config/analyzer_struct.rs`).
#![allow(clippy::redundant_field_names)]

use super::stats::ErrorStats;
use crate::neural_operators::chebyshev::{
    TRANSFER_DEGREE, Transfers, cc_norm_sq_nd, cc_rel_l2_error_weighted_nd, clencurt,
};
use crate::neural_operators::data::loaders::poisson::SidecarRecord;
use crate::neural_operators::runs::EpochMetrics;
use burn::config::Config;
use ndarray::{Array1, ArrayView, ArrayView3, ArrayViewD, Axis, Dimension, RemoveAxis};
use std::fmt;
use std::path::{Path, PathBuf};

/// Per-sample errors from [`poisson_errors`] (2D) or [`poisson_errors_nd`] (2D or 3D), one
/// entry per sample, in sample order.
#[derive(Debug, Clone, Default)]
pub struct PoissonErrors {
    /// Error 2: ‖P R u − u‖ / ‖u‖, Clenshaw–Curtis norm.
    pub transfer: Vec<f64>,
    /// Error 3: ‖prediction − R u‖ / ‖R u‖, equal weights on the learning grid.
    pub model: Vec<f64>,
    /// Error 4: ‖P(prediction) − u‖ / ‖u‖, Clenshaw–Curtis norm.
    pub total: Vec<f64>,
    /// ‖u‖ in the Clenshaw–Curtis norm, the denominator of errors 2 and 4 (design §12,
    /// decision 22: does a high error 4 come from a small ‖u‖?).
    pub u_norm: Vec<f64>,
    /// Error 5, max ratio: max over ∂Ω of |û| / max over Ω of |u|, on the CGL nodes.
    pub boundary_max_ratio: Vec<f64>,
    /// Error 5, relative boundary RMS: in 2D √(∮ û² ds / 8) / √(∫∫ u² dA / 4), in 3D
    /// √(∯ û² dS / 24) / √(∭ u² dV / 8) (CONVENTIONS §12).
    pub boundary_rel_rms: Vec<f64>,
}

/// The errors 2–5 of design §7 for each sample (see the module docs).
///
/// - `pred`: decoded predictions on the uniform learning grid, `[N, s, s]`, physical scale,
///   stored 'ij' (CONVENTIONS §12).
/// - `u`: the raw solutions on the CGL grid, `[N, n, n]`, the reference.
/// - `transfers`: the run's R and P between those grids (`Transfers::with_s(n, s)`;
///   `Transfers::new(n)` for s = n − 1).
///
/// The Clenshaw–Curtis norms use [`clencurt`]`(n)` weights: over Ω
/// as a tensor product, and along each of the four edges of ∂Ω, so each corner node
/// carries the end weights of both its edges; the boundary maximum visits it twice, which
/// does not change the maximum. A sample with u ≡ 0 gives NaN or infinity, as
/// [`cc_rel_l2_error_weighted`](crate::neural_operators::chebyshev::cc_rel_l2_error_weighted)
/// does.
///
/// # Panics
/// If `pred` and `u` hold different numbers of samples, either is not square in its
/// spatial axes, or their grids are not the transfers' s and n.
pub fn poisson_errors(
    pred: ArrayView3<f64>,
    u: ArrayView3<f64>,
    transfers: &Transfers,
) -> PoissonErrors {
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
        (s, n),
        (transfers.s(), transfers.n()),
        "poisson_errors: pred and u must be on the transfers' grids (s, n)"
    );

    per_sample(pred, u, transfers)
}

/// [`poisson_errors`] for 2D or 3D fields (rank r = 2 or 3, the ranks CONVENTIONS §12
/// defines): axis 0 the samples, the rest the spatial axes, stored 'ij' or 'ijk'.
///
/// - `pred`: decoded predictions on the uniform learning grid, `[N, s, …, s]`.
/// - `u`: the raw solutions on the CGL grid, `[N, n, …, n]`, of the same rank.
/// - `transfers`: the run's R and P, applied along every axis.
///
/// Rank 3 arrays (2D fields) give [`poisson_errors`]' bits: both run one per-sample loop
/// (3D D1, D2; CL4.1). The errors are those of the module docs with tensor Clenshaw–Curtis weights
/// over every axis, and error 5 sums the 2r faces of ∂Ω, each with the tensor weights of
/// its r − 1 coordinates, so a node on an edge or corner is counted once per face that
/// holds it. The relative boundary RMS normalises by the total face area r · 2^r against
/// the volume 2^r: in 3D √(∯ û² dS / 24) / √(∭ u² dV / 8) (CONVENTIONS §12).
///
/// # Panics
/// If `u` has not two or three spatial axes, `pred` and `u` differ in rank or sample count,
/// or an axis is not the transfers' s (for `pred`) or n (for `u`).
pub fn poisson_errors_nd<D: RemoveAxis>(
    pred: ArrayView<f64, D>,
    u: ArrayView<f64, D>,
    transfers: &Transfers,
) -> PoissonErrors {
    assert!(
        matches!(u.ndim(), 3 | 4),
        "poisson_errors_nd: u must be [N, n, n] or [N, n, n, n] (two or three spatial axes), got {:?}",
        u.shape()
    );
    assert_eq!(
        pred.ndim(),
        u.ndim(),
        "poisson_errors_nd: pred has rank {} but u has rank {}",
        pred.ndim(),
        u.ndim()
    );
    let (np, nu) = (pred.len_of(Axis(0)), u.len_of(Axis(0)));
    assert_eq!(
        np, nu,
        "poisson_errors_nd: pred has {np} samples but u has {nu}"
    );
    let (s, n) = (transfers.s(), transfers.n());
    assert!(
        pred.shape()[1..].iter().all(|&a| a == s) && u.shape()[1..].iter().all(|&a| a == n),
        "poisson_errors_nd: pred {:?} and u {:?} must be on the transfers' grids (s, n) = ({s}, {n})",
        pred.shape(),
        u.shape()
    );

    per_sample(pred, u, transfers)
}

/// The errors of every sample, for [`poisson_errors`] and [`poisson_errors_nd`] after their
/// checks: `pred: [N, s, …, s]` and `u: [N, n, …, n]`, two or three spatial axes. At rank 2
/// it gives the bits of `poisson_errors` before 3D CL4.1 (the D2 oracle in the tests).
fn per_sample<D: RemoveAxis>(
    pred: ArrayView<f64, D>,
    u: ArrayView<f64, D>,
    transfers: &Transfers,
) -> PoissonErrors {
    let w = clencurt(transfers.n());
    let ws = vec![w.view(); u.ndim() - 1];
    let mut errors = PoissonErrors::default();
    for (p, ui) in pred.axis_iter(Axis(0)).zip(u.axis_iter(Axis(0))) {
        let u_uni = transfers.to_uniform_nd(ui.view());
        let round_trip = transfers.to_cheb_nd(u_uni.view());
        let u_hat = transfers.to_cheb_nd(p.view());

        errors.transfer.push(cc_rel_l2_error_weighted_nd(
            round_trip.view(),
            ui.view(),
            &ws,
        ));
        errors.model.push(rel_l2_equal_weights(p, u_uni.view()));
        errors
            .total
            .push(cc_rel_l2_error_weighted_nd(u_hat.view(), ui.view(), &ws));
        let u_sq = cc_norm_sq_nd(ui.view(), &ws);
        errors.u_norm.push(u_sq.sqrt());
        let max_u = ui.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let (ratio, rms) = boundary_errors_nd(u_hat.view().into_dyn(), &w, u_sq, max_u);
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
    let rel_diff = rel_diff(mean_error3, test_l2);
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

/// Error 1 of a 3D run, quoted from the 3D solver A3's results, as [`evaluate_3d`] records it.
pub const ERROR1_QUOTE_3D: &str = "solver A3 floor <= 1e-12 from n = 33 (manufactured and sine-series closed forms, max norm and CC L2; 3D CL2.1)";

/// The error 2 cross-check against the test split's sidecar, from [`evaluate`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Error2Check {
    /// Error 2 as the sidecar records it.
    pub sidecar: ErrorStats,
    /// The sidecar's sample count.
    pub sidecar_count: usize,
    /// The samples evaluated.
    pub n_test: usize,
    /// The learning grid of the sidecar's error 2 (n − 1).
    pub sidecar_s: usize,
    /// The run's learning grid.
    pub s: usize,
    /// (mean, max) of |error 2 − sidecar| / |sidecar|; `None` when the comparison was
    /// skipped because the sidecar's error 2 is at another s, or covers other samples than
    /// those evaluated.
    pub rel_diff: Option<(f64, f64)>,
    /// Whether both relative differences are ≤ [`ERROR2_TOL`]; a skipped comparison
    /// counts as passed.
    pub passed: bool,
}

impl fmt::Display for Error2Check {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.rel_diff {
            Some((mean, max)) => write!(
                f,
                "error 2 vs sidecar mean {:.6e} / max {:.6e}: rel diff {mean:.1e} / {max:.1e} (tol {ERROR2_TOL:e}) {}",
                self.sidecar.mean,
                self.sidecar.max,
                if self.passed { "ok" } else { "FAILED" }
            ),
            None if self.sidecar_s != self.s => write!(
                f,
                "error 2 vs sidecar skipped: the sidecar's error 2 is at s = {}, this run's at s = {}",
                self.sidecar_s, self.s
            ),
            None => write!(
                f,
                "error 2 vs sidecar skipped: the sidecar covers {} samples but {} were evaluated",
                self.sidecar_count, self.n_test
            ),
        }
    }
}

/// The file in a run directory that holds its [`PoissonErrorsRecord`].
pub const ERRORS_JSON: &str = "errors.json";

/// What a run's `errors.json` records, from [`evaluate`].
#[derive(Config, Debug)]
pub struct PoissonErrorsRecord {
    /// Test samples evaluated.
    pub n_test: usize,
    /// CGL nodes per axis.
    pub n: usize,
    /// The run's learning grid, uniform points per axis (n − 1 before Phase 5).
    pub s: usize,
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
    /// Whether error 2 was compared with the sidecar: only when `s` is the sidecar's
    /// (n − 1) and `n_test` equals its sample count, since it covers its whole split.
    pub error2_compared: bool,
    /// Error 3 (model).
    pub error3: ErrorStats,
    /// The last test_l2 in the run's metrics.csv.
    pub error3_metrics_csv: f64,
    /// Error 4 (total).
    pub error4: ErrorStats,
    /// The test sample with the max error 4, an index into the test split.
    pub error4_worst_sample: usize,
    /// Error 4 per test sample, in sample order (design §12, decision 22).
    pub error4_per_sample: Vec<f64>,
    /// ‖u‖ per test sample, Clenshaw–Curtis norm, in sample order.
    pub u_norm_per_sample: Vec<f64>,
    /// Pearson correlation of ln(error 4) with ln ‖u‖ over the test samples: near −1 if a
    /// high error 4 comes from a small ‖u‖ (decision 22).
    pub error4_u_norm_log_correlation: f64,
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
}

/// [`evaluate`]'s result: the record `errors.json` holds and the two cross-checks behind
/// it. Its [`Display`](fmt::Display) is the report `predict_poisson` prints.
#[derive(Debug)]
pub struct PoissonEvaluation {
    /// What `errors.json` holds.
    pub record: PoissonErrorsRecord,
    /// The error 2 cross-check against the test sidecar.
    pub error2: Error2Check,
    /// The error 3 cross-check against the run's last test_l2.
    pub error3: Error3Check,
    /// The run directory for the report's `poisson_plot` command (2D, from [`evaluate`]);
    /// `None` from [`evaluate_3d`], as no example plots a 3D sample.
    pub plot_run_dir: Option<PathBuf>,
}

/// Evaluates the per-sample `errors` of a run's test split, as `predict_poisson` reports
/// them: the five errors, the worst sample, both cross-checks and the boundary verdict.
///
/// - `transfers`: the run's transfers, as `errors` was computed with; they give n and s.
/// - `sidecar`: the test split's sidecar. Error 2 is compared with it, to
///   [`ERROR2_TOL`], only when its error 2 is at the run's s (s = n − 1) and it covers
///   exactly the evaluated samples.
/// - `last`: the run's last epoch; the mean error 3 is checked with [`check_error3`].
/// - `device`: the device the forward pass ran on, as Burn prints it.
/// - `backend_features`: the backend features the binary was built with
///   ([`backend_features`](crate::neural_operators::utils::backend::backend_features)).
/// - `run_dir`: the run directory, only for the report.
///
/// The worst sample is the first holding the max error 4, NaN included, as
/// [`ErrorStats`] reports it. The ablation is warranted if the mean relative boundary RMS
/// / mean error 4 ≥ [`BOUNDARY_THRESHOLD`].
///
/// # Panics
/// If the sidecar's n or transfer degree differ from the transfers' n and
/// [`TRANSFER_DEGREE`], or `errors` holds no samples.
pub fn evaluate(
    errors: &PoissonErrors,
    transfers: &Transfers,
    sidecar: &SidecarRecord,
    last: &EpochMetrics,
    device: String,
    backend_features: Vec<String>,
    run_dir: &Path,
) -> PoissonEvaluation {
    PoissonEvaluation {
        plot_run_dir: Some(run_dir.to_path_buf()),
        ..evaluate_core(
            errors,
            transfers,
            sidecar,
            last,
            device,
            backend_features,
            ERROR1_QUOTE,
        )
    }
}

/// [`evaluate`] for a 3D run, whose `errors` come from [`poisson_errors_nd`] on 3D fields
/// (3D CL4.1): the same record, the 2D type (no 3D field is new), with error 1 quoted as
/// [`ERROR1_QUOTE_3D`], and a report without the `poisson_plot` command.
///
/// # Panics
/// As [`evaluate`].
pub fn evaluate_3d(
    errors: &PoissonErrors,
    transfers: &Transfers,
    sidecar: &SidecarRecord,
    last: &EpochMetrics,
    device: String,
    backend_features: Vec<String>,
) -> PoissonEvaluation {
    evaluate_core(
        errors,
        transfers,
        sidecar,
        last,
        device,
        backend_features,
        ERROR1_QUOTE_3D,
    )
}

/// [`evaluate`] and [`evaluate_3d`] but for the plot's run directory (`None` here): the
/// record with error 1 quoted as `error1`, and both cross-checks.
fn evaluate_core(
    errors: &PoissonErrors,
    transfers: &Transfers,
    sidecar: &SidecarRecord,
    last: &EpochMetrics,
    device: String,
    backend_features: Vec<String>,
    error1: &str,
) -> PoissonEvaluation {
    let (n, s) = (transfers.n(), transfers.s());
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
        (sidecar.n, sidecar.error2.d),
        (n, TRANSFER_DEGREE),
        "sidecar n, d differ from this evaluation"
    );
    let e2_sidecar = ErrorStats {
        mean: sidecar.error2.mean,
        max: sidecar.error2.max,
    };
    // the sidecar's error 2 is at s = n - 1 over its whole split; compare only when this
    // evaluation matches both, and count a skipped comparison as passed
    let e2_rel_diff = (sidecar.error2.s == s && sidecar.count == n_test).then(|| {
        (
            rel_diff(e2.mean, e2_sidecar.mean),
            rel_diff(e2.max, e2_sidecar.max),
        )
    });
    let e2_check = Error2Check {
        sidecar: e2_sidecar,
        sidecar_count: sidecar.count,
        n_test,
        sidecar_s: sidecar.error2.s,
        s,
        rel_diff: e2_rel_diff,
        passed: e2_rel_diff.is_none_or(|(mean, max)| mean <= ERROR2_TOL && max <= ERROR2_TOL),
    };

    // Cross-check 2: mean error 3 against the run's last test_l2
    let e3_check = check_error3(last, e3.mean);

    let boundary_to_total = e5_rms.mean / e4.mean;
    let record = PoissonErrorsRecord {
        n_test,
        n,
        s,
        device,
        backend_features,
        error1: error1.to_string(),
        error2: e2,
        error2_sidecar: e2_sidecar,
        error2_compared: e2_rel_diff.is_some(),
        error3: e3,
        error3_metrics_csv: e3_check.test_l2,
        error4: e4,
        error4_worst_sample: worst,
        error4_per_sample: errors.total.clone(),
        u_norm_per_sample: errors.u_norm.clone(),
        error4_u_norm_log_correlation: log_correlation(&errors.total, &errors.u_norm),
        error5_max_ratio: e5_ratio,
        error5_rel_rms: e5_rms,
        boundary_to_total,
        ablation_warranted: boundary_to_total >= BOUNDARY_THRESHOLD,
        cross_checks_passed: e2_check.passed && e3_check.passed,
    };
    PoissonEvaluation {
        record,
        error2: e2_check,
        error3: e3_check,
        plot_run_dir: None,
    }
}

fn rel_diff(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

/// Pearson correlation of ln a with ln b; NaN for fewer than two samples or a constant
/// series.
pub fn log_correlation(a: &[f64], b: &[f64]) -> f64 {
    let (x, y): (Vec<f64>, Vec<f64>) = a.iter().zip(b).map(|(a, b)| (a.ln(), b.ln())).unzip();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (mx, my) = (mean(&x), mean(&y));
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (x, y) in x.iter().zip(&y) {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx) * (x - mx);
        syy += (y - my) * (y - my);
    }
    sxy / (sxx * syy).sqrt()
}

impl fmt::Display for PoissonEvaluation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let r = &self.record;
        let (n, s, n_test) = (r.n, r.s, r.n_test);
        writeln!(
            f,
            "errors on the test split (n = {n}, s = {s}, {n_test} samples), device {}, backend features {:?}",
            r.device, r.backend_features
        )?;
        writeln!(f, "{:<34} {:>12} {:>12}", "error", "mean", "max")?;
        writeln!(f, "{:<34} {}", "1 solver (quoted)", r.error1)?;
        for (name, st) in [
            ("2 transfer (CC norm)", &r.error2),
            ("3 model (uniform, equal weights)", &r.error3),
            ("4 total (CC norm)", &r.error4),
            ("5 boundary max ratio", &r.error5_max_ratio),
            ("5 boundary relative RMS", &r.error5_rel_rms),
        ] {
            writeln!(f, "{name:<34} {:>12.4e} {:>12.4e}", st.mean, st.max)?;
        }
        let worst = r.error4_worst_sample;
        if let Some(dir) = &self.plot_run_dir {
            writeln!(
                f,
                "worst sample (max error 4): {worst}; plot it with: poisson_plot -- {} {worst}",
                dir.display()
            )?;
        } else {
            writeln!(f, "worst sample (max error 4): {worst}")?;
        }
        writeln!(
            f,
            "correlation of ln(error 4) with ln ||u|| over the test split: {:.3}",
            r.error4_u_norm_log_correlation
        )?;

        writeln!(f, "\ncheck: {}", self.error2)?;
        writeln!(f, "check: {}", self.error3)?;

        let warranted = r.ablation_warranted;
        write!(
            f,
            "\nboundary verdict: mean relative boundary RMS / mean error 4 = {:.4e} / {:.4e} = {:.3} \
             ({} the {BOUNDARY_THRESHOLD} threshold): the hard-constraint ablation is {}",
            r.error5_rel_rms.mean,
            r.error4.mean,
            r.boundary_to_total,
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
fn rel_l2_equal_weights<D: Dimension>(a: ArrayView<f64, D>, b: ArrayView<f64, D>) -> f64 {
    let num: f64 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum();
    (num / den).sqrt()
}

/// Error 5 for one sample of rank r: `(max ratio, relative boundary RMS)` of
/// `u_hat: [n, …, n]` on the CGL grid, with Clenshaw–Curtis weights `w: [n]` on every
/// axis, against u through `u_sq`, ‖u‖² in that norm, and `max_u`, max over Ω of |u| (see
/// [`poisson_errors_nd`]).
fn boundary_errors_nd(
    u_hat: ArrayViewD<f64>,
    w: &Array1<f64>,
    u_sq: f64,
    max_u: f64,
) -> (f64, f64) {
    let r = u_hat.ndim();
    let last = u_hat.len_of(Axis(0)) - 1;
    let face_ws = vec![w.view(); r - 1];
    let (mut surface, mut max_face) = (0.0, 0.0_f64);
    for axis in 0..r {
        for index in [0, last] {
            let face = u_hat.index_axis(Axis(axis), index);
            max_face = face.iter().fold(max_face, |m, v| m.max(v.abs()));
            surface += cc_norm_sq_nd(face, &face_ws);
        }
    }
    // 2r faces of area 2^(r − 1) each, against the volume 2^r: 8 and 4 in 2D, 24 and 8 in 3D
    let cube = 2.0_f64.powi(r as i32);
    let faces = r as f64 * cube;
    (
        max_face / max_u,
        (surface / faces).sqrt() / (u_sq / cube).sqrt(),
    )
}

#[cfg(test)]
mod tests;

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
//! error is summarised over the samples with [`ErrorStats`](super::stats::ErrorStats).
//!
//! Error 1 (the solver) needs the `chebyshev` feature and is quoted, not recomputed (design
//! §12, decision 16). Everything here is ndarray and f64, built without `chebyshev`
//! (decision 15). Grids, transfers and norms follow CONVENTIONS §12.

use crate::neural_operators::chebyshev::{
    Transfers, cc_norm_sq, cc_rel_l2_error_weighted, clencurt,
};
use crate::neural_operators::runs::EpochMetrics;
use ndarray::{Array1, Array2, ArrayView2, ArrayView3, Axis};
use std::fmt;

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
    use crate::neural_operators::chebyshev::nodes_closed_form;
    use crate::neural_operators::losses::data_losses::{LpLoss, Reduction};
    use burn::prelude::*;
    use ndarray::Array3;

    /// `count` smooth fields on the n × n CGL grid that vanish on ∂Ω; sample k is scaled
    /// and shifted so the samples differ.
    fn vanishing_u(count: usize, n: usize) -> Array3<f64> {
        let x = nodes_closed_form(n);
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
        let x = nodes_closed_form(n);
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

//! Proper orthogonal decomposition (POD) of 2D snapshots in a weighted L² inner product
//! (design §13.4, Phase 6): the thin SVD of the centred, weighted snapshot matrix. Behind
//! `datagen`, as the SVD is RLST's (LAPACK `gesdd`). f64 throughout.
//!
//! The inner product of two fields on an a × b tensor grid is
//! ⟨u, v⟩ = Σ_{i,j} wx_i wy_j u[i, j] v[i, j]; with Clenshaw–Curtis weights on the CGL
//! grid (`chebyshev::clencurt`) it is the L²(Ω) inner product of design §3.4. Each grid
//! point is scaled by √(wx_i wy_j), so the Euclidean SVD of the scaled matrix is the POD in
//! that inner product, and σ_i² / (N − 1) estimate the covariance eigenvalues. The SVD is
//! taken of the snapshot matrix itself, not of the N × N Gram matrix, which would square
//! its dynamic range (docs/phase6/README.md).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3, Axis, s};
use rlst::dense::linalg::lapack::singular_value_decomposition::SvdMode;
use rlst::{DynArray, SingularValueDecomposition};

/// The POD of N snapshots on an a × b grid: the snapshot mean, every singular value of the
/// centred weighted snapshot matrix, and its leading left singular vectors (the modes, in
/// weighted coordinates).
#[derive(Debug, Clone)]
pub struct Pod {
    /// The mean of the snapshots, `[a, b]`.
    mean: Array2<f64>,
    /// √(wx_i wy_j), `[a, b]`.
    sqrt_w: Array2<f64>,
    /// σ_1 ≥ σ_2 ≥ …, `[min(a·b, N)]`.
    singular_values: Array1<f64>,
    /// The first `keep` left singular vectors as orthonormal columns, `[a·b, keep]`, rows
    /// in the row-major order of the grid (row i·b + j is point (i, j)).
    modes: Array2<f64>,
    /// N, the number of fitted snapshots.
    n_samples: usize,
}

impl Pod {
    /// Fits the POD of `snapshots: [N, a, b]` (`snapshots[k, i, j]` = sample k at
    /// (x_i, y_j)) in the inner product with weights `wx: [a]` and `wy: [b]`: subtracts the
    /// mean over the N samples, scales each grid point by √(wx_i wy_j), and takes the thin
    /// SVD of the `[a·b, N]` matrix with one column per sample. Keeps the first `keep`
    /// modes for [`reconstruction_errors`](Self::reconstruction_errors).
    ///
    /// # Panics
    /// If N < 2, `wx` has not a entries or `wy` not b, a weight is negative or not finite,
    /// `keep` exceeds min(a·b, N), or LAPACK's SVD fails.
    pub fn fit(
        snapshots: ArrayView3<f64>,
        wx: ArrayView1<f64>,
        wy: ArrayView1<f64>,
        keep: usize,
    ) -> Self {
        let (n, a, b) = snapshots.dim();
        assert!(n >= 2, "Pod::fit: needs at least 2 snapshots, got {n}");
        assert_eq!(wx.len(), a, "Pod::fit: wx must have a = {a} entries");
        assert_eq!(wy.len(), b, "Pod::fit: wy must have b = {b} entries");
        assert!(
            wx.iter()
                .chain(wy.iter())
                .all(|w| w.is_finite() && *w >= 0.0),
            "Pod::fit: weights must be finite and non-negative"
        );
        assert!(
            keep <= n.min(a * b),
            "Pod::fit: keep = {keep} exceeds min(a·b, N) = {}",
            n.min(a * b)
        );
        let mean = snapshots.mean_axis(Axis(0)).expect("N >= 2");
        let sqrt_w = Array2::from_shape_fn((a, b), |(i, j)| (wx[i] * wy[j]).sqrt());

        let mut m = DynArray::<f64, 2>::from_shape([a * b, n]);
        weighted_centred(snapshots, mean.view(), sqrt_w.view(), |p, k, v| {
            m[[p, k]] = v
        });
        let (sv, u, _) = m.svd(SvdMode::Compact).expect("Pod::fit: SVD failed");
        Self {
            mean,
            sqrt_w,
            singular_values: Array1::from_shape_fn(sv.shape()[0], |i| sv[[i]]),
            modes: Array2::from_shape_fn((a * b, keep), |(p, q)| u[[p, q]]),
            n_samples: n,
        }
    }

    /// The singular values σ_1 ≥ σ_2 ≥ … of the centred weighted snapshot matrix,
    /// `[min(a·b, N)]`.
    pub fn singular_values(&self) -> ArrayView1<'_, f64> {
        self.singular_values.view()
    }

    /// σ_i² / (N − 1), `[min(a·b, N)]`: the eigenvalues of the sample covariance in the
    /// weighted inner product, descending.
    pub fn eigenvalues(&self) -> Array1<f64> {
        let n = self.n_samples as f64;
        self.singular_values.mapv(|s| s * s / (n - 1.0))
    }

    /// The number of modes holding `fraction` of the sample's energy (variance about the
    /// mean): [`energy_rank_of`] the σ_i².
    ///
    /// # Panics
    /// If `fraction` is not in (0, 1].
    pub fn energy_rank(&self, fraction: f64) -> usize {
        energy_rank_of(&self.singular_values.mapv(|s| s * s).to_vec(), fraction)
    }

    /// For each r of `ranks`, per sample of `samples: [M, a, b]`, the relative error
    /// ‖v − v_r‖ / ‖v‖ in the weighted norm, where v is the sample centred with the fitted
    /// mean and v_r its projection on the first r modes; one `[M]` array per rank, in the
    /// order of `ranks`. 1 at r = 0, non-increasing in r. On the fitted snapshots
    /// Σ_k ‖v_k − v_{k,r}‖² = Σ_{i > r} σ_i²; on other samples it is the held-out error.
    /// NaN for a sample equal to the mean. Centres the samples and projects them on the
    /// modes once, for the largest rank; the residual is formed explicitly, not as
    /// ‖v‖² − ‖v_r‖², which would lose the small errors to cancellation.
    ///
    /// # Panics
    /// If the samples are not on the fitted a × b grid, or a rank exceeds the kept modes.
    pub fn reconstruction_errors(
        &self,
        samples: ArrayView3<f64>,
        ranks: &[usize],
    ) -> Vec<Array1<f64>> {
        let (m, a, b) = samples.dim();
        assert_eq!(
            (a, b),
            self.mean.dim(),
            "Pod::reconstruction_errors: samples must be on the fitted a × b grid"
        );
        let r_max = ranks.iter().copied().max().unwrap_or(0);
        assert!(
            r_max <= self.modes.ncols(),
            "Pod::reconstruction_errors: rank {r_max} exceeds the {} kept modes",
            self.modes.ncols()
        );
        let mut v = Array2::zeros((a * b, m));
        weighted_centred(samples, self.mean.view(), self.sqrt_w.view(), |p, k, x| {
            v[[p, k]] = x
        });
        let norm = |c: ArrayView1<f64>| c.dot(&c).sqrt();
        let v_norm: Vec<f64> = v.columns().into_iter().map(norm).collect();
        let coeffs = self.modes.slice(s![.., ..r_max]).t().dot(&v);
        ranks
            .iter()
            .map(|&r| {
                let res = &v
                    - &self
                        .modes
                        .slice(s![.., ..r])
                        .dot(&coeffs.slice(s![..r, ..]));
                res.columns()
                    .into_iter()
                    .zip(&v_norm)
                    .map(|(e, n)| norm(e) / n)
                    .collect()
            })
            .collect()
    }
}

/// The smallest r with Σ_{i ≤ r} e_i ≥ `fraction` · Σ_i e_i: the number of leading
/// entries of `energies` (descending, non-negative) holding that fraction of their sum.
/// Serves the empirical spectrum ([`Pod::energy_rank`]) and a closed-form one alike.
///
/// # Panics
/// If `fraction` is not in (0, 1].
pub fn energy_rank_of(energies: &[f64], fraction: f64) -> usize {
    assert!(
        fraction > 0.0 && fraction <= 1.0,
        "energy_rank_of: fraction must be in (0, 1], got {fraction}"
    );
    cumulative_energy(energies)
        .iter()
        .position(|c| *c >= fraction)
        // round-off in the sum at fraction = 1
        .map_or(energies.len(), |i| i + 1)
}

/// The cumulative energy fractions of `energies`: entry r − 1 is
/// Σ_{i ≤ r} e_i / Σ_i e_i, so the last entry is 1 up to round-off.
pub fn cumulative_energy(energies: &[f64]) -> Vec<f64> {
    let total: f64 = energies.iter().sum();
    let mut acc = 0.0;
    energies
        .iter()
        .map(|e| {
            acc += e;
            acc / total
        })
        .collect()
}

/// Calls `put(p, k, √w[i, j] · (samples[k, i, j] − mean[i, j]))` for each sample k and
/// grid point p = i·b + j (row-major), with `samples: [M, a, b]`, `mean` and `sqrt_w`
/// `[a, b]`: column k of the weighted, centred `[a·b, M]` snapshot matrix, written into
/// whichever store the caller holds.
fn weighted_centred(
    samples: ArrayView3<f64>,
    mean: ArrayView2<f64>,
    sqrt_w: ArrayView2<f64>,
    mut put: impl FnMut(usize, usize, f64),
) {
    let b = mean.ncols();
    for (k, x) in samples.outer_iter().enumerate() {
        for ((i, j), x) in x.indexed_iter() {
            put(i * b + j, k, sqrt_w[[i, j]] * (x - mean[[i, j]]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::chebyshev::{cc_norm_sq, clencurt, nodes};
    use ndarray::Array3;
    use std::f64::consts::PI;

    const A: usize = 9;
    const B: usize = 7;

    /// `count` samples m + Σ_q c_q g_q of three fixed smooth fields on the A × B CGL
    /// grid, with deterministic irregular coefficients offset by `seed`.
    fn rank3_family(count: usize, seed: usize) -> Array3<f64> {
        let (x, y) = (nodes(A), nodes(B));
        Array3::from_shape_fn((count, A, B), |(k, i, j)| {
            let (x, y) = (x[i], y[j]);
            // one frequency per field: phases of one sinusoid in k would span only 2 dims
            let c = |q: usize| ((k + seed) as f64 * (1.1 + 0.7 * q as f64) + q as f64).sin();
            1.0 + x + c(0) * (x * x + y) + c(1) * (PI * x).sin() * y.cos() + c(2) * (x * y).exp()
        })
    }

    fn fit_rank3(keep: usize) -> Pod {
        Pod::fit(
            rank3_family(12, 0).view(),
            clencurt(A).view(),
            clencurt(B).view(),
            keep,
        )
    }

    #[test]
    fn rank_three_family_has_three_modes() {
        let pod = fit_rank3(5);
        let sv = pod.singular_values();
        assert_eq!(sv.len(), 12);
        assert!(sv[2] > 1e-3 * sv[0], "{sv}");
        assert!(sv.iter().skip(3).all(|s| *s <= 1e-12 * sv[0]), "{sv}");
        assert_eq!(pod.energy_rank(0.999), 3);
        // fresh samples of the same family lie in the span of the fitted modes
        let err = pod.reconstruction_errors(rank3_family(5, 100).view(), &[3]);
        assert!(err[0].iter().all(|e| *e <= 1e-12), "{}", err[0]);
    }

    #[test]
    fn tail_energy_is_the_fitted_residual() {
        let snapshots = rank3_family(12, 0);
        let pod = fit_rank3(3);
        let sv = pod.singular_values();
        let (w_x, w_y) = (clencurt(A), clencurt(B));
        let mean = snapshots.mean_axis(Axis(0)).unwrap();
        let norm_sq: Vec<f64> = snapshots
            .outer_iter()
            .map(|v| cc_norm_sq((&v - &mean).view(), w_x.view(), w_y.view()))
            .collect();
        let errs = pod.reconstruction_errors(snapshots.view(), &[0, 1, 2, 3]);
        for (r, err) in errs.iter().enumerate() {
            let resid: f64 = err.iter().zip(&norm_sq).map(|(e, n)| e * e * n).sum();
            let tail: f64 = sv.iter().skip(r).map(|s| s * s).sum();
            assert!((resid - tail).abs() <= 1e-12 * sv[0] * sv[0], "r = {r}");
        }
    }

    #[test]
    fn reconstruction_error_starts_at_one_and_decreases() {
        let pod = fit_rank3(5);
        let samples = rank3_family(4, 50);
        let errs = pod.reconstruction_errors(samples.view(), &[0, 1, 2, 3, 4, 5]);
        assert!(
            errs[0].iter().all(|e| (e - 1.0).abs() <= 1e-15),
            "{}",
            errs[0]
        );
        for (r, pair) in errs.windows(2).enumerate() {
            assert!(
                pair[1].iter().zip(&pair[0]).all(|(e, l)| *e <= l + 1e-15),
                "r = {}",
                r + 1
            );
        }
    }

    #[test]
    fn cc_weights_give_the_l2_eigenvalues() {
        // two L²(Ω)-orthonormal sine modes (design §3.3: mean square ¼ on [−1, 1]²) with
        // orthogonal, zero-mean coefficients of sample variance 4 and 1 over N − 1 = 3
        let n = 33;
        let x = nodes(n);
        let phi = |k: f64, l: f64, i: usize, j: usize| {
            (k * PI * (x[i] + 1.0) / 2.0).sin() * (l * PI * (x[j] + 1.0) / 2.0).sin()
        };
        let c1 = [1.0, 1.0, -1.0, -1.0].map(|c| c * 3f64.sqrt());
        let c2 = [1.0, -1.0, 1.0, -1.0].map(|c| c * 3f64.sqrt() / 2.0);
        let snapshots = Array3::from_shape_fn((4, n, n), |(k, i, j)| {
            0.5 + c1[k] * phi(1.0, 1.0, i, j) + c2[k] * phi(2.0, 1.0, i, j)
        });
        let w = clencurt(n);
        let pod = Pod::fit(snapshots.view(), w.view(), w.view(), 2);
        let eig = pod.eigenvalues();
        assert!((eig[0] - 4.0).abs() <= 1e-12, "{eig}");
        assert!((eig[1] - 1.0).abs() <= 1e-12, "{eig}");
        // the trace is the summed sample variance in the CC norm
        let mean = snapshots.mean_axis(Axis(0)).unwrap();
        let var: f64 = snapshots
            .outer_iter()
            .map(|v| cc_norm_sq((&v - &mean).view(), w.view(), w.view()))
            .sum::<f64>()
            / 3.0;
        assert!(
            (eig.sum() - var).abs() <= 1e-12 * var,
            "{} vs {var}",
            eig.sum()
        );
    }

    #[test]
    #[should_panic(expected = "Pod::fit: wx must have a = 9 entries")]
    fn wrong_weight_length_panics() {
        Pod::fit(
            rank3_family(4, 0).view(),
            clencurt(B).view(),
            clencurt(B).view(),
            1,
        );
    }

    #[test]
    #[should_panic(expected = "Pod::fit: keep = 13 exceeds min(a·b, N) = 12")]
    fn keeping_too_many_modes_panics() {
        fit_rank3(13);
    }

    #[test]
    #[should_panic(expected = "Pod::reconstruction_errors: samples must be on the fitted")]
    fn wrong_sample_grid_panics() {
        fit_rank3(3).reconstruction_errors(Array3::zeros((2, A, A)).view(), &[1]);
    }

    #[test]
    #[should_panic(expected = "Pod::reconstruction_errors: rank 4 exceeds the 3 kept modes")]
    fn too_large_r_panics() {
        fit_rank3(3).reconstruction_errors(rank3_family(2, 0).view(), &[1, 4]);
    }

    #[test]
    fn energy_rank_of_reads_the_cumulative_fractions() {
        let e = [4.0, 3.0, 2.0, 1.0];
        assert_eq!(cumulative_energy(&e), [0.4, 0.7, 0.9, 1.0]);
        assert_eq!(energy_rank_of(&e, 0.4), 1);
        assert_eq!(energy_rank_of(&e, 0.5), 2);
        assert_eq!(energy_rank_of(&e, 0.9), 3);
        assert_eq!(energy_rank_of(&e, 1.0), 4);
    }

    #[test]
    #[should_panic(expected = "energy_rank_of: fraction must be in (0, 1], got 0")]
    fn energy_rank_of_rejects_zero_fraction() {
        energy_rank_of(&[1.0], 0.0);
    }
}

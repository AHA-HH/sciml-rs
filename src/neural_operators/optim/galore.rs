//! Tensor GaLore's gradient projector (George et al. 2025, arXiv 2501.02379; GaLore, Zhao et
//! al. 2024): `neuraloperator` 2.0.0's `TensorGaLoreProjector` for real tensors of any rank
//! (library CM3.6a; complex tensors are CM3.6b).
//!
//! A gradient G of shape `[I_0, .., I_{D−1}]` is projected into a low-rank space by the
//! factors Fₖ (`[I_k, r_k]`, orthonormal columns) of a Tucker decomposition of G:
//! `project` gives G ×ₖ Fₖᵀ for every mode k (shape `[r_0, .., r_{D−1}]`), and
//! `project_back` gives `scale` · (Y ×ₖ Fₖ). An optimiser keeps its moments in the low-rank
//! space, which is where GaLore saves memory.
//!
//! As 2.0.0 (`neuralop/training/tensor_galore_projector.py`), with tensorly 0.9.0's `tucker`:
//! - the factors are computed on the first `project` (its `iter` a multiple of
//!   `update_proj_gap`) and never again: 2.0.0 never resets them, so its `update_proj_gap`
//!   only gates the first call and its `warm_restart` and `tucker_n_iter_max` never take
//!   effect (they are not offered);
//! - the factors come from the higher-order orthogonal iteration: each Fₖ starts as the
//!   leading left singular vectors of G's mode-k unfolding, then each sweep recomputes every
//!   Fₖ from the unfolding of G multiplied by the other factors, transposed; up to 100
//!   sweeps, stopping from the third on when the relative reconstruction error
//!   √|‖G‖² − ‖core‖²| / ‖G‖ changes by less than 1e-4;
//! - each singular vector's sign is fixed as tensorly's `svd_flip`: the entry of largest
//!   magnitude (the first, on a tie) positive.
//!
//! The SVDs are Burn's `linalg::svd` (host code, in the tensor's precision), as in
//! [`soap`](super::soap). Nothing here imports from `models/` or `training/`; the wiring
//! into an optimiser is a later task (`docs/operators/tasks/CM3.6a.md`, "Wiring later").

use burn::prelude::*;

#[cfg(test)]
mod tests;

/// The QR sweeps allowed to Burn's `svd` per singular value (as [`soap`](super::soap)).
const SVD_SWEEPS: usize = 30;
/// tensorly's `tucker` defaults, which 2.0.0 does not override.
const HOOI_MAX_SWEEPS: usize = 100;
const HOOI_TOL: f64 = 1e-4;

/// The Tucker rank: 2.0.0's `rank` argument, validated as tensorly's `validate_tucker_rank`.
#[derive(Clone, Debug, PartialEq)]
pub enum GaLoreRank {
    /// A fraction of the parameters to keep (2.0.0's float): ranks rₖ = max(round(Iₖ c), 1),
    /// rounding half to even, with c the root in [0, max(fraction, 1)] of
    /// N cᴰ + (Σₖ Iₖ²) c = fraction · N, N = ∏ₖ Iₖ (the core and factors' parameters).
    Fraction(f64),
    /// One rank for every mode (2.0.0's int).
    Uniform(usize),
    /// One rank per mode (2.0.0's list).
    PerMode(Vec<usize>),
}

impl GaLoreRank {
    /// The rank of each mode of a tensor of shape `shape`.
    ///
    /// # Panics
    /// If a fraction is not positive and finite, or a `PerMode` list's length is not the
    /// tensor's rank.
    pub fn ranks(&self, shape: &[usize]) -> Vec<usize> {
        match self {
            GaLoreRank::Uniform(r) => vec![*r; shape.len()],
            GaLoreRank::PerMode(r) => {
                assert_eq!(r.len(), shape.len(), "one rank per mode");
                r.clone()
            }
            GaLoreRank::Fraction(fraction) => {
                assert!(
                    fraction.is_finite() && *fraction > 0.0,
                    "the fraction {fraction} must be positive"
                );
                let n: f64 = shape.iter().map(|&s| s as f64).product();
                let squares: f64 = shape.iter().map(|&s| (s * s) as f64).sum();
                let d = shape.len() as i32;
                let excess = |c: f64| n * c.powi(d) + squares * c - fraction * n;
                // tensorly's brentq bracket; excess(0) < 0 ≤ excess(hi), excess increasing.
                let (mut lo, mut hi) = (0.0, fraction.max(1.0));
                for _ in 0..200 {
                    let mid = 0.5 * (lo + hi);
                    if excess(mid) < 0.0 {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                let c = 0.5 * (lo + hi);
                shape
                    .iter()
                    .map(|&s| ((s as f64 * c).round_ties_even() as usize).max(1))
                    .collect()
            }
        }
    }
}

/// The projector's settings: 2.0.0's `TensorGaLoreProjector` arguments that take effect.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorGaLoreConfig {
    /// The Tucker rank of the projected gradient.
    pub rank: GaLoreRank,
    /// The first `project` must have an `iter` that is a multiple of this (2.0.0's default
    /// 200; see the module documentation).
    pub update_proj_gap: usize,
    /// The factor on `project_back`'s output (2.0.0's default 1.0).
    pub scale: f64,
}

impl TensorGaLoreConfig {
    /// `rank` with 2.0.0's defaults: `update_proj_gap` 200, `scale` 1.0.
    pub fn new(rank: GaLoreRank) -> Self {
        Self {
            rank,
            update_proj_gap: 200,
            scale: 1.0,
        }
    }

    /// A projector for rank-`D` gradients, without factors until its first `project`.
    pub fn init<const D: usize>(&self) -> TensorGaLoreProjector<D> {
        TensorGaLoreProjector {
            config: self.clone(),
            factors: None,
        }
    }
}

/// 2.0.0's `TensorGaLoreProjector` for one rank-`D` parameter's gradients.
#[derive(Clone, Debug)]
pub struct TensorGaLoreProjector<const D: usize> {
    config: TensorGaLoreConfig,
    factors: Option<Vec<Tensor<2>>>,
}

impl<const D: usize> TensorGaLoreProjector<D> {
    /// `grad` (`[I_0, .., I_{D−1}]`) in the low-rank space, `[r_0, .., r_{D−1}]`: G ×ₖ Fₖᵀ
    /// for every mode k. The first call computes the factors from `grad` by
    /// [`tucker_factors`]; later calls reuse them, whatever `iter` (as 2.0.0).
    ///
    /// # Panics
    /// On the first call, if `iter` is not a multiple of `update_proj_gap` (2.0.0 has no
    /// factors to apply then), or as [`tucker_factors`].
    pub fn project(&mut self, grad: Tensor<D>, iter: usize) -> Tensor<D> {
        let factors = self.factors.get_or_insert_with(|| {
            assert!(
                iter.is_multiple_of(self.config.update_proj_gap),
                "the first project needs iter ({iter}) a multiple of update_proj_gap ({})",
                self.config.update_proj_gap
            );
            let ranks = self.config.rank.ranks(&grad.dims());
            tucker_factors(grad.clone(), &ranks)
        });
        multi_mode_dot(grad, factors, true)
    }

    /// `low` (`[r_0, .., r_{D−1}]`) back in the full space, `[I_0, .., I_{D−1}]`:
    /// `scale` · (Y ×ₖ Fₖ).
    ///
    /// # Panics
    /// Before the first [`project`](Self::project), or if `low`'s shape is not the ranks.
    pub fn project_back(&self, low: Tensor<D>) -> Tensor<D> {
        let factors = self.factors.as_ref().expect("project_back before project");
        multi_mode_dot(low, factors, false).mul_scalar(self.config.scale)
    }

    /// The factors Fₖ, `[I_k, r_k]` each, once [`project`](Self::project) has made them.
    pub fn factors(&self) -> Option<&[Tensor<2>]> {
        self.factors.as_deref()
    }
}

/// The factors of a rank-`ranks` Tucker decomposition of `x` by tensorly's higher-order
/// orthogonal iteration with SVD initialisation (see the module documentation): one
/// `[I_k, r_k]` matrix with orthonormal columns per mode.
///
/// # Panics
/// If `ranks` does not have one entry per mode, a rank is 0 or above min(Iₖ, ∏_{j≠k} Iⱼ)
/// (tensorly would fall back to a full-matrix SVD there), `x` requires gradients, or an SVD
/// does not converge.
pub fn tucker_factors<const D: usize>(x: Tensor<D>, ranks: &[usize]) -> Vec<Tensor<2>> {
    assert_eq!(ranks.len(), D, "one rank per mode");
    let mut factors: Vec<Tensor<2>> = (0..D)
        .map(|k| leading_vectors(unfold(x.clone(), k), ranks[k]))
        .collect();
    let norm_sq = sum_sq(x.clone());
    let mut previous = None;
    for sweep in 0..HOOI_MAX_SWEEPS {
        for k in 0..D {
            let partial = mode_products(x.clone(), &factors, Some(k), true);
            factors[k] = leading_vectors(unfold(partial, k), ranks[k]);
        }
        let core = multi_mode_dot(x.clone(), &factors, true);
        let error = (norm_sq - sum_sq(core)).abs().sqrt() / norm_sq.sqrt();
        if sweep > 1 && previous.is_some_and(|p: f64| (p - error).abs() < HOOI_TOL) {
            break;
        }
        previous = Some(error);
    }
    factors
}

/// `x` multiplied along every mode k by its factor: by Fₖᵀ (`transpose`, `[I_k, r_k]` to
/// `r_k`) or by Fₖ (`r_k` to `I_k`); tensorly's `multi_mode_dot`.
///
/// # Panics
/// If `factors` does not have one matrix per mode or a matrix does not fit its mode.
pub fn multi_mode_dot<const D: usize>(
    x: Tensor<D>,
    factors: &[Tensor<2>],
    transpose: bool,
) -> Tensor<D> {
    mode_products(x, factors, None, transpose)
}

/// [`multi_mode_dot`] leaving mode `skip` as it is. Each pass contracts the leading
/// dimension and moves the result last, so D passes restore the order.
fn mode_products<const D: usize>(
    mut x: Tensor<D>,
    factors: &[Tensor<2>],
    skip: Option<usize>,
    transpose: bool,
) -> Tensor<D> {
    assert_eq!(factors.len(), D, "one factor per mode");
    for (k, f) in factors.iter().enumerate() {
        let mut dims = x.dims();
        let rows: Tensor<2> = x.reshape([dims[0] as i32, -1]).transpose();
        let rows = match () {
            _ if skip == Some(k) => rows,
            _ if transpose => rows.matmul(f.clone()),
            _ => rows.matmul(f.clone().transpose()),
        };
        dims[0] = rows.dims()[1];
        dims.rotate_left(1);
        x = rows.reshape(dims);
    }
    x
}

/// The mode-k unfolding of `x`, `[I_k, ∏_{j≠k} I_j]`. Its columns are in another order than
/// tensorly's, which leaves the left singular vectors unchanged.
fn unfold<const D: usize>(x: Tensor<D>, k: usize) -> Tensor<2> {
    let rows = x.dims()[k] as i32;
    x.swap_dims(0, k).reshape([rows, -1])
}

/// The `r` leading left singular vectors of `m`, signs fixed as tensorly's `svd_flip`.
fn leading_vectors(m: Tensor<2>, r: usize) -> Tensor<2> {
    let [rows, cols] = m.dims();
    assert!(
        (1..=rows.min(cols)).contains(&r),
        "rank {r} outside 1..={} for a [{rows}, {cols}] unfolding",
        rows.min(cols)
    );
    let (u, _, _) = burn::linalg::svd::<2, 1>(m, SVD_SWEEPS);
    let u = u.slice([0..rows, 0..r]);
    let largest = u.clone().abs().argmax(0);
    let signs = u.clone().gather(0, largest).sign();
    u.mul(signs)
}

/// ‖x‖², on the host in f64.
fn sum_sq<const D: usize>(x: Tensor<D>) -> f64 {
    x.square().sum().into_scalar::<f64>()
}

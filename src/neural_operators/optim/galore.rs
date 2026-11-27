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

use super::SVD_SWEEPS;

#[cfg(test)]
mod tests;

/// tensorly's `tucker` defaults, which 2.0.0 does not override.
const HOOI_MAX_SWEEPS: usize = 100;
const HOOI_TOL: f64 = 1e-4;
/// Bisection steps for the fraction rule's root: enough to reach f64 resolution on its
/// bracket, which is at most [0, max(fraction, 1)].
const ROOT_BISECTIONS: usize = 200;

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
    /// If a fraction is not positive and finite, or a `PerMode` list's length is not `D`.
    pub fn ranks<const D: usize>(&self, shape: [usize; D]) -> [usize; D] {
        match self {
            GaLoreRank::Uniform(r) => [*r; D],
            GaLoreRank::PerMode(r) => r.as_slice().try_into().expect("one rank per mode"),
            GaLoreRank::Fraction(fraction) => {
                assert!(
                    fraction.is_finite() && *fraction > 0.0,
                    "the fraction {fraction} must be positive"
                );
                let n: f64 = shape.iter().map(|&s| s as f64).product();
                let squares: f64 = shape.iter().map(|&s| (s * s) as f64).sum();
                let d = D as i32;
                let excess = |c: f64| n * c.powi(d) + squares * c - fraction * n;
                // tensorly's brentq bracket; excess(0) < 0 ≤ excess(hi), excess increasing.
                let (mut lo, mut hi) = (0.0, fraction.max(1.0));
                for _ in 0..ROOT_BISECTIONS {
                    let mid = 0.5 * (lo + hi);
                    if excess(mid) < 0.0 {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                let c = 0.5 * (lo + hi);
                shape.map(|s| ((s as f64 * c).round_ties_even() as usize).max(1))
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
    factors: Option<[Tensor<2>; D]>,
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
            let ranks = self.config.rank.ranks(grad.dims());
            tucker_factors(grad.clone(), ranks)
        });
        to_low_rank(grad, factors)
    }

    /// `low` (`[r_0, .., r_{D−1}]`) back in the full space, `[I_0, .., I_{D−1}]`:
    /// `scale` · (Y ×ₖ Fₖ).
    ///
    /// # Panics
    /// Before the first [`project`](Self::project), or if `low`'s shape is not the ranks.
    pub fn project_back(&self, low: Tensor<D>) -> Tensor<D> {
        let factors = self.factors.as_ref().expect("project_back before project");
        from_low_rank(low, factors).mul_scalar(self.config.scale)
    }

    /// The factors Fₖ, `[I_k, r_k]` each, once [`project`](Self::project) has made them.
    pub fn factors(&self) -> Option<&[Tensor<2>; D]> {
        self.factors.as_ref()
    }
}

/// The factors of a rank-`ranks` Tucker decomposition of `x` by tensorly's higher-order
/// orthogonal iteration with SVD initialisation (see the module documentation): one
/// `[I_k, r_k]` matrix with orthonormal columns per mode.
///
/// An all-zero `x` makes the error NaN, so every sweep runs (as tensorly).
///
/// # Panics
/// If a rank is 0 or above min(Iₖ, ∏_{j≠k} Iⱼ) (tensorly would fall back to a full-matrix
/// SVD there), `x` requires gradients, or an SVD does not converge.
pub fn tucker_factors<const D: usize>(x: Tensor<D>, ranks: [usize; D]) -> [Tensor<2>; D] {
    let mut factors: [Tensor<2>; D] =
        std::array::from_fn(|k| leading_vectors(unfold(x.clone(), k), ranks[k]));
    let norm_sq = sum_sq(x.clone());
    let mut previous = None;
    for sweep in 0..HOOI_MAX_SWEEPS {
        for k in 0..D {
            let partial = mode_products(x.clone(), &factors, Some(k), true);
            factors[k] = leading_vectors(unfold(partial, k), ranks[k]);
        }
        let core = to_low_rank(x.clone(), &factors);
        let error = (norm_sq - sum_sq(core)).abs().sqrt() / norm_sq.sqrt();
        if sweep > 1 && previous.is_some_and(|p: f64| (p - error).abs() < HOOI_TOL) {
            break;
        }
        previous = Some(error);
    }
    factors
}

/// `x` (`[I_0, .., I_{D−1}]`) multiplied along every mode k by Fₖᵀ, giving
/// `[r_0, .., r_{D−1}]`: tensorly's `multi_mode_dot(x, factors, transpose=True)`.
fn to_low_rank<const D: usize>(x: Tensor<D>, factors: &[Tensor<2>; D]) -> Tensor<D> {
    mode_products(x, factors, None, true)
}

/// `y` (`[r_0, .., r_{D−1}]`) multiplied along every mode k by Fₖ, giving
/// `[I_0, .., I_{D−1}]`: tensorly's `multi_mode_dot(y, factors)`.
fn from_low_rank<const D: usize>(y: Tensor<D>, factors: &[Tensor<2>; D]) -> Tensor<D> {
    mode_products(y, factors, None, false)
}

/// `x` multiplied along every mode but `skip` by Fₖᵀ (`transpose`) or Fₖ. Each pass
/// contracts the leading dimension and moves the result last, so D passes restore the order.
fn mode_products<const D: usize>(
    mut x: Tensor<D>,
    factors: &[Tensor<2>; D],
    skip: Option<usize>,
    transpose: bool,
) -> Tensor<D> {
    for (k, f) in factors.iter().enumerate() {
        let mut dims = x.dims();
        let mut rows: Tensor<2> = x.reshape([dims[0] as i32, -1]).transpose();
        if skip != Some(k) {
            let f = if transpose {
                f.clone()
            } else {
                f.clone().transpose()
            };
            rows = rows.matmul(f);
        }
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

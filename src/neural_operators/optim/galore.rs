//! Tensor GaLore's gradient projector (George et al. 2025, arXiv 2501.02379; GaLore, Zhao et
//! al. 2024): `neuraloperator` 2.0.0's `TensorGaLoreProjector` for real tensors of any rank
//! (library CM3.6a) and complex ones, [`ComplexTensor`], 2.0.0's target, the spectral
//! weights (CM3.6b).
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
//!   effect (they are not offered). GaLore's intended refresh, the factors recomputed at
//!   every `iter` that is a multiple of `update_proj_gap`, is opt-in here:
//!   [`TensorGaLoreProjector::with_refresh`] (CM3.6c; off by default, as 2.0.0);
//! - the factors come from the higher-order orthogonal iteration: each Fₖ starts as the
//!   leading left singular vectors of G's mode-k unfolding, then each sweep recomputes every
//!   Fₖ from the unfolding of G multiplied by the other factors, transposed; up to 100
//!   sweeps, stopping from the third on when the relative reconstruction error
//!   √|‖G‖² − ‖core‖²| / ‖G‖ changes by less than 1e-4;
//! - each singular vector's sign is fixed as tensorly's `svd_flip`: the entry of largest
//!   magnitude (the first, on a tie) positive.
//!
//! For a complex tensor, as tensorly: the products with Fₖᵀ use the conjugate transpose Fₖᴴ,
//! the norms are √Σ|x|², and `svd_flip` multiplies each column by sgn(u_max) = u_max / |u_max|,
//! u_max its entry of largest modulus. That rule doubles the column's phase rather than
//! removing it, so the result depends on the phase the SVD returns, which torch's LAPACK
//! `gesdd` fixes by its bidiagonalisation: for an m × n matrix, the first row of V is real
//! when m ≥ n, the first row of U when m < n (up to a sign the rule cancels). Each column is
//! rotated to that convention before the rule, which reproduces 2.0.0. The complex SVD is a
//! host one-sided Jacobi in f64 (Burn has none, and its real `svd` of the embedding
//! [[B, −C], [C, B]] stalls in f64 on the embedding's paired singular values). 2.0.0 casts
//! complex gradients to `cfloat` (and then fails on `cdouble`); here f64 stays f64.
//!
//! Where a sweep's unfolding is tall (Iₖ > ∏_{j≠k} rⱼ), LAPACK's reference entry is a
//! column of the other factors' product, so each sweep turns the phases again: they do not
//! converge, and 2.0.0's factors there change by O(1) under a 1e-7 change of the gradient.
//! Only their subspaces, and so `project_back ∘ project`, are well defined there (the
//! parity tests compare those; `tools/parity/README.md`).
//!
//! The real SVDs are Burn's `linalg::svd` (host code, in the tensor's precision), as in
//! [`soap`](super::soap); the complex ones are `complex_svd`'s (host, f64). Nothing here imports from `models/` or `training/`; the wiring
//! into an optimiser is a later task (`docs/operators/tasks/CM3.6a.md`, "Wiring later").

use burn::prelude::*;

use super::SVD_SWEEPS;
use complex_svd::complex_leading_vectors;

mod complex_svd;
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

/// A complex tensor by its real and imaginary parts, of the same shape: a complex gradient
/// for the projector (2.0.0's `cfloat` spectral weight), or a complex factor.
#[derive(Clone, Debug)]
pub struct ComplexTensor<const D: usize> {
    /// The real part.
    pub re: Tensor<D>,
    /// The imaginary part.
    pub im: Tensor<D>,
}

impl<const D: usize> ComplexTensor<D> {
    /// `re` + i `im`.
    ///
    /// # Panics
    /// If the shapes differ.
    pub fn new(re: Tensor<D>, im: Tensor<D>) -> Self {
        assert_eq!(re.dims(), im.dims(), "real and imaginary parts' shapes");
        Self { re, im }
    }
}

impl<const D: usize> std::ops::Sub for ComplexTensor<D> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }
}

mod sealed {
    pub trait Sealed {}
    impl<const D: usize> Sealed for burn::prelude::Tensor<D> {}
    impl<const D: usize> Sealed for super::ComplexTensor<D> {}
}

/// A matrix in the projector's field: `Tensor<2>` or `ComplexTensor<2>` (a factor). Sealed.
pub trait GaLoreMatrix: sealed::Sealed + Clone + std::fmt::Debug {
    /// `self` (`[m, k]`) times `other` (`[k, n]`), `[m, n]`.
    fn matmul(self, other: Self) -> Self;
    /// The transpose (not conjugated), `[n, m]`.
    fn transpose(self) -> Self;
    /// The complex conjugate (the identity when real).
    fn conj(self) -> Self;
    /// The shape `[m, n]`.
    fn shape(&self) -> [usize; 2];
}

impl GaLoreMatrix for Tensor<2> {
    fn matmul(self, other: Self) -> Self {
        Tensor::matmul(self, other)
    }

    fn transpose(self) -> Self {
        Tensor::transpose(self)
    }

    fn conj(self) -> Self {
        self
    }

    fn shape(&self) -> [usize; 2] {
        self.dims()
    }
}

impl GaLoreMatrix for ComplexTensor<2> {
    fn matmul(self, other: Self) -> Self {
        let (a, b, f, g) = (self.re, self.im, other.re, other.im);
        Self::new(
            a.clone().matmul(f.clone()) - b.clone().matmul(g.clone()),
            a.matmul(g) + b.matmul(f),
        )
    }

    fn transpose(self) -> Self {
        Self::new(self.re.transpose(), self.im.transpose())
    }

    fn conj(self) -> Self {
        Self::new(self.re, self.im.neg())
    }

    fn shape(&self) -> [usize; 2] {
        self.re.dims()
    }
}

/// A gradient the projector takes: a real `Tensor<D>`, or a [`ComplexTensor<D>`]. Sealed;
/// its methods are the Tucker decomposition's operations in that field.
pub trait GaLoreTensor<const D: usize>: sealed::Sealed + Clone + std::fmt::Debug {
    /// A factor's type: `Tensor<2>`, or `ComplexTensor<2>`.
    type Factor: GaLoreMatrix;

    /// The shape `[I_0, .., I_{D−1}]`.
    fn shape(&self) -> [usize; D];

    /// The mode-k unfolding, `[I_k, ∏_{j≠k} I_j]`. Its columns are in another order than
    /// tensorly's, which leaves the left singular vectors unchanged.
    fn unfolding(self, k: usize) -> Self::Factor;

    /// `m` (`[I_0 .. I_{D−1}]` entries in row-major order) reshaped to `dims`.
    fn fold(m: Self::Factor, dims: [usize; D]) -> Self;

    /// The `r` leading left singular vectors of the mode-`k` unfolding, `[I_k, r]`, as
    /// tensorly's `svd_flip` (and, complex, LAPACK's phase) leaves them.
    fn leading_factor(self, k: usize, r: usize) -> Self::Factor;

    /// ‖self‖² = Σ|x|², on the host in f64.
    fn norm_sq(self) -> f64;

    /// `self` times `scale`.
    fn scaled(self, scale: f64) -> Self;

    /// `self` (`[I_0, .., I_{D−1}]`) multiplied along every mode k by Fₖᴴ (Fₖᵀ when real),
    /// giving `[r_0, .., r_{D−1}]`: tensorly's `multi_mode_dot(x, factors, transpose=True)`.
    fn to_low_rank(self, factors: &[Self::Factor; D]) -> Self {
        mode_products(self, factors, None, true)
    }

    /// `self` (`[r_0, .., r_{D−1}]`) multiplied along every mode k by Fₖ, giving
    /// `[I_0, .., I_{D−1}]`: tensorly's `multi_mode_dot(y, factors)`.
    fn to_full_rank(self, factors: &[Self::Factor; D]) -> Self {
        mode_products(self, factors, None, false)
    }
}

impl<const D: usize> GaLoreTensor<D> for Tensor<D> {
    type Factor = Tensor<2>;

    fn shape(&self) -> [usize; D] {
        self.dims()
    }

    fn unfolding(self, k: usize) -> Tensor<2> {
        let rows = self.dims()[k] as i32;
        self.swap_dims(0, k).reshape([rows, -1])
    }

    fn fold(m: Tensor<2>, dims: [usize; D]) -> Self {
        m.reshape(dims)
    }

    fn leading_factor(self, k: usize, r: usize) -> Tensor<2> {
        leading_vectors(self.unfolding(k), r)
    }

    fn norm_sq(self) -> f64 {
        self.square().sum().into_scalar::<f64>()
    }

    fn scaled(self, scale: f64) -> Self {
        self.mul_scalar(scale)
    }
}

impl<const D: usize> GaLoreTensor<D> for ComplexTensor<D> {
    type Factor = ComplexTensor<2>;

    fn shape(&self) -> [usize; D] {
        self.re.dims()
    }

    fn unfolding(self, k: usize) -> ComplexTensor<2> {
        ComplexTensor::new(self.re.unfolding(k), self.im.unfolding(k))
    }

    fn fold(m: ComplexTensor<2>, dims: [usize; D]) -> Self {
        Self::new(m.re.reshape(dims), m.im.reshape(dims))
    }

    fn leading_factor(self, k: usize, r: usize) -> ComplexTensor<2> {
        complex_leading_vectors(self.unfolding(k), r)
    }

    fn norm_sq(self) -> f64 {
        self.re.norm_sq() + self.im.norm_sq()
    }

    fn scaled(self, scale: f64) -> Self {
        Self::new(self.re.mul_scalar(scale), self.im.mul_scalar(scale))
    }
}

/// The projector's settings: 2.0.0's `TensorGaLoreProjector` arguments that take effect.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorGaLoreConfig {
    /// The Tucker rank of the projected gradient.
    pub rank: GaLoreRank,
    /// The first `project` must have an `iter` that is a multiple of this (2.0.0's default
    /// 200; see the module documentation); with
    /// [`with_refresh`](TensorGaLoreProjector::with_refresh), the refresh period too.
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

    /// A projector for rank-`D` gradients of type `T` (`Tensor<D>` or [`ComplexTensor<D>`]),
    /// without factors until its first `project`.
    pub fn init<const D: usize, T: GaLoreTensor<D>>(&self) -> TensorGaLoreProjector<D, T> {
        TensorGaLoreProjector {
            config: self.clone(),
            factors: None,
            refresh: false,
        }
    }
}

/// 2.0.0's `TensorGaLoreProjector` for one rank-`D` parameter's gradients, real (`T` =
/// `Tensor<D>`, the default) or complex (`T` = [`ComplexTensor<D>`]).
#[derive(Clone, Debug)]
pub struct TensorGaLoreProjector<const D: usize, T: GaLoreTensor<D> = Tensor<D>> {
    config: TensorGaLoreConfig,
    factors: Option<[T::Factor; D]>,
    refresh: bool,
}

impl<const D: usize, T: GaLoreTensor<D>> TensorGaLoreProjector<D, T> {
    /// With `refresh`, [`project`](Self::project) recomputes the factors from its gradient at
    /// every `iter` that is a multiple of `update_proj_gap` (GaLore's schedule, a fresh
    /// [`tucker_factors`] each time); without it (the default, 2.0.0's behaviour) only the
    /// first call computes them.
    pub fn with_refresh(mut self, refresh: bool) -> Self {
        self.refresh = refresh;
        self
    }

    /// `grad` (`[I_0, .., I_{D−1}]`) in the low-rank space, `[r_0, .., r_{D−1}]`: G ×ₖ Fₖᴴ
    /// for every mode k (Fₖᵀ when real). The first call computes the factors from `grad` by
    /// [`tucker_factors`]; later calls reuse them, whatever `iter` (as 2.0.0), unless
    /// [`with_refresh`](Self::with_refresh) is on and `iter` is a multiple of
    /// `update_proj_gap`, when they are computed again from `grad`.
    ///
    /// # Panics
    /// On the first call, if `update_proj_gap` is 0 or `iter` is not a multiple of it
    /// (2.0.0 has no factors to apply then), or as [`tucker_factors`].
    pub fn project(&mut self, grad: T, iter: usize) -> T {
        // A zero gap is caught below: the refresh drops the factors, so the first call's check runs.
        if self.refresh && iter.is_multiple_of(self.config.update_proj_gap) {
            self.factors = None;
        }
        let factors = self.factors.get_or_insert_with(|| {
            assert!(
                self.config.update_proj_gap > 0,
                "update_proj_gap must be positive"
            );
            assert!(
                iter.is_multiple_of(self.config.update_proj_gap),
                "the first project needs iter ({iter}) a multiple of update_proj_gap ({})",
                self.config.update_proj_gap
            );
            let ranks = self.config.rank.ranks(grad.shape());
            tucker_factors(grad.clone(), ranks)
        });
        grad.to_low_rank(factors)
    }

    /// `low` (`[r_0, .., r_{D−1}]`) back in the full space, `[I_0, .., I_{D−1}]`:
    /// `scale` · (Y ×ₖ Fₖ).
    ///
    /// # Panics
    /// Before the first [`project`](Self::project), or if `low`'s shape is not the ranks.
    pub fn project_back(&self, low: T) -> T {
        let factors = self.factors.as_ref().expect("project_back before project");
        low.to_full_rank(factors).scaled(self.config.scale)
    }

    /// The factors Fₖ, `[I_k, r_k]` each, once [`project`](Self::project) has made them.
    pub fn factors(&self) -> Option<&[T::Factor; D]> {
        self.factors.as_ref()
    }
}

/// The factors of a rank-`ranks` Tucker decomposition of `x` by tensorly's higher-order
/// orthogonal iteration with SVD initialisation (see the module documentation): one
/// `[I_k, r_k]` matrix with orthonormal columns per mode, real or complex as `x`.
///
/// An all-zero `x` makes the error NaN, so every sweep runs (as tensorly).
///
/// # Panics
/// If a rank rₖ is 0, above Iₖ, or above ∏_{j≠k} rⱼ (the columns of a sweep's unfolding;
/// tensorly falls back to a full-matrix SVD there, e.g. ranks [3, 1]), `x` requires
/// gradients, an SVD does not converge, or (complex) a singular vector's phase reference
/// entry is zero.
pub fn tucker_factors<const D: usize, T: GaLoreTensor<D>>(
    x: T,
    ranks: [usize; D],
) -> [T::Factor; D] {
    let dims = x.shape();
    for k in 0..D {
        let others: usize = (0..D).filter(|&j| j != k).map(|j| ranks[j]).product();
        assert!(
            ranks[k] >= 1 && ranks[k] <= dims[k] && ranks[k] <= others,
            "rank {} of mode {k} outside 1..=min({}, {others}) (ranks {ranks:?}, shape {dims:?})",
            ranks[k],
            dims[k]
        );
    }
    let mut factors: [T::Factor; D] =
        std::array::from_fn(|k| x.clone().leading_factor(k, ranks[k]));
    let norm_sq = x.clone().norm_sq();
    let mut previous = None;
    for sweep in 0..HOOI_MAX_SWEEPS {
        for k in 0..D {
            let partial = mode_products(x.clone(), &factors, Some(k), true);
            factors[k] = partial.leading_factor(k, ranks[k]);
        }
        let core = x.clone().to_low_rank(&factors);
        let error = (norm_sq - core.norm_sq()).abs().sqrt() / norm_sq.sqrt();
        if sweep > 1 && previous.is_some_and(|p: f64| (p - error).abs() < HOOI_TOL) {
            break;
        }
        previous = Some(error);
    }
    factors
}

/// `x` multiplied along every mode but `skip` by Fₖᴴ (`conj_transpose`; Fₖᵀ when real) or
/// Fₖ. Each pass contracts the leading dimension and moves the result last, so D passes
/// restore the order.
fn mode_products<const D: usize, T: GaLoreTensor<D>>(
    mut x: T,
    factors: &[T::Factor; D],
    skip: Option<usize>,
    conj_transpose: bool,
) -> T {
    for (k, f) in factors.iter().enumerate() {
        let mut dims = x.shape();
        let mut rows = x.unfolding(0).transpose();
        if skip != Some(k) {
            let f = if conj_transpose {
                f.clone().conj()
            } else {
                f.clone().transpose()
            };
            rows = rows.matmul(f);
        }
        dims[0] = rows.shape()[1];
        dims.rotate_left(1);
        x = T::fold(rows, dims);
    }
    x
}

/// The `r` leading left singular vectors of `m`, signs fixed as tensorly's `svd_flip`;
/// `r` ≤ min(rows, columns), as [`tucker_factors`] checks.
fn leading_vectors(m: Tensor<2>, r: usize) -> Tensor<2> {
    let [rows, _] = m.dims();
    let (u, _, _) = burn::linalg::svd::<2, 1>(m, SVD_SWEEPS);
    let u = u.slice([0..rows, 0..r]);
    let largest = u.clone().abs().argmax(0);
    let signs = u.clone().gather(0, largest).sign();
    u.mul(signs)
}

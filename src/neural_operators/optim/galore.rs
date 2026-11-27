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
//!   effect (they are not offered);
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

mod sealed {
    pub trait Sealed {}
    impl<const D: usize> Sealed for burn::prelude::Tensor<D> {}
    impl<const D: usize> Sealed for super::ComplexTensor<D> {}
}

/// A gradient the projector takes: a real `Tensor<D>`, or a [`ComplexTensor<D>`]. Sealed;
/// its methods are the Tucker decomposition's operations in that field.
pub trait GaLoreTensor<const D: usize>: sealed::Sealed + Clone + std::fmt::Debug {
    /// A factor's type: `Tensor<2>`, or `ComplexTensor<2>`.
    type Factor: Clone + std::fmt::Debug;

    /// The shape `[I_0, .., I_{D−1}]`.
    fn shape(&self) -> [usize; D];

    /// `self` multiplied along every mode k but `skip` by Fₖᴴ (`conj_transpose`) or Fₖ.
    fn mode_products(
        self,
        factors: &[Self::Factor; D],
        skip: Option<usize>,
        conj_transpose: bool,
    ) -> Self;

    /// The `r` leading left singular vectors of the mode-`k` unfolding, `[I_k, r]`, as
    /// tensorly's `svd_flip` (and, complex, LAPACK's phase) leaves them.
    fn leading_factor(self, k: usize, r: usize) -> Self::Factor;

    /// ‖self‖² = Σ|x|², on the host in f64.
    fn norm_sq(self) -> f64;

    /// `self` times `scale`.
    fn scaled(self, scale: f64) -> Self;
}

impl<const D: usize> GaLoreTensor<D> for Tensor<D> {
    type Factor = Tensor<2>;

    fn shape(&self) -> [usize; D] {
        self.dims()
    }

    fn mode_products(
        self,
        factors: &[Tensor<2>; D],
        skip: Option<usize>,
        conj_transpose: bool,
    ) -> Self {
        mode_products(self, factors, skip, conj_transpose)
    }

    fn leading_factor(self, k: usize, r: usize) -> Tensor<2> {
        leading_vectors(unfold(self, k), r)
    }

    fn norm_sq(self) -> f64 {
        sum_sq(self)
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

    fn mode_products(
        self,
        factors: &[ComplexTensor<2>; D],
        skip: Option<usize>,
        conj_transpose: bool,
    ) -> Self {
        complex_mode_products(self, factors, skip, conj_transpose)
    }

    fn leading_factor(self, k: usize, r: usize) -> ComplexTensor<2> {
        complex_leading_vectors(unfold(self.re, k), unfold(self.im, k), r)
    }

    fn norm_sq(self) -> f64 {
        sum_sq(self.re) + sum_sq(self.im)
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

    /// A projector for rank-`D` gradients of type `T` (`Tensor<D>` or [`ComplexTensor<D>`]),
    /// without factors until its first `project`.
    pub fn init<const D: usize, T: GaLoreTensor<D>>(&self) -> TensorGaLoreProjector<D, T> {
        TensorGaLoreProjector {
            config: self.clone(),
            factors: None,
        }
    }
}

/// 2.0.0's `TensorGaLoreProjector` for one rank-`D` parameter's gradients, real (`T` =
/// `Tensor<D>`, the default) or complex (`T` = [`ComplexTensor<D>`]).
#[derive(Clone, Debug)]
pub struct TensorGaLoreProjector<const D: usize, T: GaLoreTensor<D> = Tensor<D>> {
    config: TensorGaLoreConfig,
    factors: Option<[T::Factor; D]>,
}

impl<const D: usize, T: GaLoreTensor<D>> TensorGaLoreProjector<D, T> {
    /// `grad` (`[I_0, .., I_{D−1}]`) in the low-rank space, `[r_0, .., r_{D−1}]`: G ×ₖ Fₖᴴ
    /// for every mode k (Fₖᵀ when real). The first call computes the factors from `grad` by
    /// [`tucker_factors`]; later calls reuse them, whatever `iter` (as 2.0.0).
    ///
    /// # Panics
    /// On the first call, if `update_proj_gap` is 0 or `iter` is not a multiple of it
    /// (2.0.0 has no factors to apply then), or as [`tucker_factors`].
    pub fn project(&mut self, grad: T, iter: usize) -> T {
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
        grad.mode_products(factors, None, true)
    }

    /// `low` (`[r_0, .., r_{D−1}]`) back in the full space, `[I_0, .., I_{D−1}]`:
    /// `scale` · (Y ×ₖ Fₖ).
    ///
    /// # Panics
    /// Before the first [`project`](Self::project), or if `low`'s shape is not the ranks.
    pub fn project_back(&self, low: T) -> T {
        let factors = self.factors.as_ref().expect("project_back before project");
        low.mode_products(factors, None, false)
            .scaled(self.config.scale)
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
/// gradients, or an SVD does not converge.
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
            let partial = x.clone().mode_products(&factors, Some(k), true);
            factors[k] = partial.leading_factor(k, ranks[k]);
        }
        let core = x.clone().mode_products(&factors, None, true);
        let error = (norm_sq - core.norm_sq()).abs().sqrt() / norm_sq.sqrt();
        if sweep > 1 && previous.is_some_and(|p: f64| (p - error).abs() < HOOI_TOL) {
            break;
        }
        previous = Some(error);
    }
    factors
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

/// [`mode_products`] for complex `x` and factors, the transpose conjugated: tensorly's
/// `multi_mode_dot(x, factors, transpose=True)` and `multi_mode_dot(y, factors)`.
fn complex_mode_products<const D: usize>(
    x: ComplexTensor<D>,
    factors: &[ComplexTensor<2>; D],
    skip: Option<usize>,
    conj_transpose: bool,
) -> ComplexTensor<D> {
    let (mut re, mut im) = (x.re, x.im);
    for (k, f) in factors.iter().enumerate() {
        let mut dims = re.dims();
        let rows = |t: Tensor<D>| -> Tensor<2> { t.reshape([dims[0] as i32, -1]).transpose() };
        let (mut a, mut b) = (rows(re), rows(im));
        if skip != Some(k) {
            // (A + iB)(F + iG), with F + iG = conj(Fₖ) or Fₖᵀ.
            let (f, g) = if conj_transpose {
                (f.re.clone(), f.im.clone().neg())
            } else {
                (f.re.clone().transpose(), f.im.clone().transpose())
            };
            (a, b) = (
                a.clone().matmul(f.clone()) - b.clone().matmul(g.clone()),
                a.matmul(g) + b.matmul(f),
            );
        }
        dims[0] = a.dims()[1];
        dims.rotate_left(1);
        (re, im) = (a.reshape(dims), b.reshape(dims));
    }
    ComplexTensor::new(re, im)
}

/// The mode-k unfolding of `x`, `[I_k, ∏_{j≠k} I_j]`. Its columns are in another order than
/// tensorly's, which leaves the left singular vectors unchanged.
fn unfold<const D: usize>(x: Tensor<D>, k: usize) -> Tensor<2> {
    let rows = x.dims()[k] as i32;
    x.swap_dims(0, k).reshape([rows, -1])
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

/// The `r` leading left singular vectors of B + iC (`b`, `c`: `[m, n]`), `[m, r]`, with the
/// phase of LAPACK's `gesdd` (the first entry of each right singular vector real when
/// m ≥ n, of each left one when m < n) and then tensorly's complex `svd_flip` (each column
/// times sgn of its entry of largest modulus; the first on a tie). Computed on the host in
/// f64 by [`jacobi_left_vectors`]; returned in `b`'s precision. `r` ≤ min(m, n).
///
/// A reference entry of zero leaves the phase undefined (NaN), as it leaves LAPACK's
/// arbitrary.
fn complex_leading_vectors(b: Tensor<2>, c: Tensor<2>, r: usize) -> ComplexTensor<2> {
    let [m, n] = b.dims();
    let (dtype, device) = (b.dtype(), b.device());
    let host = |t: Tensor<2>| {
        t.into_data()
            .convert::<f64>()
            .try_into_vec::<f64>()
            .unwrap()
    };
    let a: Vec<Cx> = host(b)
        .into_iter()
        .zip(host(c))
        .map(|(x, y)| Cx(x, y))
        .collect();
    let mut u = jacobi_left_vectors(a.clone(), m, n, r);
    for col in u.iter_mut() {
        // gesdd's phase: σ v_j[0] = (Aᴴ u_j)₀ = Σᵢ conj(aᵢ₀) uᵢⱼ real, or u₀ⱼ real.
        let w = if m >= n {
            (0..m).fold(Cx(0.0, 0.0), |acc, i| acc + a[i * n].conj() * col[i])
        } else {
            col[0]
        };
        let phase = w.conj().unit();
        col.iter_mut().for_each(|x| *x = *x * phase);
        // svd_flip with torch.sgn: times u_max / |u_max|.
        let largest = col.iter().enumerate().fold(0, |best, (i, x)| {
            if x.norm_sq() > col[best].norm_sq() {
                i
            } else {
                best
            }
        });
        let sgn = col[largest].unit();
        col.iter_mut().for_each(|x| *x = *x * sgn);
    }
    let part = |f: fn(&Cx) -> f64| {
        let values: Vec<f64> = (0..m)
            .flat_map(|i| u.iter().map(move |col| col[i]))
            .map(|x| f(&x))
            .collect();
        Tensor::from_data(
            TensorData::new(values, [m, r]).convert_dtype(dtype),
            &device,
        )
    };
    ComplexTensor::new(part(|x| x.0), part(|x| x.1))
}

/// Jacobi sweeps allowed before [`jacobi_left_vectors`] gives up (it converges
/// quadratically; a handful suffice).
const JACOBI_SWEEPS: usize = 60;

/// The `r` leading left singular vectors (columns, by singular value, descending) of the
/// complex `m × n` matrix `a` (row-major), phases arbitrary: one-sided Jacobi (Hestenes) on
/// its rows, Wᴴ A = Z with orthogonal rows, so A = W Z and the columns of W, ordered by the
/// rows' norms, are U. Robust to repeated singular values, which stall Burn's QR `svd` on a
/// complex matrix's real embedding (each value twice) in f64.
///
/// # Panics
/// If the rows are not orthogonal after [`JACOBI_SWEEPS`] sweeps.
fn jacobi_left_vectors(mut a: Vec<Cx>, m: usize, n: usize, r: usize) -> Vec<Vec<Cx>> {
    // W accumulated as Q = Wᴴ, row by row with A's rows.
    let mut q: Vec<Cx> = (0..m * m)
        .map(|k| Cx(if k / m == k % m { 1.0 } else { 0.0 }, 0.0))
        .collect();
    let tol = m as f64 * f64::EPSILON;
    // Rows below this (squared) norm are rounding noise (the m − n surplus rows of a tall
    // matrix end there): no relative orthogonality is reachable between them.
    let negligible = (tol * tol) * a.iter().map(Cx::norm_sq).sum::<f64>();
    let mut converged = false;
    for _ in 0..JACOBI_SWEEPS {
        converged = true;
        for p in 0..m {
            for t in p + 1..m {
                let (row_p, row_t) = (&a[p * n..(p + 1) * n], &a[t * n..(t + 1) * n]);
                let alpha: f64 = row_p.iter().map(Cx::norm_sq).sum();
                let beta: f64 = row_t.iter().map(Cx::norm_sq).sum();
                let gamma = row_p
                    .iter()
                    .zip(row_t)
                    .fold(Cx(0.0, 0.0), |acc, (&x, &y)| acc + x * y.conj());
                let g = gamma.norm_sq().sqrt();
                if g <= tol * (alpha * beta).sqrt() || alpha.min(beta) <= negligible {
                    continue;
                }
                converged = false;
                // Row t times e^{iφ} (γ = |γ| e^{iφ}) makes the pair's inner product real;
                // then the real rotation that zeroes it, the smaller root of
                // t² + 2ζt − 1 = 0, ζ = (β − α) / 2|γ|.
                let phase = gamma.unit();
                let zeta = (beta - alpha) / (2.0 * g);
                let tan = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let cos = 1.0 / (1.0 + tan * tan).sqrt();
                let sin = cos * tan;
                for (rows, len) in [(&mut a, n), (&mut q, m)] {
                    for k in 0..len {
                        let (x, y) = (rows[p * len + k], phase * rows[t * len + k]);
                        rows[p * len + k] = x.scale(cos) - y.scale(sin);
                        rows[t * len + k] = x.scale(sin) + y.scale(cos);
                    }
                }
            }
        }
        if converged {
            break;
        }
    }
    assert!(
        converged,
        "complex SVD: no convergence in {JACOBI_SWEEPS} Jacobi sweeps"
    );
    let norm = |i: usize| -> f64 { a[i * n..(i + 1) * n].iter().map(Cx::norm_sq).sum() };
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&i, &j| norm(j).total_cmp(&norm(i)));
    // U = Qᴴ: column i of U is row i of Q conjugated.
    order[..r]
        .iter()
        .map(|&i| q[i * m..(i + 1) * m].iter().map(|x| x.conj()).collect())
        .collect()
}

/// A complex number on the host, f64: (re, im).
#[derive(Clone, Copy, Debug)]
struct Cx(f64, f64);

impl Cx {
    fn conj(self) -> Self {
        Cx(self.0, -self.1)
    }

    fn norm_sq(&self) -> f64 {
        self.0 * self.0 + self.1 * self.1
    }

    /// self / |self|.
    fn unit(self) -> Self {
        self.scale(1.0 / self.norm_sq().sqrt())
    }

    fn scale(self, s: f64) -> Self {
        Cx(self.0 * s, self.1 * s)
    }
}

impl std::ops::Add for Cx {
    type Output = Cx;
    fn add(self, o: Cx) -> Cx {
        Cx(self.0 + o.0, self.1 + o.1)
    }
}

impl std::ops::Sub for Cx {
    type Output = Cx;
    fn sub(self, o: Cx) -> Cx {
        Cx(self.0 - o.0, self.1 - o.1)
    }
}

impl std::ops::Mul for Cx {
    type Output = Cx;
    fn mul(self, o: Cx) -> Cx {
        Cx(self.0 * o.0 - self.1 * o.1, self.0 * o.1 + self.1 * o.0)
    }
}

/// ‖x‖², on the host in f64.
fn sum_sq<const D: usize>(x: Tensor<D>) -> f64 {
    x.square().sum().into_scalar::<f64>()
}

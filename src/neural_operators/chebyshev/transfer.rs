//! Grid transfers between the CGL grid and the model's uniform grid (CONVENTIONS §12).
//!
//! Both transfers are rational barycentric interpolants, built once per (n, s) as dense
//! 1D matrices in f64 and applied along each axis: Tx F Tyᵀ for 2D fields ([`apply`]), and
//! along x, then y, then z for 3D 'ijk' fields ([`apply_nd`], any rank; 3D CL1.2):
//! - Chebyshev → uniform, T_cu ([`cheb_to_uniform`]): polynomial interpolation on the CGL
//!   nodes, second-kind weights.
//! - Uniform → Chebyshev, T_uc ([`uniform_to_cheb`]): Floater–Hormann interpolation of
//!   degree d on the uniform nodes.
//!
//! Production uses d = [`TRANSFER_DEGREE`] = 2 and, by default, s = n − 1 (design §12,
//! decisions 5 and 8), bundled as [`Transfers`]; [`Transfers::with_s`] takes any learning
//! grid s, with T_cu as the restriction (decision 30, D1). The functions take n, s and d as
//! arguments so the tests can vary them. Nodes and weights are closed forms, so this
//! module needs only ndarray and is built without the `datagen` feature (decision 13).

use ndarray::{
    Array, Array1, Array2, Array3, ArrayView, ArrayView1, ArrayView2, ArrayView3, Axis, Dimension,
    RemoveAxis,
};
use std::f64::consts::PI;

use super::check_n;
use super::tensor::apply_nd;

/// The Floater–Hormann degree d of T_uc in production (design §12, decision 8). The
/// generator records it in each split's sidecar as `error2.d`.
pub const TRANSFER_DEGREE: usize = 2;

/// Panic with a uniform message when a uniform grid has fewer than two points.
fn check_s(caller: &str, s: usize) {
    assert!(
        s >= 2,
        "{caller}: s must be >= 2 (the uniform grid includes both endpoints), got {s}"
    );
}

/// The `s` uniform nodes x̃_j = −1 + 2j/(s − 1), j = 0..s − 1, ascending, with both
/// endpoints (CONVENTIONS §12).
///
/// Returns shape `[s]`, with `x[0] = −1` and `x[s − 1] = 1` exactly. With s = n − 1 this
/// is the model's coordinate grid (CONVENTIONS §2) mapped to [−1, 1].
///
/// # Panics
/// If `s < 2`.
pub fn uniform_nodes(s: usize) -> Array1<f64> {
    check_s("uniform_nodes", s);
    let h = (s - 1) as f64;
    Array1::from_shape_fn(s, |j| -1.0 + 2.0 * j as f64 / h)
}

/// The `n` CGL nodes x_j = −cos(πj/(n − 1)), j = 0..n − 1, ascending, with both
/// endpoints (CONVENTIONS §12), in closed form (no RLST).
///
/// With N = n − 1 and θ = πj/N, −cos θ = sin(θ − π/2) = sin(π(2j − N)/(2N)), which is
/// the form computed here. It gives exact ±1, an exact 0 for odd n, and exact
/// antisymmetry x_j = −x_{N−j}. It agrees with RLST's nodes (the `points::nodes` test
/// oracle, feature `datagen`) to round-off. Returns shape `[n]`.
///
/// # Panics
/// If `n < 2`.
pub fn nodes(n: usize) -> Array1<f64> {
    check_n("nodes", n);
    let nn = (n - 1) as f64;
    Array1::from_shape_fn(n, |j| (PI * (2.0 * j as f64 - nn) / (2.0 * nn)).sin())
}

/// Second-kind barycentric weights on the `n` CGL nodes: w_j = (−1)^j δ_j, with
/// δ_0 = δ_{n−1} = 1/2 and δ_j = 1 otherwise (CONVENTIONS §12). Shape `[n]`, n ≥ 2.
fn cheb_bary_weights(n: usize) -> Array1<f64> {
    let mut w = Array1::from_shape_fn(n, |j| if j.is_multiple_of(2) { 1.0 } else { -1.0 });
    w[0] *= 0.5;
    w[n - 1] *= 0.5;
    w
}

/// Floater–Hormann weights of degree `d` on ascending nodes `x: [N + 1]`
/// (CONVENTIONS §12; Floater & Hormann 2007, eq. (18)):
///
/// w_k = (−1)^(k−d) Σ_{i ∈ J_k} Π_{j = i, j ≠ k}^{i + d} 1 / |x_k − x_j|,
/// J_k = {i : max(0, k − d) ≤ i ≤ min(k, N − d)}.
///
/// The caller guarantees `d <= N`. Shape `[N + 1]`.
fn fh_weights(x: ArrayView1<f64>, d: usize) -> Array1<f64> {
    let nn = x.len() - 1;
    Array1::from_shape_fn(nn + 1, |k| {
        let mut sum = 0.0;
        for i in k.saturating_sub(d)..=k.min(nn - d) {
            let mut prod = 1.0;
            for j in (i..=i + d).filter(|&j| j != k) {
                prod /= (x[k] - x[j]).abs();
            }
            sum += prod;
        }
        // (−1)^(k−d) has the parity of k + d.
        if (k + d).is_multiple_of(2) { sum } else { -sum }
    })
}

/// Dense evaluation matrix `[tgt.len(), src.len()]` of the rational barycentric
/// interpolant with nodes `src` and weights `w`, at the points `tgt`
/// (CONVENTIONS §12). Row i holds the coefficients of the node values in p(tgt_i). A
/// target bitwise equal to a node gets that node's unit row.
fn bary_matrix(src: ArrayView1<f64>, w: ArrayView1<f64>, tgt: ArrayView1<f64>) -> Array2<f64> {
    let mut t = Array2::zeros((tgt.len(), src.len()));
    for (i, &x) in tgt.iter().enumerate() {
        if let Some(k) = src.iter().position(|&z| z == x) {
            t[[i, k]] = 1.0;
            continue;
        }
        let mut row = t.row_mut(i);
        for (k, (&z, &wk)) in src.iter().zip(w).enumerate() {
            row[k] = wk / (x - z);
        }
        let denom = row.sum();
        row /= denom;
    }
    t
}

/// The Chebyshev → uniform transfer T_cu, shape `[s, n]` (CONVENTIONS §12).
///
/// Polynomial interpolation on the `n` ascending CGL nodes in barycentric form, with
/// second-kind weights, evaluated at the `s` [`uniform_nodes`]. For a field `f: [n]` on
/// the CGL nodes, `T_cu · f` is its interpolant on the uniform grid; for 2D fields use
/// [`apply`]. Production uses s = n − 1 (design §12, decision 5).
///
/// # Panics
/// If `n < 2` or `s < 2`.
pub fn cheb_to_uniform(n: usize, s: usize) -> Array2<f64> {
    check_n("cheb_to_uniform", n);
    check_s("cheb_to_uniform", s);
    let x = nodes(n);
    bary_matrix(
        x.view(),
        cheb_bary_weights(n).view(),
        uniform_nodes(s).view(),
    )
}

/// The uniform → Chebyshev transfer T_uc, shape `[n, s]` (CONVENTIONS §12).
///
/// Floater–Hormann interpolation of degree `d` on the `s` [`uniform_nodes`], evaluated
/// at the `n` ascending CGL nodes. It reproduces polynomials of degree ≤ d and converges
/// at order d + 1. For a field `g: [s]` on the uniform grid, `T_uc · g` is its
/// interpolant on the CGL nodes; for 2D fields use [`apply`]. Production uses d = 2 and
/// s = n − 1 (design §12, decision 8).
///
/// # Panics
/// If `s < 2`, `n < 2`, or `d > s − 1`.
pub fn uniform_to_cheb(s: usize, n: usize, d: usize) -> Array2<f64> {
    check_s("uniform_to_cheb", s);
    check_n("uniform_to_cheb", n);
    assert!(
        d < s,
        "uniform_to_cheb: d must be <= s - 1 (degree at most the number of intervals), \
         got d = {d}, s = {s}"
    );
    let x = uniform_nodes(s);
    bary_matrix(x.view(), fh_weights(x.view(), d).view(), nodes(n).view())
}

/// Apply 1D transfers along both axes of a 2D field: Tx · F · Tyᵀ (CONVENTIONS §12).
///
/// `f: [n_x, n_y]` is stored 'ij' (`f[i, j] = f(x_i, y_j)`); `tx: [m_x, n_x]` acts on
/// the first axis (x) and `ty: [m_y, n_y]` on the second (y). Returns `[m_x, m_y]`,
/// also 'ij'. The grid may be rectangular.
///
/// # Panics
/// If `tx` has not `n_x` columns or `ty` has not `n_y` columns.
pub fn apply(tx: ArrayView2<f64>, ty: ArrayView2<f64>, f: ArrayView2<f64>) -> Array2<f64> {
    assert_eq!(
        tx.ncols(),
        f.nrows(),
        "apply: tx must have f.nrows() (n_x) columns"
    );
    assert_eq!(
        ty.ncols(),
        f.ncols(),
        "apply: ty must have f.ncols() (n_y) columns"
    );
    tx.dot(&f).dot(&ty.t())
}

/// The production transfer pair for the n × n CGL grid (CONVENTIONS §12): the restriction
/// R = T_cu ([`cheb_to_uniform`]) and the prolongation P = T_uc ([`uniform_to_cheb`]) of
/// degree [`TRANSFER_DEGREE`], between the CGL nodes and the uniform learning grid of s
/// points: s = n − 1 from [`Transfers::new`] (design §12, decisions 5 and 8), any s from
/// [`Transfers::with_s`] (decision 30, D1). Both matrices are built once and applied along
/// both axes of 'ij' fields with [`apply`], or along every axis of a field of any rank
/// with [`apply_nd`] (the `_nd` methods).
#[derive(Clone, Debug)]
pub struct Transfers {
    /// `[s, n]`
    t_cu: Array2<f64>,
    /// `[n, s]`
    t_uc: Array2<f64>,
}

impl Transfers {
    /// The smallest supported CGL grid: the default s = n − 1 is at least
    /// [`Transfers::MIN_S`].
    pub const MIN_N: usize = TRANSFER_DEGREE + 2;

    /// The smallest learning grid s: T_uc of degree [`TRANSFER_DEGREE`] needs d ≤ s − 1
    /// (CONVENTIONS §12).
    pub const MIN_S: usize = TRANSFER_DEGREE + 1;

    /// The transfers for `n` CGL nodes per axis, s = n − 1.
    ///
    /// # Panics
    /// If `n <` [`Transfers::MIN_N`].
    pub fn new(n: usize) -> Self {
        assert!(
            n >= Self::MIN_N,
            "Transfers::new: n must be >= {} (T_uc of degree {TRANSFER_DEGREE} needs \
             s = n - 1 > d), got {n}",
            Self::MIN_N
        );
        Self::with_s(n, n - 1)
    }

    /// The transfers between `n` CGL nodes and a learning grid of `s` uniform points per
    /// axis (CONVENTIONS §12): R = T_cu `[s, n]` and P = T_uc `[n, s]` of degree
    /// [`TRANSFER_DEGREE`]. `with_s(n, n - 1)` equals [`Transfers::new`]`(n)`. Any n ≥ 2
    /// would give valid matrices; n is held to the same floor as [`Transfers::new`] so both
    /// constructors support the same grids.
    ///
    /// # Panics
    /// If `n <` [`Transfers::MIN_N`] or `s <` [`Transfers::MIN_S`].
    pub fn with_s(n: usize, s: usize) -> Self {
        assert!(
            n >= Self::MIN_N,
            "Transfers::with_s: n must be >= {}, got {n}",
            Self::MIN_N
        );
        assert!(
            s >= Self::MIN_S,
            "Transfers::with_s: s must be >= {} (T_uc of degree {TRANSFER_DEGREE} needs \
             s - 1 >= d), got {s}",
            Self::MIN_S
        );
        Self {
            t_cu: cheb_to_uniform(n, s),
            t_uc: uniform_to_cheb(s, n, TRANSFER_DEGREE),
        }
    }

    /// CGL nodes per axis.
    pub fn n(&self) -> usize {
        self.t_cu.ncols()
    }

    /// Uniform (learning) grid points per axis: n − 1 from [`Transfers::new`].
    pub fn s(&self) -> usize {
        self.t_cu.nrows()
    }

    /// `f: [n, n]` on the CGL grid, transferred to the uniform grid: T_cu · F · T_cuᵀ,
    /// `[s, s]`.
    ///
    /// # Panics
    /// If `f` is not `[n, n]`.
    pub fn to_uniform(&self, f: ArrayView2<f64>) -> Array2<f64> {
        apply(self.t_cu.view(), self.t_cu.view(), f)
    }

    /// Every sample of `f: [N, n, n]` on the CGL grid transferred to the uniform grid with
    /// [`Transfers::to_uniform`]: `[N, s, s]`.
    ///
    /// # Panics
    /// If `f` is not `[N, n, n]`.
    pub fn to_uniform_all(&self, f: ArrayView3<f64>) -> Array3<f64> {
        let s = self.s();
        let mut out = Array3::zeros((f.len_of(Axis(0)), s, s));
        for (mut dst, src) in out.axis_iter_mut(Axis(0)).zip(f.axis_iter(Axis(0))) {
            dst.assign(&self.to_uniform(src));
        }
        out
    }

    /// `g: [s, s]` on the uniform grid, transferred to the CGL grid: T_uc · G · T_ucᵀ,
    /// `[n, n]`.
    ///
    /// # Panics
    /// If `g` is not `[s, s]`.
    pub fn to_cheb(&self, g: ArrayView2<f64>) -> Array2<f64> {
        apply(self.t_uc.view(), self.t_uc.view(), g)
    }

    /// `f: [n, …, n]` on the CGL grid, of any rank, transferred to the uniform grid with
    /// T_cu along every axis ([`apply_nd`]; CONVENTIONS §12): `[s, …, s]`. Rank 2 equals
    /// [`Transfers::to_uniform`] bit for bit.
    ///
    /// # Panics
    /// If an axis of `f` has not n points, or `f` has rank 0.
    pub fn to_uniform_nd<D: Dimension>(&self, f: ArrayView<f64, D>) -> Array<f64, D> {
        apply_nd(&vec![self.t_cu.view(); f.ndim()], f)
    }

    /// Every sample of `f: [N, n, …, n]` (axis 0 the samples) transferred with
    /// [`Transfers::to_uniform_nd`]: `[N, s, …, s]`. Rank 3 equals
    /// [`Transfers::to_uniform_all`] bit for bit.
    ///
    /// # Panics
    /// If an axis of `f` other than the first has not n points, or `f` has rank below 2.
    pub fn to_uniform_all_nd<D: RemoveAxis>(&self, f: ArrayView<f64, D>) -> Array<f64, D> {
        assert!(
            f.ndim() >= 2,
            "Transfers::to_uniform_all_nd: f must be [N, n, ...], got rank {}",
            f.ndim()
        );
        let mut shape = f.raw_dim();
        for a in 1..shape.ndim() {
            shape[a] = self.s();
        }
        let mut out = Array::zeros(shape);
        for (mut dst, src) in out.axis_iter_mut(Axis(0)).zip(f.axis_iter(Axis(0))) {
            dst.assign(&self.to_uniform_nd(src));
        }
        out
    }

    /// `g: [s, …, s]` on the uniform grid, of any rank, transferred to the CGL grid with
    /// T_uc along every axis ([`apply_nd`]; CONVENTIONS §12): `[n, …, n]`. Rank 2 equals
    /// [`Transfers::to_cheb`] bit for bit.
    ///
    /// # Panics
    /// If an axis of `g` has not s points, or `g` has rank 0.
    pub fn to_cheb_nd<D: Dimension>(&self, g: ArrayView<f64, D>) -> Array<f64, D> {
        apply_nd(&vec![self.t_uc.view(); g.ndim()], g)
    }
}

#[cfg(test)]
mod tests;

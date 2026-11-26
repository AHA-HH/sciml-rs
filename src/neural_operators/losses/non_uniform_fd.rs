//! First derivatives on a point cloud, as `neuraloperator` 2.0.0's `non_uniform_fd` and
//! `get_non_uniform_fd_weights` (`neuralop/losses/differentiation.py`; library CM1.6c).
//!
//! Each point's derivative is a weighted sum of the values at its k nearest points (itself
//! first). The weights make the stencil exact on affine functions: for derivative j they solve
//! A w = e_{j+1}, A the (d + 1) × k matrix of a row of ones and the neighbours' coordinate
//! offsets, by the minimum-norm solution (2.0.0's `torch.linalg.lstsq`) or by 2.0.0's ridge
//! system. The weights depend on the points only and are computed on the host in f64; the
//! derivatives are a Burn gather and sum, so gradients flow to the values.

use burn::prelude::Device;
use burn::tensor::{Int, Tensor, TensorData};
use ndarray::{Array1, Array2, Array3, ArrayView2, s};

/// 2.0.0's ridge parameter for `regularize_lstsq`.
const RIDGE: f64 = 1e-6;

/// The point-cloud finite differences, with 2.0.0's arguments and defaults
/// (`num_neighbors=5`, `derivative_indices=[0]`, `radius=None`, `regularize_lstsq=False`).
#[derive(Clone, Debug, PartialEq)]
pub struct NonUniformFd {
    /// The stencil size k, the point itself included; clamped to [3, N] as 2.0.0.
    pub num_neighbors: usize,
    /// The coordinates to differentiate along, e.g. `[0, 1]` for ∂/∂x and ∂/∂y.
    pub derivative_indices: Vec<usize>,
    /// Neighbours farther than this get weight 0; the nearest 3 are always kept.
    pub radius: Option<f64>,
    /// 2.0.0's `regularize_lstsq`: solve (AᵀA + 1e-6 I) w = Aᵀb instead of the
    /// minimum-norm A w = b.
    pub regularize: bool,
}

impl Default for NonUniformFd {
    fn default() -> Self {
        Self {
            num_neighbors: 5,
            derivative_indices: vec![0],
            radius: None,
            regularize: false,
        }
    }
}

impl NonUniformFd {
    /// The stencils of `points` `(N, d)`, as 2.0.0's `get_non_uniform_fd_weights`; they depend
    /// on the points only, so one set serves any number of value sets. Ties in distance go to
    /// the lower index (torch's `topk` leaves them unspecified), so with a duplicate point of
    /// lower index a point's own index is not first.
    ///
    /// # Panics
    /// If `N == 0`, a derivative index is not below d, or (without `regularize`) a stencil's
    /// A has rank below d + 1: fewer than d + 1 neighbours kept, or all on one hyperplane.
    pub fn stencils(&self, points: &Array2<f64>) -> Stencils {
        let (n, d) = points.dim();
        assert!(n > 0, "NonUniformFd: no points");
        for &j in &self.derivative_indices {
            assert!(j < d, "NonUniformFd: derivative index {j} in {d}D");
        }
        let k = self.num_neighbors.max(3).min(n);
        let mut indices = Array2::zeros((n, k));
        let mut weights = Array3::zeros((n, self.derivative_indices.len(), k));
        for i in 0..n {
            let p = points.row(i);
            let dist: Vec<f64> = points
                .rows()
                .into_iter()
                .map(|q| {
                    q.iter()
                        .zip(&p)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .collect();
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| dist[a].total_cmp(&dist[b]));
            let near = &order[..k];
            let kept = |m: usize| m < 3 || self.radius.is_none_or(|r| dist[near[m]] <= r);
            let a = Array2::from_shape_fn((d + 1, k), |(r, m)| match (kept(m), r) {
                (false, _) => 0.0,
                (true, 0) => 1.0,
                (true, _) => points[[near[m], r - 1]] - p[r - 1],
            });
            for (jj, &j) in self.derivative_indices.iter().enumerate() {
                let mut b = Array1::zeros(d + 1);
                b[j + 1] = 1.0;
                let w = match self.regularize {
                    true => ridge(a.view(), &b),
                    false => min_norm(a.view(), &b),
                };
                weights.slice_mut(s![i, jj, ..]).assign(&w);
            }
            indices.row_mut(i).assign(&Array1::from(near.to_vec()));
        }
        Stencils { indices, weights }
    }

    /// The derivatives of `values` `[N]` at `points` `(N, d)`, as 2.0.0's `non_uniform_fd`:
    /// [`stencils`](Self::stencils), then [`Stencils::apply`].
    ///
    /// # Panics
    /// As [`stencils`](Self::stencils) and [`Stencils::apply`].
    pub fn derivatives(&self, points: &Array2<f64>, values: Tensor<1>) -> Tensor<2> {
        self.stencils(points).apply(values)
    }
}

/// The stencils of a point cloud ([`NonUniformFd::stencils`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Stencils {
    /// Each point's k nearest points, nearest first: `(N, k)`.
    pub indices: Array2<usize>,
    /// Their weights for each of the j derivatives: `(N, j, k)`.
    pub weights: Array3<f64>,
}

impl Stencils {
    /// The derivatives of `values` `[N]`: `[j, N]`, row r the derivative along the r-th
    /// derivative index. On `values`'s device and float type; gradients flow to `values`.
    ///
    /// # Panics
    /// If `values` has not N entries.
    pub fn apply(&self, values: Tensor<1>) -> Tensor<2> {
        let (n, j, k) = self.weights.dim();
        assert_eq!(
            values.dims()[0],
            n,
            "NonUniformFd: {} values at {n} points",
            values.dims()[0]
        );
        let device: Device = values.device();
        let flat: Vec<i64> = self.indices.iter().map(|&i| i as i64).collect();
        let idx = Tensor::<1, Int>::from_data(TensorData::new(flat, [n * k]), &device);
        let neighbours: Tensor<3> = values.select(0, idx).reshape([1, n, k]);
        // (N, j, k) to (j, N, k), row-major.
        let w: Vec<f64> = self
            .weights
            .view()
            .permuted_axes([1, 0, 2])
            .iter()
            .copied()
            .collect();
        let w = Tensor::<3>::from_data(TensorData::new(w, [j, n, k]), &device);
        (w * neighbours).sum_dim(2).reshape([j, n])
    }
}

/// The minimum-norm solution of A w = b for A `(m, k)` of rank m ≤ k: with Aᵀ = QR,
/// w = Q R⁻ᵀ b, Q and R by modified Gram–Schmidt with one reorthogonalisation pass.
///
/// # Panics
/// If A's rank is below m.
fn min_norm(a: ArrayView2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let (m, k) = a.dim();
    // shortcut: rank-deficient stencils panic; 2.0.0's gelsy returns the pseudo-inverse
    // solution there. Upgrade if a point cloud needs it (collinear or radius-starved).
    assert!(
        m <= k,
        "NonUniformFd: {k} neighbours for {m} unknowns; needs at least {m}"
    );
    let mut q = a.t().to_owned(); // (k, m), orthonormalised in place
    let mut r = Array2::<f64>::zeros((m, m));
    let scale = a.iter().fold(0.0_f64, |s, x| s.max(x.abs()));
    for c in 0..m {
        for _ in 0..2 {
            for p in 0..c {
                let dot = q.column(p).dot(&q.column(c));
                r[[p, c]] += dot;
                let qp = q.column(p).to_owned();
                q.column_mut(c).scaled_add(-dot, &qp);
            }
        }
        let norm = q.column(c).dot(&q.column(c)).sqrt();
        assert!(
            norm > 1e-12 * scale.max(1.0),
            "NonUniformFd: a stencil of rank below {m} (points on one hyperplane, or too few kept)"
        );
        r[[c, c]] = norm;
        q.column_mut(c).mapv_inplace(|x| x / norm);
    }
    // Rᵀ y = b, forward substitution.
    let mut y = Array1::<f64>::zeros(m);
    for c in 0..m {
        let s: f64 = (0..c).map(|p| r[[p, c]] * y[p]).sum();
        y[c] = (b[c] - s) / r[[c, c]];
    }
    q.dot(&y)
}

/// 2.0.0's regularised solution (AᵀA + 1e-6 I) w = Aᵀb, by Cholesky.
fn ridge(a: ArrayView2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let k = a.ncols();
    let mut g = a.t().dot(&a);
    g.diag_mut().mapv_inplace(|x| x + RIDGE);
    let rhs = a.t().dot(b);
    // G = L Lᵀ, L in the lower triangle of `l`.
    let mut l = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..=i {
            let s: f64 = (0..j).map(|p| l[[i, p]] * l[[j, p]]).sum();
            l[[i, j]] = match i == j {
                true => (g[[i, i]] - s).sqrt(),
                false => (g[[i, j]] - s) / l[[j, j]],
            };
        }
    }
    let mut y = Array1::<f64>::zeros(k);
    for i in 0..k {
        let s: f64 = (0..i).map(|p| l[[i, p]] * y[p]).sum();
        y[i] = (rhs[i] - s) / l[[i, i]];
    }
    let mut w = Array1::<f64>::zeros(k);
    for i in (0..k).rev() {
        let s: f64 = (i + 1..k).map(|p| l[[p, i]] * w[p]).sum();
        w[i] = (y[i] - s) / l[[i, i]];
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// A scattered 2D cloud in [0, 1)², pseudo-random by the sine hash.
    fn cloud(n: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, 2), |(i, c)| {
            ((i as f64 * 12.9898 + c as f64 * 78.233).sin() * 43_758.545_3)
                .fract()
                .abs()
        })
    }

    /// The stencils are exact on an affine function: f = 3 + 2x − 5y gives ∂x = 2, ∂y = −5 at
    /// every point, both solvers (f32 rounding of the default device: 1e-4).
    #[test]
    fn exact_on_affine_functions() {
        let points = cloud(20);
        let f: Vec<f64> = points
            .rows()
            .into_iter()
            .map(|p| 3.0 + 2.0 * p[0] - 5.0 * p[1])
            .collect();
        for regularize in [false, true] {
            let op = NonUniformFd {
                derivative_indices: vec![0, 1],
                regularize,
                ..NonUniformFd::default()
            };
            let values =
                Tensor::<1>::from_data(TensorData::new(f.clone(), [20]), &Device::default());
            let d = op.derivatives(&points, values);
            let d: Vec<f64> = d.into_data().convert::<f64>().try_to_vec::<f64>().unwrap();
            for (i, x) in d.iter().enumerate() {
                let want = if i < 20 { 2.0 } else { -5.0 };
                assert!((x - want).abs() < 1e-3, "{regularize} {i}: {x}");
            }
        }
    }

    /// `min_norm` is the closed form Aᵀ(AAᵀ)⁻¹b on a small full-rank system.
    #[test]
    fn min_norm_is_the_pseudo_inverse_solution() {
        let a = array![
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.3, -0.2, 0.1],
            [0.0, 0.1, 0.4, -0.3]
        ];
        let b = array![0.0, 1.0, 0.0];
        let w = min_norm(a.view(), &b);
        let closed = a.t().dot(&solve3(&a.dot(&a.t()), &b));
        let err = (&w - &closed).iter().fold(0.0_f64, |m, e| m.max(e.abs()));
        assert!(err < 1e-13, "{w} against {closed}: {err:e}");
    }

    /// x of G x = r for a 3 × 3 G by Cramer's rule (test only).
    fn solve3(g: &Array2<f64>, r: &Array1<f64>) -> Array1<f64> {
        let det = |m: &Array2<f64>| {
            m[[0, 0]] * (m[[1, 1]] * m[[2, 2]] - m[[1, 2]] * m[[2, 1]])
                - m[[0, 1]] * (m[[1, 0]] * m[[2, 2]] - m[[1, 2]] * m[[2, 0]])
                + m[[0, 2]] * (m[[1, 0]] * m[[2, 1]] - m[[1, 1]] * m[[2, 0]])
        };
        let d = det(g);
        Array1::from_shape_fn(3, |c| {
            let mut m = g.clone();
            m.column_mut(c).assign(r);
            det(&m) / d
        })
    }

    /// The point itself is the first neighbour, and the radius zeroes far neighbours' weights
    /// but keeps the nearest 3.
    #[test]
    fn self_first_and_radius_mask() {
        let points = cloud(12);
        let op = NonUniformFd {
            radius: Some(0.0),
            ..NonUniformFd::default()
        };
        let Stencils {
            indices: idx,
            weights: w,
        } = op.stencils(&points);
        for i in 0..12 {
            assert_eq!(idx[[i, 0]], i);
            assert_eq!(w[[i, 0, 3]], 0.0);
            assert_eq!(w[[i, 0, 4]], 0.0);
        }
    }

    /// Gradients flow to the values: d Σᵢ (∂f/∂x)ᵢ / d f_p is the sum of the weights that
    /// stencils give point p (f32: 1e-5 relative).
    #[test]
    fn gradients_flow_to_values() {
        let device = Device::default().autodiff();
        let points = cloud(10);
        let st = NonUniformFd::default().stencils(&points);
        let values =
            Tensor::<1>::from_data(TensorData::new(vec![0.5; 10], [10]), &device).require_grad();
        let out = st.apply(values.clone());
        let g = values.grad(&out.sum().backward()).unwrap();
        let g: Vec<f64> = g.into_data().convert::<f64>().try_to_vec::<f64>().unwrap();
        let mut want = vec![0.0; 10];
        for ((i, m), &p) in st.indices.indexed_iter() {
            want[p] += st.weights[[i, 0, m]];
        }
        let scale = want.iter().fold(0.0_f64, |s, w| s.max(w.abs()));
        for (p, (a, e)) in g.iter().zip(&want).enumerate() {
            assert!((a - e).abs() <= 1e-5 * scale, "point {p}: {a} against {e}");
        }
    }

    #[test]
    #[should_panic(expected = "needs at least 4")]
    fn too_few_neighbours_panic() {
        let points = Array2::from_shape_fn((3, 3), |(i, c)| (i * 3 + c) as f64);
        NonUniformFd::default().stencils(&points);
    }

    #[test]
    #[should_panic(expected = "rank below 3")]
    fn collinear_points_panic() {
        let points = Array2::from_shape_fn((5, 2), |(i, c)| i as f64 * (c + 1) as f64);
        NonUniformFd::default().stencils(&points);
    }
}

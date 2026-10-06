//! Nodes, quadrature weights and weighted relative errors.

use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Chebyshev–Gauss–Lobatto nodes x_j = −cos(πj/(n − 1)), ascending.
///
/// Computed as sin(π(2j − N)/(2N)), N = n − 1, which gives exact ±1, an exact 0 for odd n
/// and exact antisymmetry.
pub fn cheb_nodes(n: usize) -> Array1<f64> {
    assert!(n >= 2);
    let nn = (n - 1) as f64;
    Array1::from_iter((0..n).map(|j| (PI * (2.0 * j as f64 - nn) / (2.0 * nn)).sin()))
}

/// `s` equispaced points on [−1, 1], both endpoints included.
pub fn uniform_nodes(s: usize) -> Array1<f64> {
    assert!(s >= 2);
    let h = (s - 1) as f64;
    Array1::from_iter((0..s).map(|k| -1.0 + 2.0 * k as f64 / h))
}

/// Clenshaw–Curtis weights on the `n` CGL nodes (Trefethen, *Spectral Methods in
/// MATLAB*, `clencurt`). Symmetric, so the node order does not matter.
pub fn clenshaw_curtis(n: usize) -> Array1<f64> {
    assert!(n >= 2);
    let nn = n - 1;
    let nf = nn as f64;
    let mut w = Array1::zeros(n);
    if nn == 1 {
        w.fill(1.0);
        return w;
    }
    let end = if nn.is_multiple_of(2) {
        1.0 / (nf * nf - 1.0)
    } else {
        1.0 / (nf * nf)
    };
    w[0] = end;
    w[nn] = end;
    for j in 1..nn {
        let theta = PI * j as f64 / nf;
        let mut v = 1.0;
        if nn.is_multiple_of(2) {
            for k in 1..nn / 2 {
                let kf = k as f64;
                v -= 2.0 * (2.0 * kf * theta).cos() / (4.0 * kf * kf - 1.0);
            }
            v -= (nf * theta).cos() / (nf * nf - 1.0);
        } else {
            for k in 1..=(nn - 1) / 2 {
                let kf = k as f64;
                v -= 2.0 * (2.0 * kf * theta).cos() / (4.0 * kf * kf - 1.0);
            }
        }
        w[j] = 2.0 * v / nf;
    }
    w
}

/// Trapezoidal weights on `s` equispaced points of [−1, 1].
pub fn trapezoid(s: usize) -> Array1<f64> {
    assert!(s >= 2);
    let h = 2.0 / (s - 1) as f64;
    let mut w = Array1::from_elem(s, h);
    w[0] = h / 2.0;
    w[s - 1] = h / 2.0;
    w
}

/// Tensor-product weighted norm sqrt(Σ wx_i wy_j v_ij²).
pub fn wnorm(v: &Array2<f64>, wx: &Array1<f64>, wy: &Array1<f64>) -> f64 {
    let mut acc = 0.0;
    for ((i, j), x) in v.indexed_iter() {
        acc += wx[i] * wy[j] * x * x;
    }
    acc.sqrt()
}

/// Relative weighted L² error ‖approx − exact‖ / ‖exact‖.
pub fn rel_l2(
    approx: &Array2<f64>,
    exact: &Array2<f64>,
    wx: &Array1<f64>,
    wy: &Array1<f64>,
) -> f64 {
    wnorm(&(approx - exact), wx, wy) / wnorm(exact, wx, wy)
}

/// Evaluate a separable-or-not field on the tensor grid xs × ys, 'ij' storage.
pub fn sample(xs: &Array1<f64>, ys: &Array1<f64>, f: impl Fn(f64, f64) -> f64) -> Array2<f64> {
    Array2::from_shape_fn((xs.len(), ys.len()), |(i, j)| f(xs[i], ys[j]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cheb_nodes_ascending_exact_endpoints_antisymmetric() {
        for n in [2, 3, 33, 64, 257] {
            let x = cheb_nodes(n);
            assert_eq!(x[0], -1.0);
            assert_eq!(x[n - 1], 1.0);
            for j in 1..n {
                assert!(x[j] > x[j - 1]);
                assert_eq!(x[j], -x[n - 1 - j]);
            }
            for j in 0..n {
                let reference = -(PI * j as f64 / (n - 1) as f64).cos();
                assert!((x[j] - reference).abs() < 1e-15);
            }
        }
    }

    // Tolerance 1e-14: CC on n nodes integrates polynomials of degree ≤ n − 1 exactly.
    #[test]
    fn clenshaw_curtis_exact_to_degree_n_minus_1() {
        for n in [3, 4, 5, 33, 65, 129] {
            let x = cheb_nodes(n);
            let w = clenshaw_curtis(n);
            for k in 0..n {
                let quad: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(k as i32)).sum();
                let exact = if k % 2 == 1 {
                    0.0
                } else {
                    2.0 / (k as f64 + 1.0)
                };
                assert!(
                    (quad - exact).abs() < 1e-14,
                    "n={n} k={k}: {quad} vs {exact}"
                );
            }
        }
    }

    #[test]
    fn trapezoid_sums_to_two() {
        for s in [2, 32, 33] {
            assert!((trapezoid(s).sum() - 2.0).abs() < 1e-14);
        }
    }
}

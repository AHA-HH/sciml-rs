//! Chebyshev differentiation on the ascending CGL nodes (design §3.2).

use ndarray::{Array1, Array2, s};
use transfers_spike::grids::cheb_nodes;

/// Trefethen's `cheb` (Spectral Methods in MATLAB, ch. 6) on the ascending nodes
/// x_j = −cos(πj/(n − 1)). Off-diagonal entries c_i/c_j (−1)^(i+j) / (x_i − x_j), with
/// c = 2 at the ends and 1 inside; the diagonal is the negative row sum.
pub fn cheb_d(n: usize) -> Array2<f64> {
    assert!(n >= 2);
    let x = cheb_nodes(n);
    let c = |i: usize| if i == 0 || i == n - 1 { 2.0 } else { 1.0 };
    let mut d = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
                d[[i, j]] = c(i) / c(j) * sign / (x[i] - x[j]);
            }
        }
        let row: f64 = d.row(i).sum();
        d[[i, i]] = -row;
    }
    d
}

/// Trefethen's closed-form entries, for the sanity check (same ordering as `cheb_d`).
/// On the ascending ordering the corners flip sign: D_00 = −(2N² + 1)/6.
pub fn cheb_d_closed_form(n: usize) -> Array2<f64> {
    let x = cheb_nodes(n);
    let nn = (n - 1) as f64;
    let mut d = cheb_d(n);
    for j in 1..n - 1 {
        d[[j, j]] = -x[j] / (2.0 * (1.0 - x[j] * x[j]));
    }
    d[[0, 0]] = -(2.0 * nn * nn + 1.0) / 6.0;
    d[[n - 1, n - 1]] = (2.0 * nn * nn + 1.0) / 6.0;
    d
}

/// D² = D · D on all n nodes.
pub fn cheb_d2(n: usize) -> Array2<f64> {
    let d = cheb_d(n);
    d.dot(&d)
}

/// Interior block of D², (n − 2) × (n − 2): the Dirichlet second-derivative matrix.
pub fn d2_interior(n: usize) -> Array2<f64> {
    cheb_d2(n).slice(s![1..n - 1, 1..n - 1]).to_owned()
}

/// Max error of D² on x^k, k < n, relative to max |k(k − 1)x^(k−2)| over the nodes.
pub fn d2_poly_error(n: usize) -> f64 {
    let x = cheb_nodes(n);
    let d2 = cheb_d2(n);
    let mut worst = 0.0_f64;
    for k in 0..n {
        let p: Array1<f64> = x.mapv(|t| t.powi(k as i32));
        let exact: Array1<f64> = x.mapv(|t| {
            if k < 2 {
                0.0
            } else {
                (k * (k - 1)) as f64 * t.powi(k as i32 - 2)
            }
        });
        let got = d2.dot(&p);
        let scale = exact.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
        let err = (&got - &exact).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        worst = worst.max(err / scale);
    }
    worst
}

/// Max |cheb_d − closed form| relative to max |D|.
pub fn d_closed_form_error(n: usize) -> f64 {
    crate::rel_max(&cheb_d(n), &cheb_d_closed_form(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d_matches_closed_form() {
        for n in [3, 9, 17, 33] {
            let e = d_closed_form_error(n);
            assert!(e < 1e-13, "n={n}: {e:e}");
        }
    }

    // D² is exact on polynomials of degree < n; round-off grows like n⁴ ε.
    #[test]
    fn d2_exact_on_polynomials() {
        for (n, tol) in [(9, 1e-12), (17, 1e-11), (33, 1e-10)] {
            let e = d2_poly_error(n);
            assert!(e < tol, "n={n}: {e:e}");
        }
    }
}

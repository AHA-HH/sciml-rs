//! Chebyshev differentiation matrices on the ascending CGL nodes.

use ndarray::Array2;
use rlst::chebychev::{ChebychevCoefficientsToData, DerivativeOfChebychevSeries1d};

use super::{check_n, unit_coeffs};

/// First-derivative matrix D, shape `[n, n]`, on the ascending CGL nodes
/// (CONVENTIONS §12): `(D v)[i]` is the derivative at x_i of the degree-(n − 1)
/// polynomial interpolating `v: [n]` at the nodes.
///
/// Column j is RLST's coefficient transform of the unit vector e_j, differentiated
/// once in coefficient space and transformed back; the descending result is
/// reordered as `D[i][j] = D_desc[n − 1 − i][n − 1 − j]`, with no sign change.
///
/// # Panics
/// If `n < 2`.
pub fn diff_matrix(n: usize) -> Array2<f64> {
    check_n("diff_matrix", n);
    derivative_matrix(n, 1)
}

/// Second-derivative matrix D², shape `[n, n]`, on the ascending CGL nodes
/// (CONVENTIONS §12).
///
/// Built like [`diff_matrix`] with two derivatives in coefficient space, so it equals
/// D · D up to round-off. Exact on polynomials of degree < n.
///
/// # Panics
/// If `n < 2`.
pub fn diff2_matrix(n: usize) -> Array2<f64> {
    check_n("diff2_matrix", n);
    derivative_matrix(n, 2)
}

/// Matrix of the `order`-th derivative of the CGL interpolant, ascending order.
fn derivative_matrix(n: usize, order: usize) -> Array2<f64> {
    let mut desc = Array2::zeros((n, n));
    for j in 0..n {
        let mut coeffs = unit_coeffs(n, j);
        coeffs.chebychev_derivative_second_kind(order);
        let column = coeffs.chebychev_data_from_coeffs_second_kind();
        for i in 0..n {
            desc[[i, j]] = column[[i]];
        }
    }
    Array2::from_shape_fn((n, n), |(i, j)| desc[[n - 1 - i, n - 1 - j]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use std::f64::consts::PI;

    fn max_abs(a: &Array2<f64>) -> f64 {
        a.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
    }

    /// Trefethen's `cheb` (Spectral Methods in MATLAB, ch. 6) on the closed-form
    /// ascending nodes, with the closed-form diagonal. On the ascending ordering the
    /// corners flip sign: D_00 = −(2N² + 1)/6.
    fn cheb_closed_form(n: usize) -> Array2<f64> {
        let nn = (n - 1) as f64;
        let x = Array1::from_shape_fn(n, |j| -(PI * j as f64 / nn).cos());
        let c = |i: usize| if i == 0 || i == n - 1 { 2.0 } else { 1.0 };
        Array2::from_shape_fn((n, n), |(i, j)| {
            if i != j {
                let sign = if (i + j).is_multiple_of(2) { 1.0 } else { -1.0 };
                c(i) / c(j) * sign / (x[i] - x[j])
            } else if i == 0 {
                -(2.0 * nn * nn + 1.0) / 6.0
            } else if i == n - 1 {
                (2.0 * nn * nn + 1.0) / 6.0
            } else {
                -x[i] / (2.0 * (1.0 - x[i] * x[i]))
            }
        })
    }

    #[test]
    fn diff_matrix_matches_closed_form() {
        for n in [2, 3, 9, 33, 65, 129, 257] {
            let d = diff_matrix(n);
            let exact = cheb_closed_form(n);
            let scale = max_abs(&exact);
            let err = max_abs(&(&d - &exact)) / scale;
            assert!(
                err <= 1e-10,
                "n={n}: entry error {err:e} relative to max|D|"
            );
            let row_sum = d
                .rows()
                .into_iter()
                .fold(0.0_f64, |m, r| m.max(r.sum().abs()));
            assert!(
                row_sum <= 1e-10 * scale,
                "n={n}: row sum {row_sum:e}, max|D| {scale:e}"
            );
        }
    }

    // Error relative to ‖D²‖_∞ · ‖x^k‖_∞, not to the value: T4 saw value-relative
    // errors of 7.2e-10 at n = 65 from round-off alone (T1 brief).
    #[test]
    fn d2_exact_on_polynomials() {
        for n in [9, 17, 33, 65, 129, 257] {
            let nn = (n - 1) as f64;
            let x = Array1::from_shape_fn(n, |j| -(PI * j as f64 / nn).cos());
            let d2 = diff2_matrix(n);
            let d2_inf = d2
                .rows()
                .into_iter()
                .fold(0.0_f64, |m, r| m.max(r.iter().map(|v| v.abs()).sum()));
            for k in 0..n {
                let p = x.mapv(|t| t.powi(k as i32));
                let exact = x.mapv(|t| {
                    if k < 2 {
                        0.0
                    } else {
                        (k * (k - 1)) as f64 * t.powi(k as i32 - 2)
                    }
                });
                let p_inf = p.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                let err = (&d2.dot(&p) - &exact)
                    .iter()
                    .fold(0.0_f64, |m, v| m.max(v.abs()));
                assert!(
                    err <= 1e-10 * d2_inf * p_inf,
                    "n={n} k={k}: error {err:e}, ‖D²‖_∞ {d2_inf:e}"
                );
            }
        }
    }

    // Much tighter than the ‖D²‖_∞-scaled bound above: catches a wrong derivative
    // order or a wrong reordering of D².
    #[test]
    fn diff2_matrix_equals_d_squared() {
        for n in [2, 3, 9, 33, 65, 129, 257] {
            let d = diff_matrix(n);
            let d2 = diff2_matrix(n);
            let err = max_abs(&(&d.dot(&d) - &d2)) / max_abs(&d2).max(1.0);
            assert!(err <= 1e-12, "n={n}: ‖D·D − D²‖_max relative {err:e}");
        }
    }

    #[test]
    fn diff2_matrix_small_n() {
        // n = 2: interpolants are linear, so D² = 0.
        assert!(max_abs(&diff2_matrix(2)) <= 1e-14);
        // n = 3: nodes −1, 0, 1 and the quadratic interpolant has p'' = v_0 − 2v_1 + v_2.
        let d2 = diff2_matrix(3);
        for row in d2.rows() {
            for (v, e) in row.iter().zip([1.0, -2.0, 1.0]) {
                assert!((v - e).abs() <= 1e-14, "{d2}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "n must be >= 2")]
    fn diff_matrix_panics_below_two() {
        diff_matrix(1);
    }

    #[test]
    #[should_panic(expected = "n must be >= 2")]
    fn diff2_matrix_panics_below_two() {
        diff2_matrix(1);
    }
}

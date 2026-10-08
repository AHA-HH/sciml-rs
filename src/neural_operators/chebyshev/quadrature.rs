//! Clenshaw–Curtis quadrature weights on the ascending CGL nodes.

use ndarray::Array1;

use super::{check_n, unit_coeffs};

/// Clenshaw–Curtis weights w, shape `[n]`, on the ascending CGL nodes
/// (CONVENTIONS §12): Σ_j w_j f(x_j) approximates ∫_{−1}^{1} f(x) dx, exactly for
/// polynomials of degree ≤ n − 1. The weights sum to 2.
///
/// w_j integrates the interpolant of the unit vector e_j:
/// w_j = Σ_k c_k(e_j) I_k, with c_k RLST's Chebyshev coefficients and
/// I_k = ∫ T_k = 2/(1 − k²) for even k, 0 for odd k. Reordered to ascending nodes.
///
/// # Panics
/// If `n < 2`.
pub fn clenshaw_curtis(n: usize) -> Array1<f64> {
    check_n("clenshaw_curtis", n);
    let integral = |k: usize| {
        if k.is_multiple_of(2) {
            let k = k as f64;
            2.0 / (1.0 - k * k)
        } else {
            0.0
        }
    };
    let desc: Vec<f64> = (0..n)
        .map(|j| {
            let coeffs = unit_coeffs(n, j);
            (0..n).map(|k| coeffs[[k]] * integral(k)).sum()
        })
        .collect();
    Array1::from_shape_fn(n, |j| desc[n - 1 - j])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn clenshaw_curtis_exact_on_polynomials() {
        for n in [2, 3, 9, 33, 65] {
            let nn = (n - 1) as f64;
            let w = clenshaw_curtis(n);
            for k in 0..n {
                let quad: f64 = (0..n)
                    .map(|j| w[j] * (-(PI * j as f64 / nn).cos()).powi(k as i32))
                    .sum();
                let exact = if k.is_multiple_of(2) {
                    2.0 / (k as f64 + 1.0)
                } else {
                    0.0
                };
                assert!(
                    (quad - exact).abs() <= 1e-14,
                    "n={n} k={k}: {quad} vs {exact}"
                );
            }
        }
    }

    #[test]
    fn clenshaw_curtis_symmetric_and_positive() {
        for n in [2, 3, 4, 9, 33, 65, 129, 257] {
            let w = clenshaw_curtis(n);
            for j in 0..n {
                assert!(w[j] > 0.0, "n={n} j={j}: {}", w[j]);
                assert!(
                    (w[j] - w[n - 1 - j]).abs() <= 1e-15,
                    "n={n} j={j}: {} vs {}",
                    w[j],
                    w[n - 1 - j]
                );
            }
        }
        // n = 3 is Simpson's rule.
        let w = clenshaw_curtis(3);
        for (v, e) in w.iter().zip([1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0]) {
            assert!((v - e).abs() <= 1e-15, "{w}");
        }
    }

    #[test]
    #[should_panic(expected = "n must be >= 2")]
    fn clenshaw_curtis_panics_below_two() {
        clenshaw_curtis(1);
    }
}

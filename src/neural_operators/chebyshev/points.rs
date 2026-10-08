//! Chebyshev–Gauss–Lobatto nodes in ascending order.

use ndarray::Array1;
use rlst::interpolation::{Kind, chebychev_points};

use super::check_n;

/// The `n` CGL nodes x_j = −cos(πj/(n − 1)), j = 0..n − 1, ascending, with both
/// endpoints (CONVENTIONS §12).
///
/// RLST's `chebychev_points(Kind::Second, n)` reversed. Returns shape `[n]`, with
/// `x[0] = −1` and `x[n − 1] = 1` exactly.
///
/// # Panics
/// If `n < 2`.
pub fn nodes(n: usize) -> Array1<f64> {
    check_n("nodes", n);
    let desc = chebychev_points::<f64>(Kind::Second, n);
    Array1::from_shape_fn(n, |j| desc[[n - 1 - j]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn cgl_nodes_ascending_match_rlst_reversed() {
        for n in [2, 3, 9, 33, 257] {
            let x = nodes(n);
            assert_eq!(x.len(), n);
            assert_eq!(x[0], -1.0, "n={n}");
            assert_eq!(x[n - 1], 1.0, "n={n}");
            for j in 0..n {
                let exact = -(PI * j as f64 / (n - 1) as f64).cos();
                assert!(
                    (x[j] - exact).abs() <= 1e-14,
                    "n={n} j={j}: {} vs {exact}",
                    x[j]
                );
                if j > 0 {
                    assert!(x[j] > x[j - 1], "n={n}: not ascending at j={j}");
                }
                // Symmetric about 0; for odd n the middle node is 0 up to round-off.
                assert!(
                    (x[j] + x[n - 1 - j]).abs() <= 1e-15,
                    "n={n} j={j}: not symmetric"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "n must be >= 2")]
    fn nodes_panics_below_two() {
        nodes(1);
    }
}

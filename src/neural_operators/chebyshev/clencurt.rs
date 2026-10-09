//! Clenshaw–Curtis weights in closed form and the relative Clenshaw–Curtis L² error
//! (CONVENTIONS §12, design §3.4).
//!
//! The weights follow Trefethen's `clencurt` (*Spectral Methods in MATLAB*, ch. 12), so
//! this module needs only ndarray and is built without the `chebyshev` feature: evaluation
//! on the Chebyshev grid needs no RLST, BLAS or FFTW (design §12, decision 15). They agree
//! with the RLST-backed `chebyshev::clenshaw_curtis` (feature `chebyshev`) to round-off.

use ndarray::{Array1, ArrayView2, Zip};
use std::f64::consts::PI;

use super::check_n;

/// Clenshaw–Curtis weights w, shape `[n]`, on the ascending CGL nodes (CONVENTIONS §12),
/// in closed form (Trefethen, *Spectral Methods in MATLAB*, ch. 12, `clencurt`):
/// Σ_j w_j f(x_j) approximates ∫_{−1}^{1} f(x) dx, exactly for polynomials of degree
/// ≤ n − 1 (design §3.4). The weights are positive, symmetric and sum to 2.
///
/// With N = n − 1 and θ_j = πj/N, the end weights are w_0 = w_N = 1/(N² − 1) for even N
/// and 1/N² for odd N, and for 0 < j < N
///
/// w_j = (2/N) (1 − Σ_{k=1}^{⌊(N−1)/2⌋} 2 cos(2kθ_j)/(4k² − 1) − [N even] cos(Nθ_j)/(N² − 1)).
///
/// Trefethen's nodes are descending; the weights are symmetric, so the same array serves
/// the ascending nodes. Only j ≤ N/2 is computed and mirrored, so w_j = w_{N−j} exactly.
/// O(n²). Agrees with `clenshaw_curtis` (feature `chebyshev`) to round-off.
///
/// # Panics
/// If `n < 2`.
pub fn clencurt(n: usize) -> Array1<f64> {
    check_n("clencurt", n);
    let nn = n - 1;
    let nf = nn as f64;
    let mut w = Array1::zeros(n);
    let end = if nn.is_multiple_of(2) {
        1.0 / (nf * nf - 1.0)
    } else {
        1.0 / (nf * nf)
    };
    w[0] = end;
    w[nn] = end;
    for j in 1..=nn / 2 {
        let theta = PI * j as f64 / nf;
        let mut v = 1.0;
        for k in 1..=(nn - 1) / 2 {
            let k2 = (k * k) as f64;
            v -= 2.0 * (2.0 * k as f64 * theta).cos() / (4.0 * k2 - 1.0);
        }
        if nn.is_multiple_of(2) {
            // cos(Nθ_j) = cos(πj) = (−1)^j.
            let sign = if j.is_multiple_of(2) { 1.0 } else { -1.0 };
            v -= sign / (nf * nf - 1.0);
        }
        w[j] = 2.0 * v / nf;
        w[nn - j] = w[j];
    }
    w
}

/// Relative Clenshaw–Curtis L² error ‖a − b‖ / ‖b‖ on the CGL tensor grid, `b` the
/// reference (CONVENTIONS §12, design §3.4):
///
/// ‖v‖² ≈ Σ_{i,j} wx_i · wy_j · v[i, j]², with wx = [`clencurt`]`(n_x)`, wy = [`clencurt`]`(n_y)`.
///
/// `a` and `b` are `[n_x, n_y]` fields stored 'ij' (`v[i, j] = v(x_i, y_j)`). Before
/// squaring, a − b is divided by max |a − b| and b by max |b|, so fields far from unit
/// size (below about 1e-154 or above about 1e154) neither underflow nor overflow, even
/// when `a` and `b` differ greatly in magnitude. Returns NaN or infinity when ‖b‖ = 0, as
/// `rel_l2_error` (feature `chebyshev`) does, and equals it with `clenshaw_curtis` weights
/// to round-off.
///
/// # Panics
/// If `a` and `b` differ in shape, or either axis has fewer than 2 nodes.
pub fn cc_rel_l2_error(a: ArrayView2<f64>, b: ArrayView2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim(), "cc_rel_l2_error: shapes differ");
    let (nx, ny) = b.dim();
    check_n("cc_rel_l2_error", nx);
    check_n("cc_rel_l2_error", ny);
    let (wx, wy) = (clencurt(nx), clencurt(ny));
    // ‖a − b‖ / ‖b‖ = (sd / sb) · ‖(a − b)/sd‖ / ‖b/sb‖, with each scaled norm of unit
    // size. Dividing (not multiplying by 1/s) keeps a subnormal s finite. A scale of 0 or
    // infinity is replaced by 1, so ‖b‖ = 0 still gives NaN or infinity.
    let (mut sd, mut sb) = (0.0_f64, 0.0_f64);
    Zip::from(a).and(b).for_each(|&ai, &bi| {
        sd = sd.max((ai - bi).abs());
        sb = sb.max(bi.abs());
    });
    let unit = |s: f64| if s > 0.0 && s.is_finite() { s } else { 1.0 };
    let (sd, sb) = (unit(sd), unit(sb));
    let (mut num, mut den) = (0.0, 0.0);
    Zip::indexed(a).and(b).for_each(|(i, j), &ai, &bi| {
        let w = wx[i] * wy[j];
        let (d, r) = ((ai - bi) / sd, bi / sb);
        num += w * d * d;
        den += w * r * r;
    });
    (sd / sb) * (num / den).sqrt()
}

#[cfg(test)]
mod tests {
    use super::super::cgl_nodes;
    use super::*;
    use ndarray::{Array2, s};

    fn sample(nx: usize, ny: usize, f: impl Fn(f64, f64) -> f64) -> Array2<f64> {
        let (x, y) = (cgl_nodes(nx), cgl_nodes(ny));
        Array2::from_shape_fn((nx, ny), |(i, j)| f(x[i], y[j]))
    }

    #[test]
    fn clencurt_integrates_polynomials() {
        for n in [2, 3, 9, 33, 65] {
            let x = cgl_nodes(n);
            let w = clencurt(n);
            for k in 0..n {
                let quad: f64 = (0..n).map(|j| w[j] * x[j].powi(k as i32)).sum();
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
    fn clencurt_is_symmetric_and_sums_to_two() {
        for n in [2, 3, 4, 9, 33, 65, 129, 257] {
            let w = clencurt(n);
            for j in 0..n {
                assert!(w[j] > 0.0, "n={n} j={j}: {}", w[j]);
                assert_eq!(w[j], w[n - 1 - j], "n={n} j={j}");
            }
            let sum = w.sum();
            assert!((sum - 2.0).abs() <= 1e-14, "n={n}: {sum}");
        }
        assert_eq!(clencurt(2).to_vec(), [1.0, 1.0]);
        // n = 3 is Simpson's rule.
        let w = clencurt(3);
        for (v, e) in w.iter().zip([1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0]) {
            assert!((v - e).abs() <= 1e-15, "{w}");
        }
    }

    #[test]
    fn cc_rel_l2_error_known_pair() {
        // Non-square, so an axis swap would change the weights.
        let b = sample(17, 33, |x, y| x.exp() * (2.0 * y).cos() + 0.5);
        for eps in [1e-3, 0.25, -0.5] {
            let a = (1.0 + eps) * &b;
            let err = cc_rel_l2_error(a.view(), b.view());
            // Absolute: a − b = ε b carries round-off of order 1e-16 in b, not in ε.
            assert!((err - eps.abs()).abs() <= 1e-14, "eps={eps}: {err}");
        }
        assert_eq!(cc_rel_l2_error(b.view(), b.view()), 0.0);
    }

    #[test]
    fn cc_rel_l2_error_accepts_strided_views() {
        let b = sample(9, 17, |x, y| x.exp() * (2.0 * y).cos() + 0.5);
        let a = 1.25 * &b;
        let expected = cc_rel_l2_error(a.view(), b.view());
        // Every other row of a larger array, offset by one column: a strided view.
        let embed = |v: &Array2<f64>| {
            let mut big = Array2::<f64>::zeros((2 * 9, 17 + 3));
            big.slice_mut(s![..;2, 1..=17]).assign(v);
            big
        };
        let (big_a, big_b) = (embed(&a), embed(&b));
        let (va, vb) = (big_a.slice(s![..;2, 1..=17]), big_b.slice(s![..;2, 1..=17]));
        let err = cc_rel_l2_error(va, vb);
        assert!((err - expected).abs() <= 1e-15, "{err} vs {expected}");
        // A transposed view swaps the axes and their weights; the error is unchanged.
        let err = cc_rel_l2_error(a.t(), b.t());
        assert!((err - expected).abs() <= 1e-14, "{err} vs {expected}");
    }

    #[test]
    fn cc_rel_l2_error_extreme_magnitudes() {
        // Squaring unscaled values would underflow (1e-170) or overflow (1e170) to 0/0 or
        // ∞/∞, both NaN.
        let b = sample(9, 5, |x, y| x.exp() * (2.0 * y).cos() + 0.5);
        for scale in [1e-170, 1e170] {
            let bs = scale * &b;
            let a = 1.25 * &bs;
            let err = cc_rel_l2_error(a.view(), bs.view());
            assert!((err - 0.25).abs() <= 1e-14, "scale={scale}: {err}");
        }
        // a far larger than b: with max |b| as the only scale, (a − b)/s ≈ 1e200 would
        // overflow when squared and give infinity instead of about 1e200.
        for (sa, sb) in [(1e200, 1.0), (1.0, 1e-200)] {
            let (a, bs) = (sa * &b, sb * &b);
            let err = cc_rel_l2_error(a.view(), bs.view());
            let expected = sa / sb - 1.0;
            assert!(
                (err - expected).abs() <= 1e-14 * expected,
                "sa={sa} sb={sb}: {err}"
            );
        }
        let z = Array2::<f64>::zeros((3, 3));
        assert!(cc_rel_l2_error(z.view(), z.view()).is_nan());
        let one = Array2::from_elem((3, 3), 1.0);
        assert_eq!(cc_rel_l2_error(one.view(), z.view()), f64::INFINITY);
    }

    #[test]
    #[should_panic(expected = "clencurt: n must be >= 2")]
    fn clencurt_panics_below_two() {
        clencurt(1);
    }

    #[test]
    #[should_panic(expected = "cc_rel_l2_error: shapes differ")]
    fn cc_rel_l2_error_panics_on_shape_mismatch() {
        cc_rel_l2_error(Array2::zeros((3, 3)).view(), Array2::zeros((3, 4)).view());
    }

    #[test]
    #[should_panic(expected = "cc_rel_l2_error: n must be >= 2")]
    fn cc_rel_l2_error_panics_for_n_below_2() {
        let v = Array2::zeros((1, 3));
        cc_rel_l2_error(v.view(), v.view());
    }

    #[test]
    #[should_panic(expected = "cc_rel_l2_error: n must be >= 2")]
    fn cc_rel_l2_error_panics_for_n_y_below_2() {
        let v = Array2::zeros((3, 1));
        cc_rel_l2_error(v.view(), v.view());
    }

    #[cfg(feature = "chebyshev")]
    #[test]
    fn clencurt_matches_rlst_weights() {
        use super::super::clenshaw_curtis;
        for n in 2..=129 {
            let diff = (&clencurt(n) - &clenshaw_curtis(n))
                .iter()
                .fold(0.0_f64, |m, d| m.max(d.abs()));
            assert!(
                diff <= 1e-14,
                "n={n}: max |clencurt − clenshaw_curtis| = {diff}"
            );
        }
    }

    #[cfg(feature = "chebyshev")]
    #[test]
    fn cc_rel_l2_matches_gated() {
        use super::super::{clenshaw_curtis, rel_l2_error};
        type Field = fn(f64, f64) -> f64;
        let fields: [(usize, usize, Field, Field); 3] = [
            (9, 9, |x, y| x * x + y, |x, y| x * x + y + 0.1 * x * y),
            (
                17,
                33,
                |x, y| (x + 2.0 * y).sin(),
                |x, y| (x + 2.0 * y).sin() + 0.01 * x.exp(),
            ),
            (
                65,
                33,
                |x, y| (1.0 - x * x) * (1.0 - y * y),
                |x, y| (3.0 * x).cos() * y,
            ),
        ];
        for (nx, ny, fa, fb) in fields {
            let (a, b) = (sample(nx, ny, fa), sample(nx, ny, fb));
            let ours = cc_rel_l2_error(a.view(), b.view());
            let (wx, wy) = (clenshaw_curtis(nx), clenshaw_curtis(ny));
            let gated = rel_l2_error(a.view(), b.view(), wx.view(), wy.view());
            assert!(
                (ours - gated).abs() <= 1e-14 * gated,
                "{nx}x{ny}: {ours} vs {gated}"
            );
        }
    }
}

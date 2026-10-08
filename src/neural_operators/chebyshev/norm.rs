//! The Clenshaw–Curtis L² norm on a tensor-product CGL grid.

use ndarray::{ArrayView1, ArrayView2};

/// Discrete L²(Ω) norm of a field on the CGL tensor grid (CONVENTIONS §12, design §3.4):
///
/// ‖v‖² ≈ Σ_{i,j} wx_i · wy_j · v[i, j]²
///
/// `v: [n_x, n_y]` is stored 'ij' (`v[i, j] = v(x_i, y_j)`); `wx: [n_x]` and `wy: [n_y]`
/// are the per-axis weights, normally [`clenshaw_curtis`](super::clenshaw_curtis). Exact
/// when v² is a polynomial of degree ≤ n_x − 1 in x and ≤ n_y − 1 in y.
///
/// # Panics
/// If `v`'s shape is not `[wx.len(), wy.len()]`.
pub fn l2_norm(v: ArrayView2<f64>, wx: ArrayView1<f64>, wy: ArrayView1<f64>) -> f64 {
    assert_eq!(
        v.dim(),
        (wx.len(), wy.len()),
        "l2_norm: field shape must be [wx.len(), wy.len()]"
    );
    v.indexed_iter()
        .map(|((i, j), x)| wx[i] * wy[j] * x * x)
        .sum::<f64>()
        .sqrt()
}

/// Relative L² error ‖a − b‖ / ‖b‖ with [`l2_norm`], `b` the reference
/// (CONVENTIONS §12).
///
/// `a` and `b` are `[n_x, n_y]` fields stored 'ij'; `wx: [n_x]`, `wy: [n_y]`. Returns
/// NaN or infinity when ‖b‖ = 0.
///
/// # Panics
/// If `a` and `b` differ in shape, or their shape is not `[wx.len(), wy.len()]`.
pub fn rel_l2_error(
    a: ArrayView2<f64>,
    b: ArrayView2<f64>,
    wx: ArrayView1<f64>,
    wy: ArrayView1<f64>,
) -> f64 {
    assert_eq!(a.dim(), b.dim(), "rel_l2_error: shapes differ");
    l2_norm((&a - &b).view(), wx, wy) / l2_norm(b, wx, wy)
}

#[cfg(test)]
mod tests {
    use super::super::{clenshaw_curtis, nodes};
    use super::*;
    use ndarray::{Array2, s};

    #[test]
    fn l2_norm_of_constant_is_area() {
        let (nx, ny) = (9, 17);
        let (wx, wy) = (clenshaw_curtis(nx), clenshaw_curtis(ny));
        let v = Array2::from_elem((nx, ny), 3.0);
        // ‖3‖² = 9 · |Ω| = 36 on [−1, 1]².
        let norm = l2_norm(v.view(), wx.view(), wy.view());
        assert!((norm - 6.0).abs() <= 1e-14, "{norm}");

        // A strided slice of a larger array works without copying.
        let big = Array2::from_elem((2 * nx, ny + 3), 3.0);
        let norm = l2_norm(big.slice(s![..;2, 1..=ny]), wx.view(), wy.view());
        assert!((norm - 6.0).abs() <= 1e-14, "{norm}");
    }

    #[test]
    fn l2_norm_of_separable_polynomial_is_exact() {
        // v = x² y³: ∫∫ x⁴ y⁶ = (2/5)(2/7); degrees 4 and 6 need n_x ≥ 5, n_y ≥ 7.
        let (nx, ny) = (5, 9);
        let (x, y) = (nodes(nx), nodes(ny));
        let (wx, wy) = (clenshaw_curtis(nx), clenshaw_curtis(ny));
        let v = Array2::from_shape_fn((nx, ny), |(i, j)| x[i].powi(2) * y[j].powi(3));
        let exact = (2.0 / 5.0 * 2.0 / 7.0_f64).sqrt();
        let norm = l2_norm(v.view(), wx.view(), wy.view());
        assert!((norm - exact).abs() <= 1e-14, "{norm} vs {exact}");

        let err = rel_l2_error((2.0 * &v).view(), v.view(), wx.view(), wy.view());
        assert!((err - 1.0).abs() <= 1e-14, "{err}");
    }

    #[test]
    #[should_panic(expected = "field shape must be")]
    fn l2_norm_panics_on_shape_mismatch() {
        let w = clenshaw_curtis(3);
        l2_norm(Array2::zeros((3, 4)).view(), w.view(), w.view());
    }

    #[test]
    #[should_panic(expected = "shapes differ")]
    fn rel_l2_error_panics_on_shape_mismatch() {
        let w = clenshaw_curtis(3);
        rel_l2_error(
            Array2::zeros((3, 3)).view(),
            Array2::zeros((3, 4)).view(),
            w.view(),
            w.view(),
        );
    }
}

//! Barycentric transfer matrices: polynomial on CGL nodes, Floater–Hormann on uniform nodes.

use ndarray::{Array1, Array2};

/// Second-kind barycentric weights on CGL nodes: (−1)^j, endpoints halved.
///
/// Ascending and descending order give the same weights up to a global sign, which
/// cancels in the barycentric formula.
pub fn cheb_bary_weights(n: usize) -> Array1<f64> {
    let mut w = Array1::from_iter((0..n).map(|j| if j % 2 == 0 { 1.0 } else { -1.0 }));
    w[0] *= 0.5;
    w[n - 1] *= 0.5;
    w
}

/// Floater–Hormann weights of degree `d` on arbitrary ascending nodes (Floater & Hormann
/// 2007, eq. (18)):
/// w_k = (−1)^(k−d) Σ_{i ∈ J_k} Π_{j = i, j ≠ k}^{i + d} 1 / |x_k − x_j|,
/// J_k = {i : max(0, k − d) ≤ i ≤ min(k, N − d)}, N = len − 1.
///
/// Panics if `d > N`.
pub fn fh_weights(x: &Array1<f64>, d: usize) -> Array1<f64> {
    let nn = x.len() - 1;
    assert!(d <= nn, "FH degree {d} exceeds N = {nn}");
    Array1::from_iter((0..=nn).map(|k| {
        let lo = k.saturating_sub(d);
        let hi = k.min(nn - d);
        let mut sum = 0.0;
        for i in lo..=hi {
            let mut prod = 1.0;
            for j in i..=i + d {
                if j != k {
                    prod /= (x[k] - x[j]).abs();
                }
            }
            sum += prod;
        }
        if (k + d).is_multiple_of(2) { sum } else { -sum }
    }))
}

/// Dense barycentric evaluation matrix (len(tgt) × len(src)) of the rational interpolant
/// with weights `w` on nodes `src`. A target equal to a source node gets the unit row.
pub fn bary_matrix(src: &Array1<f64>, w: &Array1<f64>, tgt: &Array1<f64>) -> Array2<f64> {
    let (m, n) = (tgt.len(), src.len());
    let mut t = Array2::zeros((m, n));
    for i in 0..m {
        if let Some(k) = src.iter().position(|&x| x == tgt[i]) {
            t[[i, k]] = 1.0;
            continue;
        }
        let mut denom = 0.0;
        for k in 0..n {
            let c = w[k] / (tgt[i] - src[k]);
            t[[i, k]] = c;
            denom += c;
        }
        for k in 0..n {
            t[[i, k]] /= denom;
        }
    }
    t
}

/// Chebyshev → uniform matrix T_cu (s × n).
pub fn t_cu(cheb: &Array1<f64>, uni: &Array1<f64>) -> Array2<f64> {
    bary_matrix(cheb, &cheb_bary_weights(cheb.len()), uni)
}

/// Uniform → Chebyshev Floater–Hormann matrix T_uc (n × s) of degree `d`.
pub fn t_uc(uni: &Array1<f64>, cheb: &Array1<f64>, d: usize) -> Array2<f64> {
    bary_matrix(uni, &fh_weights(uni, d), cheb)
}

/// Apply a 1D transfer along both axes: T F Tᵀ.
pub fn apply2d(t: &Array2<f64>, f: &Array2<f64>) -> Array2<f64> {
    t.dot(f).dot(&t.t())
}

/// 1D Lebesgue constant of a transfer matrix: max_i Σ_k |T_ik|. The 2D tensor operator's
/// is its square.
pub fn lebesgue(t: &Array2<f64>) -> f64 {
    t.rows()
        .into_iter()
        .map(|r| r.iter().map(|v| v.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grids::{cheb_nodes, uniform_nodes};

    fn cheb_t(k: usize, x: f64) -> f64 {
        (k as f64 * x.clamp(-1.0, 1.0).acos()).cos()
    }

    /// Lagrange basis by the product formula, independent of the barycentric code.
    fn lagrange_matrix(src: &Array1<f64>, tgt: &Array1<f64>) -> Array2<f64> {
        Array2::from_shape_fn((tgt.len(), src.len()), |(i, k)| {
            let mut p = 1.0;
            for j in 0..src.len() {
                if j != k {
                    p *= (tgt[i] - src[j]) / (src[k] - src[j]);
                }
            }
            p
        })
    }

    fn max_abs(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        (a - b).iter().fold(0.0, |m, v| m.max(v.abs()))
    }

    // Acceptance: FH with d = s − 1 is polynomial interpolation on uniform nodes, 1e-10, s ≤ 17.
    #[test]
    fn fh_full_degree_equals_polynomial_interpolation() {
        let tgt = cheb_nodes(41);
        for s in 2..=17 {
            let x = uniform_nodes(s);
            let fh = t_uc(&x, &tgt, s - 1);
            let lag = lagrange_matrix(&x, &tgt);
            let err = max_abs(&fh, &lag);
            assert!(err < 1e-10, "s={s}: {err:e}");
        }
    }

    // Acceptance: FH of degree d reproduces polynomials of degree ≤ d, 1e-12.
    #[test]
    fn fh_reproduces_polynomials_up_to_degree_d() {
        for (s, n) in [
            (16, 17),
            (17, 17),
            (32, 33),
            (64, 65),
            (128, 129),
            (256, 257),
        ] {
            let x = uniform_nodes(s);
            let tgt = cheb_nodes(n);
            for d in 0..=8 {
                let t = t_uc(&x, &tgt, d);
                for m in 0..=d {
                    let f = x.mapv(|x| cheb_t(m, x));
                    let got = t.dot(&f);
                    let err = got
                        .iter()
                        .zip(&tgt)
                        .fold(0.0_f64, |e, (g, &y)| e.max((g - cheb_t(m, y)).abs()));
                    assert!(err < 1e-12, "s={s} d={d} m={m}: {err:e}");
                }
            }
        }
    }

    // Acceptance: T_cu reproduces polynomials of degree < n, 1e-12 (Chebyshev basis T_k).
    #[test]
    fn t_cu_reproduces_polynomials_below_degree_n() {
        for n in [5, 17, 33, 65, 129, 257] {
            let x = cheb_nodes(n);
            for s in [n - 1, n] {
                let u = uniform_nodes(s);
                let t = t_cu(&x, &u);
                for k in 0..n {
                    let f = x.mapv(|x| cheb_t(k, x));
                    let got = t.dot(&f);
                    let err = got
                        .iter()
                        .zip(&u)
                        .fold(0.0_f64, |e, (g, &y)| e.max((g - cheb_t(k, y)).abs()));
                    assert!(err < 1e-12, "n={n} s={s} k={k}: {err:e}");
                }
            }
        }
    }

    // Closed-form CGL weights against the general w_k = 1 / Π_{j≠k}(x_k − x_j), up to scale.
    #[test]
    fn cheb_weights_match_general_formula() {
        for n in [3, 9, 17] {
            let x = cheb_nodes(n);
            let general = Array1::from_iter((0..n).map(|k| {
                let mut p = 1.0;
                for j in 0..n {
                    if j != k {
                        p *= x[k] - x[j];
                    }
                }
                1.0 / p
            }));
            let closed = cheb_bary_weights(n);
            let scale = general[0] / closed[0];
            for k in 0..n {
                assert!(
                    (general[k] / scale - closed[k]).abs() < 1e-12,
                    "n={n} k={k}"
                );
            }
        }
    }

    #[test]
    fn coincident_target_gives_unit_row() {
        let u = uniform_nodes(33);
        let c = cheb_nodes(33);
        let t = t_uc(&u, &c, 3);
        assert_eq!(t[[16, 16]], 1.0);
        assert_eq!(t.row(16).sum(), 1.0);
    }
}

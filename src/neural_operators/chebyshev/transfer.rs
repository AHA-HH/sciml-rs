//! Grid transfers between the CGL grid and the model's uniform grid (CONVENTIONS §12).
//!
//! Both transfers are rational barycentric interpolants, built once per (n, s) as dense
//! 1D matrices in f64 and applied along each axis, Tx F Tyᵀ ([`apply`]):
//! - Chebyshev → uniform, T_cu ([`cheb_to_uniform`]): polynomial interpolation on the CGL
//!   nodes, second-kind weights.
//! - Uniform → Chebyshev, T_uc ([`uniform_to_cheb`]): Floater–Hormann interpolation of
//!   degree d on the uniform nodes.
//!
//! Production uses d = 2 and s = n − 1 (design §12, decisions 5 and 8); both are
//! arguments so the tests can vary them. Nodes and weights are closed forms, so this
//! module needs only ndarray and is built without the `chebyshev` feature (decision 13).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::f64::consts::PI;

use super::check_n;

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
/// endpoints (CONVENTIONS §12), without RLST.
///
/// With N = n − 1 and θ = πj/N, −cos θ = sin(θ − π/2) = sin(π(2j − N)/(2N)), which is
/// the form computed here. It gives exact ±1, an exact 0 for odd n, and exact
/// antisymmetry x_j = −x_{N−j}. It agrees with `chebyshev::nodes` (feature
/// `chebyshev`) to round-off. Returns shape `[n]`.
///
/// # Panics
/// If `n < 2`.
pub(crate) fn cgl_nodes(n: usize) -> Array1<f64> {
    check_n("cgl_nodes", n);
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
    let x = cgl_nodes(n);
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
    bary_matrix(
        x.view(),
        fh_weights(x.view(), d).view(),
        cgl_nodes(n).view(),
    )
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

#[cfg(test)]
mod tests {
    use super::*;

    /// f sampled on the tensor grid xs × ys, stored 'ij'.
    fn sample(xs: &Array1<f64>, ys: &Array1<f64>, f: impl Fn(f64, f64) -> f64) -> Array2<f64> {
        Array2::from_shape_fn((xs.len(), ys.len()), |(i, j)| f(xs[i], ys[j]))
    }

    fn sin_sin(x: f64, y: f64) -> f64 {
        (PI * x).sin() * (PI * y).sin()
    }

    fn bubble_exp(x: f64, y: f64) -> f64 {
        (1.0 - x * x) * (1.0 - y * y) * (x + 2.0 * y).exp()
    }

    /// Trapezoidal weights on the `s` uniform nodes.
    fn trapezoid(s: usize) -> Array1<f64> {
        let h = 2.0 / (s - 1) as f64;
        let mut w = Array1::from_elem(s, h);
        w[0] = h / 2.0;
        w[s - 1] = h / 2.0;
        w
    }

    /// Relative trapezoidal L² error on the uniform tensor grid, `b` the reference.
    fn trapezoid_rel_l2(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let (wx, wy) = (trapezoid(a.nrows()), trapezoid(a.ncols()));
        let norm2 = |v: &Array2<f64>| -> f64 {
            v.indexed_iter()
                .map(|((i, j), x)| wx[i] * wy[j] * x * x)
                .sum()
        };
        (norm2(&(a - b)) / norm2(b)).sqrt()
    }

    /// 1D Lebesgue constant of a transfer matrix: max_i Σ_k |T_ik|.
    fn lebesgue(t: &Array2<f64>) -> f64 {
        t.rows()
            .into_iter()
            .map(|r| r.iter().map(|v| v.abs()).sum::<f64>())
            .fold(0.0, f64::max)
    }

    #[test]
    fn uniform_grid_has_endpoints_and_n_minus_1_points() {
        for n in [3, 9, 33, 65, 129, 257] {
            let s = n - 1;
            let x = uniform_nodes(s);
            assert_eq!(x.len(), s, "n={n}");
            assert_eq!(x[0], -1.0, "n={n}");
            assert_eq!(x[s - 1], 1.0, "n={n}");
            let h = 2.0 / (s - 1) as f64;
            for j in 1..s {
                assert!(x[j] > x[j - 1], "n={n}: not ascending at j={j}");
                let gap = x[j] - x[j - 1];
                assert!((gap - h).abs() <= 1e-15, "n={n} j={j}: spacing {gap}");
            }
        }
    }

    #[test]
    fn cgl_closed_form_is_ascending_and_antisymmetric() {
        for n in [2, 3, 9, 33, 64, 257] {
            let x = cgl_nodes(n);
            assert_eq!(x[0], -1.0, "n={n}");
            assert_eq!(x[n - 1], 1.0, "n={n}");
            for j in 0..n {
                assert_eq!(x[j], -x[n - 1 - j], "n={n} j={j}");
                let cos_form = -(PI * j as f64 / (n - 1) as f64).cos();
                assert!((x[j] - cos_form).abs() <= 1e-15, "n={n} j={j}");
                if j > 0 {
                    assert!(x[j] > x[j - 1], "n={n}: not ascending at j={j}");
                }
            }
        }
    }

    #[test]
    fn fh_reproduces_polynomials_to_degree_d() {
        for s in [16, 32, 64] {
            let u = uniform_nodes(s);
            let c = cgl_nodes(s + 1);
            for d in 0..=4 {
                let t = uniform_to_cheb(s, s + 1, d);
                for k in 0..=d as i32 {
                    let got = t.dot(&u.mapv(|x| x.powi(k)));
                    let err = got
                        .iter()
                        .zip(&c)
                        .fold(0.0_f64, |e, (g, &y)| e.max((g - y.powi(k)).abs()));
                    assert!(err <= 1e-12, "s={s} d={d} k={k}: {err:e}");
                }
            }
        }
    }

    #[test]
    fn transfers_preserve_ij_layout() {
        let (nx, ny) = (9, 17);
        let (sx, sy) = (nx - 1, ny - 1);
        let (cx, cy) = (cgl_nodes(nx), cgl_nodes(ny));
        let (ux, uy) = (uniform_nodes(sx), uniform_nodes(sy));
        let f = |x: f64, y: f64| x + 10.0 * y;

        let on_uniform = apply(
            cheb_to_uniform(nx, sx).view(),
            cheb_to_uniform(ny, sy).view(),
            sample(&cx, &cy, f).view(),
        );
        assert_eq!(on_uniform.dim(), (sx, sy));
        let err = (&on_uniform - &sample(&ux, &uy, f))
            .iter()
            .fold(0.0_f64, |e, v| e.max(v.abs()));
        assert!(err <= 1e-12, "cheb -> uniform: {err:e}");

        let on_cheb = apply(
            uniform_to_cheb(sx, nx, 2).view(),
            uniform_to_cheb(sy, ny, 2).view(),
            sample(&ux, &uy, f).view(),
        );
        assert_eq!(on_cheb.dim(), (nx, ny));
        let err = (&on_cheb - &sample(&cx, &cy, f))
            .iter()
            .fold(0.0_f64, |e, v| e.max(v.abs()));
        assert!(err <= 1e-12, "uniform -> cheb: {err:e}");
    }

    #[test]
    fn coincident_target_gives_unit_row() {
        // Both grids contain ±1, so the first and last rows are unit vectors.
        for t in [cheb_to_uniform(33, 32), uniform_to_cheb(32, 33, 2)] {
            let (m, n) = t.dim();
            assert_eq!(t[[0, 0]], 1.0);
            assert_eq!(t.row(0).sum(), 1.0);
            assert_eq!(t[[m - 1, n - 1]], 1.0);
            assert_eq!(t.row(m - 1).sum(), 1.0);
        }
    }

    // Design §11: within 1e-12 relative; Phase 0 table 1 measured ≤ 1.0e-15.
    #[test]
    fn cheb_to_uniform_on_analytic_functions() {
        for n in [33, 65, 129] {
            let s = n - 1;
            let t = cheb_to_uniform(n, s);
            let (c, u) = (cgl_nodes(n), uniform_nodes(s));
            for (name, f) in [
                ("sin(pi x) sin(pi y)", sin_sin as fn(f64, f64) -> f64),
                ("(1-x^2)(1-y^2)e^(x+2y)", bubble_exp),
            ] {
                let got = apply(t.view(), t.view(), sample(&c, &c, f).view());
                let err = trapezoid_rel_l2(&got, &sample(&u, &u, f));
                println!("T_cu n={n} s={s} {name}: {err:.1e}");
                assert!(err <= 1e-12, "n={n} {name}: {err:e}");
            }
        }
    }

    // Phase 0 T3 table 4: 3.32 / 3.91 / 4.38 / 4.87.
    #[test]
    fn lebesgue_constant_bounded() {
        for n in [33, 65, 129, 257] {
            let lambda = lebesgue(&uniform_to_cheb(n - 1, n, 2));
            println!("Lambda(T_uc) d=2 n={n}: {lambda:.2}");
            assert!(lambda <= 5.0, "n={n}: {lambda}");
        }
    }

    #[test]
    #[should_panic(expected = "uniform_nodes: s must be >= 2")]
    fn uniform_nodes_panics_below_two() {
        uniform_nodes(1);
    }

    #[test]
    #[should_panic(expected = "cheb_to_uniform: n must be >= 2")]
    fn cheb_to_uniform_panics_on_n_below_two() {
        cheb_to_uniform(1, 8);
    }

    #[test]
    #[should_panic(expected = "cheb_to_uniform: s must be >= 2")]
    fn cheb_to_uniform_panics_on_s_below_two() {
        cheb_to_uniform(9, 1);
    }

    #[test]
    #[should_panic(expected = "uniform_to_cheb: s must be >= 2")]
    fn uniform_to_cheb_panics_on_s_below_two() {
        uniform_to_cheb(1, 9, 0);
    }

    #[test]
    #[should_panic(expected = "uniform_to_cheb: n must be >= 2")]
    fn uniform_to_cheb_panics_on_n_below_two() {
        uniform_to_cheb(8, 1, 2);
    }

    #[test]
    #[should_panic(expected = "uniform_to_cheb: d must be <= s - 1")]
    fn uniform_to_cheb_panics_on_degree_above_s_minus_1() {
        uniform_to_cheb(4, 5, 4);
    }

    #[test]
    #[should_panic(expected = "apply: tx must have f.nrows() (n_x) columns")]
    fn apply_panics_on_x_mismatch() {
        let f = Array2::<f64>::zeros((9, 17));
        apply(
            cheb_to_uniform(17, 16).view(),
            cheb_to_uniform(17, 16).view(),
            f.view(),
        );
    }

    #[test]
    #[should_panic(expected = "apply: ty must have f.ncols() (n_y) columns")]
    fn apply_panics_on_y_mismatch() {
        let f = Array2::<f64>::zeros((9, 17));
        apply(
            cheb_to_uniform(9, 8).view(),
            cheb_to_uniform(9, 8).view(),
            f.view(),
        );
    }

    /// Oracle tests against RLST and the Clenshaw–Curtis norm; they need the feature.
    #[cfg(feature = "chebyshev")]
    mod rlst_oracle {
        use super::super::super::{clenshaw_curtis, nodes, rel_l2_error};
        use super::*;
        use rlst::DynArray;
        use rlst::interpolation::{
            Kind, barycentric_chebychev_weights, barycentric_evaluate_1d, chebychev_points,
        };

        #[test]
        fn cgl_closed_form_matches_nodes() {
            for n in [2, 3, 9, 33, 65, 129, 257] {
                let err = (&cgl_nodes(n) - &nodes(n))
                    .iter()
                    .fold(0.0_f64, |e, v| e.max(v.abs()));
                assert!(err <= 1e-14, "n={n}: {err:e}");
            }
        }

        #[test]
        fn cheb_bary_weights_match_closed_form() {
            for n in [2, 3, 9, 33, 257] {
                let ours = cheb_bary_weights(n);
                let desc = barycentric_chebychev_weights::<f64>(Kind::Second, n);
                let theirs = Array1::from_shape_fn(n, |j| desc[[n - 1 - j]]);
                let (ours, theirs) = (&ours / ours[0], &theirs / theirs[0]);
                let err = (&ours - &theirs)
                    .iter()
                    .fold(0.0_f64, |e, v| e.max(v.abs()));
                assert!(err <= 1e-14, "n={n}: {err:e}");
            }
        }

        #[test]
        fn cheb_to_uniform_matches_rlst_barycentric() {
            let f = |x: f64| (2.0 * x).sin() * x.exp() + 1.0 / (2.0 + x);
            for n in [33, 65, 129] {
                let s = n - 1;
                let u = uniform_nodes(s);
                let ours = cheb_to_uniform(n, s).dot(&cgl_nodes(n).mapv(f));

                let desc_nodes = chebychev_points::<f64>(Kind::Second, n);
                let values: DynArray<f64, 1> =
                    desc_nodes.iter_value().map(f).collect::<Vec<_>>().into();
                let eval: DynArray<f64, 1> = u.to_vec().into();
                let weights = barycentric_chebychev_weights::<f64>(Kind::Second, n);
                let mut theirs = DynArray::<f64, 1>::from_shape([s]);
                barycentric_evaluate_1d(&eval, &desc_nodes, &weights, &values, &mut theirs);

                let scale = ours.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                let err = (0..s).fold(0.0_f64, |e, i| e.max((ours[i] - theirs[[i]]).abs()));
                assert!(err / scale <= 1e-12, "n={n}: {:e}", err / scale);
            }
        }

        /// CC relative L² error of T_uc(d = 2) on sin(πx) sin(πy), sampled on s = n − 1.
        fn fh_error(n: usize) -> f64 {
            let (c, u) = (cgl_nodes(n), uniform_nodes(n - 1));
            let t = uniform_to_cheb(n - 1, n, 2);
            let got = apply(t.view(), t.view(), sample(&u, &u, sin_sin).view());
            let w = clenshaw_curtis(n);
            rel_l2_error(
                got.view(),
                sample(&c, &c, sin_sin).view(),
                w.view(),
                w.view(),
            )
        }

        // Phase 0 T3 table 2: 1.2e-4, 1.2e-5, 1.5e-6, observed orders about 3.3 and 3.0.
        #[test]
        fn fh_degree_two_rate() {
            let e: Vec<f64> = [33, 65, 129].into_iter().map(fh_error).collect();
            for (k, pair) in e.windows(2).enumerate() {
                let order = (pair[0] / pair[1]).log2();
                println!(
                    "FH d=2 sin sin: e={:.1e} -> {:.1e}, order {order:.2} (step {k})",
                    pair[0], pair[1]
                );
                assert!((2.5..=3.5).contains(&order), "step {k}: order {order}");
            }
        }

        // Phase 0 T3 table 2: T_uc alone on sin(πx) sin(πy) at n = 65, d = 2 is 1.2e-5.
        #[test]
        fn round_trip_matches_spike() {
            let n = 65;
            let c = cgl_nodes(n);
            let u = sample(&c, &c, sin_sin);
            let (t_cu, t_uc) = (cheb_to_uniform(n, n - 1), uniform_to_cheb(n - 1, n, 2));
            let back = apply(
                t_uc.view(),
                t_uc.view(),
                apply(t_cu.view(), t_cu.view(), u.view()).view(),
            );
            let w = clenshaw_curtis(n);
            let err = rel_l2_error(back.view(), u.view(), w.view(), w.view());
            println!("round trip n={n} d=2 sin sin: {err:.2e}");
            assert!((0.6e-5..=2.4e-5).contains(&err), "{err:e}");
        }
    }
}

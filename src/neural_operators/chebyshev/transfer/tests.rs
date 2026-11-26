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
        let x = nodes(n);
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
        let c = nodes(s + 1);
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

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |e, v| e.max(v.abs()))
}

/// 'ij' layout of both transfers on an nx × ny CGL grid and an sx × sy uniform grid,
/// checked on a linear field neither axis can confuse with the other.
fn check_ij_layout(nx: usize, ny: usize, sx: usize, sy: usize) {
    let (cx, cy) = (nodes(nx), nodes(ny));
    let (ux, uy) = (uniform_nodes(sx), uniform_nodes(sy));
    let f = |x: f64, y: f64| x + 10.0 * y;

    let on_uniform = apply(
        cheb_to_uniform(nx, sx).view(),
        cheb_to_uniform(ny, sy).view(),
        sample(&cx, &cy, f).view(),
    );
    assert_eq!(on_uniform.dim(), (sx, sy));
    let err = max_abs(&(&on_uniform - &sample(&ux, &uy, f)));
    assert!(err <= 1e-12, "cheb -> uniform: {err:e}");

    let on_cheb = apply(
        uniform_to_cheb(sx, nx, TRANSFER_DEGREE).view(),
        uniform_to_cheb(sy, ny, TRANSFER_DEGREE).view(),
        sample(&ux, &uy, f).view(),
    );
    assert_eq!(on_cheb.dim(), (nx, ny));
    let err = max_abs(&(&on_cheb - &sample(&cx, &cy, f)));
    assert!(err <= 1e-12, "uniform -> cheb: {err:e}");
}

#[test]
fn transfers_preserve_ij_layout() {
    check_ij_layout(9, 17, 8, 16);
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
        let (c, u) = (nodes(n), uniform_nodes(s));
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
fn transfers_round_trip_matches_manual_recipe() {
    for n in [9, 33] {
        let s = n - 1;
        let t = Transfers::new(n);
        assert_eq!((t.n(), t.s()), (n, s));
        let c = nodes(n);
        let u = sample(&c, &c, bubble_exp);
        // the inline recipe the four sites used before `Transfers`
        let t_cu = cheb_to_uniform(n, s);
        let t_uc = uniform_to_cheb(s, n, 2);
        let on_uniform = apply(t_cu.view(), t_cu.view(), u.view());
        let back = apply(t_uc.view(), t_uc.view(), on_uniform.view());

        assert_eq!(t.to_uniform(u.view()), on_uniform, "n={n}");
        assert_eq!(t.to_cheb(on_uniform.view()), back, "n={n}");
    }
}

#[test]
#[should_panic(expected = "Transfers::new: n must be >= 4")]
fn transfers_panic_below_min_n() {
    Transfers::new(3);
}

// A sine mode well inside what s resolves: R samples the n-point interpolant, exact
// to round-off on it (CONVENTIONS §12).
#[test]
fn restriction_exact_on_resolved_mode() {
    let n = 129;
    let mode = |x: f64, y: f64| (3.0 * PI * (x + 1.0) / 2.0).sin() * (PI * (y + 1.0)).sin();
    let c = nodes(n);
    for s in [16, 32, 64] {
        let t = Transfers::with_s(n, s);
        assert_eq!((t.n(), t.s()), (n, s));
        let u = uniform_nodes(s);
        let got = t.to_uniform(sample(&c, &c, mode).view());
        let err = max_abs(&(&got - &sample(&u, &u, mode)));
        assert!(err <= 1e-12, "s={s}: {err:e}");
    }
}

#[test]
fn restriction_rows_sum_to_one() {
    for (n, s) in [(129, 16), (129, 32), (129, 64), (257, 128), (33, 3)] {
        let t = Transfers::with_s(n, s);
        let err = t
            .t_cu
            .rows()
            .into_iter()
            .fold(0.0_f64, |e, r| e.max((r.sum() - 1.0).abs()));
        assert!(err <= 1e-12, "n={n} s={s}: {err:e}");
    }
}

#[test]
fn transfers_preserve_ij_layout_at_any_s() {
    check_ij_layout(17, 33, 5, 12);
}

#[test]
#[should_panic(expected = "Transfers::with_s: s must be >= 3")]
fn transfers_with_s_panic_below_min_s() {
    Transfers::with_s(129, 2);
}

#[test]
#[should_panic(expected = "Transfers::with_s: n must be >= 4")]
fn transfers_with_s_panic_below_min_n() {
    Transfers::with_s(3, 32);
}

/// f sampled on the tensor grid xs × ys × zs, stored 'ijk' (CONVENTIONS §12).
fn sample3(
    xs: &Array1<f64>,
    ys: &Array1<f64>,
    zs: &Array1<f64>,
    f: impl Fn(f64, f64, f64) -> f64,
) -> Array3<f64> {
    Array3::from_shape_fn((xs.len(), ys.len(), zs.len()), |(i, j, k)| {
        f(xs[i], ys[j], zs[k])
    })
}

fn max_abs3(a: &Array3<f64>) -> f64 {
    a.iter().fold(0.0_f64, |e, v| e.max(v.abs()))
}

// 'ijk' layout both ways on a non-cubic grid: a linear field no axis swap preserves.
#[test]
fn apply_nd_preserves_ijk_layout() {
    let (n, s) = ([9, 17, 5], [8, 12, 4]);
    let c: Vec<_> = n.iter().map(|&k| nodes(k)).collect();
    let u: Vec<_> = s.iter().map(|&k| uniform_nodes(k)).collect();
    let f = |x: f64, y: f64, z: f64| x + 10.0 * y + 100.0 * z;

    let t_cu: Vec<_> = (0..3).map(|a| cheb_to_uniform(n[a], s[a])).collect();
    let views: Vec<_> = t_cu.iter().map(|t| t.view()).collect();
    let got = apply_nd(&views, sample3(&c[0], &c[1], &c[2], f).view());
    assert_eq!(got.dim(), (8, 12, 4));
    let err = max_abs3(&(&got - &sample3(&u[0], &u[1], &u[2], f)));
    assert!(err <= 1e-12, "cheb -> uniform: {err:e}");

    let t_uc: Vec<_> = (0..3)
        .map(|a| uniform_to_cheb(s[a], n[a], TRANSFER_DEGREE))
        .collect();
    let views: Vec<_> = t_uc.iter().map(|t| t.view()).collect();
    let got = apply_nd(&views, sample3(&u[0], &u[1], &u[2], f).view());
    assert_eq!(got.dim(), (9, 17, 5));
    let err = max_abs3(&(&got - &sample3(&c[0], &c[1], &c[2], f)));
    assert!(err <= 1e-12, "uniform -> cheb: {err:e}");
}

// T(g ⊗ h ⊗ k) = (Tx g) ⊗ (Ty h) ⊗ (Tz k): each axis gets its own 1D matrix.
#[test]
fn apply_nd_on_separable_field_is_tensor_of_1d() {
    let (n, s) = ([9, 17, 13], [6, 10, 7]);
    let c: Vec<_> = n.iter().map(|&k| nodes(k)).collect();
    let g = [
        c[0].mapv(|x| (2.0 * x).sin() + 1.5),
        c[1].mapv(f64::exp),
        c[2].mapv(|z| 1.0 / (2.0 + z)),
    ];
    let t: Vec<_> = (0..3).map(|a| cheb_to_uniform(n[a], s[a])).collect();
    let f = Array3::from_shape_fn((n[0], n[1], n[2]), |(i, j, k)| g[0][i] * g[1][j] * g[2][k]);
    let views: Vec<_> = t.iter().map(|m| m.view()).collect();
    let got = apply_nd(&views, f.view());
    let h: Vec<_> = (0..3).map(|a| t[a].dot(&g[a])).collect();
    let want = Array3::from_shape_fn((s[0], s[1], s[2]), |(i, j, k)| h[0][i] * h[1][j] * h[2][k]);
    let err = max_abs3(&(&got - &want)) / max_abs3(&want);
    assert!(err <= 1e-14, "{err:e}");
}

// T_cu is exact below degree n and FH of degree 2 reproduces degree 2, so the round
// trip returns a field of degree ≤ 2 per axis to round-off.
#[test]
fn transfers_round_trip_3d_exact_on_quadratics() {
    let n = 17;
    let t = Transfers::new(n);
    let c = nodes(n);
    let f = |x: f64, y: f64, z: f64| (x + 2.0 * y - z).powi(2) + x * y * z - 0.5;
    let u = sample3(&c, &c, &c, f);
    let on_uniform = t.to_uniform_nd(u.view());
    assert_eq!(on_uniform.dim(), (n - 1, n - 1, n - 1));
    let back = t.to_cheb_nd(on_uniform.view());
    let err = max_abs3(&(&back - &u));
    assert!(err <= 1e-12, "{err:e}");
}

#[test]
fn restriction_exact_on_resolved_mode_3d() {
    let n = 65;
    let mode = |x: f64, y: f64, z: f64| {
        (3.0 * PI * (x + 1.0) / 2.0).sin() * (PI * (y + 1.0)).sin() * (z + 1.0) / 2.0
    };
    let c = nodes(n);
    for s in [16, 32] {
        let t = Transfers::with_s(n, s);
        let u = uniform_nodes(s);
        let got = t.to_uniform_nd(sample3(&c, &c, &c, mode).view());
        let err = max_abs3(&(&got - &sample3(&u, &u, &u, mode)));
        assert!(err <= 1e-12, "s={s}: {err:e}");
    }
}

#[test]
fn to_uniform_all_nd_matches_per_sample() {
    let n = 9;
    let t = Transfers::with_s(n, 6);
    let f = ndarray::Array4::from_shape_fn((3, n, n, n), |(m, i, j, k)| {
        ((m + 1) as f64 * (i as f64 - 2.0 * j as f64 + 0.5 * k as f64)).sin()
    });
    let all = t.to_uniform_all_nd(f.view());
    assert_eq!(all.dim(), (3, 6, 6, 6));
    for (got, src) in all.outer_iter().zip(f.outer_iter()) {
        assert_eq!(got, t.to_uniform_nd(src));
    }
}

// D2: rank 2 is routed to the 2D code, so every output is bit for bit the old one.
#[test]
fn rank_two_is_bit_identical_to_2d() {
    let (n, s) = (17, 12);
    let t = Transfers::with_s(n, s);
    let (cx, cy) = (nodes(n), nodes(9));
    let rect = sample(&cx, &cy, bubble_exp);
    let (tx, ty) = (cheb_to_uniform(n, s), cheb_to_uniform(9, 5));
    assert_eq!(
        apply_nd(&[tx.view(), ty.view()], rect.view()),
        apply(tx.view(), ty.view(), rect.view())
    );
    let square = sample(&cx, &cx, bubble_exp);
    let on_uniform = t.to_uniform(square.view());
    for f in [square.view(), square.t()] {
        assert_eq!(t.to_uniform_nd(f), t.to_uniform(f));
    }
    assert_eq!(
        t.to_cheb_nd(on_uniform.view()),
        t.to_cheb(on_uniform.view())
    );
    let batch = Array3::from_shape_fn((4, n, n), |(m, i, j)| square[[i, j]] * (m as f64 + 0.5));
    assert_eq!(
        t.to_uniform_all_nd(batch.view()),
        t.to_uniform_all(batch.view())
    );
}

#[test]
fn apply_nd_rank_one_is_matrix_vector() {
    let t = cheb_to_uniform(9, 5);
    let f = nodes(9).mapv(f64::exp);
    // a [n, 1] matrix product, so equal to the matrix-vector product to round-off
    let err = (&apply_nd(&[t.view()], f.view()) - &t.dot(&f))
        .iter()
        .fold(0.0_f64, |e, v| e.max(v.abs()));
    assert!(err <= 1e-15, "{err:e}");
}

#[test]
#[should_panic(expected = "apply_nd: one transfer per axis")]
fn apply_nd_panics_on_transfer_count() {
    let t = cheb_to_uniform(5, 4);
    apply_nd(
        &[t.view(), t.view()],
        Array3::<f64>::zeros((5, 5, 5)).view(),
    );
}

#[test]
#[should_panic(expected = "apply_nd: ts[2] must have 5 columns")]
fn apply_nd_panics_on_axis_mismatch() {
    let (t, u) = (cheb_to_uniform(5, 4), cheb_to_uniform(6, 4));
    apply_nd(
        &[t.view(), t.view(), u.view()],
        Array3::<f64>::zeros((5, 5, 5)).view(),
    );
}

#[test]
#[should_panic(expected = "Transfers::to_uniform_all_nd: f must be [N, n, ...], got rank 1")]
fn to_uniform_all_nd_panics_below_rank_two() {
    Transfers::new(5).to_uniform_all_nd(Array1::<f64>::zeros(0).view());
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
#[cfg(feature = "datagen")]
mod rlst_oracle {
    use super::super::super::cc_rel_l2_error_weighted;
    use super::super::super::points::nodes as rlst_nodes;
    use super::super::super::quadrature::clenshaw_curtis;
    use super::*;
    use rlst::DynArray;
    use rlst::interpolation::{
        Kind, barycentric_chebychev_weights, barycentric_evaluate_1d, chebychev_points,
    };

    #[test]
    fn cgl_closed_form_matches_nodes() {
        for n in [2, 3, 9, 33, 65, 129, 257] {
            let err = (&nodes(n) - &rlst_nodes(n))
                .iter()
                .fold(0.0_f64, |e, v| e.max(v.abs()));
            // 1e-14 is the loader's NODE_TOL: old dataset files load (decision 26).
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
            let ours = cheb_to_uniform(n, s).dot(&nodes(n).mapv(f));

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
        let (c, u) = (nodes(n), uniform_nodes(n - 1));
        let t = uniform_to_cheb(n - 1, n, 2);
        let got = apply(t.view(), t.view(), sample(&u, &u, sin_sin).view());
        let w = clenshaw_curtis(n);
        cc_rel_l2_error_weighted(
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

    // Phase 0 T3 table 2: T_uc alone on sin(πx) sin(πy) at n = 65, d = 2 is 1.2e-5. The
    // round trip T_uc(T_cu u) measured here has the same error, since T_cu is exact to
    // about 1e-16 on this function.
    #[test]
    fn round_trip_matches_spike() {
        let n = 65;
        let c = nodes(n);
        let u = sample(&c, &c, sin_sin);
        let (t_cu, t_uc) = (cheb_to_uniform(n, n - 1), uniform_to_cheb(n - 1, n, 2));
        let back = apply(
            t_uc.view(),
            t_uc.view(),
            apply(t_cu.view(), t_cu.view(), u.view()).view(),
        );
        let w = clenshaw_curtis(n);
        let err = cc_rel_l2_error_weighted(back.view(), u.view(), w.view(), w.view());
        println!("round trip n={n} d=2 sin sin: {err:.2e}");
        assert!((0.6e-5..=2.4e-5).contains(&err), "{err:e}");
    }
}

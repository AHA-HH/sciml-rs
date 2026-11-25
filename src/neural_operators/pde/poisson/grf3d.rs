//! Gaussian random field forcings for −Δu = f on [−1, 1]³ (3D proposal §3.3): the 2D
//! family of [`grf`](super::grf) with a third sine factor,
//!
//! f_K = (1/√S_K) Σ_{k,l,m=1}^K ξ_klm λ_klm^½ φ_k(x) φ_l(y) φ_m(z), λ_klm = (μ_klm + τ²)^(−α),
//! μ_klm = (π²/4)(k² + l² + m²), S_K = ⅛ Σ_{k,l,m=1}^K λ_klm, ξ_klm ~ N(0, 1),
//!
//! with φ_k(x) = sin(kπ(x + 1)/2), τ = [`TAU`] as in 2D and α = [`ALPHA`]. The exact solution
//! of −Δu = f_K, u = 0 on ∂Ω, has coefficients c_klm / μ_klm. Both are evaluated exactly at
//! given nodes. The draw of ξ from a seed is part of the dataset format: see [`draw_xi`].

use super::grf::{K_MAX, TAU, sine_basis};
use ndarray::{Array3, ArrayView1, ArrayView3, Axis, s};
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use std::f64::consts::PI;

/// α of the 3D covariance (−Δ + τ²)^(−α) on [−1, 1]³ (proposal §3.3, D3): 3, not 2, since
/// ε_K ~ K^(d − 2α) decays only as K⁻¹ in 3D at α = 2. τ is [`TAU`], as in 2D.
pub const ALPHA: f64 = 3.0;

/// S_∞ = ⅛ Σ_{k,l,m ≥ 1} λ_klm in 3D, which cannot be brute-forced as in 2D: the cube sums
/// to m = 80 and 160 with their (m + ½)^(3 − 2α) tail extrapolated (proposal §3.3; the
/// tail beyond 160 is 5e-6 relative, the extrapolation good to about 2e-7).
/// `eps_k_3d_matches_design_table` recomputes it.
pub const S_INF: f64 = 1.232538e-4;

/// The ChaCha8 stream of the 3D draws ([`draw_xi`]). The 2D draws ([`grf::draw_xi`](super::grf::draw_xi)) and the
/// Phase 7 training subsets use stream 0, so a 3D seed never repeats their normals.
pub const STREAM: u64 = 3;

/// The RNG behind [`draw_xi`], as recorded in 3D dataset sidecars. The versions follow
/// [`grf::RNG`](super::grf::RNG)'s pins (`rng_pins_match_2d` checks it).
pub const RNG: &str = "rand_chacha 0.10.0 ChaCha8Rng::seed_from_u64, set_stream(3), \
                          rand_distr 0.6.0 StandardNormal";

/// μ_klm = (π²/4)(k² + l² + m²), the eigenvalue of −Δ on [−1, 1]³ for the sine product
/// φ_k(x) φ_l(y) φ_m(z) (proposal §3.3). The indices start at 1.
pub fn mu(k: usize, l: usize, m: usize) -> f64 {
    PI * PI / 4.0 * ((k * k + l * l + m * m) as f64)
}

/// λ_klm = (μ_klm + τ²)^(−α), α = [`ALPHA`] (proposal §3.3).
pub fn lambda(k: usize, l: usize, m: usize) -> f64 {
    (mu(k, l, m) + TAU * TAU).powf(-ALPHA)
}

/// S_K = ⅛ Σ_{k,l,m=1}^K λ_klm, the expected mean square of the unnormalised 3D truncation
/// K (proposal §3.3). Each sine product has mean square ⅛ on [−1, 1]³.
pub fn s_k(k: usize) -> f64 {
    let mut acc = 0.0;
    for i in 1..=k {
        for j in 1..=k {
            for l in 1..=k {
                acc += lambda(i, j, l);
            }
        }
    }
    acc / 8.0
}

/// ε_K = 1 − S_K / S_∞ in 3D (proposal §3.3): 2.7e-2, 4.3e-3, 5.9e-4, 7.7e-5 at K = 8, 16,
/// 32, 64. The tolerance for training and evaluation sets is 5e-3 (D3).
pub fn eps_k(k: usize) -> f64 {
    1.0 - s_k(k) / S_INF
}

/// The standard normals ξ_klm of the 3D sample with this seed, shape
/// `[K_MAX, K_MAX, K_MAX]`, index `[k − 1, l − 1, m − 1]` (proposal §3.3).
///
/// As [`grf::draw_xi`](super::grf::draw_xi), on the ChaCha8 stream [`STREAM`] ([`RNG`]): the K_MAX³ draws fill
/// the block in row-major order (m fastest), so ξ_klm is draw number
/// ((k − 1)·K_MAX + (l − 1))·K_MAX + (m − 1), and a truncation K uses the leading K³ block.
pub fn draw_xi(seed: u64) -> Array3<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(STREAM);
    Array3::from_shape_simple_fn((K_MAX, K_MAX, K_MAX), || StandardNormal.sample(&mut rng))
}

/// The 3D forcing coefficients c_klm = ξ_klm λ_klm^½ / √S_K of the truncation K, shape
/// `[K, K, K]`, index `[k − 1, l − 1, m − 1]` (proposal §3.3).
///
/// `xi: [K_MAX, K_MAX, K_MAX]` is a draw from [`draw_xi`]; only its leading K³ block is
/// read.
///
/// # Panics
/// If `k` is 0 or above [`K_MAX`], or `xi` is not `[K_MAX, K_MAX, K_MAX]`.
pub fn forcing_coeffs(xi: ArrayView3<f64>, k: usize) -> Array3<f64> {
    assert!(
        (1..=K_MAX).contains(&k),
        "forcing_coeffs: K must be in 1..={K_MAX}, got {k}"
    );
    assert_eq!(
        xi.dim(),
        (K_MAX, K_MAX, K_MAX),
        "forcing_coeffs: xi must be [K_MAX, K_MAX, K_MAX]"
    );
    let norm = s_k(k).sqrt();
    let mut c = xi.slice(s![..k, ..k, ..k]).to_owned();
    for ((i, j, l), v) in c.indexed_iter_mut() {
        *v *= lambda(i + 1, j + 1, l + 1).sqrt() / norm;
    }
    c
}

/// The coefficients c_klm / μ_klm of the exact solution of −Δu = f, u = 0 on ∂Ω, for the 3D
/// forcing with coefficients `c: [K, K, K]` (proposal §3.3). Returns `[K, K, K]`, same
/// indexing.
///
/// # Panics
/// If `c` is not a cube.
pub fn solution_coeffs(c: ArrayView3<f64>) -> Array3<f64> {
    let k = c.dim().0;
    assert_eq!(c.dim(), (k, k, k), "solution_coeffs: c must be [K, K, K]");
    let mut u = c.to_owned();
    for ((i, j, l), v) in u.indexed_iter_mut() {
        *v /= mu(i + 1, j + 1, l + 1);
    }
    u
}

/// Evaluates the 3D sine series with coefficients `c: [K, K, K]` (index
/// `[k − 1, l − 1, m − 1]`) on the tensor grid `xs × ys × zs` by three mode products: for
/// each k the 2D product S_y C_k S_zᵀ, then S_x over the stacked results (proposal §3.3).
///
/// `xs: [n_x]`, `ys: [n_y]`, `zs: [n_z]` are any points in [−1, 1], normally the CGL nodes.
/// Returns `[n_x, n_y, n_z]`, stored 'ijk' (CONVENTIONS §12). Costs
/// O(K n³ + K² n² + K³ n) for n_x = n_y = n_z = n.
///
/// # Panics
/// If `c` is not a cube.
pub fn evaluate(
    c: ArrayView3<f64>,
    xs: ArrayView1<f64>,
    ys: ArrayView1<f64>,
    zs: ArrayView1<f64>,
) -> Array3<f64> {
    let k = c.dim().0;
    assert_eq!(c.dim(), (k, k, k), "evaluate: c must be [K, K, K]");
    let (sy, sz) = (sine_basis(ys, k), sine_basis(zs, k));
    let (ny, nz) = (ys.len(), zs.len());
    // g[a] = S_y C[a] S_zᵀ, then S_x over a with g flattened to [K, n_y n_z]
    let mut g = Array3::zeros((k, ny, nz));
    for (a, mut ga) in g.outer_iter_mut().enumerate() {
        ga.assign(&sy.dot(&c.index_axis(Axis(0), a)).dot(&sz.t()));
    }
    let g = g
        .into_shape_with_order((k, ny * nz))
        .expect("a fresh [K, n_y, n_z] is standard layout");
    sine_basis(xs, k)
        .dot(&g)
        .into_shape_with_order((xs.len(), ny, nz))
        .expect("the product [n_x, n_y n_z] is standard layout")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::chebyshev::nodes;
    use crate::neural_operators::pde::poisson::grf;
    use ndarray::Array1;

    /// The one-dimensional sine values sin(kπ(x + 1)/2), k = 1..K, at one point.
    fn sines_at(x: f64, k: usize) -> Array1<f64> {
        Array1::from_shape_fn(k, |j| ((j + 1) as f64 * PI * (x + 1.0) / 2.0).sin())
    }

    /// Max |a − b| over all entries, rank 3.
    fn max_abs_diff(a: ArrayView3<f64>, b: ArrayView3<f64>) -> f64 {
        (&a - &b).iter().fold(0.0_f64, |m, v| m.max(v.abs()))
    }

    #[test]
    fn evaluate_3d_matches_direct_sum() {
        let k = 4;
        let c = forcing_coeffs(draw_xi(5).view(), k);
        let (x, y, z) = (nodes(9), nodes(7), nodes(5));
        let fast = evaluate(c.view(), x.view(), y.view(), z.view());
        assert_eq!(fast.dim(), (9, 7, 5));
        let direct = Array3::from_shape_fn((9, 7, 5), |(i, j, l)| {
            let (sx, sy, sz) = (sines_at(x[i], k), sines_at(y[j], k), sines_at(z[l], k));
            c.indexed_iter()
                .map(|((a, b, d), v)| v * sx[a] * sy[b] * sz[d])
                .sum::<f64>()
        });
        let scale = direct.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let err = max_abs_diff(fast.view(), direct.view());
        println!("evaluate against the direct sum: max |diff| {err:.1e}, scale {scale:.2}");
        assert!(err <= 1e-14 * scale, "{err:e} against scale {scale:e}");
    }

    #[test]
    fn eps_k_3d_matches_design_table() {
        // S_∞ by extrapolating the cube sums' tail, which falls as (m + ½)^(3 − 2α) (proposal
        // §3.3): S_∞ ≈ S_b + (S_b − S_a) L_b^(−3) / (L_a^(−3) − L_b^(−3)), L = m + ½. Two
        // pairs of cutoffs agreeing shows the extrapolation has converged.
        let sums = [40, 80, 160].map(|m| (m as f64 + 0.5, s_k(m)));
        let extrapolate = |(la, sa): (f64, f64), (lb, sb): (f64, f64)| {
            let p = 2.0 * ALPHA - 3.0;
            sb + (sb - sa) * lb.powf(-p) / (la.powf(-p) - lb.powf(-p))
        };
        let coarse = extrapolate(sums[0], sums[1]);
        let s_inf = extrapolate(sums[1], sums[2]);
        let (spread, rel) = (
            (coarse - s_inf).abs() / s_inf,
            (s_inf - S_INF).abs() / S_INF,
        );
        let tail = (s_inf - sums[2].1) / s_inf;
        println!(
            "3D S_∞ {s_inf:.7e} (tail beyond m = 160 {tail:.1e} relative; cutoffs 40/80 \
             differ by {spread:.1e}), relative to S_INF {rel:.1e}"
        );
        // measured 7.9e-7 (deterministic); the 80/160 value is then good to about 2e-7
        assert!(spread <= 1e-6, "extrapolation not converged: {spread:e}");
        assert!(rel <= 1e-6, "S_∞ recomputed {s_inf:e}, constant {S_INF:e}");

        // Proposal §3.3, the α = 3 row, two significant digits.
        for (k, table) in [(8, 2.7e-2), (16, 4.3e-3), (32, 5.9e-4), (64, 7.7e-5)] {
            let e = eps_k(k);
            println!("3D ε_{k} = {e:.3e} (proposal {table:.1e})");
            assert!(
                (e - table).abs() <= 0.05 * table,
                "ε_{k} = {e:e}, proposal {table:e}"
            );
        }
    }

    #[test]
    fn draws_3d_are_nested() {
        let xi = draw_xi(11);
        let c8 = forcing_coeffs(xi.view(), 8);
        let c16 = forcing_coeffs(xi.view(), 16);
        let leading = c16.slice(s![..8, ..8, ..8]).to_owned() * (s_k(16) / s_k(8)).sqrt();
        let scale = c8.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let err = max_abs_diff(c8.view(), leading.view());
        assert!(err <= 1e-14 * scale, "{err:e} against scale {scale:e}");
    }

    #[test]
    fn stream_3d_differs_from_2d() {
        let (xi3, xi2) = (draw_xi(0), grf::draw_xi(0));
        assert_eq!(xi3.dim(), (K_MAX, K_MAX, K_MAX));
        // The first K_MAX² normals of each stream, in draw order.
        let first3: Vec<f64> = xi3.iter().take(K_MAX * K_MAX).copied().collect();
        let first2: Vec<f64> = xi2.iter().copied().collect();
        assert!(first3.iter().zip(&first2).all(|(a, b)| a != b));
        // Reproducible: the same seed gives the same draws.
        assert_eq!(xi3, draw_xi(0));
    }

    #[test]
    fn mean_square_3d_is_one() {
        // Parseval: each φ_klm has mean square ⅛ on Ω, so the mean square of f_K is
        // ⅛ Σ c_klm². Its expectation is 1 for every K.
        let k = 16;
        // 30 samples (64³ normals each) keep it fast; 4 s.e. ≈ 0.3 still catches ¼ for ⅛.
        let n_samples = 30;
        let ms: Vec<f64> = (0..n_samples)
            .map(|seed| {
                let c = forcing_coeffs(draw_xi(seed).view(), k);
                c.iter().map(|v| v * v).sum::<f64>() / 8.0
            })
            .collect();
        let nf = n_samples as f64;
        let mean = ms.iter().sum::<f64>() / nf;
        let var = ms.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (nf - 1.0);
        let se = (var / nf).sqrt();
        println!("3D mean square at K = {k}, N = {n_samples}: {mean:.4} ± {se:.4}");
        assert!(
            (mean - 1.0).abs() <= 4.0 * se,
            "mean square {mean} ± {se}, more than 4 s.e. from 1"
        );
    }

    #[test]
    fn field_and_solution_3d_vanish_on_boundary() {
        let n = 17;
        let x = nodes(n);
        let c = forcing_coeffs(draw_xi(3).view(), 8);
        for field in [
            evaluate(c.view(), x.view(), x.view(), x.view()),
            evaluate(
                solution_coeffs(c.view()).view(),
                x.view(),
                x.view(),
                x.view(),
            ),
        ] {
            let scale = field.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            for axis in 0..3 {
                for i in [0, n - 1] {
                    let face = field.index_axis(Axis(axis), i);
                    let e = face.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                    assert!(
                        e <= 1e-14 * scale,
                        "axis {axis}, index {i}: {e:e} against {scale:e}"
                    );
                }
            }
        }
    }

    #[test]
    fn solution_3d_solves_poisson() {
        // −Δu by the 7-point stencil at interior points equals f to O(h²).
        let c = forcing_coeffs(draw_xi(9).view(), 8);
        let u = solution_coeffs(c.view());
        let h = 1e-3;
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for p in [[0.1, -0.3, 0.55], [-0.7, 0.2, 0.0], [0.45, 0.8, -0.6]] {
            let axis = |v: f64| Array1::from(vec![v - h, v, v + h]);
            let (x, y, z) = (axis(p[0]), axis(p[1]), axis(p[2]));
            let uu = evaluate(u.view(), x.view(), y.view(), z.view());
            let lap = (uu[[0, 1, 1]]
                + uu[[2, 1, 1]]
                + uu[[1, 0, 1]]
                + uu[[1, 2, 1]]
                + uu[[1, 1, 0]]
                + uu[[1, 1, 2]]
                - 6.0 * uu[[1, 1, 1]])
                / (h * h);
            let f = evaluate(c.view(), x.view(), y.view(), z.view())[[1, 1, 1]];
            worst = worst.max((-lap - f).abs());
            scale = scale.max(f.abs());
        }
        println!("3D −Δu − f by the stencil: {worst:.1e} against max |f| {scale:.2}");
        assert!(worst <= 1e-4 * scale, "{worst:e} against {scale:e}");
    }

    #[test]
    fn draw_order_is_row_major() {
        // ξ_klm is draw number ((k − 1)·K_MAX + (l − 1))·K_MAX + (m − 1) of the stream.
        let xi = draw_xi(4);
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        rng.set_stream(STREAM);
        let draws: Vec<f64> = (0..K_MAX * K_MAX + 2)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        assert_eq!(xi[[0, 0, 1]], draws[1]);
        assert_eq!(xi[[0, 1, 0]], draws[K_MAX]);
        assert_eq!(xi[[1, 0, 1]], draws[K_MAX * K_MAX + 1]);
    }

    #[test]
    fn rng_pins_match_2d() {
        // Both sidecar strings name the same pinned crate versions.
        for pin in ["rand_chacha 0.10.0", "rand_distr 0.6.0 StandardNormal"] {
            assert!(grf::RNG.contains(pin) && RNG.contains(pin), "{pin}");
        }
    }

    #[test]
    #[should_panic(expected = "K must be in 1..=64")]
    fn k_above_k_max_panics_3d() {
        forcing_coeffs(draw_xi(0).view(), K_MAX + 1);
    }
}

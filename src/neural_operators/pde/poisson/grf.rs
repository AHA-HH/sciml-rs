//! Gaussian random field forcings for −Δu = f on [−1, 1]² (design §3.3).
//!
//! f_K is the Karhunen–Loève expansion of the covariance (−Δ + τ²)^(−α) in the Dirichlet
//! sine basis φ_kl(x, y) = sin(kπ(x + 1)/2) sin(lπ(y + 1)/2), truncated at k, l ≤ K and
//! normalised to unit expected mean square:
//!
//! f_K = (1/√S_K) Σ_{k,l=1}^K ξ_kl λ_kl^½ φ_kl, λ_kl = (μ_kl + τ²)^(−α),
//! μ_kl = (π²/4)(k² + l²), S_K = ¼ Σ_{k,l=1}^K λ_kl, ξ_kl ~ N(0, 1).
//!
//! φ_kl is an eigenfunction of −Δ with eigenvalue μ_kl and vanishes on ∂Ω, so the exact
//! solution of −Δu = f_K, u = 0 on ∂Ω, is the same series with coefficients c_kl / μ_kl
//! (design §4.3). Both are evaluated exactly at given nodes, with no FFT or interpolation.
//!
//! The draw of ξ from a seed is part of the dataset format: see [`draw_xi`].

use ndarray::{Array2, ArrayView1, ArrayView2, s};
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use std::f64::consts::PI;

/// τ of the covariance (−Δ + τ²)^(−α), as in Li et al.'s Darcy data (design §3.3).
pub const TAU: f64 = 3.0;

/// α of the covariance (−Δ + τ²)^(−α) (design §3.3).
pub const ALPHA: f64 = 2.0;

/// The largest production truncation, and the side of the ξ block every sample draws
/// (design §3.3, §12 decision 3).
pub const K_MAX: usize = 64;

/// S_∞ = ¼ Σ_{k,l ≥ 1} λ_kl, the expected mean square of the untruncated, unnormalised
/// series.
///
/// Computed in Phase 0 T3 (`spikes/transfers/src/grf.rs`, `s_inf(3000)`) as the sum over
/// k, l ≤ 3000 plus the integral of λ's r⁻⁴ asymptote outside that square, about 1e-6
/// relative. `eps_k_matches_design_table` recomputes it.
pub const S_INF: f64 = 4.984158e-3;

/// The RNG behind [`draw_xi`], as recorded in dataset sidecars (design §5.1).
///
/// The versions are written by hand: keep them equal to the `=` pins of `rand_chacha` and
/// `rand_distr` in `Cargo.toml`.
pub const RNG: &str =
    "rand_chacha 0.10.0 ChaCha8Rng::seed_from_u64, rand_distr 0.6.0 StandardNormal";

/// μ_kl = (π²/4)(k² + l²), the eigenvalue of −Δ on [−1, 1]² for the sine product φ_kl
/// (design §3.3). The basis indices k, l start at 1.
pub fn mu(k: usize, l: usize) -> f64 {
    PI * PI / 4.0 * ((k * k + l * l) as f64)
}

/// λ_kl = (μ_kl + τ²)^(−α), the covariance eigenvalue of φ_kl (design §3.3).
pub fn lambda(k: usize, l: usize) -> f64 {
    (mu(k, l) + TAU * TAU).powf(-ALPHA)
}

/// S_K = ¼ Σ_{k,l=1}^K λ_kl, the expected mean square of the unnormalised truncation K
/// (design §3.3). Each φ_kl has mean square ¼ on [−1, 1]².
pub fn s_k(k: usize) -> f64 {
    let mut acc = 0.0;
    for i in 1..=k {
        for j in 1..=k {
            acc += lambda(i, j);
        }
    }
    acc / 4.0
}

/// ε_K = 1 − S_K / S_∞, the fraction of the full series' expected energy outside the
/// truncation K (design §3.3): 1.9e-2, 4.9e-3, 1.3e-3, 3.2e-4 at K = 16, 32, 64, 128.
/// The tolerance for training and evaluation sets is 5e-3.
pub fn eps_k(k: usize) -> f64 {
    1.0 - s_k(k) / S_INF
}

/// The default truncation K(n) = min((n − 1)/2, [`K_MAX`]) for an n-point CGL grid
/// (design §3.3): 16 at n = 33, 32 at 65, 64 at 129 and 257.
///
/// # Panics
/// If `n < 3`, where K(n) would be 0.
pub fn k_default(n: usize) -> usize {
    assert!(n >= 3, "k_default: n must be >= 3, got {n}");
    ((n - 1) / 2).min(K_MAX)
}

/// The standard normals ξ_kl of the sample with this seed, shape `[K_MAX, K_MAX]`, index
/// `[k − 1, l − 1]` (design §3.3).
///
/// The stream is part of the dataset format: `ChaCha8Rng::seed_from_u64(seed)`
/// (rand_chacha 0.10.0) feeds `StandardNormal` (rand_distr 0.6.0), and the K_MAX² draws
/// fill the block in row-major order, so ξ_kl is draw number (k − 1)·K_MAX + (l − 1). A
/// truncation K uses the leading K × K block, so draws are nested: f_K at a smaller K is
/// the truncation of the same sample, apart from the factor √S_K. Both crates are pinned
/// exactly, so a seed gives the same ξ on every machine, and the same field up to
/// floating-point rounding in its evaluation.
pub fn draw_xi(seed: u64) -> Array2<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    Array2::from_shape_simple_fn((K_MAX, K_MAX), || StandardNormal.sample(&mut rng))
}

/// The forcing coefficients c_kl = ξ_kl λ_kl^½ / √S_K of the truncation K, shape
/// `[K, K]`, index `[k − 1, l − 1]` (design §3.3).
///
/// `xi: [K_MAX, K_MAX]` is a draw from [`draw_xi`]; only its leading K × K block is read.
///
/// # Panics
/// If `k` is 0 or above [`K_MAX`], or `xi` is not `[K_MAX, K_MAX]`.
pub fn forcing_coeffs(xi: ArrayView2<f64>, k: usize) -> Array2<f64> {
    assert!(
        (1..=K_MAX).contains(&k),
        "forcing_coeffs: K must be in 1..={K_MAX}, got {k}"
    );
    assert_eq!(
        xi.dim(),
        (K_MAX, K_MAX),
        "forcing_coeffs: xi must be [K_MAX, K_MAX]"
    );
    let norm = s_k(k).sqrt();
    let mut c = xi.slice(s![..k, ..k]).to_owned();
    for ((i, j), v) in c.indexed_iter_mut() {
        *v *= lambda(i + 1, j + 1).sqrt() / norm;
    }
    c
}

/// The coefficients c_kl / μ_kl of the exact solution of −Δu = f, u = 0 on ∂Ω, for the
/// forcing with coefficients `c: [K, K]` (design §4.3). Returns `[K, K]`, same indexing.
///
/// # Panics
/// If `c` is not square.
pub fn solution_coeffs(c: ArrayView2<f64>) -> Array2<f64> {
    assert_eq!(c.nrows(), c.ncols(), "solution_coeffs: c must be [K, K]");
    let mut u = c.to_owned();
    for ((i, j), v) in u.indexed_iter_mut() {
        *v /= mu(i + 1, j + 1);
    }
    u
}

/// Evaluates the sine series with coefficients `c: [K, K]` (index `[k − 1, l − 1]`) on
/// the tensor grid `xs × ys`, as S_x C S_yᵀ with S[i, k − 1] = sin(kπ(x_i + 1)/2)
/// (design §3.3).
///
/// `xs: [n_x]` and `ys: [n_y]` are any points in [−1, 1], normally the CGL nodes. Returns
/// `[n_x, n_y]`, stored 'ij' (CONVENTIONS §12). Costs O(K n² + K² n) for n_x = n_y = n.
///
/// # Panics
/// If `c` is not square.
pub fn evaluate(c: ArrayView2<f64>, xs: ArrayView1<f64>, ys: ArrayView1<f64>) -> Array2<f64> {
    assert_eq!(c.nrows(), c.ncols(), "evaluate: c must be [K, K]");
    let k = c.nrows();
    sine_basis(xs, k).dot(&c).dot(&sine_basis(ys, k).t())
}

/// S[i, k − 1] = sin(kπ(x_i + 1)/2), shape `[xs.len(), k]`.
fn sine_basis(xs: ArrayView1<f64>, k: usize) -> Array2<f64> {
    Array2::from_shape_fn((xs.len(), k), |(i, j)| {
        ((j + 1) as f64 * PI * (xs[i] + 1.0) / 2.0).sin()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::chebyshev::nodes;
    use ndarray::Array1;

    /// The one-dimensional sine values sin(kπ(x + 1)/2), k = 1..K, at one point.
    fn sines_at(x: f64, k: usize) -> Array1<f64> {
        Array1::from_shape_fn(k, |j| ((j + 1) as f64 * PI * (x + 1.0) / 2.0).sin())
    }

    /// Max |a − b| over all entries.
    fn max_abs_diff(a: ArrayView2<f64>, b: ArrayView2<f64>) -> f64 {
        (&a - &b).iter().fold(0.0_f64, |m, v| m.max(v.abs()))
    }

    #[test]
    fn same_seed_same_field_on_common_nodes() {
        let k = 8;
        let c = forcing_coeffs(draw_xi(7).view(), k);
        let field = |n: usize| {
            let x = nodes(n);
            evaluate(c.view(), x.view(), x.view())
        };
        let coarse = field(17);
        for (n, step) in [(33, 2), (65, 4)] {
            let fine = field(n);
            let common = fine.slice(s![..;step, ..;step]);
            let err = max_abs_diff(coarse.view(), common);
            println!("n = 17 vs n = {n} on common nodes: max |diff| {err:.1e}");
            assert!(err <= 1e-14, "n = {n}: {err:e}");
        }
        // A different seed gives a different field.
        let other = forcing_coeffs(draw_xi(8).view(), k);
        assert!(max_abs_diff(c.view(), other.view()) > 0.1);
    }

    #[test]
    fn draws_are_nested() {
        let xi = draw_xi(11);
        let c32 = forcing_coeffs(xi.view(), 32);
        let c64 = forcing_coeffs(xi.view(), 64);
        let leading = c64.slice(s![..32, ..32]).to_owned() * (s_k(64) / s_k(32)).sqrt();
        let scale = c32.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let err = max_abs_diff(c32.view(), leading.view());
        assert!(err <= 1e-14 * scale, "{err:e} against scale {scale:e}");
    }

    #[test]
    fn eps_k_matches_design_table() {
        // S_∞ at cutoff m, as in the spike: the sum over k, l ≤ m plus ¼ times the
        // integral of λ's asymptote (4/π²)^α r⁻⁴ outside the summed cells, bounded by the
        // quarter plane outside radius m + ½: (π/2) ∫ r⁻³ dr = π / (4 (m + ½)²).
        let m = 1000;
        let mf = m as f64 + 0.5;
        let s_inf = s_k(m) + (4.0 / (PI * PI)).powf(ALPHA) * PI / (4.0 * mf * mf) / 4.0;
        let rel = (s_inf - S_INF).abs() / S_INF;
        println!("S_∞ at m = {m}: {s_inf:.7e}, relative to S_INF {rel:.1e}");
        assert!(rel <= 1e-4, "S_∞ recomputed {s_inf:e}, constant {S_INF:e}");

        // Design §3.3, two significant digits.
        for (k, table) in [(16, 1.9e-2), (32, 4.9e-3), (64, 1.3e-3), (128, 3.2e-4)] {
            let e = eps_k(k);
            println!("ε_{k} = {e:.3e} (design {table:.1e})");
            assert!(
                (e - table).abs() <= 0.05 * table,
                "ε_{k} = {e:e}, design {table:e}"
            );
        }
    }

    #[test]
    fn mean_square_is_one() {
        // Parseval: each φ_kl has mean square ¼ on Ω, so the mean square of f_K is
        // ¼ Σ c_kl². Its expectation is 1 for every K.
        let k = 32;
        let n_samples = 1000;
        let ms: Vec<f64> = (0..n_samples)
            .map(|seed| {
                let c = forcing_coeffs(draw_xi(seed).view(), k);
                c.iter().map(|v| v * v).sum::<f64>() / 4.0
            })
            .collect();
        let nf = n_samples as f64;
        let mean = ms.iter().sum::<f64>() / nf;
        let var = ms.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (nf - 1.0);
        let se = (var / nf).sqrt();
        println!("mean square at K = {k}, N = {n_samples}: {mean:.4} ± {se:.4}");
        assert!(
            (mean - 1.0).abs() <= 4.0 * se,
            "mean square {mean} ± {se}, more than 4 s.e. from 1"
        );
    }

    #[test]
    fn point_covariance_matches_formula() {
        let (n, k, n_samples) = (17, 8, 2000);
        let x = nodes(n);
        // (i, j) node pairs: a variance, a neighbour and a distant pair; all interior.
        let pairs = [((8, 8), (8, 8)), ((8, 8), (9, 8)), ((4, 12), (12, 5))];
        let fields: Vec<Array2<f64>> = (0..n_samples)
            .map(|seed| {
                evaluate(
                    forcing_coeffs(draw_xi(seed).view(), k).view(),
                    x.view(),
                    x.view(),
                )
            })
            .collect();
        // Λ[k − 1, l − 1] = λ_kl.
        let lam = Array2::from_shape_fn((k, k), |(i, j)| lambda(i + 1, j + 1));
        let sk = s_k(k);
        for (p, q) in pairs {
            // E f(p) f(q) = (1/S_K) Σ λ_kl φ_kl(p) φ_kl(q), and E f = 0.
            let sp = sines_at(x[p.0], k) * sines_at(x[q.0], k);
            let sq = sines_at(x[p.1], k) * sines_at(x[q.1], k);
            let exact = sp.dot(&lam.dot(&sq)) / sk;
            let prods: Vec<f64> = fields.iter().map(|f| f[p] * f[q]).collect();
            let nf = n_samples as f64;
            let mean = prods.iter().sum::<f64>() / nf;
            let var = prods.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (nf - 1.0);
            let se = (var / nf).sqrt();
            println!(
                "cov {p:?}, {q:?}: empirical {mean:.4} ± {se:.4}, formula {exact:.4}, \
                 {:.2} s.e.",
                (mean - exact).abs() / se
            );
            assert!(
                (mean - exact).abs() <= 5.0 * se,
                "{p:?}, {q:?}: {mean} ± {se} against {exact}"
            );
        }
    }

    #[test]
    fn field_and_solution_vanish_on_boundary() {
        let x = nodes(33);
        let c = forcing_coeffs(draw_xi(3).view(), 16);
        for field in [
            evaluate(c.view(), x.view(), x.view()),
            evaluate(solution_coeffs(c.view()).view(), x.view(), x.view()),
        ] {
            let scale = field.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            for edge in [
                field.row(0),
                field.row(32),
                field.column(0),
                field.column(32),
            ] {
                let e = edge.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                assert!(e <= 1e-14 * scale, "{e:e} against scale {scale:e}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "K must be in 1..=64")]
    fn k_above_k_max_panics() {
        forcing_coeffs(draw_xi(0).view(), K_MAX + 1);
    }
}

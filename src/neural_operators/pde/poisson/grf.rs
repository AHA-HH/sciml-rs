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
//!
//! The `_3d` items are the same family on [−1, 1]³ with a third sine factor, α =
//! [`ALPHA_3D`] and S_K = ⅛ Σ λ_klm (3D proposal §3.3), drawn by [`draw_xi_3d`] from a
//! ChaCha8 stream of their own.

use ndarray::{Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis, s};
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

/// One sine mode φ_kl of the covariance spectrum of [`covariance_spectrum`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectrumMode {
    /// k² + l²: λ_kl and μ_kl depend on (k, l) only through it, so modes with equal q
    /// share their eigenvalues.
    pub q: usize,
    /// λ_kl / S_K, the eigenvalue of f_K's covariance on φ_kl.
    pub f: f64,
    /// λ_kl / (S_K μ_kl²), the eigenvalue of u's covariance on φ_kl, since u's sine
    /// coefficients are f's over μ_kl ([`solution_coeffs`]).
    pub u: f64,
}

/// The covariance spectrum of f_K and of the solution u at truncation `k`: the k² modes
/// (k, l ≤ `k`), sorted by ascending q = k² + l² (ties in (k, l) order). Every φ_kl has
/// unit L²(Ω) norm (mean square ¼ on [−1, 1]²), so these are the eigenvalues of the
/// covariance operators in L²(Ω) (design §3.3, §13.4). λ_kl falls and μ_kl rises with q,
/// so both fields' values are descending in this order; Σ f = 4 = |Ω| × f_K's unit
/// expected mean square.
pub fn covariance_spectrum(k: usize) -> Vec<SpectrumMode> {
    let s = s_k(k);
    let mut modes: Vec<SpectrumMode> = (1..=k)
        .flat_map(|i| (1..=k).map(move |j| (i, j)))
        .map(|(i, j)| {
            let f = lambda(i, j) / s;
            SpectrumMode {
                q: i * i + j * j,
                f,
                u: f / mu(i, j).powi(2),
            }
        })
        .collect();
    // stable: equal q keep (k, l) order
    modes.sort_by_key(|m| m.q);
    modes
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

/// α of the 3D covariance (−Δ + τ²)^(−α) on [−1, 1]³ (proposal §3.3, D3): 3, not 2, since
/// ε_K ~ K^(d − 2α) decays only as K⁻¹ in 3D at α = 2. τ is [`TAU`], as in 2D.
pub const ALPHA_3D: f64 = 3.0;

/// S_∞ = ⅛ Σ_{k,l,m ≥ 1} λ_klm in 3D, which cannot be brute-forced as in 2D: the cube sums
/// to m = 80 and 160 with their (m + ½)^(3 − 2α) tail extrapolated (proposal §3.3; the
/// tail beyond 160 is 5e-6 relative, the extrapolation good to about 2e-7).
/// `eps_k_3d_matches_design_table` recomputes it.
pub const S_INF_3D: f64 = 1.232538e-4;

/// The ChaCha8 stream of the 3D draws ([`draw_xi_3d`]). The 2D draws ([`draw_xi`]) and the
/// Phase 7 training subsets use stream 0, so a 3D seed never repeats their normals.
pub const STREAM_3D: u64 = 3;

/// The RNG behind [`draw_xi_3d`], as recorded in 3D dataset sidecars. The versions follow
/// [`RNG`]'s pins.
pub const RNG_3D: &str = "rand_chacha 0.10.0 ChaCha8Rng::seed_from_u64, set_stream(3), \
                          rand_distr 0.6.0 StandardNormal";

/// μ_klm = (π²/4)(k² + l² + m²), the eigenvalue of −Δ on [−1, 1]³ for the sine product
/// φ_k(x) φ_l(y) φ_m(z) (proposal §3.3). The indices start at 1.
pub fn mu_3d(k: usize, l: usize, m: usize) -> f64 {
    PI * PI / 4.0 * ((k * k + l * l + m * m) as f64)
}

/// λ_klm = (μ_klm + τ²)^(−α), α = [`ALPHA_3D`] (proposal §3.3).
pub fn lambda_3d(k: usize, l: usize, m: usize) -> f64 {
    (mu_3d(k, l, m) + TAU * TAU).powf(-ALPHA_3D)
}

/// S_K = ⅛ Σ_{k,l,m=1}^K λ_klm, the expected mean square of the unnormalised 3D truncation
/// K (proposal §3.3). Each sine product has mean square ⅛ on [−1, 1]³.
pub fn s_k_3d(k: usize) -> f64 {
    let mut acc = 0.0;
    for i in 1..=k {
        for j in 1..=k {
            for l in 1..=k {
                acc += lambda_3d(i, j, l);
            }
        }
    }
    acc / 8.0
}

/// ε_K = 1 − S_K / S_∞ in 3D (proposal §3.3): 2.7e-2, 4.3e-3, 5.9e-4, 7.7e-5 at K = 8, 16,
/// 32, 64. The tolerance for training and evaluation sets is 5e-3 (D3).
pub fn eps_k_3d(k: usize) -> f64 {
    1.0 - s_k_3d(k) / S_INF_3D
}

/// The standard normals ξ_klm of the 3D sample with this seed, shape
/// `[K_MAX, K_MAX, K_MAX]`, index `[k − 1, l − 1, m − 1]` (proposal §3.3).
///
/// As [`draw_xi`], on the ChaCha8 stream [`STREAM_3D`] ([`RNG_3D`]): the K_MAX³ draws fill
/// the block in row-major order (m fastest), so ξ_klm is draw number
/// ((k − 1)·K_MAX + (l − 1))·K_MAX + (m − 1), and a truncation K uses the leading K³ block.
pub fn draw_xi_3d(seed: u64) -> Array3<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(STREAM_3D);
    Array3::from_shape_simple_fn((K_MAX, K_MAX, K_MAX), || StandardNormal.sample(&mut rng))
}

/// The 3D forcing coefficients c_klm = ξ_klm λ_klm^½ / √S_K of the truncation K, shape
/// `[K, K, K]`, index `[k − 1, l − 1, m − 1]` (proposal §3.3).
///
/// `xi: [K_MAX, K_MAX, K_MAX]` is a draw from [`draw_xi_3d`]; only its leading K³ block is
/// read.
///
/// # Panics
/// If `k` is 0 or above [`K_MAX`], or `xi` is not `[K_MAX, K_MAX, K_MAX]`.
pub fn forcing_coeffs_3d(xi: ArrayView3<f64>, k: usize) -> Array3<f64> {
    assert!(
        (1..=K_MAX).contains(&k),
        "forcing_coeffs_3d: K must be in 1..={K_MAX}, got {k}"
    );
    assert_eq!(
        xi.dim(),
        (K_MAX, K_MAX, K_MAX),
        "forcing_coeffs_3d: xi must be [K_MAX, K_MAX, K_MAX]"
    );
    let norm = s_k_3d(k).sqrt();
    let mut c = xi.slice(s![..k, ..k, ..k]).to_owned();
    for ((i, j, l), v) in c.indexed_iter_mut() {
        *v *= lambda_3d(i + 1, j + 1, l + 1).sqrt() / norm;
    }
    c
}

/// The coefficients c_klm / μ_klm of the exact solution of −Δu = f, u = 0 on ∂Ω, for the 3D
/// forcing with coefficients `c: [K, K, K]` (proposal §3.3). Returns `[K, K, K]`, same
/// indexing.
///
/// # Panics
/// If `c` is not a cube.
pub fn solution_coeffs_3d(c: ArrayView3<f64>) -> Array3<f64> {
    let k = c.dim().0;
    assert_eq!(
        c.dim(),
        (k, k, k),
        "solution_coeffs_3d: c must be [K, K, K]"
    );
    let mut u = c.to_owned();
    for ((i, j, l), v) in u.indexed_iter_mut() {
        *v /= mu_3d(i + 1, j + 1, l + 1);
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
pub fn evaluate_3d(
    c: ArrayView3<f64>,
    xs: ArrayView1<f64>,
    ys: ArrayView1<f64>,
    zs: ArrayView1<f64>,
) -> Array3<f64> {
    let k = c.dim().0;
    assert_eq!(c.dim(), (k, k, k), "evaluate_3d: c must be [K, K, K]");
    let (sy, sz) = (sine_basis(ys, k), sine_basis(zs, k));
    let (ny, nz) = (ys.len(), zs.len());
    // row a: S_y C[a] S_zᵀ, flattened 'ij'
    let mut g = Array2::zeros((k, ny * nz));
    for (a, mut row) in g.outer_iter_mut().enumerate() {
        let ga = sy.dot(&c.index_axis(Axis(0), a)).dot(&sz.t());
        row.assign(&ArrayView1::from(
            ga.as_slice().expect("dot is standard layout"),
        ));
    }
    sine_basis(xs, k)
        .dot(&g)
        .into_shape_with_order((xs.len(), ny, nz))
        .expect("[n_x, n_y n_z] reshapes to [n_x, n_y, n_z]")
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
    fn covariance_spectrum_matches_the_phase6_facts() {
        use crate::neural_operators::metrics::pod::energy_rank_of;
        let spectrum = covariance_spectrum(K_MAX);
        assert_eq!(spectrum.len(), K_MAX * K_MAX);
        assert!(spectrum.windows(2).all(|p| p[0].q <= p[1].q));
        assert!(
            spectrum
                .windows(2)
                .all(|p| p[0].f >= p[1].f && p[0].u >= p[1].u)
        );
        let f: Vec<f64> = spectrum.iter().map(|m| m.f).collect();
        let u: Vec<f64> = spectrum.iter().map(|m| m.u).collect();
        assert!((f.iter().sum::<f64>() - 4.0).abs() <= 1e-12);
        // docs/poisson2d/phase6/README.md: trace 4.998e-2, leading values 1.035 and 4.25e-2
        assert!((u.iter().sum::<f64>() - 4.998e-2).abs() <= 5e-6);
        assert!((f[0] - 1.035).abs() <= 5e-4 && (u[0] - 4.25e-2).abs() <= 5e-5);
        let ranks = |v: &[f64]| [0.9, 0.99, 0.999].map(|x| energy_rank_of(v, x));
        assert_eq!(ranks(&f), [38, 418, 2175]);
        assert_eq!(ranks(&u), [2, 6, 17]);
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

    /// Max |a − b| over all entries, rank 3.
    fn max_abs_diff_3d(a: ArrayView3<f64>, b: ArrayView3<f64>) -> f64 {
        (&a - &b).iter().fold(0.0_f64, |m, v| m.max(v.abs()))
    }

    #[test]
    fn evaluate_3d_matches_direct_sum() {
        let k = 4;
        let c = forcing_coeffs_3d(draw_xi_3d(5).view(), k);
        let (x, y, z) = (nodes(9), nodes(7), nodes(5));
        let fast = evaluate_3d(c.view(), x.view(), y.view(), z.view());
        assert_eq!(fast.dim(), (9, 7, 5));
        let direct = Array3::from_shape_fn((9, 7, 5), |(i, j, l)| {
            let (sx, sy, sz) = (sines_at(x[i], k), sines_at(y[j], k), sines_at(z[l], k));
            c.indexed_iter()
                .map(|((a, b, d), v)| v * sx[a] * sy[b] * sz[d])
                .sum::<f64>()
        });
        let scale = direct.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let err = max_abs_diff_3d(fast.view(), direct.view());
        println!("evaluate_3d against the direct sum: max |diff| {err:.1e}, scale {scale:.2}");
        assert!(err <= 1e-14 * scale, "{err:e} against scale {scale:e}");
    }

    #[test]
    fn eps_k_3d_matches_design_table() {
        // S_∞ by extrapolating the cube sums' tail, which falls as (m + ½)^(3 − 2α) (proposal
        // §3.3): S_∞ ≈ S_b + (S_b − S_a) L_b^(−3) / (L_a^(−3) − L_b^(−3)), L = m + ½. Two
        // pairs of cutoffs agreeing shows the extrapolation has converged.
        let sums = [40, 80, 160].map(|m| (m as f64 + 0.5, s_k_3d(m)));
        let extrapolate = |(la, sa): (f64, f64), (lb, sb): (f64, f64)| {
            let p = 2.0 * ALPHA_3D - 3.0;
            sb + (sb - sa) * lb.powf(-p) / (la.powf(-p) - lb.powf(-p))
        };
        let coarse = extrapolate(sums[0], sums[1]);
        let s_inf = extrapolate(sums[1], sums[2]);
        let (spread, rel) = (
            (coarse - s_inf).abs() / s_inf,
            (s_inf - S_INF_3D).abs() / S_INF_3D,
        );
        let tail = (s_inf - sums[2].1) / s_inf;
        println!(
            "3D S_∞ {s_inf:.7e} (tail beyond m = 160 {tail:.1e} relative; cutoffs 40/80 \
             differ by {spread:.1e}), relative to S_INF_3D {rel:.1e}"
        );
        assert!(spread <= 1e-6, "extrapolation not converged: {spread:e}");
        assert!(
            rel <= 1e-6,
            "S_∞ recomputed {s_inf:e}, constant {S_INF_3D:e}"
        );

        // Proposal §3.3, the α = 3 row, two significant digits.
        for (k, table) in [(8, 2.7e-2), (16, 4.3e-3), (32, 5.9e-4), (64, 7.7e-5)] {
            let e = eps_k_3d(k);
            println!("3D ε_{k} = {e:.3e} (proposal {table:.1e})");
            assert!(
                (e - table).abs() <= 0.05 * table,
                "ε_{k} = {e:e}, proposal {table:e}"
            );
        }
    }

    #[test]
    fn draws_3d_are_nested() {
        let xi = draw_xi_3d(11);
        let c8 = forcing_coeffs_3d(xi.view(), 8);
        let c16 = forcing_coeffs_3d(xi.view(), 16);
        let leading = c16.slice(s![..8, ..8, ..8]).to_owned() * (s_k_3d(16) / s_k_3d(8)).sqrt();
        let scale = c8.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let err = max_abs_diff_3d(c8.view(), leading.view());
        assert!(err <= 1e-14 * scale, "{err:e} against scale {scale:e}");
    }

    #[test]
    fn stream_3d_differs_from_2d() {
        let (xi3, xi2) = (draw_xi_3d(0), draw_xi(0));
        assert_eq!(xi3.dim(), (K_MAX, K_MAX, K_MAX));
        // The first K_MAX² normals of each stream, in draw order.
        let first3: Vec<f64> = xi3.iter().take(K_MAX * K_MAX).copied().collect();
        let first2: Vec<f64> = xi2.iter().copied().collect();
        assert!(first3.iter().zip(&first2).all(|(a, b)| a != b));
        // Reproducible: the same seed gives the same draws.
        assert_eq!(xi3, draw_xi_3d(0));
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
                let c = forcing_coeffs_3d(draw_xi_3d(seed).view(), k);
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
        let c = forcing_coeffs_3d(draw_xi_3d(3).view(), 8);
        for field in [
            evaluate_3d(c.view(), x.view(), x.view(), x.view()),
            evaluate_3d(
                solution_coeffs_3d(c.view()).view(),
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
        let c = forcing_coeffs_3d(draw_xi_3d(9).view(), 8);
        let u = solution_coeffs_3d(c.view());
        let h = 1e-3;
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for p in [[0.1, -0.3, 0.55], [-0.7, 0.2, 0.0], [0.45, 0.8, -0.6]] {
            let axis = |v: f64| Array1::from(vec![v - h, v, v + h]);
            let (x, y, z) = (axis(p[0]), axis(p[1]), axis(p[2]));
            let uu = evaluate_3d(u.view(), x.view(), y.view(), z.view());
            let lap = (uu[[0, 1, 1]]
                + uu[[2, 1, 1]]
                + uu[[1, 0, 1]]
                + uu[[1, 2, 1]]
                + uu[[1, 1, 0]]
                + uu[[1, 1, 2]]
                - 6.0 * uu[[1, 1, 1]])
                / (h * h);
            let f = evaluate_3d(c.view(), x.view(), y.view(), z.view())[[1, 1, 1]];
            worst = worst.max((-lap - f).abs());
            scale = scale.max(f.abs());
        }
        println!("3D −Δu − f by the stencil: {worst:.1e} against max |f| {scale:.2}");
        assert!(worst <= 1e-4 * scale, "{worst:e} against {scale:e}");
    }

    #[test]
    #[should_panic(expected = "K must be in 1..=64")]
    fn k_above_k_max_panics_3d() {
        forcing_coeffs_3d(draw_xi_3d(0).view(), K_MAX + 1);
    }
}

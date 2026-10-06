//! Sine-basis GRF of design §3.3: covariance (−Δ + τ²)^(−α), τ = 3, α = 2.

use ndarray::{Array1, Array2, s};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::StandardNormal;
use std::f64::consts::PI;

pub const TAU: f64 = 3.0;
pub const ALPHA: f64 = 2.0;
/// Largest truncation drawn in this spike; every sample draws K_MAX² normals.
pub const K_MAX: usize = 128;

/// μ_kl = π²/4 (k² + l²): the eigenvalue of −Δ on Ω for the sine product (k, l).
pub fn mu(k: usize, l: usize) -> f64 {
    PI * PI / 4.0 * ((k * k + l * l) as f64)
}

/// λ_kl = (μ_kl + τ²)^(−α), k, l ≥ 1.
pub fn lambda(k: usize, l: usize) -> f64 {
    (mu(k, l) + TAU * TAU).powf(-ALPHA)
}

/// S_K = ¼ Σ_{k,l=1}^K λ_kl.
pub fn s_k(k_trunc: usize) -> f64 {
    let mut acc = 0.0;
    for k in 1..=k_trunc {
        for l in 1..=k_trunc {
            acc += lambda(k, l);
        }
    }
    acc / 4.0
}

/// S_∞: the sum over k, l ≤ `m`, plus the integral of the r⁻⁴ asymptote of λ over the
/// region outside the square (upper bound on the tail; ~1e-6 relative at m = 3000).
pub fn s_inf(m: usize) -> f64 {
    let c = (4.0 / (PI * PI)).powf(ALPHA);
    // ∫∫ over {x > M} ∪ {y > M} of (x² + y²)^(−2) ≤ 2 · π / (8 M²).
    let mf = m as f64 + 0.5;
    s_k(m) + c * PI / (4.0 * mf * mf) / 4.0
}

/// Unresolved-energy fraction ε_K = 1 − S_K / S_∞.
pub fn eps_k(k_trunc: usize, s_inf: f64) -> f64 {
    1.0 - s_k(k_trunc) / s_inf
}

/// ξ_kl ~ N(0, 1), drawn from `seed` in row-major order over K_MAX × K_MAX. Index [k − 1,
/// l − 1]. A truncation K uses the leading K × K block, so draws are nested.
pub fn draw_xi(seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::from_shape_simple_fn((K_MAX, K_MAX), || rng.sample(StandardNormal))
}

fn sine_basis(xs: &Array1<f64>, k_trunc: usize) -> Array2<f64> {
    Array2::from_shape_fn((xs.len(), k_trunc), |(i, k)| {
        ((k + 1) as f64 * PI * (xs[i] + 1.0) / 2.0).sin()
    })
}

/// Coefficients c_kl = ξ_kl √λ_kl / √S_K of the truncation K.
pub fn coeffs(xi: &Array2<f64>, k_trunc: usize) -> Array2<f64> {
    let norm = s_k(k_trunc).sqrt();
    let mut c = xi.slice(s![..k_trunc, ..k_trunc]).to_owned();
    for ((k, l), v) in c.indexed_iter_mut() {
        *v *= lambda(k + 1, l + 1).sqrt() / norm;
    }
    c
}

/// Exact evaluation of f_K on xs × ys ('ij'): S_x C S_yᵀ.
pub fn eval(xi: &Array2<f64>, k_trunc: usize, xs: &Array1<f64>, ys: &Array1<f64>) -> Array2<f64> {
    let c = coeffs(xi, k_trunc);
    sine_basis(xs, k_trunc)
        .dot(&c)
        .dot(&sine_basis(ys, k_trunc).t())
}

/// Exact solution of −Δu = f_K on Ω, u = 0 on ∂Ω, on xs × ys ('ij'): coefficients
/// c_kl / μ_kl in the same sine basis.
pub fn eval_solution(
    xi: &Array2<f64>,
    k_trunc: usize,
    xs: &Array1<f64>,
    ys: &Array1<f64>,
) -> Array2<f64> {
    let mut c = coeffs(xi, k_trunc);
    for ((k, l), v) in c.indexed_iter_mut() {
        *v /= mu(k + 1, l + 1);
    }
    sine_basis(xs, k_trunc)
        .dot(&c)
        .dot(&sine_basis(ys, k_trunc).t())
}

/// Mean square of f_K over Ω by Parseval: ¼ Σ c_kl² (each sine product has ∫∫ = 1).
pub fn mean_square_parseval(xi: &Array2<f64>, k_trunc: usize) -> f64 {
    coeffs(xi, k_trunc).iter().map(|c| c * c).sum::<f64>() / 4.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grids::cheb_nodes;

    // Nesting: f_32 is the K = 32 truncation of the same draw that gives f_64, up to
    // the √S_K factor. 1e-13 relative.
    #[test]
    fn draws_are_nested() {
        let xi = draw_xi(7);
        let x = cheb_nodes(65);
        let f32_ = eval(&xi, 32, &x, &x);
        let c64 = coeffs(&xi, 64);
        let mut c = c64.slice(s![..32, ..32]).to_owned();
        c *= (s_k(64) / s_k(32)).sqrt();
        let b = sine_basis(&x, 32);
        let g = b.dot(&c).dot(&b.t());
        let scale = f32_.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let err = (&f32_ - &g).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(err < 1e-13 * scale, "{err:e}");
    }

    // −Δ_h u against f with the 5-point stencil on a fine uniform grid: O(h²) agreement,
    // 1e-3 relative (max norm) at K = 8, h = 2/400.
    #[test]
    fn solution_satisfies_poisson() {
        let xi = draw_xi(11);
        let x = crate::grids::uniform_nodes(401);
        let h = x[1] - x[0];
        let u = eval_solution(&xi, 8, &x, &x);
        let f = eval(&xi, 8, &x, &x);
        let scale = f.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let mut err = 0.0_f64;
        for i in 1..400 {
            for j in 1..400 {
                let lap = (u[[i + 1, j]] + u[[i - 1, j]] + u[[i, j + 1]] + u[[i, j - 1]]
                    - 4.0 * u[[i, j]])
                    / (h * h);
                err = err.max((-lap - f[[i, j]]).abs());
            }
        }
        assert!(err < 1e-3 * scale, "{err:e} vs scale {scale:e}");
        for j in 0..401 {
            assert!(u[[0, j]].abs() < 1e-14 && u[[400, j]].abs() < 1e-14);
        }
    }

    #[test]
    fn same_seed_same_draw() {
        assert_eq!(draw_xi(3), draw_xi(3));
        assert_ne!(draw_xi(3), draw_xi(4));
    }

    #[test]
    fn field_vanishes_on_boundary() {
        let xi = draw_xi(1);
        let x = cheb_nodes(33);
        let f = eval(&xi, 16, &x, &x);
        for j in 0..33 {
            assert!(f[[0, j]].abs() < 1e-14 && f[[32, j]].abs() < 1e-14);
        }
    }
}

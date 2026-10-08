//! Manufactured solutions of design §4.3 and grid helpers, shared by the solver tests.
//!
//! Each pair has u = 0 on ∂[−1, 1]² and f = −Δu in closed form.

use ndarray::Array2;
use std::f64::consts::PI;

use crate::neural_operators::chebyshev::nodes;

/// A manufactured pair (u, f = −Δu) of design §4.3.
pub(super) struct Manufactured {
    pub(super) name: &'static str,
    pub(super) u: fn(f64, f64) -> f64,
    pub(super) f: fn(f64, f64) -> f64,
}

pub(super) fn sin_u(x: f64, y: f64) -> f64 {
    (PI * x).sin() * (PI * y).sin()
}
pub(super) fn sin_f(x: f64, y: f64) -> f64 {
    2.0 * PI * PI * sin_u(x, y)
}
pub(super) fn poly_u(x: f64, y: f64) -> f64 {
    (1.0 - x * x) * (1.0 - y * y) * (x + y * y)
}
// u_xx = (1 − y²)(−6x − 2y²), u_yy = (1 − x²)(2 − 2x − 12y²).
pub(super) fn poly_f(x: f64, y: f64) -> f64 {
    let uxx = (1.0 - y * y) * (-6.0 * x - 2.0 * y * y);
    let uyy = (1.0 - x * x) * (2.0 - 2.0 * x - 12.0 * y * y);
    -(uxx + uyy)
}
pub(super) fn exp_u(x: f64, y: f64) -> f64 {
    (1.0 - x * x) * (1.0 - y * y) * (x + 2.0 * y).exp()
}
// u_xx = e^(x+2y)(1 − y²)(−1 − 4x − x²), u_yy = e^(x+2y)(1 − x²)(2 − 8y − 4y²).
pub(super) fn exp_f(x: f64, y: f64) -> f64 {
    let uxx = (1.0 - y * y) * (-1.0 - 4.0 * x - x * x);
    let uyy = (1.0 - x * x) * (2.0 - 8.0 * y - 4.0 * y * y);
    -(x + 2.0 * y).exp() * (uxx + uyy)
}

// Polynomial of degree 5 in x, sin(8y) in y: exact in x from n_x = 6, but needs
// n_y ≈ 33 for the floor, so it tells the two axes apart.
// u = p(x) q(y), p = (1 − x²)(1 + x), q = (1 − y²) sin 8y;
// p'' = −2 − 6x, q'' = −2 sin 8y − 32y cos 8y − 64(1 − y²) sin 8y.
pub(super) fn aniso_u(x: f64, y: f64) -> f64 {
    (1.0 - x * x) * (1.0 + x) * (1.0 - y * y) * (8.0 * y).sin()
}
pub(super) fn aniso_f(x: f64, y: f64) -> f64 {
    let (p, pxx) = ((1.0 - x * x) * (1.0 + x), -2.0 - 6.0 * x);
    let (s, c) = ((8.0 * y).sin(), (8.0 * y).cos());
    let q = (1.0 - y * y) * s;
    let qyy = -2.0 * s - 32.0 * y * c - 64.0 * (1.0 - y * y) * s;
    -(pxx * q + p * qyy)
}
pub(super) fn aniso_t_u(x: f64, y: f64) -> f64 {
    aniso_u(y, x)
}
pub(super) fn aniso_t_f(x: f64, y: f64) -> f64 {
    aniso_f(y, x)
}

pub(super) const SIN: Manufactured = Manufactured {
    name: "sin(πx)sin(πy)",
    u: sin_u,
    f: sin_f,
};
pub(super) const POLY: Manufactured = Manufactured {
    name: "(1−x²)(1−y²)(x+y²)",
    u: poly_u,
    f: poly_f,
};
pub(super) const EXP: Manufactured = Manufactured {
    name: "(1−x²)(1−y²)e^(x+2y)",
    u: exp_u,
    f: exp_f,
};

pub(super) const ANISO: Manufactured = Manufactured {
    name: "(1−x²)(1+x)(1−y²)sin 8y",
    u: aniso_u,
    f: aniso_f,
};
pub(super) const ANISO_T: Manufactured = Manufactured {
    name: "ANISO with x and y swapped",
    u: aniso_t_u,
    f: aniso_t_f,
};

/// `g` on the `[nx, ny]` CGL grid, 'ij'.
pub(super) fn sample(nx: usize, ny: usize, g: impl Fn(f64, f64) -> f64) -> Array2<f64> {
    let (x, y) = (nodes(nx), nodes(ny));
    Array2::from_shape_fn((nx, ny), |(i, j)| g(x[i], y[j]))
}

/// max |a − b| / max |b|.
pub(super) fn rel_max(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let err = a
        .iter()
        .zip(b)
        .fold(0.0_f64, |m, (p, q)| m.max((p - q).abs()));
    err / b.iter().fold(0.0_f64, |m, q| m.max(q.abs()))
}

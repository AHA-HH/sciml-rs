//! Manufactured solutions of design §4.3 and grid helpers, shared by the solver tests.
//!
//! Each pair has u = 0 on ∂[−1, 1]² (∂[−1, 1]³ for the 3D pairs of 3D proposal §5) and
//! f = −Δu in closed form.

use ndarray::{Array, Array2, Array3, Dimension};
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
pub(super) fn rel_max<D: Dimension>(a: &Array<f64, D>, b: &Array<f64, D>) -> f64 {
    let err = a
        .iter()
        .zip(b)
        .fold(0.0_f64, |m, (p, q)| m.max((p - q).abs()));
    err / b.iter().fold(0.0_f64, |m, q| m.max(q.abs()))
}

// ---- 3D pairs for A3 (3D proposal §5) ----

pub(super) type Field3 = fn(f64, f64, f64) -> f64;

/// A 3D manufactured pair: u and f = −Δu.
pub(super) struct Manufactured3 {
    pub(super) name: &'static str,
    pub(super) u: Field3,
    pub(super) f: Field3,
}

pub(super) const SIN3: Manufactured3 = Manufactured3 {
    name: "sin πx sin πy sin πz",
    u: |x, y, z| (PI * x).sin() * (PI * y).sin() * (PI * z).sin(),
    f: |x, y, z| 3.0 * PI * PI * (PI * x).sin() * (PI * y).sin() * (PI * z).sin(),
};
pub(super) const MODES3: Manufactured3 = Manufactured3 {
    name: "sin πx sin 2πy sin 3πz",
    u: |x, y, z| (PI * x).sin() * (2.0 * PI * y).sin() * (3.0 * PI * z).sin(),
    f: |x, y, z| 14.0 * PI * PI * (PI * x).sin() * (2.0 * PI * y).sin() * (3.0 * PI * z).sin(),
};
/// Neither a polynomial nor a sine product: u = (1 − x²) eˣ (1 − y²)(1 − z²).
pub(super) const EXP3: Manufactured3 = Manufactured3 {
    name: "(1−x²)eˣ(1−y²)(1−z²)",
    u: |x, y, z| (1.0 - x * x) * x.exp() * (1.0 - y * y) * (1.0 - z * z),
    f: |x, y, z| {
        let (bx, by, bz) = (1.0 - x * x, 1.0 - y * y, 1.0 - z * z);
        x.exp() * ((x * x + 4.0 * x + 1.0) * by * bz + 2.0 * bx * (by + bz))
    },
};
/// Exact along x at any n ≥ 3; resolved along y by 17 nodes, along z only by 33.
pub(super) const ANISO3: Manufactured3 = Manufactured3 {
    name: "(1−x²) sin πy sin 3πz",
    u: |x, y, z| (1.0 - x * x) * (PI * y).sin() * (3.0 * PI * z).sin(),
    f: |x, y, z| (PI * y).sin() * (3.0 * PI * z).sin() * (2.0 + 10.0 * PI * PI * (1.0 - x * x)),
};

/// `g` on the `[nx, ny, nz]` CGL grid, 'ijk'.
pub(super) fn sample3(nx: usize, ny: usize, nz: usize, g: Field3) -> Array3<f64> {
    let (x, y, z) = (nodes(nx), nodes(ny), nodes(nz));
    Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| g(x[i], y[j], z[k]))
}

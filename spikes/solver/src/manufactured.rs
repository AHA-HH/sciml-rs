//! Manufactured solutions of design §4.3, with f = −Δu in closed form.

use std::f64::consts::PI;

/// A manufactured pair (u, f = −Δu).
pub struct Manufactured {
    pub name: &'static str,
    pub u: fn(f64, f64) -> f64,
    pub f: fn(f64, f64) -> f64,
}

fn sin_u(x: f64, y: f64) -> f64 {
    (PI * x).sin() * (PI * y).sin()
}
fn sin_f(x: f64, y: f64) -> f64 {
    2.0 * PI * PI * sin_u(x, y)
}

fn poly_u(x: f64, y: f64) -> f64 {
    (1.0 - x * x) * (1.0 - y * y) * (x + y * y)
}
// u_xx = (1 − y²)(−2(x + y²) − 4x), u_yy = (1 − x²)(2 − 2x − 12y²).
fn poly_f(x: f64, y: f64) -> f64 {
    let uxx = (1.0 - y * y) * (-2.0 * (x + y * y) - 4.0 * x);
    let uyy = (1.0 - x * x) * (2.0 - 2.0 * x - 12.0 * y * y);
    -(uxx + uyy)
}

fn exp_u(x: f64, y: f64) -> f64 {
    (1.0 - x * x) * (1.0 - y * y) * (x + 2.0 * y).exp()
}
// u_xx = e^(x+2y)(1 − y²)(−1 − 4x − x²), u_yy = e^(x+2y)(1 − x²)(2 − 8y − 4y²).
fn exp_f(x: f64, y: f64) -> f64 {
    let e = (x + 2.0 * y).exp();
    let uxx = (1.0 - y * y) * (-1.0 - 4.0 * x - x * x);
    let uyy = (1.0 - x * x) * (2.0 - 8.0 * y - 4.0 * y * y);
    -e * (uxx + uyy)
}

/// The three solutions, in the brief's order.
pub const ALL: [Manufactured; 3] = [
    Manufactured {
        name: "sin(πx)sin(πy)",
        u: sin_u,
        f: sin_f,
    },
    Manufactured {
        name: "(1−x²)(1−y²)(x+y²)",
        u: poly_u,
        f: poly_f,
    },
    Manufactured {
        name: "(1−x²)(1−y²)e^(x+2y)",
        u: exp_u,
        f: exp_f,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    // f against a 4th-order finite-difference Laplacian of u, h = 1e-3.
    #[test]
    fn f_is_minus_laplacian_of_u() {
        let h = 1e-3;
        for m in &ALL {
            for &(x, y) in &[(0.3, -0.7), (-0.55, 0.2), (0.9, 0.9), (0.0, 0.0)] {
                let d2 = |g: &dyn Fn(f64) -> f64| {
                    (-g(2.0 * h) + 16.0 * g(h) - 30.0 * g(0.0) + 16.0 * g(-h) - g(-2.0 * h))
                        / (12.0 * h * h)
                };
                let lap = d2(&|t| (m.u)(x + t, y)) + d2(&|t| (m.u)(x, y + t));
                let f = (m.f)(x, y);
                assert!(
                    (f + lap).abs() < 1e-6 * (1.0 + f.abs()),
                    "{}: {f} vs {}",
                    m.name,
                    -lap
                );
            }
        }
    }
}

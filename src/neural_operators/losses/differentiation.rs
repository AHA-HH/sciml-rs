//! Finite differences on a uniform grid, as `neuraloperator` 2.0.0's `FiniteDiff`
//! (`neuralop/losses/differentiation.py`; library CM1.6a).
//!
//! The last `D` axes of a tensor are the spatial axes x, y, z, …; every axis before them is a
//! batch-like axis the stencils leave alone. Along a periodic axis the stencils wrap round;
//! along a non-periodic one they are central inside and one-sided at the two ends, with
//! 2.0.0's coefficients. 2.0.0's `central_diff_{1,2,3}d` are [`FiniteDiff::partial`] on each
//! axis.

use burn::tensor::Tensor;

/// First and second derivatives on a uniform grid of `D` spatial axes.
///
/// Coefficients, step h along the axis, first derivative: periodic and interior
/// (u₊ − u₋) / 2h; non-periodic ends (−11u₀ + 18u₁ − 9u₂ + 2u₃) / 6h and its mirror (third
/// order). Second derivative: periodic and interior (u₊ − 2u + u₋) / h²; non-periodic ends
/// (2u₀ − 5u₁ + 4u₂ − u₃) / h² and its mirror (second order).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FiniteDiff<const D: usize> {
    /// The grid step along each spatial axis.
    pub h: [f64; D],
    /// Whether each spatial axis is periodic; a non-periodic one needs at least 4 points.
    pub periodic: [bool; D],
}

impl<const D: usize> FiniteDiff<D> {
    /// The derivative of order `order` (1 or 2) of `u` along spatial axis `axis` (0 = x).
    ///
    /// `u`: `[.., s_1, .., s_D]`; the result has the same shape.
    ///
    /// # Panics
    /// If `R < D`, `axis >= D`, `order` is not 1 or 2, or the axis is non-periodic with fewer
    /// than 4 points.
    pub fn partial<const R: usize>(&self, u: Tensor<R>, axis: usize, order: usize) -> Tensor<R> {
        assert!(R >= D, "FiniteDiff: rank {R} below the {D} spatial axes");
        assert!(axis < D, "FiniteDiff: axis {axis} of {D}");
        assert!(
            order == 1 || order == 2,
            "FiniteDiff: order {order}; only 1 and 2"
        );
        let dim = R - D + axis;
        let h = self.h[axis];
        if self.periodic[axis] {
            let next = u.clone().roll_dim(-1, dim);
            let prev = u.clone().roll_dim(1, dim);
            return match order {
                1 => (next - prev).div_scalar(2.0 * h),
                _ => (next - u.mul_scalar(2.0) + prev).div_scalar(h * h),
            };
        }
        let n = u.dims()[dim];
        assert!(
            n >= 4,
            "FiniteDiff: non-periodic axis {axis} has {n} points; needs at least 4"
        );
        let at = |start: usize, len: usize| u.clone().narrow(dim, start, len);
        // a·u[i] + b·u[i+1] + c·u[i+2] + d·u[i+3], one point wide.
        let four = |i: usize, [a, b, c, d]: [f64; 4]| {
            at(i, 1).mul_scalar(a)
                + at(i + 1, 1).mul_scalar(b)
                + at(i + 2, 1).mul_scalar(c)
                + at(i + 3, 1).mul_scalar(d)
        };
        let (first, inside, last) = match order {
            1 => (
                four(0, [-11.0, 18.0, -9.0, 2.0]).div_scalar(6.0 * h),
                (at(2, n - 2) - at(0, n - 2)).div_scalar(2.0 * h),
                four(n - 4, [-2.0, 9.0, -18.0, 11.0]).div_scalar(6.0 * h),
            ),
            _ => (
                four(0, [2.0, -5.0, 4.0, -1.0]).div_scalar(h * h),
                (at(2, n - 2) - at(1, n - 2).mul_scalar(2.0) + at(0, n - 2)).div_scalar(h * h),
                four(n - 4, [-1.0, 4.0, -5.0, 2.0]).div_scalar(h * h),
            ),
        };
        Tensor::cat(vec![first, inside, last], dim)
    }

    /// ∂u/∂x: [`partial`](Self::partial) on axis 0.
    ///
    /// # Panics
    /// As [`partial`](Self::partial).
    pub fn dx<const R: usize>(&self, u: Tensor<R>, order: usize) -> Tensor<R> {
        self.partial(u, 0, order)
    }

    /// ∂u/∂y: [`partial`](Self::partial) on axis 1.
    ///
    /// # Panics
    /// As [`partial`](Self::partial).
    pub fn dy<const R: usize>(&self, u: Tensor<R>, order: usize) -> Tensor<R> {
        self.partial(u, 1, order)
    }

    /// ∂u/∂z: [`partial`](Self::partial) on axis 2.
    ///
    /// # Panics
    /// As [`partial`](Self::partial).
    pub fn dz<const R: usize>(&self, u: Tensor<R>, order: usize) -> Tensor<R> {
        self.partial(u, 2, order)
    }

    /// Σₐ ∂ᵏu/∂xₐᵏ over every spatial axis, k = `order`: for k = 1 the sum of the first
    /// partials (2.0.0's `HdivLoss` term), for k = 2 the Laplacian. `u`: `[.., s_1, .., s_D]`;
    /// the result has the same shape.
    ///
    /// # Panics
    /// As [`partial`](Self::partial).
    pub fn partial_sum<const R: usize>(&self, u: Tensor<R>, order: usize) -> Tensor<R> {
        (1..D).fold(self.partial(u.clone(), 0, order), |acc, a| {
            acc + self.partial(u.clone(), a, order)
        })
    }

    /// The Laplacian Σₐ ∂²u/∂xₐ², [`partial_sum`](Self::partial_sum) of order 2.
    ///
    /// # Panics
    /// As [`partial`](Self::partial).
    pub fn laplacian<const R: usize>(&self, u: Tensor<R>) -> Tensor<R> {
        self.partial_sum(u, 2)
    }

    /// The gradient: `u` `[.., s_1, .., s_D]` to `[.., D, s_1, .., s_D]`, the components
    /// stacked on a new axis before the spatial ones. 2.0.0 omits that axis in 1D only; here
    /// it is always present.
    ///
    /// # Panics
    /// If `S != R + 1`, or as [`partial`](Self::partial).
    pub fn gradient<const R: usize, const S: usize>(&self, u: Tensor<R>) -> Tensor<S> {
        assert_eq!(S, R + 1, "FiniteDiff::gradient: output rank must be R + 1");
        let parts = (0..D).map(|a| self.partial(u.clone(), a, 1)).collect();
        Tensor::stack(parts, R - D)
    }

    /// The divergence Σₐ ∂vₐ/∂xₐ of a vector field `v` `[.., D, s_1, .., s_D]`, to
    /// `[.., s_1, .., s_D]`.
    ///
    /// # Panics
    /// If `S + 1 != R`, the component axis is not of length `D`, or as
    /// [`partial`](Self::partial).
    pub fn divergence<const R: usize, const S: usize>(&self, v: Tensor<R>) -> Tensor<S> {
        assert_eq!(
            S + 1,
            R,
            "FiniteDiff::divergence: output rank must be R − 1"
        );
        let c = self.component_axis(&v);
        (1..D)
            .fold(self.partial(v.clone().narrow(c, 0, 1), 0, 1), |acc, a| {
                acc + self.partial(v.clone().narrow(c, a, 1), a, 1)
            })
            .squeeze_dim(c)
    }

    /// The 2D curl ∂v₁/∂x − ∂v₀/∂y of a vector field `v` `[.., 2, s_1, s_2]`, to
    /// `[.., s_1, s_2]`.
    ///
    /// # Panics
    /// If `D != 2`, `S + 1 != R`, the component axis is not of length 2, or as
    /// [`partial`](Self::partial).
    pub fn curl_2d<const R: usize, const S: usize>(&self, v: Tensor<R>) -> Tensor<S> {
        assert_eq!(D, 2, "FiniteDiff::curl_2d in {D}D");
        assert_eq!(S + 1, R, "FiniteDiff::curl_2d: output rank must be R − 1");
        let c = self.component_axis(&v);
        let d = |i: usize, axis: usize| self.partial(v.clone().narrow(c, i, 1), axis, 1);
        (d(1, 0) - d(0, 1)).squeeze_dim(c)
    }

    /// The 3D curl (∂v₂/∂y − ∂v₁/∂z, ∂v₀/∂z − ∂v₂/∂x, ∂v₁/∂x − ∂v₀/∂y) of a vector field `v`
    /// `[.., 3, s_1, s_2, s_3]`, the same shape.
    ///
    /// # Panics
    /// If `D != 3`, the component axis is not of length 3, or as [`partial`](Self::partial).
    pub fn curl_3d<const R: usize>(&self, v: Tensor<R>) -> Tensor<R> {
        assert_eq!(D, 3, "FiniteDiff::curl_3d in {D}D");
        let c = self.component_axis(&v);
        let d = |i: usize, axis: usize| self.partial(v.clone().narrow(c, i, 1), axis, 1);
        Tensor::cat(
            vec![d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1)],
            c,
        )
    }

    /// The axis of a vector field's components, just before the spatial axes.
    fn component_axis<const R: usize>(&self, v: &Tensor<R>) -> usize {
        assert!(R > D, "FiniteDiff: a vector field needs a component axis");
        let c = R - D - 1;
        assert_eq!(
            v.dims()[c],
            D,
            "FiniteDiff: {} components on axis {c}, expected {D}",
            v.dims()[c]
        );
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::prelude::*;
    use std::f64::consts::PI;

    fn row(values: Vec<f64>) -> Tensor<2> {
        let n = values.len();
        Tensor::from_data(TensorData::new(values, [1, n]), &Device::default())
    }

    fn values(t: Tensor<2>) -> Vec<f64> {
        t.into_data().convert::<f64>().try_to_vec::<f64>().unwrap()
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
    }

    /// sin on a periodic grid of n points over [0, 2π): the central differences are within
    /// their truncation error h²/6 (first) and h²/12 (second) of cos and −sin.
    #[test]
    fn periodic_derivatives_of_sin_within_truncation_error() {
        let n = 64;
        let h = 2.0 * PI / n as f64;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 * h).collect();
        let u = row(xs.iter().map(|x| x.sin()).collect());
        let fd = FiniteDiff {
            h: [h],
            periodic: [true],
        };
        let cos: Vec<f64> = xs.iter().map(|x| x.cos()).collect();
        let neg_sin: Vec<f64> = xs.iter().map(|x| -x.sin()).collect();
        let e1 = max_abs_diff(&values(fd.dx(u.clone(), 1)), &cos);
        let e2 = max_abs_diff(&values(fd.dx(u, 2)), &neg_sin);
        assert!(
            e1 <= h * h / 6.0 && e1 > 0.0,
            "first derivative error {e1:e}"
        );
        assert!(
            e2 <= h * h / 12.0 && e2 > 0.0,
            "second derivative error {e2:e}"
        );
    }

    /// The one-sided ends are exact on a cubic (first derivative, third order) and the whole
    /// second derivative on a quadratic; the interior first derivative on a quadratic. The
    /// tests run in f32 on the default device: 1e-4 covers its rounding over h² = 1/16.
    #[test]
    fn one_sided_ends_exact_on_polynomials() {
        let h = 0.25;
        let xs: Vec<f64> = (0..7).map(|i| i as f64 * h).collect();
        let fd = FiniteDiff {
            h: [h],
            periodic: [false],
        };

        let cubic = row(xs.iter().map(|x| x * x * x - 2.0 * x).collect());
        let d1 = values(fd.dx(cubic, 1));
        let exact = |x: f64| 3.0 * x * x - 2.0;
        assert!((d1[0] - exact(xs[0])).abs() < 1e-4, "{}", d1[0]);
        assert!((d1[6] - exact(xs[6])).abs() < 1e-4, "{}", d1[6]);

        let quad = row(xs.iter().map(|x| 3.0 * x * x + x).collect());
        let d1 = values(fd.dx(quad.clone(), 1));
        let d2 = values(fd.dx(quad, 2));
        let exact1: Vec<f64> = xs.iter().map(|x| 6.0 * x + 1.0).collect();
        assert!(max_abs_diff(&d1, &exact1) < 1e-4, "{d1:?}");
        assert!(max_abs_diff(&d2, &[6.0; 7]) < 1e-4, "{d2:?}");
    }

    /// A 2D field: each axis differentiates only its own direction, the Laplacian sums them,
    /// and curl ∘ gradient vanishes up to f32 rounding (periodic central differences commute).
    #[test]
    fn axes_laplacian_and_curl_of_gradient() {
        let (nx, ny) = (8, 6);
        let data: Vec<f64> = (0..nx * ny).map(|k| ((k * 37 % 11) as f64).sin()).collect();
        let u: Tensor<3> =
            Tensor::from_data(TensorData::new(data, [1, nx, ny]), &Device::default());
        let fd = FiniteDiff {
            h: [0.1, 0.3],
            periodic: [true, true],
        };
        let lap = fd.laplacian(u.clone());
        let sum = fd.dx(u.clone(), 2) + fd.dy(u.clone(), 2);
        let err = (lap - sum).abs().max().into_scalar::<f64>();
        assert_eq!(err, 0.0);

        let grad: Tensor<4> = fd.gradient(u);
        assert_eq!(grad.dims(), [1, 2, nx, ny]);
        let curl: Tensor<3> = fd.curl_2d(grad);
        let err = curl.abs().max().into_scalar::<f64>();
        assert!(err < 1e-4, "curl of a gradient: {err:e}");
    }

    #[test]
    #[should_panic(expected = "needs at least 4")]
    fn non_periodic_axis_needs_four_points() {
        FiniteDiff {
            h: [1.0],
            periodic: [false],
        }
        .dx(row(vec![0.0, 1.0, 2.0]), 1);
    }

    #[test]
    #[should_panic(expected = "axis 1 of 1")]
    fn axis_beyond_d_panics() {
        FiniteDiff {
            h: [1.0],
            periodic: [true],
        }
        .dy(row(vec![0.0; 4]), 1);
    }

    #[test]
    #[should_panic(expected = "only 1 and 2")]
    fn third_order_panics() {
        FiniteDiff {
            h: [1.0],
            periodic: [true],
        }
        .dx(row(vec![0.0; 4]), 3);
    }

    #[test]
    #[should_panic(expected = "curl_2d in 1D")]
    fn curl_in_1d_panics() {
        let v: Tensor<3> = Tensor::zeros([1, 1, 4], &Device::default());
        let _: Tensor<3> = FiniteDiff {
            h: [1.0],
            periodic: [true],
        }
        .curl_2d(v);
    }
}

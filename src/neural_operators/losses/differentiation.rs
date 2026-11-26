//! Derivatives on a uniform grid, as `neuraloperator` 2.0.0's `FiniteDiff` (library CM1.6a)
//! and `FourierDiff` (CM1.6c; `neuralop/losses/differentiation.py`).
//!
//! The last `D` axes of a tensor are the spatial axes x, y, z, …; every axis before them is a
//! batch-like axis the stencils leave alone. Along a periodic axis the stencils wrap round;
//! along a non-periodic one they are central inside and one-sided at the two ends, with
//! 2.0.0's coefficients. 2.0.0's `central_diff_{1,2,3}d` are [`FiniteDiff::partial`] on each
//! axis. [`FourierDiff`] differentiates spectrally on a periodic grid. [`GridDiff`] is any such
//! operator, with the vector-calculus operators on top; [`MatrixDiff`] one by a matrix per axis
//! (library CM2.5).

use burn::prelude::Device;
use burn::tensor::{Tensor, TensorData, signal};
use ndarray::Array2;
use std::f64::consts::TAU;

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
}

/// Partial derivatives along the spatial axes of a grid: the last `D` axes of a tensor, x
/// first, as [`FiniteDiff`], [`FourierDiff`] and [`MatrixDiff`] (library CM2.5, for the
/// equation losses), and the vector-calculus operators built on them (CM1.6c).
pub trait GridDiff<const D: usize> {
    /// The derivative of order `order` of `u` along spatial axis `axis` (0 = x): orders 1 and
    /// 2 for [`FiniteDiff`] and [`MatrixDiff`], any order from 1 for [`FourierDiff`].
    /// `u`: `[.., s_1, .., s_D]`; the result has the same shape.
    ///
    /// # Panics
    /// If `R < D`, `axis >= D`, the operator has no derivative of that order, or the axis
    /// does not fit the operator.
    fn partial<const R: usize>(&self, u: Tensor<R>, axis: usize, order: usize) -> Tensor<R>;

    /// The Laplacian Σₐ ∂²u/∂xₐ². `u`: `[.., s_1, .., s_D]`; the result has the same shape.
    ///
    /// # Panics
    /// As [`partial`](Self::partial).
    fn laplacian<const R: usize>(&self, u: Tensor<R>) -> Tensor<R> {
        (1..D).fold(self.partial(u.clone(), 0, 2), |acc, a| {
            acc + self.partial(u.clone(), a, 2)
        })
    }

    /// ∂u/∂x: [`partial`](Self::partial) on axis 0.
    ///
    /// # Panics
    /// As [`partial`](Self::partial).
    fn dx<const R: usize>(&self, u: Tensor<R>, order: usize) -> Tensor<R> {
        self.partial(u, 0, order)
    }

    /// ∂u/∂y: [`partial`](Self::partial) on axis 1.
    ///
    /// # Panics
    /// As [`partial`](Self::partial).
    fn dy<const R: usize>(&self, u: Tensor<R>, order: usize) -> Tensor<R> {
        self.partial(u, 1, order)
    }

    /// ∂u/∂z: [`partial`](Self::partial) on axis 2.
    ///
    /// # Panics
    /// As [`partial`](Self::partial).
    fn dz<const R: usize>(&self, u: Tensor<R>, order: usize) -> Tensor<R> {
        self.partial(u, 2, order)
    }

    /// The gradient: `u` `[.., s_1, .., s_D]` to `[.., D, s_1, .., s_D]`, the components
    /// stacked on a new axis before the spatial ones. 2.0.0's `FiniteDiff` omits that axis in
    /// 1D only (its `FourierDiff` keeps it); here it is always present.
    ///
    /// # Panics
    /// If `S != R + 1`, or as [`partial`](Self::partial).
    fn gradient<const R: usize, const S: usize>(&self, u: Tensor<R>) -> Tensor<S> {
        assert_eq!(S, R + 1, "GridDiff::gradient: output rank must be R + 1");
        let parts = (0..D).map(|a| self.partial(u.clone(), a, 1)).collect();
        Tensor::stack(parts, R - D)
    }

    /// The divergence Σₐ ∂vₐ/∂xₐ of a vector field `v` `[.., D, s_1, .., s_D]`, to
    /// `[.., s_1, .., s_D]`.
    ///
    /// # Panics
    /// If `S + 1 != R`, the component axis is not of length `D`, or as
    /// [`partial`](Self::partial).
    fn divergence<const R: usize, const S: usize>(&self, v: Tensor<R>) -> Tensor<S> {
        assert_eq!(S + 1, R, "GridDiff::divergence: output rank must be R − 1");
        let c = component_axis::<D, R>(&v);
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
    fn curl_2d<const R: usize, const S: usize>(&self, v: Tensor<R>) -> Tensor<S> {
        assert_eq!(D, 2, "GridDiff::curl_2d in {D}D");
        assert_eq!(S + 1, R, "GridDiff::curl_2d: output rank must be R − 1");
        let c = component_axis::<D, R>(&v);
        let d = |i: usize, axis: usize| self.partial(v.clone().narrow(c, i, 1), axis, 1);
        (d(1, 0) - d(0, 1)).squeeze_dim(c)
    }

    /// The 3D curl (∂v₂/∂y − ∂v₁/∂z, ∂v₀/∂z − ∂v₂/∂x, ∂v₁/∂x − ∂v₀/∂y) of a vector field `v`
    /// `[.., 3, s_1, s_2, s_3]`, the same shape.
    ///
    /// # Panics
    /// If `D != 3`, the component axis is not of length 3, or as [`partial`](Self::partial).
    fn curl_3d<const R: usize>(&self, v: Tensor<R>) -> Tensor<R> {
        assert_eq!(D, 3, "GridDiff::curl_3d in {D}D");
        let c = component_axis::<D, R>(&v);
        let d = |i: usize, axis: usize| self.partial(v.clone().narrow(c, i, 1), axis, 1);
        Tensor::cat(
            vec![d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1)],
            c,
        )
    }
}

/// The axis of a vector field's components, just before the spatial axes.
fn component_axis<const D: usize, const R: usize>(v: &Tensor<R>) -> usize {
    assert!(R > D, "GridDiff: a vector field needs a component axis");
    let c = R - D - 1;
    assert_eq!(
        v.dims()[c],
        D,
        "GridDiff: {} components on axis {c}, expected {D}",
        v.dims()[c]
    );
    c
}

impl<const D: usize> GridDiff<D> for FiniteDiff<D> {
    fn partial<const R: usize>(&self, u: Tensor<R>, axis: usize, order: usize) -> Tensor<R> {
        FiniteDiff::partial(self, u, axis, order)
    }
}

/// Spectral derivatives on a periodic uniform grid, as 2.0.0's `FourierDiff` without Fourier
/// continuation and without its low-pass filter (library CM1.6c).
///
/// Along an axis of n points on a domain of length L (step L / n, the point L left out), the
/// derivative of order p multiplies the `rfft` coefficient of wavenumber k = 2π m / L,
/// m = 0, …, ⌊n/2⌋, by (i k)ᵖ and transforms back by `irfft`. 2.0.0 takes `fftn` and the real
/// part of `ifftn` in 2D and 3D; for a derivative along one axis that is the same per-axis
/// transform. At the Nyquist bin of an even n an odd order contributes nothing to the real
/// part, so its multiplier is 0 here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FourierDiff<const D: usize> {
    /// The length of the periodic domain along each spatial axis (2.0.0's `L`; its default
    /// is 2π, [`TAU`](std::f64::consts::TAU)).
    pub l: [f64; D],
}

impl<const D: usize> FourierDiff<D> {
    /// The derivative of order `order` (any from 1) of `u` along spatial axis `axis` (0 = x).
    ///
    /// `u`: `[.., s_1, .., s_D]`; the result has the same shape.
    ///
    /// # Panics
    /// If `R < D`, `axis >= D` or `order == 0`.
    pub fn partial<const R: usize>(&self, u: Tensor<R>, axis: usize, order: usize) -> Tensor<R> {
        assert!(R >= D, "FourierDiff: rank {R} below the {D} spatial axes");
        assert!(axis < D, "FourierDiff: axis {axis} of {D}");
        assert!(order >= 1, "FourierDiff: order 0");
        let dim = R - D + axis;
        let n = u.dims()[dim];
        let bins = n / 2 + 1;
        // (i k)ᵖ = iᵖ kᵖ: the real factor ±kᵖ, times i for an odd p.
        let sign = if order % 4 < 2 { 1.0 } else { -1.0 };
        let odd = order % 2 == 1;
        let factor: Vec<f64> = (0..bins)
            .map(|m| match odd && 2 * m == n {
                true => 0.0,
                false => sign * (TAU * m as f64 / self.l[axis]).powi(order as i32),
            })
            .collect();
        let mut shape = [1; R];
        shape[dim] = bins;
        let factor =
            Tensor::<1>::from_data(TensorData::new(factor, [bins]), &u.device()).reshape(shape);
        let (re, im) = signal::rfft(u, dim, None);
        let (re, im) = match odd {
            // i · (re + i im) = −im + i re
            true => ((im * factor.clone()).neg(), re * factor),
            false => (re * factor.clone(), im * factor),
        };
        signal::irfft(re, im, dim, Some(n))
    }
}

impl<const D: usize> GridDiff<D> for FourierDiff<D> {
    fn partial<const R: usize>(&self, u: Tensor<R>, axis: usize, order: usize) -> Tensor<R> {
        FourierDiff::partial(self, u, axis, order)
    }
}

/// Differentiation by a matrix per spatial axis: `(∂ᵏu)[.., i, ..] = Σⱼ Mₖ[i, j] u[.., j, ..]`
/// along the axis, with `M₁` the first- and `M₂` the second-derivative matrix.
#[derive(Clone, Debug)]
pub struct MatrixDiff<const D: usize> {
    /// Per axis, the transposes of M₁ and M₂: a contraction over the last axis is `u · Mᵀ`.
    first_t: [Tensor<2>; D],
    second_t: [Tensor<2>; D],
}

impl<const D: usize> MatrixDiff<D> {
    /// From the first- and second-derivative matrices of each axis, each `[n_a, n_a]` for
    /// that axis's `n_a` points, on `device`.
    ///
    /// # Panics
    /// If a matrix is not square, or an axis's two matrices differ in size.
    pub fn new(first: [Array2<f64>; D], second: [Array2<f64>; D], device: &Device) -> Self {
        let transposed = |m: &Array2<f64>| {
            let (rows, cols) = m.dim();
            assert_eq!(rows, cols, "MatrixDiff: a {rows} × {cols} matrix");
            let values = m.t().iter().copied().collect();
            Tensor::from_data(TensorData::new(values, [rows, cols]), device)
        };
        for (a, (m1, m2)) in first.iter().zip(&second).enumerate() {
            assert_eq!(
                m1.dim(),
                m2.dim(),
                "MatrixDiff: axis {a}'s matrices differ in size"
            );
        }
        Self {
            first_t: first.each_ref().map(transposed),
            second_t: second.each_ref().map(transposed),
        }
    }

    /// The Chebyshev differentiation matrices ([`diff_matrix`], [`diff2_matrix`]) on the
    /// `n[a]` ascending CGL nodes of [−1, 1] per axis (CONVENTIONS §12): spectral
    /// derivatives of the polynomial interpolant. On another interval [a, b], scale the
    /// first derivative by 2 / (b − a) and the second by its square.
    ///
    /// [`diff_matrix`]: crate::neural_operators::chebyshev::diff_matrix
    /// [`diff2_matrix`]: crate::neural_operators::chebyshev::diff2_matrix
    ///
    /// # Panics
    /// If some `n[a] < 2`.
    #[cfg(feature = "datagen")]
    pub fn chebyshev(n: [usize; D], device: &Device) -> Self {
        use crate::neural_operators::chebyshev::{diff_matrix, diff2_matrix};
        Self::new(n.map(diff_matrix), n.map(diff2_matrix), device)
    }
}

impl<const D: usize> GridDiff<D> for MatrixDiff<D> {
    fn partial<const R: usize>(&self, u: Tensor<R>, axis: usize, order: usize) -> Tensor<R> {
        assert!(R >= D, "MatrixDiff: rank {R} below the {D} spatial axes");
        assert!(axis < D, "MatrixDiff: axis {axis} of {D}");
        let m = match order {
            1 => &self.first_t[axis],
            2 => &self.second_t[axis],
            _ => panic!("MatrixDiff: order {order}; only 1 and 2"),
        };
        let n = m.dims()[0];
        let dim = R - D + axis;
        assert_eq!(
            u.dims()[dim],
            n,
            "MatrixDiff: axis {axis} has {} points, its matrix {n}",
            u.dims()[dim]
        );
        let moved = u.swap_dims(dim, R - 1);
        let shape = moved.dims();
        let rows: Tensor<2> = moved.reshape([-1, n as i32]);
        rows.matmul(m.clone()).reshape(shape).swap_dims(dim, R - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    /// A trigonometric polynomial below the Nyquist mode, u = sin(3ωx) + cos(5ωx) with
    /// ω = 2π / L, on odd and even grids: orders 1–3 equal the closed forms up to f32 rounding
    /// (relative 1e-4).
    #[test]
    fn fourier_exact_on_a_trigonometric_polynomial() {
        let l = 2.0;
        let w = TAU / l;
        for n in [15, 16] {
            let xs: Vec<f64> = (0..n).map(|i| i as f64 * l / n as f64).collect();
            let u = row(xs
                .iter()
                .map(|x| (3.0 * w * x).sin() + (5.0 * w * x).cos())
                .collect());
            let fd = FourierDiff { l: [l] };
            // d^p/dx^p sin(ax) = a^p sin(ax + pπ/2), and cos likewise.
            for p in 1..=3 {
                let shift = p as f64 * PI / 2.0;
                let exact: Vec<f64> = xs
                    .iter()
                    .map(|x| {
                        (3.0 * w).powi(p) * (3.0 * w * x + shift).sin()
                            + (5.0 * w).powi(p) * (5.0 * w * x + shift).cos()
                    })
                    .collect();
                let got = values(fd.dx(u.clone(), p as usize));
                let scale = exact.iter().fold(0.0_f64, |m, e| m.max(e.abs()));
                let err = max_abs_diff(&got, &exact) / scale;
                assert!(err < 1e-4, "n {n}, order {p}: {err:e}");
            }
        }
    }

    /// The Nyquist mode (−1)ʲ of an even grid: odd orders give 0 (2.0.0's real part), order 2
    /// gives −k² u with k = π n / L.
    #[test]
    fn fourier_nyquist_mode() {
        let (n, l) = (8, 2.0);
        let u: Vec<f64> = (0..n)
            .map(|j| if j % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let fd = FourierDiff { l: [l] };
        let k = PI * n as f64 / l;
        for p in [1, 3] {
            let got = values(fd.dx(row(u.clone()), p));
            assert!(got.iter().all(|x| x.abs() < 1e-4), "order {p}: {got:?}");
        }
        let got = values(fd.dx(row(u.clone()), 2));
        let exact: Vec<f64> = u.iter().map(|x| -k * k * x).collect();
        assert!(max_abs_diff(&got, &exact) < 1e-4 * k * k, "{got:?}");
    }

    #[test]
    #[should_panic(expected = "order 0")]
    fn fourier_order_zero_panics() {
        FourierDiff { l: [1.0] }.partial(row(vec![0.0; 4]), 0, 0);
    }

    #[test]
    #[should_panic(expected = "axis 1 of 1")]
    fn fourier_axis_beyond_d_panics() {
        FourierDiff { l: [1.0] }.partial(row(vec![0.0; 4]), 1, 1);
    }

    /// The periodic central-difference matrices of `n` points with step `h`.
    fn periodic_matrices(n: usize, h: f64) -> (Array2<f64>, Array2<f64>) {
        let at = |i: usize, j: usize, c: [f64; 3]| {
            if j == (i + n - 1) % n {
                c[0]
            } else if j == i {
                c[1]
            } else if j == (i + 1) % n {
                c[2]
            } else {
                0.0
            }
        };
        let d1 = Array2::from_shape_fn((n, n), |(i, j)| at(i, j, [-0.5 / h, 0.0, 0.5 / h]));
        let d2 = Array2::from_shape_fn((n, n), |(i, j)| {
            at(i, j, [1.0 / (h * h), -2.0 / (h * h), 1.0 / (h * h)])
        });
        (d1, d2)
    }

    /// `MatrixDiff` with the periodic central-difference matrices is `FiniteDiff` periodic,
    /// on both axes of a non-square grid, up to f32 rounding (≤ 9.8e-8 measured); so is
    /// the Laplacian of `GridDiff`'s default.
    #[test]
    fn matrix_diff_matches_finite_diff() {
        let device = Device::default();
        let (h, (nx, ny)) = ([0.3, 0.2], (8, 6));
        let (ax, bx) = periodic_matrices(nx, h[0]);
        let (ay, by) = periodic_matrices(ny, h[1]);
        let md = MatrixDiff::new([ax, ay], [bx, by], &device);
        let fd = FiniteDiff {
            h,
            periodic: [true, true],
        };
        let data: Vec<f64> = (0..2 * nx * ny)
            .map(|k| ((k * 37 % 11) as f64).sin())
            .collect();
        let u: Tensor<3> = Tensor::from_data(TensorData::new(data, [2, nx, ny]), &device);
        let rel = |a: Tensor<3>, b: Tensor<3>| {
            let scale = b.clone().abs().max().into_scalar::<f64>();
            (a - b).abs().max().into_scalar::<f64>() / scale
        };
        for axis in 0..2 {
            for order in [1, 2] {
                let a = GridDiff::partial(&md, u.clone(), axis, order);
                let err = rel(a, fd.partial(u.clone(), axis, order));
                assert!(err < 1e-6, "axis {axis} order {order}: {err:e}");
            }
        }
        let err = rel(GridDiff::laplacian(&md, u.clone()), fd.laplacian(u));
        assert!(err < 1e-6, "laplacian: {err:e}");
    }

    /// Gradients flow through `MatrixDiff`: d Σ (M u)² / du = 2 Mᵀ M u, finite and non-zero.
    #[test]
    fn matrix_diff_gradient_flows() {
        let device = Device::default().autodiff();
        let (a, b) = periodic_matrices(6, 0.2);
        let md = MatrixDiff::new([a], [b], &device);
        let values: Vec<f64> = (0..6).map(|i| (i as f64).sin()).collect();
        let u = Tensor::<2>::from_data(TensorData::new(values, [1, 6]), &device).require_grad();
        let loss = GridDiff::partial(&md, u.clone(), 0, 1).square().sum();
        let g = u.grad(&loss.backward()).unwrap();
        let g = g.abs().max().into_scalar::<f64>();
        assert!(g.is_finite() && g > 0.0, "{g}");
    }

    /// The Chebyshev matrices differentiate x³ exactly on 6 CGL nodes, up to f32 rounding
    /// (≤ 3.3e-6 measured, in 2D).
    #[cfg(feature = "datagen")]
    #[test]
    fn chebyshev_exact_on_a_cubic() {
        use crate::neural_operators::chebyshev::nodes;
        let x = nodes(6).to_vec();
        let md = MatrixDiff::chebyshev([6], &Device::default());
        let u = row(x.iter().map(|x| x.powi(3)).collect());
        let d1 = values(GridDiff::partial(&md, u.clone(), 0, 1));
        let d2 = values(GridDiff::partial(&md, u, 0, 2));
        let exact1: Vec<f64> = x.iter().map(|x| 3.0 * x * x).collect();
        let exact2: Vec<f64> = x.iter().map(|x| 6.0 * x).collect();
        assert!(max_abs_diff(&d1, &exact1) < 1e-4, "{d1:?}");
        assert!(max_abs_diff(&d2, &exact2) < 1e-4, "{d2:?}");
    }

    #[test]
    #[should_panic(expected = "has 5 points, its matrix 6")]
    fn matrix_diff_size_mismatch_panics() {
        let (a, b) = periodic_matrices(6, 1.0);
        let md = MatrixDiff::new([a], [b], &Device::default());
        GridDiff::partial(&md, row(vec![0.0; 5]), 0, 1);
    }
}

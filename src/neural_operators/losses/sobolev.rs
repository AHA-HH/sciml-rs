//! The H1 and Hdiv Sobolev losses of `neuraloperator` 2.0.0 (`H1Loss`, `HdivLoss` in
//! `neuralop/losses/data_losses.py`; library CM1.6a).
//!
//! As 2.0.0, and unlike [`LpLoss`](super::data_losses::LpLoss) (CONVENTIONS §11): the
//! quadrature weight along spatial axis a is h_a = measure_a / s_a, the spacing of a periodic
//! grid of s_a points, and the same h_a is the finite-difference step; the relative loss adds
//! `eps` to the denominator.

use burn::tensor::Tensor;

use super::{
    data_losses::{Reduction, root_zero_safe},
    differentiation::FiniteDiff,
};

/// Which Sobolev norm a [`SobolevLoss`] measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SobolevNorm {
    /// ‖u‖² = Σ u² + Σₐ Σ (∂u/∂xₐ)².
    H1,
    /// ‖u‖² = Σ u² + Σ (Σₐ ∂u/∂xₐ)², 2.0.0's `HdivLoss` (the sum of the partials of each
    /// channel, not the divergence of a vector field).
    Hdiv,
}

/// The H1 or Hdiv loss between fields on `D` spatial axes, 2.0.0's `H1Loss` / `HdivLoss`.
///
/// For each index of the leading (batch, channel) axes, with h_a = measure_a / s_a and the
/// sums over the grid:
/// - `abs`: (Π h_a · ‖x − y‖²)^½;
/// - `rel`: ‖x − y‖ / (‖y‖ + eps);
///
/// then reduced over those indices by `reduction`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SobolevLoss<const D: usize> {
    /// H1 or Hdiv.
    pub norm: SobolevNorm,
    /// The domain's extent along each spatial axis (2.0.0's `measure`), default 1.
    pub measure: [f64; D],
    /// Whether each spatial axis is periodic, for the finite differences; default all.
    pub periodic: [bool; D],
    /// How the per-index losses are combined; 2.0.0 offers `Sum` (its default) and `Mean`.
    pub reduction: Reduction,
    /// Added to the denominator of `rel`, default 1e-8.
    pub eps: f64,
}

impl<const D: usize> SobolevLoss<D> {
    /// 2.0.0's `H1Loss(d=D, reduction=…)` at its other defaults.
    pub fn h1(reduction: Reduction) -> Self {
        Self::with_norm(SobolevNorm::H1, reduction)
    }

    /// 2.0.0's `HdivLoss(d=D, reduction=…)` at its other defaults.
    pub fn hdiv(reduction: Reduction) -> Self {
        Self::with_norm(SobolevNorm::Hdiv, reduction)
    }

    fn with_norm(norm: SobolevNorm, reduction: Reduction) -> Self {
        Self {
            norm,
            measure: [1.0; D],
            periodic: [true; D],
            reduction,
            eps: 1e-8,
        }
    }

    /// The absolute loss. `x`, `y`: `[.., s_1, .., s_D]`, the same shape. Returns `[1]`, or
    /// for `Reduction::None` one value per leading index, `[n]` in row-major order.
    ///
    /// # Panics
    /// If the shapes differ, or as [`FiniteDiff::partial`].
    pub fn abs<const R: usize>(&self, x: Tensor<R>, y: Tensor<R>) -> Tensor<1> {
        let h = self.quadrature(&x);
        let (diff, _) = self.squared_norms(x, y, h);
        self.reduction.apply(root_zero_safe(
            diff.mul_scalar(h.iter().product::<f64>()),
            2,
        ))
    }

    /// The relative loss. Shapes and returns as [`abs`](Self::abs).
    ///
    /// # Panics
    /// As [`abs`](Self::abs).
    pub fn rel<const R: usize>(&self, x: Tensor<R>, y: Tensor<R>) -> Tensor<1> {
        let h = self.quadrature(&x);
        let (diff, ynorm) = self.squared_norms(x, y, h);
        self.reduction
            .apply(root_zero_safe(diff, 2) / root_zero_safe(ynorm, 2).add_scalar(self.eps))
    }

    /// h_a = measure_a / s_a.
    fn quadrature<const R: usize>(&self, x: &Tensor<R>) -> [f64; D] {
        assert!(R >= D, "SobolevLoss: rank {R} below the {D} spatial axes");
        let dims = x.dims();
        std::array::from_fn(|a| self.measure[a] / dims[R - D + a] as f64)
    }

    /// Σ over the grid of the squared norm's terms, for x − y and for y: `[n]` each.
    fn squared_norms<const R: usize>(
        &self,
        x: Tensor<R>,
        y: Tensor<R>,
        h: [f64; D],
    ) -> (Tensor<1>, Tensor<1>) {
        assert_eq!(x.dims(), y.dims(), "SobolevLoss: x and y shapes differ");
        let fd = FiniteDiff {
            h,
            periodic: self.periodic,
        };
        let dims = x.dims();
        let rows: usize = dims[..R - D].iter().product();
        let sum_sq = |t: Tensor<R>| {
            t.square()
                .reshape([rows as i32, -1])
                .sum_dim(1)
                .reshape([rows])
        };
        // The derivative terms of the norm: each first partial (H1), or their sum (Hdiv).
        let derivatives = |t: Tensor<R>| match self.norm {
            SobolevNorm::H1 => (0..D).map(|a| fd.partial(t.clone(), a, 1)).collect(),
            SobolevNorm::Hdiv => vec![fd.partial_sum(t, 1)],
        };
        let mut diff = sum_sq(x.clone() - y.clone());
        let mut ynorm = sum_sq(y.clone());
        for (dx, dy) in derivatives(x).into_iter().zip(derivatives(y)) {
            diff = diff + sum_sq(dx - dy.clone());
            ynorm = ynorm + sum_sq(dy);
        }
        (diff, ynorm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::prelude::*;

    fn field(values: &[f64], shape: [usize; 3]) -> Tensor<3> {
        Tensor::from_data(TensorData::new(values.to_vec(), shape), &Device::default())
    }

    /// One field of 4 periodic points, x = (1, 0, 0, 0), y = 0: by hand, h = 1/4,
    /// ∂x = (0, −2, 0, 2), so Σ (x − y)² + Σ (∂x − ∂y)² = 1 + 8 = 9 and abs = √(9 / 4) = 1.5
    /// (H1 and, in 1D, Hdiv alike).
    #[test]
    fn abs_known_value() {
        let x = field(&[1.0, 0.0, 0.0, 0.0], [1, 1, 4]);
        let y = field(&[0.0; 4], [1, 1, 4]);
        for loss in [
            SobolevLoss::<1>::h1(Reduction::Sum),
            SobolevLoss::hdiv(Reduction::Sum),
        ] {
            let v = loss.abs(x.clone(), y.clone()).into_scalar::<f64>();
            assert!((v - 1.5).abs() < 1e-6, "{:?}: {v}", loss.norm);
        }
    }

    /// At an exact match both losses are 0 with gradient 0, not NaN.
    #[test]
    fn exact_match_gives_zero_and_zero_gradient() {
        let device = Device::default().autodiff();
        let values: Vec<f64> = (0..2 * 3 * 5).map(|k| (k as f64 * 0.7).sin()).collect();
        let y: Tensor<4> = Tensor::from_data(TensorData::new(values, [1, 2, 5, 3]), &device);
        for norm in [SobolevNorm::H1, SobolevNorm::Hdiv] {
            let loss = SobolevLoss {
                norm,
                periodic: [false, true],
                ..SobolevLoss::<2>::h1(Reduction::Sum)
            };
            for rel in [true, false] {
                let x = y.clone().detach().require_grad();
                let out = if rel {
                    loss.rel(x.clone(), y.clone().detach())
                } else {
                    loss.abs(x.clone(), y.clone().detach())
                };
                let grads = out.clone().backward();
                let g = x.grad(&grads).unwrap();
                let value = out.into_scalar::<f64>();
                let gmax = g.abs().max().into_scalar::<f64>();
                assert_eq!(value, 0.0, "{norm:?} rel={rel}");
                assert_eq!(gmax, 0.0, "{norm:?} rel={rel}: gradient {gmax}");
            }
        }
    }
}

//! The pointwise losses of `neuraloperator` 2.0.0 (`PointwiseQuantileLoss`, `MSELoss` in
//! `neuralop/losses/data_losses.py`; library CM1.6a).

use burn::tensor::Tensor;

use super::data_losses::Reduction;

/// 2.0.0's `PointwiseQuantileLoss`: the pinball loss of a predicted bound on |y|, for UQNO.
///
/// With q = 1 − alpha, d = |y| − ŷ and the scale σ = max over the batch of |y| (per point)
/// plus `eps`: each point contributes max(q·d, −(1 − q)·d) / (2 q (1 − q) σ); each example
/// is the mean over its points, and the examples are reduced by `reduction`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointwiseQuantileLoss {
    /// The miscoverage level α: the bound targets the 1 − α quantile.
    pub alpha: f64,
    /// How the per-example losses are combined; 2.0.0 offers `Sum` (its default) and `Mean`.
    pub reduction: Reduction,
    /// Added to the scale σ, default 1e-7 (2.0.0's call argument).
    pub eps: f64,
}

impl PointwiseQuantileLoss {
    /// 2.0.0's `PointwiseQuantileLoss(alpha, reduction=…)`, with its `eps` of 1e-7.
    ///
    /// # Panics
    /// Unless 0 < `alpha` < 1.
    pub fn new(alpha: f64, reduction: Reduction) -> Self {
        assert!(
            alpha > 0.0 && alpha < 1.0,
            "PointwiseQuantileLoss: alpha {alpha} outside (0, 1)"
        );
        Self {
            alpha,
            reduction,
            eps: 1e-7,
        }
    }

    /// The loss of the prediction `y_pred` against `y`, both `[batch, ..]` of one shape.
    /// Returns `[1]`, or `[batch]` for `Reduction::None`.
    ///
    /// # Panics
    /// If the shapes differ.
    pub fn loss<const R: usize>(&self, y_pred: Tensor<R>, y: Tensor<R>) -> Tensor<1> {
        assert_eq!(
            y_pred.dims(),
            y.dims(),
            "PointwiseQuantileLoss: shapes differ"
        );
        let q = 1.0 - self.alpha;
        let y_abs = y.abs();
        let diff = y_abs.clone() - y_pred;
        let scale = y_abs.max_dim(0).add_scalar(self.eps);
        let pointwise = diff
            .clone()
            .mul_scalar(q)
            .max_pair(diff.mul_scalar(-(1.0 - q)));
        let scaled = pointwise.div_scalar(2.0).div_scalar(q).div_scalar(1.0 - q) / scale;
        let batch = scaled.dims()[0];
        let per_example = scaled
            .reshape([batch as i32, -1])
            .mean_dim(1)
            .reshape([batch]);
        match self.reduction {
            Reduction::Mean => per_example.mean(),
            Reduction::Sum => per_example.sum(),
            Reduction::None => per_example,
        }
    }
}

/// 2.0.0's `MSELoss()`: the mean squared error of each example over all its non-batch
/// entries, summed over the batch. `y_pred`, `y`: `[batch, ..]` of one shape; returns `[1]`.
///
/// # Panics
/// If the shapes differ.
pub fn mse_loss<const R: usize>(y_pred: Tensor<R>, y: Tensor<R>) -> Tensor<1> {
    assert_eq!(y_pred.dims(), y.dims(), "mse_loss: shapes differ");
    let batch = y.dims()[0];
    (y_pred - y)
        .square()
        .reshape([batch as i32, -1])
        .mean_dim(1)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::prelude::*;

    fn t(values: &[f64]) -> Tensor<2> {
        Tensor::from_data(TensorData::new(values.to_vec(), [2, 2]), &Device::default())
    }

    /// By hand: errors (1, −1; 2, 0): example means 1 and 2, summed 3.
    #[test]
    fn mse_known_value() {
        let v = mse_loss(t(&[1.0, 0.0, 2.0, 0.0]), t(&[0.0, 1.0, 0.0, 0.0])).into_scalar::<f64>();
        assert!((v - 3.0).abs() < 1e-6, "{v}");
    }

    /// By hand, alpha = 0.5 (q = 0.5) and eps = 0: y = (1, −2; 3, 2), ŷ = 0, so d = |y| and
    /// σ = (3, 2); each point gives 0.5·d / (2·0.25·σ) = d / σ: (1/3, 1; 1, 1), example
    /// means 2/3 and 1, summed 5/3.
    #[test]
    fn quantile_known_value() {
        let loss = PointwiseQuantileLoss {
            eps: 0.0,
            ..PointwiseQuantileLoss::new(0.5, Reduction::Sum)
        };
        let v = loss
            .loss(t(&[0.0; 4]), t(&[1.0, -2.0, 3.0, 2.0]))
            .into_scalar::<f64>();
        assert!((v - 5.0 / 3.0).abs() < 1e-6, "{v}");
    }

    /// At an exact match (ŷ = y for MSE, ŷ = |y| for the quantile loss) both are 0. MSE's
    /// gradient is 0; the pinball loss has a kink at d = 0, so its gradient is only finite.
    #[test]
    fn exact_match_gives_zero_and_finite_gradient() {
        let device = Device::default().autodiff();
        let y = Tensor::<2>::from_data([[1.0, -2.0], [3.0, 0.5]], &device);
        let x = y.clone().detach().require_grad();
        let mse = mse_loss(x.clone(), y.clone().detach());
        let g = x.grad(&mse.clone().backward()).unwrap();
        assert_eq!(mse.into_scalar::<f64>(), 0.0);
        assert_eq!(g.abs().max().into_scalar::<f64>(), 0.0);

        let x = y.clone().abs().detach().require_grad();
        let loss = PointwiseQuantileLoss::new(0.1, Reduction::Sum).loss(x.clone(), y.detach());
        let g = x.grad(&loss.clone().backward()).unwrap();
        assert_eq!(loss.into_scalar::<f64>(), 0.0);
        assert!(g.abs().max().into_scalar::<f64>().is_finite());
    }

    #[test]
    #[should_panic(expected = "outside (0, 1)")]
    fn quantile_rejects_alpha_one() {
        PointwiseQuantileLoss::new(1.0, Reduction::Sum);
    }
}

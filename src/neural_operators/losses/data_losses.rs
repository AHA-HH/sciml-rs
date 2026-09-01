//! Lp-norm loss functions for neural operator training (absolute and relative).

use burn::tensor::Tensor;

/// Absolute or relative Lp loss between predicted and target fields.
///
/// `d`: spatial dimensionality, used only in `abs()`'s quadrature weight
/// `h^(d/p)` - it cancels out in `rel()`'s ratio, so has no effect there.
/// `p`: Lp norm order (1, 2, or general).
/// `size_average`: when `reduction` is true, mean over the batch if true,
/// sum if false.
/// `reduction`: whether to reduce over the batch at all, or return
/// per-example losses.
pub struct LpLoss {
    pub d: usize,
    pub p: usize,
    pub size_average: bool,
    pub reduction: bool,
}

impl LpLoss {
    pub fn new(d: usize, p: usize, size_average: bool, reduction: bool) -> Self {
        assert!(d > 0 && p > 0);
        Self {
            d,
            p,
            size_average,
            reduction,
        }
    }

    /// Absolute Lp loss, scaled by the uniform-mesh quadrature weight
    /// `h^(d/p)` where `h = 1/(n_points - 1)`.
    /// `x`, `y`: `[batch, n_points]`.
    pub fn abs(&self, x: Tensor<2>, y: Tensor<2>) -> Tensor<1> {
        let n_points = x.dims()[1];

        let h: f64 = 1.0 / (n_points as f64 - 1.0);
        let h_weight: f64 = h.powf(self.d as f64 / self.p as f64);

        let diff = (x - y).abs();
        // p=1: sum of abs diffs. p=2: Euclidean norm. general p: p-norm.
        let all_norms = match self.p {
            1 => diff.sum_dim(1).squeeze_dims(&[1]).mul_scalar(h_weight),
            2 => diff
                .powf_scalar(2.0)
                .sum_dim(1)
                .squeeze_dims(&[1])
                .sqrt()
                .mul_scalar(h_weight),
            p => diff
                .powf_scalar(p as f64)
                .sum_dim(1)
                .squeeze_dims(&[1])
                .powf_scalar(1.0 / p as f64)
                .mul_scalar(h_weight),
        };

        if self.reduction {
            if self.size_average {
                all_norms.mean()
            } else {
                all_norms.sum()
            }
        } else {
            all_norms // [batch] - per-example losses
        }
    }

    /// Relative Lp loss: `||x - y||_p / ||y||_p`. Quadrature weight cancels
    /// between numerator and denominator, so it's not applied here.
    /// `x`, `y`: `[batch, n_points]`.
    pub fn rel(&self, x: Tensor<2>, y: Tensor<2>) -> Tensor<1> {
        let diff_norms = match self.p {
            1 => (x - y.clone()).abs().sum_dim(1).squeeze_dims(&[1]),
            2 => (x - y.clone())
                .powf_scalar(2.0)
                .sum_dim(1)
                .squeeze_dims(&[1])
                .sqrt(),
            p => (x - y.clone())
                .abs()
                .powf_scalar(p as f64)
                .sum_dim(1)
                .squeeze_dims(&[1])
                .powf_scalar(1.0 / p as f64),
        };

        let y_norms = match self.p {
            1 => y.abs().sum_dim(1).squeeze_dims(&[1]),
            2 => y.powf_scalar(2.0).sum_dim(1).squeeze_dims(&[1]).sqrt(),
            p => y
                .abs()
                .powf_scalar(p as f64)
                .sum_dim(1)
                .squeeze_dims(&[1])
                .powf_scalar(1.0 / p as f64),
        };

        let per_example = diff_norms / y_norms;

        if self.reduction {
            if self.size_average {
                per_example.mean()
            } else {
                per_example.sum()
            }
        } else {
            per_example // [batch] - per-example losses
        }
    }

    /// Default call - relative loss.
    pub fn forward(&self, x: Tensor<2>, y: Tensor<2>) -> Tensor<1> {
        self.rel(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::prelude::*;

    fn device() -> Device {
        Default::default()
    }

    #[test]
    fn test_rel_identical_inputs_gives_zero() {
        // if pred == target, relative error should be exactly 0
        let x = Tensor::<2>::from_data([[1.0, 2.0, 3.0]], &device());
        let y = Tensor::<2>::from_data([[1.0, 2.0, 3.0]], &device());
        let loss = LpLoss::new(2, 2, true, true);
        let result = loss.forward(x, y).into_scalar::<f32>();
        approx::assert_relative_eq!(result, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rel_known_value() {
        // pred = [2, 2], target = [1, 1]
        // diff_norm = sqrt(1^2 + 1^2) = sqrt(2)
        // y_norm    = sqrt(1^2 + 1^2) = sqrt(2)
        // rel       = sqrt(2) / sqrt(2) = 1.0
        let x = Tensor::<2>::from_data([[2.0, 2.0]], &device());
        let y = Tensor::<2>::from_data([[1.0, 1.0]], &device());
        let loss = LpLoss::new(2, 2, true, true);
        let result = loss.forward(x, y).into_scalar::<f32>();
        approx::assert_relative_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rel_batch_mean_vs_sum() {
        // test 1: pred=[2,0], target=[1,0] -> diff_norm=1, y_norm=1 -> rel=1.0
        // test 2: pred=[3,0], target=[1,0] -> diff_norm=2, y_norm=1 -> rel=2.0
        let x = Tensor::<2>::from_data([[2.0, 0.0], [3.0, 0.0]], &device());
        let y = Tensor::<2>::from_data([[1.0, 0.0], [1.0, 0.0]], &device());

        let loss_mean = LpLoss::new(2, 2, true, true);
        let loss_sum = LpLoss::new(2, 2, false, true);

        let mean_result = loss_mean.forward(x.clone(), y.clone()).into_scalar::<f32>();
        let sum_result = loss_sum.forward(x, y).into_scalar::<f32>();

        approx::assert_relative_eq!(mean_result, 1.5, epsilon = 1e-10); // (1+2)/2
        approx::assert_relative_eq!(sum_result, 3.0, epsilon = 1e-10); // 1+2
    }

    #[test]
    fn test_rel_no_reduction_returns_per_example() {
        let x = Tensor::<2>::from_data([[2.0, 0.0], [3.0, 0.0]], &device());
        let y = Tensor::<2>::from_data([[1.0, 0.0], [1.0, 0.0]], &device());
        let loss = LpLoss::new(2, 2, true, false); // reduction=false

        let result = loss.forward(x, y);
        assert_eq!(result.dims(), [2]); // shape should be [batch]

        let vals = result.into_data().as_slice::<f32>().unwrap().to_vec();
        approx::assert_relative_eq!(vals[0], 1.0, epsilon = 1e-10);
        approx::assert_relative_eq!(vals[1], 2.0_f32, epsilon = 1e-10);
    }

    #[test]
    fn test_abs_identical_inputs_gives_zero() {
        let x = Tensor::<2>::from_data([[1.0, 2.0, 3.0, 4.0]], &device());
        let y = Tensor::<2>::from_data([[1.0, 2.0, 3.0, 4.0]], &device());
        let loss = LpLoss::new(1, 2, true, true);
        let result = loss.abs(x, y).into_scalar::<f32>();
        approx::assert_relative_eq!(result, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_abs_known_value() {
        // n_points=2, d=1, p=2: h = 1/(2-1) = 1.0, h_weight = 1.0^(1/2) = 1.0
        // diff = [1, 1], L2 norm = sqrt(2), * h_weight = sqrt(2)
        let x = Tensor::<2>::from_data([[2.0, 2.0]], &device());
        let y = Tensor::<2>::from_data([[1.0, 1.0]], &device());
        let loss = LpLoss::new(1, 2, true, true);
        let result = loss.abs(x, y).into_scalar::<f32>();
        approx::assert_relative_eq!(result, 2.0_f32.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_rel_l1_norm() {
        // diff = |[2,3] - [1,1]| = [1,2], L1 norm = 3
        // y L1 norm = |[1,1]| = 2
        // rel = 3/2 = 1.5
        let x = Tensor::<2>::from_data([[2.0, 3.0]], &device());
        let y = Tensor::<2>::from_data([[1.0, 1.0]], &device());
        let loss = LpLoss::new(2, 1, true, true);
        let result = loss.rel(x, y).into_scalar::<f32>();
        approx::assert_relative_eq!(result, 1.5, epsilon = 1e-10);
    }

    #[test]
    fn rel_sum_over_batch_is_the_sum_of_per_example_losses() {
        let x = Tensor::<2>::from_data([[2.0, 0.0], [3.0, 0.0]], &device());
        let y = Tensor::<2>::from_data([[1.0, 0.0], [1.0, 0.0]], &device());

        let per_example = LpLoss::new(2, 2, true, false).rel(x.clone(), y.clone());
        let summed = LpLoss::new(2, 2, false, true)
            .rel(x, y)
            .into_scalar::<f32>();

        let expected: f32 = per_example
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
            .iter()
            .sum();
        approx::assert_relative_eq!(summed, expected, epsilon = 1e-6);
    }
}

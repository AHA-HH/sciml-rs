//! Lp-norm loss functions for neural operator training (absolute and relative).

use burn::tensor::Tensor;

/// Absolute or relative Lp loss between predicted and target fields.
///
/// `d`: spatial dimensionality, used only in `abs()`'s quadrature weight
/// `h^(d/p)` — it cancels out in `rel()`'s ratio, so has no effect there.
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
        Self { d, p, size_average, reduction }
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
            2 => diff.powf_scalar(2.0).sum_dim(1).squeeze_dims(&[1]).sqrt().mul_scalar(h_weight),
            p => diff.powf_scalar(p as f64).sum_dim(1).squeeze_dims(&[1]).powf_scalar(1.0 / p as f64).mul_scalar(h_weight),
        };

        if self.reduction {
            if self.size_average { all_norms.mean() } else { all_norms.sum() }
        } else {
            all_norms // [batch] — per-example losses
        }
    }

    /// Relative Lp loss: `||x - y||_p / ||y||_p`. Quadrature weight cancels
    /// between numerator and denominator, so it's not applied here.
    /// `x`, `y`: `[batch, n_points]`.
    pub fn rel(&self, x: Tensor<2>, y: Tensor<2>) -> Tensor<1> {
        let diff_norms = match self.p {
            1 => (x - y.clone()).abs().sum_dim(1).squeeze_dims(&[1]),
            2 => (x - y.clone()).powf_scalar(2.0).sum_dim(1).squeeze_dims(&[1]).sqrt(),
            p => (x - y.clone()).abs().powf_scalar(p as f64).sum_dim(1).squeeze_dims(&[1]).powf_scalar(1.0 / p as f64),
        };

        let y_norms = match self.p {
            1 => y.abs().sum_dim(1).squeeze_dims(&[1]),
            2 => y.powf_scalar(2.0).sum_dim(1).squeeze_dims(&[1]).sqrt(),
            p => y.abs().powf_scalar(p as f64).sum_dim(1).squeeze_dims(&[1]).powf_scalar(1.0 / p as f64),
        };

        let per_example = diff_norms / y_norms;

        if self.reduction {
            if self.size_average { per_example.mean() } else { per_example.sum() }
        } else {
            per_example // [batch] — per-example losses
        }
    }

    /// Default call — relative loss.
    pub fn forward(&self, x: Tensor<2>, y: Tensor<2>) -> Tensor<1> {
        self.rel(x, y)
    }
}
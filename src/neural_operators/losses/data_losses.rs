//! Lp-norm loss functions for neural operator training (absolute and relative).

use burn::tensor::Tensor;

/// Absolute or relative Lp loss between predicted and target fields.
///
/// `d`: spatial dimensionality, used only in `abs()`, which checks it against
/// the number of spatial axes of its input. The quadrature weight cancels out
/// in `rel()`'s ratio, so `d` has no effect there.
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
    /// `Π_i h_i^(1/p)` where `h_i = 1/(s_i - 1)` is the spacing along
    /// spatial axis `i`. On a square grid this is the reference `h^(d/p)`.
    ///
    /// `x`, `y`: `[batch, s_1, ..., s_d]`, **unflattened** - the per-axis
    /// extents are needed for the weight. For `d = 1` this is `[batch, s]`.
    ///
    /// At an exact match the loss is 0 with gradient 0, as in `rel`.
    ///
    /// # Panics
    /// If `R - 1 != d` (e.g. a flattened `[batch, s^2]` tensor with `d = 2`),
    /// if `x` and `y` shapes differ, or if any spatial extent is below 2.
    pub fn abs<const R: usize>(&self, x: Tensor<R>, y: Tensor<R>) -> Tensor<1> {
        let dims = x.dims();
        assert_eq!(
            R - 1,
            self.d,
            "LpLoss::abs: input has {} spatial axes but d = {}; pass unflattened [batch, s_1, ..., s_d] tensors",
            R - 1,
            self.d
        );
        assert_eq!(dims, y.dims(), "LpLoss::abs: x and y shapes differ");
        assert!(
            dims[1..].iter().all(|&s| s >= 2),
            "LpLoss::abs: every spatial extent must be >= 2, got {:?}",
            &dims[1..]
        );

        let h_weight: f64 = dims[1..]
            .iter()
            .map(|&s| (1.0 / (s as f64 - 1.0)).powf(1.0 / self.p as f64))
            .product();

        let n_points: usize = dims[1..].iter().product();
        let diff = (x - y).abs().reshape([dims[0], n_points]);
        let all_norms = lp_norm(diff, self.p).mul_scalar(h_weight);

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
    ///
    /// At an exact match (`x == y` for an example) the loss is 0 with gradient
    /// 0, matching `torch.norm`'s subgradient convention.
    ///
    /// Undefined for an identically zero target (`||y||_p = 0`): the example
    /// evaluates to `inf`, or `NaN` if the prediction is also zero, exactly as
    /// the reference implementation does. Mask or exclude such examples
    /// upstream.
    pub fn rel(&self, x: Tensor<2>, y: Tensor<2>) -> Tensor<1> {
        let diff_norms = lp_norm(x - y.clone(), self.p);
        let y_norms = lp_norm(y, self.p);

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

/// Per-row Lp norm of `t: [batch, n_points]`, returning `[batch]`.
///
/// For `p >= 2` the final root `s^(1/p)` has an infinite derivative at
/// `s = 0`, and backward would give `inf * 0 = NaN` at an exact match. Rows
/// with `s == 0` therefore take the root of a constant 1 and are then set to
/// 0, so their norm is exactly 0 with gradient 0 - the subgradient
/// `torch.norm` uses. All other rows are computed exactly as before.
fn lp_norm(t: Tensor<2>, p: usize) -> Tensor<1> {
    // p=1: sum of abs. p=2: Euclidean norm. general p: p-norm.
    let sum = match p {
        1 => return t.abs().sum_dim(1).squeeze_dims(&[1]),
        2 => t.powf_scalar(2.0),
        p => t.abs().powf_scalar(p as f64),
    }
    .sum_dim(1)
    .squeeze_dims::<1>(&[1]);

    let is_zero = sum.clone().equal_elem(0.0);
    let safe = sum.mask_fill(is_zero.clone(), 1.0);
    let root = match p {
        2 => safe.sqrt(),
        p => safe.powf_scalar(1.0 / p as f64),
    };
    root.mask_fill(is_zero, 0.0)
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

    // Tolerances for the `abs` tests below: f32 sums of at most 32 small
    // integers are exact, and the remaining sqrt/powf/mul contribute a few
    // ulps, so max_relative = 1e-6 (~8 ulps) is tight but safe.

    #[test]
    fn abs_2d_non_square_known_value() {
        // [1, 3, 5] with x - y == 1, d=2, p=2.
        // h = (1/2, 1/4), weight = (1/2 * 1/4)^(1/2), ||e||_2 = sqrt(15)
        // => sqrt(15/8). Flattening first (the old bug) gives
        // h = 1/14, weight = 1/14 => sqrt(15)/14 ~ 0.277.
        let x = Tensor::<3>::ones([1, 3, 5], &device()).mul_scalar(2.0);
        let y = Tensor::<3>::ones([1, 3, 5], &device());
        let loss = LpLoss::new(2, 2, true, true);
        let result = loss.abs(x, y).into_scalar::<f32>();
        approx::assert_relative_eq!(result, (15.0f32 / 8.0).sqrt(), max_relative = 1e-6);
    }

    #[test]
    fn abs_2d_square_matches_reference_formula() {
        // Reference: h = 1/(s-1) along one axis, loss = h^(d/p) * ||e||_p.
        let (b, s) = (2, 4);
        let e: Vec<f32> = (0..b * s * s).map(|i| (i % 7) as f32 - 3.0).collect();
        let x = Tensor::<3>::from_data(TensorData::new(e.clone(), vec![b, s, s]), &device());
        let y = Tensor::<3>::zeros([b, s, s], &device());

        let per_example = LpLoss::new(2, 2, true, false).abs(x, y);
        assert_eq!(per_example.dims(), [b]);
        let got = per_example.into_data().try_to_vec::<f32>().unwrap();

        let h = 1.0 / (s as f64 - 1.0);
        for (k, chunk) in e.chunks(s * s).enumerate() {
            let norm = chunk
                .iter()
                .map(|&v| (v as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let expected = h.powf(2.0 / 2.0) * norm;
            approx::assert_relative_eq!(got[k] as f64, expected, max_relative = 1e-6);
        }
    }

    #[test]
    fn abs_2d_l1_and_general_p() {
        // [1, 3, 5] with |x - y| == 1: h1*h2 = 1/8, n_points = 15.
        let x = Tensor::<3>::zeros([1, 3, 5], &device());
        let y = Tensor::<3>::ones([1, 3, 5], &device());

        // p=1: weight = 1/8, ||e||_1 = 15.
        let l1 = LpLoss::new(2, 1, true, true)
            .abs(x.clone(), y.clone())
            .into_scalar::<f32>();
        approx::assert_relative_eq!(l1, 15.0 / 8.0, max_relative = 1e-6);

        // p=3: weight = (1/8)^(1/3) = 1/2, ||e||_3 = 15^(1/3).
        let l3 = LpLoss::new(2, 3, true, true).abs(x, y).into_scalar::<f32>();
        approx::assert_relative_eq!(l3, 0.5 * 15.0f32.cbrt(), max_relative = 1e-6);
    }

    #[test]
    fn abs_3d_known_value() {
        // [1, 2, 3, 2] with x - y == 1, p=2: h = (1, 1/2, 1),
        // weight = sqrt(1/2), ||e||_2 = sqrt(12) => sqrt(6).
        let x = Tensor::<4>::ones([1, 2, 3, 2], &device());
        let y = Tensor::<4>::zeros([1, 2, 3, 2], &device());
        let result = LpLoss::new(3, 2, true, true).abs(x, y).into_scalar::<f32>();
        approx::assert_relative_eq!(result, 6.0f32.sqrt(), max_relative = 1e-6);
    }

    #[test]
    #[should_panic(expected = "pass unflattened")]
    fn abs_rejects_flattened_multidim_input() {
        // The pre-fix call pattern: a 2D field flattened to [b, s^2] with d=2.
        let x = Tensor::<2>::ones([1, 15], &device());
        let y = Tensor::<2>::zeros([1, 15], &device());
        let _ = LpLoss::new(2, 2, true, true).abs(x, y);
    }

    #[test]
    #[should_panic(expected = "shapes differ")]
    fn abs_rejects_shape_mismatch() {
        let x = Tensor::<3>::ones([1, 3, 5], &device());
        let y = Tensor::<3>::ones([1, 5, 3], &device());
        let _ = LpLoss::new(2, 2, true, true).abs(x, y);
    }

    #[test]
    #[should_panic(expected = "must be >= 2")]
    fn abs_rejects_single_point_axis() {
        let x = Tensor::<3>::ones([1, 1, 5], &device());
        let y = Tensor::<3>::zeros([1, 1, 5], &device());
        let _ = LpLoss::new(2, 2, true, true).abs(x, y);
    }

    #[test]
    fn abs_gradient_matches_analytic() {
        // L = w * ||x - y||_2 with w = (1/2 * 1/4)^(1/2) on a [1, 3, 5] grid.
        // dL/dx = w * (x - y) / ||x - y||_2 (y is constant, batch mean of 1).
        let device = Device::default().autodiff();
        let e: Vec<f32> = (0..15).map(|i| (i as f32) * 0.5 - 3.25).collect();
        let x = Tensor::<3>::from_data(TensorData::new(e.clone(), vec![1, 3, 5]), &device)
            .require_grad();
        let y = Tensor::<3>::zeros([1, 3, 5], &device);

        let loss = LpLoss::new(2, 2, true, true).abs(x.clone(), y);
        let grads = loss.backward();
        let g = x
            .grad(&grads)
            .expect("x received no gradient")
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();

        let w = (0.5f64 * 0.25).sqrt();
        let norm = e.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
        for (i, (&gi, &ei)) in g.iter().zip(&e).enumerate() {
            let expected = w * ei as f64 / norm;
            approx::assert_relative_eq!(gi as f64, expected, max_relative = 1e-6, epsilon = 1e-7);
            assert!(gi.is_finite(), "grad[{i}] is not finite");
        }
    }

    // --- REVIEW.md 2.5: exact-match gradients and zero targets ---

    /// The unguarded norm used before 2.5, kept here as the forward reference.
    fn unguarded_norm(t: Tensor<2>, p: usize) -> Tensor<1> {
        match p {
            1 => t.abs().sum_dim(1).squeeze_dims(&[1]),
            2 => t.powf_scalar(2.0).sum_dim(1).squeeze_dims(&[1]).sqrt(),
            p => t
                .abs()
                .powf_scalar(p as f64)
                .sum_dim(1)
                .squeeze_dims(&[1])
                .powf_scalar(1.0 / p as f64),
        }
    }

    #[test]
    fn rel_forward_is_bit_identical_to_unguarded_formula() {
        // The mask only touches rows whose sum is exactly zero, so every
        // other value must be unchanged to the bit - no tolerance.
        let (b, n) = (3, 7);
        let xs: Vec<f32> = (0..b * n).map(|i| (i as f32 * 0.37).sin()).collect();
        let ys: Vec<f32> = (0..b * n).map(|i| (i as f32 * 0.91).cos() + 0.1).collect();
        let x = Tensor::<2>::from_data(TensorData::new(xs, vec![b, n]), &device());
        let y = Tensor::<2>::from_data(TensorData::new(ys, vec![b, n]), &device());

        for p in [1, 2, 3] {
            let got = LpLoss::new(1, p, true, false)
                .rel(x.clone(), y.clone())
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
            let expected = (unguarded_norm(x.clone() - y.clone(), p)
                / unguarded_norm(y.clone(), p))
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
            assert_eq!(got, expected, "p = {p}");
        }
    }

    #[test]
    fn rel_gradient_at_exact_match_is_zero_not_nan() {
        // Row 0: x == y (previously NaN gradient for p >= 2).
        // Row 1: d = x - y = [0.5, 0]. With mean over 2 examples,
        //   dL/dx_1 = (1/2) |d|^(p-1) sign(d) / (||d||_p^(p-1) ||y_1||_p)
        //           = [0.5 / ||y_1||_p, 0]  since ||d||_p = |d_0| = 0.5.
        for p in [2usize, 3] {
            let ad = Device::default().autodiff();
            let x = Tensor::<2>::from_data([[1.0, 2.0], [1.0, 2.0]], &ad).require_grad();
            let y = Tensor::<2>::from_data([[1.0, 2.0], [0.5, 2.0]], &ad);
            let loss = LpLoss::new(1, p, true, true).rel(x.clone(), y);
            let g = x
                .grad(&loss.backward())
                .expect("x received no gradient")
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();

            assert_eq!(&g[..2], &[0.0, 0.0], "p = {p}: exact-match row");
            let y1_norm = (0.5f64.powi(p as i32) + 2.0f64.powi(p as i32)).powf(1.0 / p as f64);
            approx::assert_relative_eq!(g[2] as f64, 0.5 / y1_norm, max_relative = 1e-6);
            assert_eq!(g[3], 0.0, "p = {p}: zero-difference element");
        }
    }

    #[test]
    fn rel_zero_target_is_inf_or_nan_as_documented() {
        // Relative error is undefined for ||y|| = 0; behaviour matches the
        // reference (x/0 = inf, 0/0 = NaN) and is documented on `rel`.
        let x = Tensor::<2>::from_data([[1.0, 2.0], [0.0, 0.0]], &device());
        let y = Tensor::<2>::zeros([2, 2], &device());
        let vals = LpLoss::new(1, 2, true, false)
            .rel(x, y)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert!(vals[0].is_infinite() && vals[0] > 0.0, "got {}", vals[0]);
        assert!(vals[1].is_nan(), "got {}", vals[1]);
    }

    #[test]
    fn abs_gradient_at_exact_match_is_zero_not_nan() {
        // abs shares the norm, so it had the same NaN at an exact match.
        let ad = Device::default().autodiff();
        let x = Tensor::<2>::from_data([[1.0, 2.0, 3.0]], &ad).require_grad();
        let y = Tensor::<2>::from_data([[1.0, 2.0, 3.0]], &ad);
        let loss = LpLoss::new(1, 2, true, true).abs(x.clone(), y);
        let g = x
            .grad(&loss.backward())
            .expect("x received no gradient")
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(g, vec![0.0, 0.0, 0.0]);
    }
}

//! Pointwise (kernel-size-1) maps on channels-first tensors of any rank, shared by the block
//! parts (library CM1.3a) and the FNO (CM1.3b): one batched matmul over the flattened grid.

use burn::{
    Tensor,
    nn::{Linear, conv::Conv1d},
};

/// `y[b, o, n] = Σ_i w[o, i] · x[b, i, n] + bias[o]` on channels-first `x`
/// (`[B, I, s..]` → `[B, O, s..]`), as one batched matmul. The reshape to `[B, I, N]` is free
/// on a contiguous tensor.
///
/// # Panics
/// If `x` is not `I` channels wide.
fn pointwise<const R: usize>(w_oi: Tensor<2>, bias: Option<Tensor<1>>, x: Tensor<R>) -> Tensor<R> {
    let mut dims = x.dims();
    let [o, i] = w_oi.dims();
    assert_eq!(dims[1], i, "expected {i} input channels, got {}", dims[1]);
    let n: usize = dims[2..].iter().product();
    let y = w_oi.unsqueeze::<3>().matmul(x.reshape([dims[0], i, n])); // [1,O,I] @ [B,I,N]
    let y = match bias {
        Some(bias) => y + bias.reshape([1, o, 1]),
        None => y,
    };
    dims[1] = o;
    y.reshape(dims)
}

/// A kernel-size-1 `Conv1d` on a channels-first `[B, I, s..]` tensor of any rank:
/// `[B, I, s..]` → `[B, O, s..]`, with the layer's own `[O, I, 1]` weight read as `[O, I]`.
///
/// # Panics
/// If `conv`'s kernel size is not 1 or the input is not `I` channels wide.
pub(crate) fn conv1x1<const R: usize>(conv: &Conv1d, x: Tensor<R>) -> Tensor<R> {
    let [o, i, k] = conv.weight.dims();
    assert_eq!(k, 1, "a pointwise convolution needs kernel size 1, got {k}");
    pointwise(
        conv.weight.val().reshape([o, i]),
        conv.bias.as_ref().map(|b| b.val()),
        x,
    )
}

/// A `Linear` along the channel axis of a channels-first `[B, I, s..]` tensor:
/// `[B, I, s..]` → `[B, O, s..]`. `Linear` stores `[I, O]` (its default row layout), hence the
/// transpose to `[O, I]`.
///
/// # Panics
/// If the input is not `I` channels wide.
pub(crate) fn linear_cf<const R: usize>(layer: &Linear, x: Tensor<R>) -> Tensor<R> {
    pointwise(
        layer.weight.val().transpose(),
        layer.bias.as_ref().map(|b| b.val()),
        x,
    )
}

#[cfg(test)]
mod tests {
    use burn::{
        nn::LinearConfig,
        tensor::{Device, Distribution, Tolerance},
    };

    use super::*;

    /// `linear_cf` is `Linear::forward` with the channel axis moved last and back.
    #[test]
    fn linear_cf_is_linear_on_the_channel_axis() {
        let device = Device::default();
        let layer = LinearConfig::new(3, 5).init(&device);
        let x = Tensor::<4>::random([2, 3, 4, 6], Distribution::Default, &device);
        let expected = layer
            .forward(x.clone().permute([0, 2, 3, 1]))
            .permute([0, 3, 1, 2]);
        linear_cf(&layer, x)
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());
    }

    #[test]
    #[should_panic(expected = "expected 3 input channels, got 4")]
    fn linear_cf_checks_the_width() {
        let device = Device::default();
        let layer = LinearConfig::new(3, 5).init(&device);
        linear_cf(&layer, Tensor::<3>::zeros([1, 4, 2], &device));
    }
}

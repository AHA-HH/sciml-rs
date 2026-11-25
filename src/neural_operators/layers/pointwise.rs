//! Pointwise (kernel-size-1) convolutions on channels-first tensors of any rank, shared by
//! the block parts (library CM1.3a).

use burn::{Tensor, nn::conv::Conv1d};

/// A kernel-size-1 `Conv1d` on a channels-first `[B, I, s..]` tensor of any rank, as one
/// batched matmul over the flattened grid: `y[b, o, n] = Σ_i w[o, i] · x[b, i, n] + bias[o]`,
/// `[B, I, s..]` → `[B, O, s..]`. The flattening is free on a contiguous tensor.
///
/// # Panics
/// If `conv`'s kernel size is not 1 or the input is not `I` channels wide.
pub(crate) fn conv1x1<const R: usize>(conv: &Conv1d, x: Tensor<R>) -> Tensor<R> {
    let [o, i, k] = conv.weight.dims();
    assert_eq!(k, 1, "a pointwise convolution needs kernel size 1, got {k}");
    let mut dims = x.dims();
    assert_eq!(dims[1], i, "expected {i} input channels, got {}", dims[1]);
    let n: usize = dims[2..].iter().product();
    let y = conv
        .weight
        .val()
        .reshape([1, o, i])
        .matmul(x.reshape([dims[0], i, n]));
    let y = match &conv.bias {
        Some(bias) => y + bias.val().reshape([1, o, 1]),
        None => y,
    };
    dims[1] = o;
    y.reshape(dims)
}

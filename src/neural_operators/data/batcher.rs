//! The batch type every loader yields: [`device_loader`](super::device_batcher::device_loader)
//! builds them; the test-only `host_oracle` builds the same batches on the host.

use burn::Tensor;

/// A batch of stacked input/target tensor pairs.
///
/// `R` = input tensor rank (batch + spatial dims + channel axis).
/// `RM1` = target tensor rank, `R - 1` (targets have no channel axis).
#[derive(Clone, Debug)]
pub struct Batch<const R: usize, const RM1: usize> {
    /// `[B, *spatial, C]`, channels-last.
    pub inputs: Tensor<R>,
    /// `[B, *spatial]`.
    pub targets: Tensor<RM1>,
}

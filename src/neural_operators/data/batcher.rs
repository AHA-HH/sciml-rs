//! Batching for operator-learning data: stacks individual [`DataItem`]s into
//! rank-`R`/`R-1` tensors for training.

use crate::neural_operators::data::dataitem::DataItem;
use burn::{Tensor, data::dataloader::batcher::Batcher, prelude::*};

/// A batch of stacked input/target tensor pairs.
///
/// `R` = input tensor rank (batch + spatial dims + channel axis).
/// `RM1` = target tensor rank, constrained to `R - 1` (targets have no
/// channel axis) - enforced at compile time by [`OperatorBatcher::new`].
#[derive(Clone, Debug)]
pub struct Batch<const R: usize, const RM1: usize> {
    pub inputs: Tensor<R>,
    pub targets: Tensor<RM1>,
}

/// Stacks [`DataItem`]s into a [`Batch`] for a fixed input/target rank pair.
pub struct OperatorBatcher<const R: usize, const RM1: usize> {
    pub device: Device,
}

impl<const R: usize, const RM1: usize> OperatorBatcher<R, RM1> {
    /// Compile-time guard: evaluated in `new`, so a wrong rank pairing fails
    /// to build instead of panicking at runtime.
    const RANK_OK: () = assert!(
        RM1 + 1 == R,
        "targets rank must be inputs rank minus one (no channel dim)"
    );

    /// Constructs a batcher for the given device.
    ///
    /// `RM1` must be `R - 1`, the only rank pairing this batcher supports:
    ///
    /// ```
    /// use burn::tensor::Device;
    /// use sciml_rs::neural_operators::data::batcher::OperatorBatcher;
    ///
    /// let _batcher = OperatorBatcher::<3, 2>::new(Device::default());
    /// ```
    ///
    /// Any other pairing is rejected at compile time:
    ///
    /// ```compile_fail
    /// use burn::tensor::Device;
    /// use sciml_rs::neural_operators::data::batcher::OperatorBatcher;
    ///
    /// let _batcher = OperatorBatcher::<3, 3>::new(Device::default());
    /// ```
    pub fn new(device: Device) -> Self {
        let () = Self::RANK_OK;
        Self { device }
    }
}

impl<const R: usize, const RM1: usize> Batcher<DataItem, Batch<R, RM1>>
    for OperatorBatcher<R, RM1>
{
    /// `_device` is the plain inner device the DataLoader always supplies.
    /// This batcher uses `self.device` instead, so batches land on the same
    /// (autodiff) backend as the model in the hand-written training loop.
    /// That's why it doesn't work with Burn's `Learner`, which controls
    /// placement across the train/validation split itself.
    fn batch(&self, items: Vec<DataItem>, _device: &Device) -> Batch<R, RM1> {
        let n = items.len();

        assert!(!items.is_empty(), "cannot construct an empty batch");

        let input_item_shape = items[0].input.shape();

        let input_shape: [usize; R] =
            core::array::from_fn(|i| if i == 0 { n } else { input_item_shape[i - 1] });

        let target_item_shape = items[0].target.shape();

        let target_shape: [usize; RM1] =
            core::array::from_fn(|i| if i == 0 { n } else { target_item_shape[i - 1] });

        let input_item_len = items[0].input.len();
        let target_item_len = items[0].target.len();

        let mut input_data = Vec::with_capacity(n * input_item_len);

        let mut target_data = Vec::with_capacity(n * target_item_len);

        // Fill both buffers in a single traversal of the batch.
        for item in &items {
            input_data.extend(item.input.iter().copied());
            target_data.extend(item.target.iter().copied());
        }

        let inputs = Tensor::<R>::from_data(
            burn::tensor::TensorData::new(input_data, input_shape),
            &self.device,
        );

        let targets = Tensor::<RM1>::from_data(
            burn::tensor::TensorData::new(target_data, target_shape),
            &self.device,
        );

        Batch { inputs, targets }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    #[test]
    fn stacks_items_in_order_with_batch_dim_prepended() {
        let device = Device::default();
        let batcher = OperatorBatcher::<3, 2>::new(device.clone());

        // Two items: input [4, 2], target [4]. Values encode their origin.
        let items: Vec<DataItem> = (0..2)
            .map(|k| DataItem {
                input: ArrayD::from_shape_fn(IxDyn(&[4, 2]), |i| {
                    (k * 100 + i[0] * 10 + i[1]) as f64
                }),
                target: ArrayD::from_shape_fn(IxDyn(&[4]), |i| (k * 100 + i[0]) as f64),
            })
            .collect();

        let batch = batcher.batch(items, &device);

        assert_eq!(batch.inputs.dims(), [2, 4, 2]);
        assert_eq!(batch.targets.dims(), [2, 4]);

        let inputs: Vec<f64> = batch.inputs.into_data().iter::<f64>().collect();
        assert_eq!(inputs[0], 0.0); // item 0, [0,0]
        assert_eq!(inputs[8], 100.0); // item 1, [0,0] — 4*2 elements per item
    }
}

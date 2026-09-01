//! Batching for operator-learning data: stacks individual [`DataItem`]s into
//! rank-`R`/`R-1` tensors for training.

use crate::neural_operators::data::dataitem::DataItem;
use burn::{Tensor, data::dataloader::batcher::Batcher, prelude::*};

/// A batch of stacked input/target tensor pairs.
///
/// `R` = input tensor rank (batch + spatial dims + channel axis).
/// `RM1` = target tensor rank, constrained to `R - 1` (targets have no
/// channel axis) - enforced at runtime in [`OperatorBatcher::new`],
/// not by the type system.
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
    /// Constructs a batcher for the given device.
    ///
    /// Panics if `RM1 != R - 1` — the only rank pairing this batcher supports.
    pub fn new(device: Device) -> Self {
        assert_eq!(
            RM1,
            R - 1,
            "targets rank must be inputs rank minus one (no channel dim)"
        );
        Self { device }
    }
}

impl<const R: usize, const RM1: usize> Batcher<DataItem, Batch<R, RM1>>
    for OperatorBatcher<R, RM1>
{
    fn batch(&self, items: Vec<DataItem>, _device: &Device) -> Batch<R, RM1> {
        let n = items.len();

        // input: prepend batch dim to item 0's shape, flatten all items' raw
        // values in order, reshape into a rank-R tensor on self.device
        let mut input_shape = vec![n];
        input_shape.extend_from_slice(items[0].input.shape());
        let input_shape: [usize; R] = input_shape.try_into().unwrap();
        let input_data: Vec<f64> = items
            .iter()
            .flat_map(|item| item.input.iter().copied())
            .collect();
        let inputs = Tensor::<R>::from_data(
            burn::tensor::TensorData::new(input_data, input_shape),
            &self.device,
        );

        // target: same construction, rank RM1 (== R - 1, no channel axis)
        let mut target_shape = vec![n];
        target_shape.extend_from_slice(items[0].target.shape());
        let target_shape: [usize; RM1] = target_shape.try_into().unwrap();
        let target_data: Vec<f64> = items
            .iter()
            .flat_map(|item| item.target.iter().copied())
            .collect();
        let targets = Tensor::<RM1>::from_data(
            burn::tensor::TensorData::new(target_data, target_shape),
            &self.device,
        );

        Batch { inputs, targets }
    }
}

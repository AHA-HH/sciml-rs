//! Batching for operator-learning data: stacks individual [`DataItem`]s into
//! rank-`IR`/`IR-1` tensors for training.

use crate::neural_operators::data::dataitem::DataItem;
use burn::{
    Tensor,
    data::dataloader::batcher::Batcher,
    prelude::*,
};

/// A batch of stacked input/target tensor pairs.
///
/// `IR` = input tensor rank (batch + spatial dims + channel axis).
/// `TR` = target tensor rank, constrained to `IR - 1` (targets have no
/// channel axis) - enforced at runtime in [`OperatorBatcher::new`],
/// not by the type system.
#[derive(Clone, Debug)]
pub struct Batch<const IR: usize, const TR: usize> {
    pub inputs: Tensor<IR>,
    pub targets: Tensor<TR>,
}

/// Stacks [`DataItem`]s into a [`BatchNd`] for a fixed input/target rank pair.
pub struct OperatorBatcher<const IR: usize, const TR: usize> {
    pub device: Device,
}

impl<const IR: usize, const TR: usize> OperatorBatcher<IR, TR> {
    /// Constructs a batcher for the given device.
    ///
    /// Panics if `TR != IR - 1` — the only rank pairing this batcher supports.
    pub fn new(device: Device) -> Self {
        assert_eq!(TR, IR - 1, "targets rank must be inputs rank minus one (no channel dim)");
        Self { device }
    }
}

impl<const IR: usize, const TR: usize> Batcher<DataItem, Batch<IR, TR>> for OperatorBatcher<IR, TR> {
    fn batch(&self, items: Vec<DataItem>, _device: &Device) -> Batch<IR, TR> {
        let n = items.len();

        // input: prepend batch dim to item 0's shape, flatten all items' raw
        // values in order, reshape into a rank-IR tensor on self.device
        let mut input_shape = vec![n];
        input_shape.extend_from_slice(items[0].input.shape());
        let input_shape: [usize; IR] = input_shape.try_into().unwrap();
        let input_data: Vec<f64> = items.iter().flat_map(|item| item.input.iter().copied()).collect();
        let inputs = Tensor::<IR>::from_data(burn::tensor::TensorData::new(input_data, input_shape), &self.device);

        // target: same construction, rank TR (== IR - 1, no channel axis)
        let mut target_shape = vec![n];
        target_shape.extend_from_slice(items[0].target.shape());
        let target_shape: [usize; TR] = target_shape.try_into().unwrap();
        let target_data: Vec<f64> = items.iter().flat_map(|item| item.target.iter().copied()).collect();
        let targets = Tensor::<TR>::from_data(burn::tensor::TensorData::new(target_data, target_shape), &self.device);

        Batch { inputs, targets }
    }
}

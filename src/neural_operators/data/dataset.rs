//! In-memory dataset for operator learning: holds a full preprocessed
//! input/target tensor pair and hands out individual [`DataItem`]s by index.

use crate::neural_operators::data::dataitem::DataItem;
use burn::data::dataset::Dataset;
use ndarray::{ArrayD, Axis};

/// Holds the entire preprocessed dataset in memory and hands out
/// individual [`DataItem`]s on demand via [`Dataset::get`].
pub struct OperatorDataset {
    inputs: ArrayD<f64>,  // [n_examples, s, 2] - signal + grid
    targets: ArrayD<f64>, // [n_examples, s]
}

impl OperatorDataset {
    pub fn new(inputs: ArrayD<f64>, targets: ArrayD<f64>) -> Self {
        assert!(
            inputs.shape()[0] == targets.shape()[0],
            "inputs and targets must have the same number of examples, got {} and {}",
            inputs.shape()[0],
            targets.shape()[0]
        );
        Self { inputs, targets }
    }
}

impl Dataset<DataItem> for OperatorDataset {
    fn get(&self, index: usize) -> Option<DataItem> {
        if index >= self.len() {
            return None;
        }

        // slice the i-th example from inputs and targets
        let input = self.inputs.index_axis(Axis(0), index).to_owned();
        let target = self.targets.index_axis(Axis(0), index).to_owned();

        Some(DataItem { input, target })
    }

    fn len(&self) -> usize {
        self.inputs.shape()[0]
    }
}
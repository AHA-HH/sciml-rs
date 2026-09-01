//! In-memory dataset for operator learning: holds a full preprocessed
//! input/target tensor pair and hands out individual [`DataItem`]s by index.

use crate::neural_operators::data::dataitem::DataItem;
use burn::data::dataset::{Dataset, DatasetError};
use ndarray::{ArrayD, Axis};
use std::io::{Error, ErrorKind};

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
    fn get(&self, index: usize) -> Result<DataItem, DatasetError> {
        if index >= self.len() {
            return Err(DatasetError::new(Error::new(
                ErrorKind::InvalidInput,
                format!("index {index} out of bounds (len {})", self.len()),
            )));
        }

        // slice the i-th example from inputs and targets
        let input = self.inputs.index_axis(Axis(0), index).to_owned();
        let target = self.targets.index_axis(Axis(0), index).to_owned();

        Ok(DataItem { input, target })
    }

    fn len(&self) -> usize {
        self.inputs.shape()[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

    #[test]
    fn get_returns_the_indexed_example() {
        let inputs = ArrayD::from_shape_fn(IxDyn(&[3, 4, 2]), |i| {
            (i[0] * 100 + i[1] * 10 + i[2]) as f64
        });
        let targets = ArrayD::from_shape_fn(IxDyn(&[3, 4]), |i| (i[0] * 100 + i[1]) as f64);
        let dataset = OperatorDataset::new(inputs, targets);

        assert_eq!(dataset.len(), 3);

        let item = dataset.get(1).expect("index 1 in range");
        assert_eq!(item.input.shape(), &[4, 2]); // batch axis dropped
        assert_eq!(item.target.shape(), &[4]);
        assert_eq!(item.input[[0, 0]], 100.0); // from example 1, not 0 or 2
        assert_eq!(item.target[[3]], 103.0);

        assert!(dataset.get(3).is_err());
    }
}

//! In-memory dataset for operator learning: holds a full preprocessed
//! input/target tensor pair and hands out individual [`DataItem`]s by index.

use crate::neural_operators::data::dataitem::{DataItem, HostFloat};
use burn::data::dataset::{Dataset, DatasetError};
use ndarray::{ArrayD, Axis};
use std::io::{Error, ErrorKind};

/// Holds the entire preprocessed dataset in memory and hands out
/// individual [`DataItem`]s on demand via [`Dataset::get`].
///
/// `T` is the host storage dtype, see [`HostFloat`].
pub struct OperatorDataset<T: HostFloat> {
    inputs: ArrayD<T>,  // [n_examples, spatial.., channels]
    targets: ArrayD<T>, // [n_examples, spatial..]
}

impl<T: HostFloat> OperatorDataset<T> {
    pub fn new(inputs: ArrayD<T>, targets: ArrayD<T>) -> Self {
        assert!(
            inputs.shape()[0] == targets.shape()[0],
            "inputs and targets must have the same number of examples, got {} and {}",
            inputs.shape()[0],
            targets.shape()[0]
        );
        Self { inputs, targets }
    }

    /// Builds a dataset from `f64` arrays, rounding each value to `T` once.
    ///
    /// The loaders' single cast point: everything before it (reading,
    /// subsampling, normalization) stays `f64`.
    pub fn from_f64(inputs: ArrayD<f64>, targets: ArrayD<f64>) -> Self {
        Self::new(inputs.mapv(T::from_f64), targets.mapv(T::from_f64))
    }

    /// All inputs, `[n_examples, spatial.., channels]`.
    pub fn inputs(&self) -> &ArrayD<T> {
        &self.inputs
    }

    /// All targets, `[n_examples, spatial..]`.
    pub fn targets(&self) -> &ArrayD<T> {
        &self.targets
    }
}

impl<T: HostFloat> Dataset<DataItem<T>> for OperatorDataset<T> {
    fn get(&self, index: usize) -> Result<DataItem<T>, DatasetError> {
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

    /// The loaders' cast point: `from_f64` rounds every value once and
    /// keeps shapes; `f64` is the identity.
    #[test]
    fn from_f64_rounds_each_value_once() {
        let inputs = ArrayD::from_shape_fn(IxDyn(&[2, 3, 2]), |i| {
            ((i[0] * 6 + i[1] * 2 + i[2]) as f64 * 0.37).sin() / 3.0
        });
        let targets = ArrayD::from_shape_fn(IxDyn(&[2, 3]), |i| (i[0] * 3 + i[1]) as f64 / 7.0);

        let ds32 = OperatorDataset::<f32>::from_f64(inputs.clone(), targets.clone());
        assert_eq!(ds32.inputs(), &inputs.mapv(|v| v as f32));
        assert_eq!(ds32.targets(), &targets.mapv(|v| v as f32));
        assert_eq!(
            ds32.get(1).unwrap().input,
            inputs.index_axis(Axis(0), 1).mapv(|v| v as f32)
        );

        let ds64 = OperatorDataset::<f64>::from_f64(inputs.clone(), targets.clone());
        assert_eq!(ds64.inputs(), &inputs);
        assert_eq!(ds64.targets(), &targets);
    }
}

//! In-memory dataset for operator learning: holds a full preprocessed
//! input/target tensor pair, batched by `device_loader`.

use crate::neural_operators::data::host_float::HostFloat;
use ndarray::ArrayD;

/// Holds the entire preprocessed dataset in memory; `device_loader` uploads it once and
/// batches it on the device.
///
/// `T` is the host storage dtype, see [`HostFloat`].
pub struct OperatorDataset<T: HostFloat> {
    inputs: ArrayD<T>,  // [n_examples, spatial.., channels]
    targets: ArrayD<T>, // [n_examples, spatial..]
}

impl<T: HostFloat> OperatorDataset<T> {
    /// A dataset of `inputs` `[n, spatial.., channels]` and `targets` `[n, spatial..]`.
    ///
    /// # Panics
    /// If `inputs` and `targets` hold different numbers of examples.
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

    /// Number of examples.
    pub fn len(&self) -> usize {
        self.inputs.shape()[0]
    }

    /// Whether the dataset holds no examples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

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
        assert_eq!(ds32.len(), 2);

        let ds64 = OperatorDataset::<f64>::from_f64(inputs.clone(), targets.clone());
        assert_eq!(ds64.inputs(), &inputs);
        assert_eq!(ds64.targets(), &targets);
    }
}

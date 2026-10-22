//! Tiny datasets and model configs shared by the training tests.

use crate::neural_operators::{data::dataset::OperatorDataset, models::fno::FNOConfig};

/// `n` Burgers-shaped samples: inputs `[n, 8, 1]`, targets `[n, 8]`,
/// targets nonzero so the relative L2 is well defined.
pub(crate) fn tiny_dataset(n: usize) -> OperatorDataset<f64> {
    let inputs = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[n, 8, 1]), |i| {
        ((i[0] * 8 + i[1]) as f64 * 0.3 + i[2] as f64).sin()
    });
    let targets = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[n, 8]), |i| {
        2.0 + ((i[0] * 8 + i[1]) as f64 * 0.7).cos()
    });
    OperatorDataset::new(inputs, targets)
}

pub(crate) fn tiny_model_cfg() -> FNOConfig {
    FNOConfig::new(vec![2], 1, 1)
        .with_hidden_channels(4)
        .with_n_layers(1)
}

/// `n` Darcy-shaped samples: inputs `[n, 4, 4, 1]`, targets `[n, 4, 4]`.
pub(crate) fn tiny_dataset_2d(n: usize) -> OperatorDataset<f64> {
    let inputs = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[n, 4, 4, 1]), |i| {
        ((i[0] * 16 + i[1] * 4 + i[2]) as f64 * 0.3).sin()
    });
    let targets = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[n, 4, 4]), |i| {
        2.0 + ((i[0] * 16 + i[1] * 4 + i[2]) as f64 * 0.7).cos()
    });
    OperatorDataset::new(inputs, targets)
}

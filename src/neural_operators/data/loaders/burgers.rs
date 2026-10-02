//! Burgers equation dataset: config and constructor.
//!
//! `BurgersConfig` holds Burgers-specific settings (subsample rate, derived
//! resolution) on top of the shared `DatasetConfig`. `load_burgers_uniform`
//! builds the full `OperatorDataset` from a `.mat` source file: reading,
//! subsampling and splitting into train/test. Inputs carry the data channel
//! only; the model appends the grid coordinates itself (see `FNO::forward`).

use crate::neural_operators::data::{
    dataitem::HostFloat,
    dataset::OperatorDataset,
    io::{errors::LoadError, readers::mat::MatFileReader, traits::FieldReader},
    loaders::base_dataset::{BaseDatasetConfig, DatasetConfig, HasBaseConfig},
    split::train_test_split,
    transforms::subsample::subsample,
};
use burn::config::Config;
use ndarray::IxDyn;
use std::path::Path;

#[derive(Config, Debug)]
pub struct BurgersConfig {
    pub base: DatasetConfig,
    pub subsample_rate: usize,
}

impl BurgersConfig {
    pub fn s(&self) -> usize {
        2usize.pow(13) / self.subsample_rate
    }
}

impl HasBaseConfig for BurgersConfig {
    fn base(&self) -> &DatasetConfig {
        &self.base
    }
}

/// Loads the 1D Burgers dataset from a `.mat` file into train/test
/// `OperatorDataset`s.
///
/// Reads fields `a` (initial condition) and `u` (solution at t=1),
/// subsamples the spatial axis by `config.subsample_rate`, splits off
/// `n_train`/`n_test` samples. No grid channel is stored: the model
/// generates the coordinates on the device for whatever resolution it is fed.
///
/// Inputs end up `[n, s, 1]` and targets `[n, s]` - the rank difference
/// the batcher's `RM1 = R - 1` invariant expects.
///
/// Everything is computed in `f64`; the datasets are stored as `T`, rounded
/// once at the end (see [`HostFloat`]).
///
/// # Errors
/// [`LoadError::Reader`] if the file can't be read or lacks `a`/`u`;
/// [`LoadError::Invalid`] if the fields aren't `[samples, 8192]`, there are
/// fewer than `n_train + n_test` samples, or the subsample rate doesn't give
/// `config.s()` points.
pub fn load_burgers_uniform<T: HostFloat>(
    path: impl AsRef<Path>,
    config: &BurgersConfig,
) -> Result<(OperatorDataset<T>, OperatorDataset<T>), LoadError> {
    if config.subsample_rate == 0 {
        return Err(LoadError::Invalid("subsample_rate must be > 0".into()));
    }

    // 1. Read raw data from .mat file
    let reader = MatFileReader::new(path.as_ref())?;
    let a_data = reader.read_field("a")?;
    let u_data = reader.read_field("u")?;

    for (name, field) in [("a", &a_data), ("u", &u_data)] {
        if field.ndim() != 2 {
            return Err(LoadError::Invalid(format!(
                "field '{name}' must be [samples, points], got shape {:?}",
                field.shape()
            )));
        }
        let n_total = field.shape()[0];
        if config.n_train() + config.n_test() > n_total {
            return Err(LoadError::Invalid(format!(
                "n_train ({}) + n_test ({}) exceeds the {n_total} samples in field '{name}'",
                config.n_train(),
                config.n_test()
            )));
        }
    }

    // 2. Subsample along spatial dimension (dim 1)
    let a_data = subsample(a_data, 1, config.subsample_rate);
    let u_data = subsample(u_data, 1, config.subsample_rate);

    // spatial size after subsampling
    let s = a_data.shape()[1];
    if s != config.s() || u_data.shape()[1] != s {
        return Err(LoadError::Invalid(format!(
            "subsampled resolution {s} (a) / {} (u) != config.s() {}",
            u_data.shape()[1],
            config.s()
        )));
    }

    // 3. Split into train and test (sizes validated above)
    let (a_train, a_test) = train_test_split(a_data, config.n_train(), config.n_test());
    let (u_train, u_test) = train_test_split(u_data, config.n_train(), config.n_test());

    // 4. Reshape inputs [n, s] -> [n, s, 1]: one data channel
    let a_train = a_train
        .into_shape_with_order(IxDyn(&[config.n_train(), s, 1]))
        .map_err(|e| LoadError::Invalid(format!("reshape a_train: {e}")))?;
    let a_test = a_test
        .into_shape_with_order(IxDyn(&[config.n_test(), s, 1]))
        .map_err(|e| LoadError::Invalid(format!("reshape a_test: {e}")))?;

    // 5. Reshape targets [n, s] -> [n, s] ensure dynamic shape
    let u_train = u_train
        .into_shape_with_order(IxDyn(&[config.n_train(), s]))
        .map_err(|e| LoadError::Invalid(format!("reshape u_train: {e}")))?;
    let u_test = u_test
        .into_shape_with_order(IxDyn(&[config.n_test(), s]))
        .map_err(|e| LoadError::Invalid(format!("reshape u_test: {e}")))?;

    // 6. Wrap in OperatorDataset, casting to the host dtype T
    let train_dataset = OperatorDataset::from_f64(a_train, u_train);
    let test_dataset = OperatorDataset::from_f64(a_test, u_test);

    println!(
        "burgers: {} train / {} test at s={}",
        config.n_train(),
        config.n_test(),
        s
    );

    Ok((train_dataset, test_dataset))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::data::io::errors::ReaderError;

    fn cfg(subsample_rate: usize) -> BurgersConfig {
        BurgersConfig::new(
            DatasetConfig {
                n_train: 1,
                n_test: 1,
            },
            subsample_rate,
        )
    }

    #[test]
    fn missing_file_is_a_reader_error() {
        let err = load_burgers_uniform::<f32>("/definitely/not/here/burgers.mat", &cfg(32))
            .err()
            .expect("loading should fail");
        assert!(
            matches!(err, LoadError::Reader(ReaderError::FileNotFound(_))),
            "{err:?}"
        );
        assert!(err.to_string().contains("burgers.mat"), "{err}");
    }

    #[test]
    fn zero_subsample_rate_is_invalid_not_a_panic() {
        // Previously `config.s()` (8192 / 0) or `subsample` would panic.
        let err = load_burgers_uniform::<f32>("/definitely/not/here/burgers.mat", &cfg(0))
            .err()
            .expect("loading should fail");
        assert!(matches!(err, LoadError::Invalid(_)), "{err:?}");
    }
}

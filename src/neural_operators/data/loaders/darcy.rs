//! Darcy flow dataset: config and constructor.
//!
//! `DarcyConfig` holds Darcy-specific settings (subsample rate, derived
//! resolution) on top of the shared `DatasetConfig`. `load_darcy_uniform`
//! builds train/test `OperatorDataset`s from separate `.mat` files: reading,
//! truncating to configured sizes, subsampling both spatial axes, fitting and
//! applying normalization and appending 2D grid coordinates.
//!
//! normalization asymmetry (matches the reference implementation exactly):
//! `x_test` is encoded using the `x` normalizer fit on `x_train`, but
//! `y_test` is left un-encoded - predictions are decoded back to physical
//! scale before comparison, not targets encoded forward. The returned
//! `y_normalizer` is what the caller needs to do that decode.

use crate::neural_operators::data::{
    dataset::OperatorDataset,
    grids::{GridPlacement, append_grid, uniform_grid},
    io::{errors::LoadError, readers::mat::MatFileReader, traits::FieldReader},
    loaders::base_dataset::{BaseDatasetConfig, DatasetConfig, HasBaseConfig},
    transforms::{
        normalizers::{Normalizer, UnitGaussianNormalizer},
        subsample::subsample,
    },
};
use burn::config::Config;
use ndarray::{Axis, IxDyn};
use std::path::Path;

/// Darcy-specific config: subsample rate and the resolution it implies.
#[derive(Config, Debug)]
pub struct DarcyConfig {
    pub base: DatasetConfig,
    pub subsample_rate: usize,
}

impl DarcyConfig {
    /// Resolution after subsampling, assuming a raw resolution of 421.
    /// Checked against the actually-read resolution in `load_darcy_uniform`.
    pub fn s(&self) -> usize {
        (421 - 1) / self.subsample_rate + 1
    }
}

impl HasBaseConfig for DarcyConfig {
    fn base(&self) -> &DatasetConfig {
        &self.base
    }
}

/// The two normalizers fitted during Darcy loading. `x` encodes inputs;
/// `y` decodes predictions back to physical scale. Named fields because
/// both have the same type and swapping them fails silently.
pub struct DarcyNormalizers {
    pub x: UnitGaussianNormalizer,
    pub y: UnitGaussianNormalizer,
}

/// Builds the Darcy flow dataset with a uniform 2D grid channel appended.
///
/// Returns `(train_dataset, test_dataset, normalizers)`. `normalizers.y` is
/// what the caller needs to decode predictions back to physical scale;
/// `normalizers.x` is only needed for encoding inputs that didn't come from
/// this dataset.
///
/// # Errors
/// [`LoadError::Reader`] if a file can't be read or lacks `coeff`/`sol`;
/// [`LoadError::Invalid`] if the fields aren't `[samples, s, s]`, a file has
/// fewer samples than requested, or the subsample rate doesn't give
/// `config.s()` points per axis.
pub fn load_darcy_uniform(
    train_path: impl AsRef<Path>,
    test_path: impl AsRef<Path>,
    config: &DarcyConfig,
) -> Result<(OperatorDataset, OperatorDataset, DarcyNormalizers), LoadError> {
    if config.subsample_rate == 0 {
        return Err(LoadError::Invalid("subsample_rate must be > 0".into()));
    }

    // read input ('coeff') and target ('sol') fields from separate train/test .mat files
    let train_reader = MatFileReader::new(train_path.as_ref())?;
    let x_train = train_reader.read_field("coeff")?;
    let y_train = train_reader.read_field("sol")?;

    let test_reader = MatFileReader::new(test_path.as_ref())?;
    let x_test = test_reader.read_field("coeff")?;
    let y_test = test_reader.read_field("sol")?;

    for (split, n, fields) in [
        ("train", config.n_train(), [&x_train, &y_train]),
        ("test", config.n_test(), [&x_test, &y_test]),
    ] {
        for (name, field) in ["coeff", "sol"].into_iter().zip(fields) {
            if field.ndim() != 3 {
                return Err(LoadError::Invalid(format!(
                    "{split} field '{name}' must be [samples, s, s], got shape {:?}",
                    field.shape()
                )));
            }
            if n > field.shape()[0] {
                return Err(LoadError::Invalid(format!(
                    "requested {n} {split} samples but '{name}' has {}",
                    field.shape()[0]
                )));
            }
        }
    }

    // truncate to configured n_train/n_test along the sample axis - without this,
    // dataset size is whatever the file happens to contain
    let x_train = x_train
        .slice_axis(Axis(0), (0..config.n_train()).into())
        .to_owned();
    let y_train = y_train
        .slice_axis(Axis(0), (0..config.n_train()).into())
        .to_owned();
    let x_test = x_test
        .slice_axis(Axis(0), (0..config.n_test()).into())
        .to_owned();
    let y_test = y_test
        .slice_axis(Axis(0), (0..config.n_test()).into())
        .to_owned();

    // downsample both spatial axes (1 and 2) by config.subsample_rate
    let x_train = subsample(
        subsample(x_train, 1, config.subsample_rate),
        2,
        config.subsample_rate,
    );
    let y_train = subsample(
        subsample(y_train, 1, config.subsample_rate),
        2,
        config.subsample_rate,
    );
    let x_test = subsample(
        subsample(x_test, 1, config.subsample_rate),
        2,
        config.subsample_rate,
    );
    let y_test = subsample(
        subsample(y_test, 1, config.subsample_rate),
        2,
        config.subsample_rate,
    );

    let s = x_train.shape()[1];
    let n_train = x_train.shape()[0];
    let n_test = x_test.shape()[0];

    // cross-check the derived resolution against what was actually subsampled
    let spatial_ok = [&x_train, &y_train, &x_test, &y_test]
        .iter()
        .all(|a| a.shape()[1] == s && a.shape()[2] == s);
    if s != config.s() || !spatial_ok {
        return Err(LoadError::Invalid(format!(
            "subsampled grids must all be {0}x{0} (config.s()), got train {1:?} / test {2:?}",
            config.s(),
            &x_train.shape()[1..],
            &x_test.shape()[1..]
        )));
    }

    // fit x-normalizer on x_train, encode both x_train and x_test;
    // fit y-normalizer on y_train, encode y_train only - y_test stays raw,
    // decoded against at eval time via the returned y_normalizer
    let x_normalizer = UnitGaussianNormalizer::fit(&x_train);
    let x_train = x_normalizer.encode(x_train);
    let x_test = x_normalizer.encode(x_test);

    let y_normalizer = UnitGaussianNormalizer::fit(&y_train);
    let y_train = y_normalizer.encode(y_train);
    // y_test intentionally NOT encoded

    // add a trailing channel axis to inputs: [n, s, s] -> [n, s, s, 1]
    let x_train = x_train
        .into_shape_with_order(IxDyn(&[n_train, s, s, 1]))
        .map_err(|e| LoadError::Invalid(format!("reshape x_train: {e}")))?;
    let x_test = x_test
        .into_shape_with_order(IxDyn(&[n_test, s, s, 1]))
        .map_err(|e| LoadError::Invalid(format!("reshape x_test: {e}")))?;

    // append 2D grid coordinates as two more channels: [n, s, s, 1] -> [n, s, s, 3]
    // Reversed 'ij' grids = Li's 'xy' meshgrid order: channel 1 varies along
    // spatial axis 2, channel 2 along axis 1. Saved checkpoints depend on it.
    let mut grid = uniform_grid(&[(0.0, 1.0); 2], &[s, s]);
    grid.reverse();
    let x_train = append_grid(x_train, &grid, GridPlacement::AfterData);
    let x_test = append_grid(x_test, &grid, GridPlacement::AfterData);

    // package into OperatorDataset
    let train_dataset = OperatorDataset::new(x_train, y_train);
    let test_dataset = OperatorDataset::new(x_test, y_test);

    println!("darcy: {n_train} train / {n_test} test at s={s}");

    Ok((
        train_dataset,
        test_dataset,
        DarcyNormalizers {
            x: x_normalizer,
            y: y_normalizer,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::data::io::errors::ReaderError;

    fn cfg(subsample_rate: usize) -> DarcyConfig {
        DarcyConfig::new(
            DatasetConfig {
                n_train: 1,
                n_test: 1,
            },
            subsample_rate,
        )
    }

    #[test]
    fn missing_file_is_a_reader_error() {
        let err = load_darcy_uniform("/nope/train.mat", "/nope/test.mat", &cfg(5))
            .err()
            .expect("loading should fail");
        assert!(
            matches!(err, LoadError::Reader(ReaderError::FileNotFound(_))),
            "{err:?}"
        );
        assert!(err.to_string().contains("train.mat"), "{err}");
    }

    #[test]
    fn zero_subsample_rate_is_invalid_not_a_panic() {
        let err = load_darcy_uniform("/nope/train.mat", "/nope/test.mat", &cfg(0))
            .err()
            .expect("loading should fail");
        assert!(matches!(err, LoadError::Invalid(_)), "{err:?}");
    }
}

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
    dataitem::HostFloat,
    dataset::OperatorDataset,
    grids::{GridPlacement, append_grid, uniform_grid},
    io::{errors::LoadError, readers::mat::MatFileReader, traits::FieldReader},
    loaders::base_dataset::{BaseDatasetConfig, DatasetConfig, HasBaseConfig},
    transforms::normalizers::{Normalizer, UnitGaussianNormalizer},
};
use burn::config::Config;
use ndarray::{ArrayD, IxDyn, s};
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
/// [`LoadError::Invalid`] if `n_train < 2` (the normalizers' sample std is
/// undefined for one sample), the fields aren't `[samples, s, s]`, a file has
/// fewer samples than requested, or the subsample rate doesn't give
/// `config.s()` points per axis.
///
/// Reading, normalization and grids are computed in `f64`, and the
/// normalizers keep `f64` statistics; the datasets are stored as `T`,
/// rounded once at the end (see [`HostFloat`]).
pub fn load_darcy_uniform<T: HostFloat>(
    train_path: impl AsRef<Path>,
    test_path: impl AsRef<Path>,
    config: &DarcyConfig,
) -> Result<(OperatorDataset<T>, OperatorDataset<T>, DarcyNormalizers), LoadError> {
    if config.subsample_rate == 0 {
        return Err(LoadError::Invalid("subsample_rate must be > 0".into()));
    }
    // ddof = 1 std of a single sample is 0/0 = NaN, which eps can't guard
    if config.n_train() < 2 {
        return Err(LoadError::Invalid(format!(
            "n_train must be >= 2: a sample std needs at least 2 training samples, got {}",
            config.n_train()
        )));
    }

    // Read input ('coeff') and target ('sol') from each split's .mat file,
    // validate them, then truncate to the configured sample count and subsample
    // both spatial axes in one strided copy. Doing this per split means the
    // reader and full-size fields of one file are freed before the next is
    // parsed. Without the truncation, dataset size is whatever the file holds.
    let r = config.subsample_rate;
    let load_split = |path: &Path, split: &str, n: usize| -> Result<_, LoadError> {
        let (x, y) = {
            let reader = MatFileReader::new(path)?;
            (reader.read_field("coeff")?, reader.read_field("sol")?)
        };
        for (name, field) in [("coeff", &x), ("sol", &y)] {
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
        Ok((take_subsampled(x, n, r), take_subsampled(y, n, r)))
    };
    let (x_train, y_train) = load_split(train_path.as_ref(), "train", config.n_train())?;
    let (x_test, y_test) = load_split(test_path.as_ref(), "test", config.n_test())?;

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

    // package into OperatorDataset, casting to the host dtype T
    let train_dataset = OperatorDataset::from_f64(x_train, y_train);
    let test_dataset = OperatorDataset::from_f64(x_test, y_test);

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

/// The first `n` samples of a `[samples, s, s]` field with every `r`-th
/// point kept on both spatial axes, as a standard-layout array.
///
/// One allocation of just the kept subset; when nothing is dropped the input
/// buffer is returned as is (the reader already produces standard layout).
fn take_subsampled(field: ArrayD<f64>, n: usize, r: usize) -> ArrayD<f64> {
    if n == field.shape()[0] && r == 1 {
        return field;
    }
    let step = r as isize;
    field.slice(s![..n, ..;step, ..;step]).to_owned().into_dyn()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::data::io::errors::ReaderError;

    fn cfg(subsample_rate: usize) -> DarcyConfig {
        DarcyConfig::new(
            DatasetConfig {
                n_train: 2,
                n_test: 1,
            },
            subsample_rate,
        )
    }

    // --- REVIEW1.md N1: one training sample gave an all-NaN normalizer ---

    #[test]
    fn single_train_sample_is_invalid_not_nan() {
        let mut config = cfg(5);
        config.base.n_train = 1;
        // nonexistent paths: the check must run before any file I/O
        let err = load_darcy_uniform::<f32>("/nope/train.mat", "/nope/test.mat", &config)
            .err()
            .expect("loading should fail");
        assert!(matches!(err, LoadError::Invalid(_)), "{err:?}");
        assert!(err.to_string().contains("at least 2"), "{err}");
    }

    #[test]
    fn missing_file_is_a_reader_error() {
        let err = load_darcy_uniform::<f32>("/nope/train.mat", "/nope/test.mat", &cfg(5))
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
        let err = load_darcy_uniform::<f32>("/nope/train.mat", "/nope/test.mat", &cfg(0))
            .err()
            .expect("loading should fail");
        assert!(matches!(err, LoadError::Invalid(_)), "{err:?}");
    }

    // --- REVIEW.md 4.7: one strided copy == old slice + 2x subsample ---

    fn old_path(field: ArrayD<f64>, n: usize, r: usize) -> ArrayD<f64> {
        use crate::neural_operators::data::transforms::subsample::subsample;
        let field = field.slice_axis(ndarray::Axis(0), (0..n).into()).to_owned();
        subsample(subsample(field, 1, r), 2, r)
    }

    #[test]
    fn take_subsampled_matches_old_slice_and_subsample() {
        // Non-square spatial extents and a rate that doesn't divide them.
        let field = ArrayD::from_shape_fn(IxDyn(&[5, 7, 9]), |i| {
            (i[0] * 10_000 + i[1] * 100 + i[2]) as f64
        });
        for (n, r) in [(5, 1), (3, 1), (5, 2), (2, 3), (1, 4)] {
            let new = take_subsampled(field.clone(), n, r);
            assert_eq!(new, old_path(field.clone(), n, r), "n = {n}, r = {r}");
            assert!(new.is_standard_layout(), "n = {n}, r = {r}");
            // the loader's row-major reshape must succeed
            let shape = [new.shape(), &[1]].concat();
            assert!(new.into_shape_with_order(IxDyn(&shape)).is_ok());
        }
    }

    #[test]
    fn take_subsampled_reuses_buffer_when_nothing_is_dropped() {
        let field = ArrayD::from_shape_fn(IxDyn(&[2, 3, 3]), |i| i[2] as f64);
        let ptr = field.as_ptr();
        assert_eq!(take_subsampled(field, 2, 1).as_ptr(), ptr);
    }
}

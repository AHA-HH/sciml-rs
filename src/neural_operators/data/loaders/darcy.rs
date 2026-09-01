//! Darcy flow dataset: config and constructor.
//!
//! `DarcyConfig` holds Darcy-specific settings (subsample rate, derived
//! resolution) on top of the shared `DatasetConfig`. `load_darcy_flow_uniform`
//! builds train/test `OperatorDataset`s from separate `.mat` files: reading,
//! truncating to configured sizes, subsampling both spatial axes, fitting and
//! applying normalization and appending 2D grid coordinates.
//!
//! normalization asymmetry (matches the reference implementation exactly):
//! `x_test` is encoded using the `x` normalizer fit on `x_train`, but
//! `y_test` is left un-encoded — predictions are decoded back to physical
//! scale before comparison, not targets encoded forward. The returned
//! `y_normalizer` is what the caller needs to do that decode.

use crate::neural_operators::data::{
    dataset::OperatorDataset,
    grids::{append_grid_2d, uniform_grid_2d},
    io::{readers::mat::MatFileReader, traits::FieldReader},
    loaders::base_dataset::{BaseDatasetConfig, DatasetConfig, HasBaseConfig},
    transforms::{
        normalizers::{Normalizer, UnitGaussianNormalizer},
        subsample::subsample,
    },
};
use burn::config::Config;
use ndarray::{Axis, IxDyn};
use std::path::{Path, PathBuf};

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

/// Builds the Darcy flow dataset with a uniform 2D grid channel appended.
/// Returns `(train_dataset, test_dataset, y_normalizer)` — the normalizer is
/// needed by the caller to decode predictions back to physical scale.
pub fn load_darcy_uniform(
    train_path: &PathBuf,
    test_path: &PathBuf,
    config: &DarcyConfig,
) -> (OperatorDataset, OperatorDataset, UnitGaussianNormalizer) {
    // read input ('coeff') and target ('sol') fields from separate train/test .mat files
    let train_reader = MatFileReader::new(Path::new(train_path));
    let x_train = train_reader
        .read_field("coeff")
        .expect("failed to read 'coeff'");
    let y_train = train_reader
        .read_field("sol")
        .expect("failed to read 'sol'");

    let test_reader = MatFileReader::new(Path::new(test_path));
    let x_test = test_reader
        .read_field("coeff")
        .expect("failed to read 'coeff'");
    let y_test = test_reader.read_field("sol").expect("failed to read 'sol'");

    // truncate to configured n_train/n_test along the sample axis — without this,
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

    // cross-check derived config values against what was actually read/subsampled
    assert_eq!(
        s,
        config.s(),
        "subsampled grid size {} does not match config.s() {}",
        s,
        config.s()
    );
    assert_eq!(
        n_train,
        config.n_train(),
        "n_train mismatch after truncation"
    );
    assert_eq!(n_test, config.n_test(), "n_test mismatch after truncation");

    // fit x-normalizer on x_train, encode both x_train and x_test;
    // fit y-normalizer on y_train, encode y_train only — y_test stays raw,
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
        .expect("reshape x_train");
    let x_test = x_test
        .into_shape_with_order(IxDyn(&[n_test, s, s, 1]))
        .expect("reshape x_test");

    // append 2D grid coordinates as two more channels: [n, s, s, 1] -> [n, s, s, 3]
    let (xx, yy) = uniform_grid_2d(0.0, 1.0, s);
    let x_train = append_grid_2d(x_train, xx.clone(), yy.clone());
    let x_test = append_grid_2d(x_test, xx, yy);

    // package into OperatorDataset
    let train_dataset = OperatorDataset::new(x_train, y_train);
    let test_dataset = OperatorDataset::new(x_test, y_test);

    println!("darcy: {n_train} train / {n_test} test at s={s}");

    (train_dataset, test_dataset, y_normalizer)
}

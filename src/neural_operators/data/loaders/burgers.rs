//! Burgers equation dataset: config and constructor.
//!
//! `BurgersConfig` holds Burgers-specific settings (subsample rate, derived
//! resolution) on top of the shared `DatasetConfig`. `load_burgers_uniform`
//! builds the full `OperatorDataset` from a `.mat` source file: reading,
//! subsampling, appending grid coordinates and splitting into train/test.

use crate::neural_operators::data::{
    dataset::OperatorDataset,
    loaders::base_dataset::{BaseDatasetConfig, DatasetConfig, HasBaseConfig},
    io::{
        traits::FieldReader,
        readers::mat::MatFileReader,
    },
    transforms::subsample::subsample,
    grids::{append_grid_1d, uniform_grid_1d},
    split::train_test_split,
};
use burn::config::Config;
use ndarray::IxDyn;
use std::path::{Path, PathBuf};

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
/// `n_train`/`n_test` samples, and appends a uniform grid on [0, 1] as a
/// second input channel.
///
/// Inputs end up `[n, s, 1 + 1]` and targets `[n, s]` — the rank difference
/// the batcher's `RM1 = R - 1` invariant expects.
pub fn load_burgers_uniform(
    path: &PathBuf,
    config: &BurgersConfig,
) -> (OperatorDataset, OperatorDataset) {
    // 1. Read raw data from .mat file
    let reader = MatFileReader::new(Path::new(path));
    let a_data = reader.read_field("a").expect("failed to read field 'a'");
    let u_data = reader.read_field("u").expect("failed to read field 'u'");

    // 2. Subsample along spatial dimension (dim 1)
    let a_data = subsample(a_data, 1, config.subsample_rate);
    let u_data = subsample(u_data, 1, config.subsample_rate);

    // spatial size after subsampling
    let s = a_data.shape()[1];
    assert_eq!(s, config.s(), "subsampled resolution {s} != config.s() {}", config.s());

    // 3. Split into train and test
    let (a_train, a_test) = train_test_split(a_data, config.n_train(), config.n_test());
    let (u_train, u_test) = train_test_split(u_data, config.n_train(), config.n_test());

    // 4. Reshape inputs [n, s] -> [n, s, 1] to prepare for grid append
    let a_train = a_train
        .into_shape_with_order(IxDyn(&[config.n_train(), s, 1]))
        .expect("failed to reshape a_train");
    let a_test = a_test
        .into_shape_with_order(IxDyn(&[config.n_test(), s, 1]))
        .expect("failed to reshape a_test");

    // 5. Generate uniform grid [0, 1] and append as second channel
    // [n, s, 1] -> [n, s, 2]
    let grid = uniform_grid_1d(0.0, 1.0, s);
    let a_train = append_grid_1d(a_train, grid.clone());
    let a_test = append_grid_1d(a_test, grid.clone());

    // 6. Reshape targets [n, s] -> [n, s] ensure dynamic shape
    let u_train = u_train
        .into_shape_with_order(IxDyn(&[config.n_train(), s]))
        .expect("failed to reshape u_train");
    let u_test = u_test
        .into_shape_with_order(IxDyn(&[config.n_test(), s]))
        .expect("failed to reshape u_test");

    // 7. Wrap in OperatorDataset
    let train_dataset = OperatorDataset::new(a_train, u_train);
    let test_dataset = OperatorDataset::new(a_test, u_test);

    println!("burgers: {} train / {} test at s={}", config.n_train(), config.n_test(), s);

    (train_dataset, test_dataset)
}
//! Base dataset configuration shared across all operator-learning datasets.
//!
//! `DatasetConfig` holds fields common to every dataset (Burgers, Darcy and
//! future additions). Dataset-specific configs (e.g. `BurgersConfig`,
//! `DarcyConfig`) embed a `DatasetConfig` via `HasBaseConfig` rather than
//! duplicating these fields and get `BaseDatasetConfig` for free through the
//! blanket impl below.
use burn::config::Config;

/// Fields common to every dataset config.
#[derive(Config, Debug)]
pub struct DatasetConfig {
    pub n_train: usize,
    pub n_test: usize,
}

/// Accessor trait so generic code can read shared config fields without
/// knowing the concrete dataset config type.
pub trait BaseDatasetConfig {
    fn n_train(&self) -> usize;
    fn n_test(&self) -> usize;
}

/// Anything that embeds a shared `DatasetConfig` gets `BaseDatasetConfig` for free.
pub trait HasBaseConfig {
    fn base(&self) -> &DatasetConfig;
}

impl<T: HasBaseConfig> BaseDatasetConfig for T {
    fn n_train(&self) -> usize { self.base().n_train }
    fn n_test(&self) -> usize { self.base().n_test }
}
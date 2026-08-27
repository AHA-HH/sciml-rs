//! Base dataset configuration shared across all operator-learning datasets.
//!
//! `DatasetConfig` holds fields common to every dataset (Burgers, Darcy and
//! future additions). Dataset-specific configs (e.g. `BurgersConfig`,
//! `DarcyFlowConfig`) embed a `DatasetConfig` via `HasBaseConfig` rather than
//! duplicating these fields and get `BaseDatasetConfig` for free through the
//! blanket impl below.

/// Fields common to every dataset config.
pub struct DatasetConfig {
    pub n_train: usize,
    pub n_test: usize,
    pub batch_size: usize,
    pub test_batch_size: usize,
}

/// Accessor trait so generic code can read shared config fields without
/// knowing the concrete dataset config type.
pub trait BaseDatasetConfig {
    fn n_train(&self) -> usize;
    fn n_test(&self) -> usize;
    fn batch_size(&self) -> usize;
    fn test_batch_size(&self) -> usize;
}

/// Anything that embeds a shared `DatasetConfig` gets `BaseDatasetConfig` for free.
pub trait HasBaseConfig {
    fn base(&self) -> &DatasetConfig;
}

impl<T: HasBaseConfig> BaseDatasetConfig for T {
    fn n_train(&self) -> usize { self.base().n_train }
    fn n_test(&self) -> usize { self.base().n_test }
    fn batch_size(&self) -> usize { self.base().batch_size }
    fn test_batch_size(&self) -> usize { self.base().test_batch_size }
}
//! Base dataset configuration shared across all operator-learning datasets.
//!
//! `DatasetConfig` holds fields common to every dataset (Burgers, Darcy,
//! Poisson). Dataset-specific configs (e.g. `BurgersConfig`, `DarcyConfig`)
//! embed it as a `base` field rather than duplicating these fields; callers
//! read `config.base.n_train` / `config.base.n_test` directly.

// Burn's `#[derive(Config)]` writes `field: field` in the `new` it generates, in an
// impl block outside the struct's attributes (burn-derive `config/analyzer_struct.rs`).
#![allow(clippy::redundant_field_names)]

use burn::config::Config;

/// Fields common to every dataset config.
#[derive(Config, Debug)]
pub struct DatasetConfig {
    pub n_train: usize,
    pub n_test: usize,
}

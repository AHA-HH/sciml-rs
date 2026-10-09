//! Dataset constructors
//!
//! Per-PDE loaders that turn raw `.mat`/`.npy/`.npz` files into preprocessed
//! `OperatorDataset` train/test splits.

pub mod base_dataset;
pub mod burgers;
pub mod darcy;
pub mod poisson;

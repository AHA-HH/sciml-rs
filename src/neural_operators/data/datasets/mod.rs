//! Dataset constructors
//!
//! Per-PDE loaders that turn raw `.mat`/`.npy` files into preprocessed
//! `OperatorDataset` train/test splits.

pub mod burgers;
pub mod darcy;
pub mod navier_stokes;

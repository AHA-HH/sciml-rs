//! Loss functions
//!
//! Data-fit losses for operator learning: relative and absolute Lp norms, and (library
//! CM1.6a) `neuraloperator` 2.0.0's Sobolev and pointwise losses with the finite
//! differences they use, (CM1.6b) its loss aggregators, and (CM2.5) its equation (PINO)
//! losses.

pub mod data_losses;
pub mod differentiation;
pub mod equation_losses;
pub mod meta_losses;
pub mod pointwise;
pub mod sobolev;

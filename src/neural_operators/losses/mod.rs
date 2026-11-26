//! Loss functions
//!
//! Data-fit losses for operator learning: relative and absolute Lp norms, and (library
//! CM1.6a) `neuraloperator` 2.0.0's Sobolev and pointwise losses with the finite
//! differences they use, (CM1.6b) its loss aggregators, (CM2.5) its equation (PINO)
//! losses, and (CM1.6c) its Fourier and point-cloud derivatives.

pub mod data_losses;
pub mod differentiation;
pub mod equation_losses;
pub mod meta_losses;
pub mod non_uniform_fd;
pub mod pointwise;
pub mod sobolev;

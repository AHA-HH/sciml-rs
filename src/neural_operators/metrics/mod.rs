//! Metrics reporting
//!
//! The Poisson errors on the Chebyshev grid and their summary statistics. Per-epoch
//! training metrics and their CSV and plot live in [`crate::neural_operators::runs`].

pub mod poisson;
pub mod stats;

//! Metrics reporting
//!
//! The Poisson errors on the Chebyshev grid and their summary statistics, and (feature
//! `datagen`) the POD of a snapshot set. Per-epoch training metrics and their CSV and plot
//! live in [`crate::neural_operators::runs`].

#[cfg(feature = "datagen")]
pub mod pod;
pub mod poisson;
pub mod stats;

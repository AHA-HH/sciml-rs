//! Metrics reporting
//!
//! Persisting per-epoch training metrics to CSV and rendering loss and
//! learning-rate curves to PNG; the Poisson errors on the Chebyshev grid and their
//! summary statistics.

pub mod io;
pub mod plots;
pub mod poisson;
pub mod stats;

//! Metrics reporting
//!
//! Persisting per-epoch training metrics to CSV and rendering loss and
//! learning-rate curves to PNG; the Poisson errors on the Chebyshev grid.

pub mod io;
pub mod plots;
pub mod poisson;

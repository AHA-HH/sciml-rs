//! Data pipeline
//!
//! Loading, preprocessing, splitting and batching of PDE datasets into the
//! in-memory examples consumed by the training loops.

pub mod batcher;
pub mod catalogue;
pub mod dataset;
pub mod device_batcher;
pub mod host_float;
#[cfg(test)]
pub(crate) mod host_oracle;
pub mod io;
pub mod loaders;
pub mod split;
pub mod transforms;

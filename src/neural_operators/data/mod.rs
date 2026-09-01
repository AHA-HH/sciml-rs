//! Data pipeline
//!
//! Loading, preprocessing, splitting and batching of PDE datasets into the
//! in-memory examples consumed by the training loops.

pub mod batcher;
pub mod dataitem;
pub mod dataset;
pub mod grids;
pub mod io;
pub mod loaders;
pub mod split;
pub mod transforms;

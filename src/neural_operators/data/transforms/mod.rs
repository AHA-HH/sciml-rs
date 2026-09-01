//! Data transforms
//!
//! Array-level preprocessing (subsampling, grid concatenation, reshaping) and
//! normalizers that fit statistics on the training set and encode/decode
//! model inputs and predictions.

pub mod normalizers;
pub mod subsample;

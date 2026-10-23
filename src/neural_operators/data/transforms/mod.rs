//! Data transforms
//!
//! Array-level preprocessing (strided subsampling) and
//! normalizers that fit statistics on the training set and encode/decode
//! model inputs and predictions.

pub mod normalizers;
pub mod strided;

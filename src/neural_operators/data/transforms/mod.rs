//! Data transforms
//!
//! Array-level preprocessing (subsampling, grid concatenation, reshaping) and
//! normalisers that fit statistics on the training set and encode/decode
//! model inputs and predictions.

pub mod normalisers;
pub mod subsample;

//! Neural operator layers
//!
//! Building blocks for spectral architectures: the spectral convolution layer, and the parts
//! of `neuraloperator` 2.0.0's FNO block (channel MLP, skips, AdaIN), its positional embeddings,
//! and its `resample`.

pub mod channel_mlp;
pub mod embedding;
pub mod norm;
pub(crate) mod pointwise;
pub mod resample;
pub mod skip;
pub mod spectral_convolution;

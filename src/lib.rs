//! Scientific machine learning in Rust.
//!
//! A Neural Operator crate built on [Burn](https://burn.dev).
//! Tensor rank is a compile-time parameter: `FNO<3>` (1D), `FNO<4>` (2D) and
//! `FNO<5>` (3D) are distinct types generated from the same code, so shape
//! invariants are checked by the compiler rather than at runtime.
//!
//! # Layout
//!
//! - [`neural_operators::models`] - the FNO
//! - [`neural_operators::layers`] - spectral convolution
//! - [`neural_operators::training`] - training loop and metrics
//! - [`neural_operators::data`] - loaders, batching, transforms
//!
//! # Examples
//!
//! Training entry points are in `examples/train/` (`burgers.rs`, `darcy.rs`);
//! re-evaluation of saved runs is in `examples/predict/`.

// Without a backend Burn's `Device::default()` panics at runtime; fail the
// build instead with an actionable message.
#[cfg(not(any(
    feature = "flex",
    feature = "metal",
    feature = "cuda",
    // feature = "rocm",
    // feature = "vulkan",
    feature = "wgpu"
)))]
compile_error!(
    "sciml-rs needs a backend feature: flex (default), metal, cuda, rocm, vulkan or wgpu"
);

#[allow(clippy::redundant_field_names)]
pub mod neural_operators;

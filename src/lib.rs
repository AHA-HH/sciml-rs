//! Scientific machine learning in Rust.
//!
//! A Neural Operator crate built on [Burn](https://burn.dev).
//! Tensor rank is a compile-time parameter: `FNO<3>` and `FNO<4>` are distinct
//! types generated from the same code, so shape invariants are checked by the
//! compiler rather than at runtime.
//!
//! # Layout
//!
//! - [`neural_operators::models`] — the FNO
//! - [`neural_operators::layers`] — spectral convolution
//! - [`neural_operators::training`] — training loop and metrics
//! - [`neural_operators::data`] — loaders, batching, transforms
//!
//! # Examples
//!
//! See `examples/burgers` and `examples/darcy`.

pub mod neural_operators;

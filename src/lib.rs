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

// rlst calls BLAS and LAPACK but leaves the provider to the final binary
// (rlst 0.9.0 `src/doc/getting_started.rs`). The provider crates are only
// linked, never called; naming them here pulls them into every binary that
// depends on this crate. Cargo.toml only declares providers for macOS and Linux.
#[cfg(all(feature = "chebyshev", any(target_os = "macos", target_os = "linux")))]
extern crate blas_src;
#[cfg(all(feature = "chebyshev", any(target_os = "macos", target_os = "linux")))]
extern crate lapack_src;
#[cfg(all(
    feature = "chebyshev",
    not(any(target_os = "macos", target_os = "linux"))
))]
compile_error!("the chebyshev feature has a BLAS/LAPACK provider only on macOS and Linux");

#[allow(clippy::redundant_field_names)]
pub mod neural_operators;

//! Scientific machine learning in Rust: neural operators on [Burn](https://burn.dev).
//!
//! The Fourier Neural Operator is the model implemented so far. Tensor rank is a
//! compile-time parameter, so the spatial dimension D is fixed by the type: `FNO<3>` is
//! 1D and `FNO<4>` is 2D (R = D + 2), and shape invariants are checked by the compiler
//! rather than at runtime. `docs/CONVENTIONS.md` defines the tensor layouts, grids,
//! padding, FFT normalisation and initialisation the code implements
//! ([`CONVENTION_VERSION`]).
//!
//! # Layout
//!
//! - [`neural_operators::models`]: the FNO and batched inference to the host
//! - [`neural_operators::layers`]: the spectral convolution
//! - [`neural_operators::losses`]: the Lp losses
//! - [`neural_operators::training`]: configs, the loss step, the training loop and the
//!   Burn `Learner` impls
//! - [`neural_operators::data`]: readers, loaders, the dataset catalogue, device batching
//!   and transforms
//! - [`neural_operators::metrics`]: the Poisson errors and summary statistics
//! - [`neural_operators::runs`]: the run directory (configs, metrics, plots, weights)
//! - [`neural_operators::chebyshev`]: Chebyshev nodes, Clenshaw–Curtis weights and the
//!   grid transfers
//! - `neural_operators::pde` (feature `datagen`): the Poisson reference solvers, forcings
//!   and dataset generation
//! - [`neural_operators::utils`]: the inverse complex FFT and backend features
//!
//! # Examples
//!
//! Training entry points are in `examples/train/` (Burgers, Darcy, Poisson and the
//! Burgers `Learner` showcase); evaluation of saved runs, resolution studies and plots
//! are in `examples/predict/`; `examples/generate/poisson.rs` (`generate_poisson`) writes
//! the Poisson datasets.
//!
//! # Features
//!
//! One backend feature is required: `flex` (default, pure-Rust CPU), `metal`, `cuda` or
//! `wgpu`. The optional `datagen` feature enables RLST (BLAS/LAPACK, FFTW), the Poisson
//! reference solvers in `neural_operators::pde` and the `generate_poisson` dataset
//! generator.

#![warn(missing_docs)]

// Without a backend Burn's `Device::default()` panics at runtime; fail the
// build instead with an actionable message.
#[cfg(not(any(
    feature = "flex",
    feature = "metal",
    feature = "cuda",
    feature = "wgpu"
)))]
compile_error!("sciml-rs needs a backend feature: flex (default), metal, cuda or wgpu");

// rlst calls BLAS and LAPACK but leaves the provider to the final binary
// (rlst 0.9.0 `src/doc/getting_started.rs`). The provider crates are only
// linked, never called; naming them here pulls them into every binary that
// depends on this crate. Cargo.toml only declares providers for macOS and Linux.
#[cfg(all(feature = "datagen", any(target_os = "macos", target_os = "linux")))]
extern crate blas_src;
#[cfg(all(feature = "datagen", any(target_os = "macos", target_os = "linux")))]
extern crate lapack_src;
#[cfg(all(
    feature = "datagen",
    not(any(target_os = "macos", target_os = "linux"))
))]
compile_error!("the datagen feature has a BLAS/LAPACK provider only on macOS and Linux");

pub mod neural_operators;

/// The version of `docs/CONVENTIONS.md` this code implements (CONVENTIONS §9).
///
/// Recorded in the sidecar of every generated dataset. It changes only with that file:
/// a change to CONVENTIONS §1–§6 or §12 bumps both, and a test checks they agree.
pub const CONVENTION_VERSION: u32 = 1;

#[cfg(test)]
mod tests {
    use super::CONVENTION_VERSION;

    #[test]
    fn convention_version_matches_conventions_md() {
        let doc = include_str!("../docs/CONVENTIONS.md");
        let stated = format!("`CONVENTION_VERSION = {CONVENTION_VERSION}`");
        assert!(
            doc.lines().take(5).any(|l| l.contains(&stated)),
            "docs/CONVENTIONS.md does not state {stated} in its header"
        );
    }
}

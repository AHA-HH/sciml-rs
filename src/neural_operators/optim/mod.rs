//! Optimisers that Burn does not provide.
//!
//! - [`gauss_newton`]: the generic core of a matrix-free Gauss–Newton optimiser (Phase 10):
//!   parameter vectors, Jacobian products and conjugate gradients, for any Burn
//!   [`Module`](burn::module::Module).

pub mod gauss_newton;

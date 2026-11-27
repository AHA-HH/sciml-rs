//! Optimisers that Burn does not provide.
//!
//! - [`galore`]: Tensor GaLore's gradient projector, `neuraloperator` 2.0.0's
//!   `TensorGaLoreProjector`, for real tensors of any rank (library CM3.6a).
//! - [`gauss_newton`]: the generic core of a matrix-free Gauss–Newton optimiser (Phase 10):
//!   parameter vectors, Jacobian products and conjugate gradients, for any Burn
//!   [`Module`](burn::module::Module).
//! - [`soap`]: SOAP, Adam in the eigenbasis of Shampoo's preconditioner, as Burn's
//!   per-parameter `Optimizer` for tensors of any rank (optimisers CO3.2).

pub mod galore;
pub mod gauss_newton;
pub mod soap;

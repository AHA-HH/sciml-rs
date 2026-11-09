//! The generic core of a matrix-free Gauss–Newton optimiser (Phase 10, T1; design §13.17).
//!
//! For a least-squares problem Φ(θ) = ½‖r(θ)‖² with Jacobian J = ∂r/∂θ, Gauss–Newton needs
//! the products J v and Jᵀu and the solve of (JᵀJ + λI) p = b, and never the Jacobian
//! itself (2.37M parameters × 4,096 outputs per sample on the Phase 5 baseline). This
//! module gives them for any Burn [`Module`](burn::module::Module):
//!
//! - [`ParamVec`], [`params`] and [`with_params`]: a model's float parameters as one
//!   vector, and back.
//! - [`LeastSquares`]: what a problem supplies, its residual and optionally an exact J v.
//! - [`vjp`] (Jᵀu, one backward), [`jvp`] (J v, exact if the problem has it, else central
//!   finite differences) and [`adjoint_error`], the runtime check of one against the other.
//! - [`cg`]: preconditioned conjugate gradients on an operator given as a closure, with
//!   the Lanczos estimates of its extreme eigenvalues.
//! - [`GaussNewton`] (T2): one damped Gauss–Newton step on a [`LeastSquares`] problem,
//!   with ρ, Levenberg–Marquardt damping, backtracking over CG's iterates, a warm start,
//!   the [`hutchinson_diagonal`] preconditioner, the diagnostics of [`GnStep`], and its
//!   state as a [`GaussNewtonState`] for checkpoints.
//!
//! The pinned Burn fork has first-order reverse-mode autodiff only, so J v cannot come
//! from Burn (phase README, "Facts verified"). Nothing here imports the FNO or the trainer.

mod cg;
mod param_vec;
mod products;
mod state;
mod step;

pub use cg::{CgConfig, CgIterate, CgResult, CgStop, cg};
pub use param_vec::{ParamVec, params, with_params};
pub use products::{LeastSquares, adjoint_error, default_rel_step, jvp, vjp};
pub use state::GaussNewtonState;
pub use step::{GaussNewton, GaussNewtonConfig, GnStep, Preconditioner, hutchinson_diagonal};

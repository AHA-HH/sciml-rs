//! Training
//!
//! The generic training loop, optimiser and schedule configuration, the AdamW
//! step, per-epoch metric types, and the per-PDE trainer entry points.

pub mod adamw;
pub mod config;
pub mod helpers;
pub mod metrics;
pub mod trainer;
pub mod trainers;

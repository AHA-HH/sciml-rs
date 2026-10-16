//! Training
//!
//! The generic training loop, optimizer and schedule configuration, the Adam
//! step, per-epoch metric types, the one `train_fno` entry point and the
//! Burn `Learner` step impls.

pub mod learner;
pub mod metrics;
pub mod trainer;

//! Training
//!
//! The generic training loop, optimizer and schedule configuration, the Adam
//! step, the one `train_fno` entry point and the Burn `Learner` step impls.
//! The per-epoch metrics type is [`crate::neural_operators::runs::EpochMetrics`].

pub mod learner;
pub mod trainer;

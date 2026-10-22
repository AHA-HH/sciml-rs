//! Training
//!
//! - `config`: optimizer and schedule settings ([`config::TrainingConfig`]).
//! - `step`: the loss step shared by training, evaluation and the `Learner`, and
//!   [`step::evaluate`].
//! - `trainer`: the components, one epoch, the loop with its per-epoch observer, and
//!   [`trainer::train_fno`].
//! - `learner`: the Burn `Learner` step impls.
//!
//! The per-epoch metrics type is [`crate::neural_operators::runs::EpochMetrics`].

pub mod config;
#[cfg(test)]
pub(crate) mod fixtures;
pub mod learner;
pub mod step;
pub mod trainer;

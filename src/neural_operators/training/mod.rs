//! Training
//!
//! - `config`: optimizer and schedule settings ([`config::TrainingConfig`]).
//! - `step`: the loss step shared by training, evaluation and the `Learner`, and
//!   [`step::evaluate`].
//! - `trainer`: the components, one epoch, the loop with its per-epoch observer, and
//!   [`trainer::train_fno`].
//! - `data_parallel`: one run on several devices (Phase 9 T2), the epoch
//!   [`trainer::training_loop`] runs when the components have replicas.
//! - `gn`: Gauss–Newton training (Phase 10 T3), [`gn::FnoResidual`] and
//!   [`gn::train_epoch_gn`], the epoch [`trainer::training_loop`] runs when the config
//!   asks for Gauss–Newton.
//! - `learner`: the Burn `Learner` step impls.
//!
//! The per-epoch metrics type is [`crate::neural_operators::runs::EpochMetrics`].

pub mod config;
pub mod data_parallel;
#[cfg(test)]
pub(crate) mod fixtures;
pub mod gn;
pub mod learner;
pub mod step;
pub mod trainer;

//! Burgers (1D) FNO training entry point.

use burn::prelude::*;

use crate::neural_operators::{
    data::{dataitem::HostFloat, dataset::OperatorDataset},
    models::fno::{FNO, FNOConfig},
    training::{
        metrics::EpochMetrics,
        trainer::{TrainingConfig, build_training_components, identity, training_loop},
    },
};

/// Trains an FNO on the 1D Burgers dataset. `R = 3` (batch + 1 spatial +
/// channel), `RM1 = 2`.
///
/// Burgers has no normalizer, so both postprocess hooks are `identity`.
pub fn train_burgers<T: HostFloat>(
    train_data: OperatorDataset<T>,
    test_data: OperatorDataset<T>,
    model_cfg: &FNOConfig,
    train_cfg: &TrainingConfig,
    device: &Device,
) -> (FNO<3>, Vec<EpochMetrics>) {
    let components =
        build_training_components::<3, 2, _>(model_cfg, train_cfg, train_data, test_data, device);

    training_loop(components, train_cfg, &identity, &identity)
}

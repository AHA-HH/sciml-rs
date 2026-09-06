//! Darcy flow (2D) FNO training entry point.

use burn::prelude::*;

use crate::neural_operators::{
    data::{
        dataset::OperatorDataset,
        loaders::base_dataset::DatasetConfig,
        transforms::normalizers::{
            UnitGaussianNormalizer, decode_flat, normalizer_to_flat_tensors,
        },
    },
    models::fno::{FNO, FNOConfig},
    training::{
        metrics::EpochMetrics,
        trainer::{TrainingConfig, build_training_components, training_loop},
    },
};

/// Trains an FNO on the 2D Darcy flow dataset. `R = 4` (batch + 2 spatial +
/// channel), `RM1 = 3`.
///
/// `y_train` was encoded during loading, `y_test` was not - so training
/// decodes both sides and evaluation decodes only the prediction.
pub fn train_darcy(
    train_data: OperatorDataset,
    test_data: OperatorDataset,
    y_normalizer: &UnitGaussianNormalizer,
    model_cfg: &FNOConfig,
    train_cfg: &TrainingConfig,
    data_cfg: &DatasetConfig,
    device: &Device,
) -> (FNO<4>, Vec<EpochMetrics>) {
    let (train_mean, train_std) = normalizer_to_flat_tensors(y_normalizer, device);

    let eval_device = device.clone().inner();

    let (eval_mean, eval_std) = normalizer_to_flat_tensors(y_normalizer, &eval_device);

    let eps = y_normalizer.eps_val();

    let train_post = move |out: Tensor<2>, target: Tensor<2>| {
        (
            decode_flat(out, &train_mean, &train_std, eps),
            decode_flat(target, &train_mean, &train_std, eps),
        )
    };

    let eval_post = move |out: Tensor<2>, target: Tensor<2>| {
        (decode_flat(out, &eval_mean, &eval_std, eps), target)
    };

    let components = build_training_components::<4, 3>(
        model_cfg, train_cfg, data_cfg, train_data, test_data, device,
    );

    training_loop(components, train_cfg, data_cfg, &train_post, &eval_post)
}

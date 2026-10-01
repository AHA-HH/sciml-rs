//! Darcy flow (2D) FNO training entry point.

use burn::prelude::*;

use crate::neural_operators::{
    data::{
        dataitem::HostFloat,
        dataset::OperatorDataset,
        transforms::normalizers::{FlatDecoder, UnitGaussianNormalizer},
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
pub fn train_darcy<T: HostFloat>(
    train_data: OperatorDataset<T>,
    test_data: OperatorDataset<T>,
    y_normalizer: &UnitGaussianNormalizer,
    model_cfg: &FNOConfig,
    train_cfg: &TrainingConfig,
    device: &Device,
) -> (FNO<4>, Vec<EpochMetrics>) {
    // One decoder per device: training tensors are autodiff, evaluation
    // tensors live on the inner device (model.valid()).
    let train_decoder = FlatDecoder::new(y_normalizer, device);
    let eval_decoder = FlatDecoder::new(y_normalizer, &device.clone().inner());

    let train_post = move |out: Tensor<2>, target: Tensor<2>| {
        (train_decoder.decode(out), train_decoder.decode(target))
    };

    let eval_post = move |out: Tensor<2>, target: Tensor<2>| (eval_decoder.decode(out), target);

    let components =
        build_training_components::<4, 3, _>(model_cfg, train_cfg, train_data, test_data, device);

    training_loop(components, train_cfg, &train_post, &eval_post)
}

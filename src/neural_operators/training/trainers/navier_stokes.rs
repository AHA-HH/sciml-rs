//! Navier-Stokes (2D + time) FNO training entry point.

use burn::prelude::*;

use crate::neural_operators::{
    data::{
        loaders::{base_dataset::DatasetConfig, navier_stokes::NavierStokesDataset},
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

/// Trains an FNO over (x, y, t) on the Navier-Stokes dataset. `R = 5` (batch + x, y, t +
/// channel), `RM1 = 4`.
///
/// Same normalization contract as Darcy: `y_train` was encoded during
/// loading, `y_test` was not - so training decodes both sides and evaluation
/// decodes only the prediction. The loss is relative L2 over every
/// (x, y, t) point of each sample, as in `fourier_3d.py`.
pub fn train_navier_stokes(
    train_data: NavierStokesDataset,
    test_data: NavierStokesDataset,
    y_normalizer: &UnitGaussianNormalizer,
    model_cfg: &FNOConfig,
    train_cfg: &TrainingConfig,
    data_cfg: &DatasetConfig,
    device: &Device,
) -> (FNO<5>, Vec<EpochMetrics>) {
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

    let components = build_training_components::<5, 4>(
        model_cfg, train_cfg, data_cfg, train_data, test_data, device,
    );

    training_loop(components, train_cfg, data_cfg, &train_post, &eval_post)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::data::loaders::navier_stokes::{
        NavierStokesConfig, navier_stokes_datasets,
    };
    use ndarray::{ArrayD, IxDyn};

    /// Overfit check: on 4 tiny smooth trajectories the full pipeline
    /// (loader, normalization, FNO<5>, decode, loss) must drive the training
    /// relative L2 well below its starting value. Slow on first compile of the
    /// Metal kernels, so ignored by default:
    /// `cargo test --lib -- --ignored navier_stokes_pipeline_overfits`.
    #[test]
    #[ignore]
    fn navier_stokes_pipeline_overfits() {
        use std::f64::consts::TAU;

        // Decaying Fourier modes with per-sample phase and amplitude; smooth
        // and periodic, so a low-mode FNO can represent the map.
        let (n, s, t) = (6, 8, 6);
        let u = ArrayD::from_shape_fn(IxDyn(&[n, s, s, t]), |i| {
            let (x, y) = (i[1] as f64 / s as f64, i[2] as f64 / s as f64);
            let (k, tk) = (i[0] as f64, i[3] as f64 / t as f64);
            (1.0 + 0.2 * k) * (TAU * x + k).sin() * (TAU * y).cos() * (-tk).exp()
        });
        let cfg = NavierStokesConfig::new(DatasetConfig::new(4, 2))
            .with_t_in(2)
            .with_t_out(4);
        let (train, test, norms) = navier_stokes_datasets(u, &cfg);

        let model_cfg = FNOConfig {
            modes: vec![2, 2, 2],
            hidden_channels: 8,
            data_channels: 2,
            out_channels: 1,
            n_layers: 2,
        };
        let train_cfg = TrainingConfig::new()
            .with_epochs(60)
            .with_batch_size(2)
            .with_test_batch_size(2)
            .with_learning_rate(1e-2)
            .with_min_lr(1e-3);

        let device = Device::default().autodiff();
        let (_, metrics) = train_navier_stokes(
            train, test, &norms.y, &model_cfg, &train_cfg, &cfg.base, &device,
        );

        let first = metrics.first().unwrap().train_l2;
        let last = metrics.last().unwrap().train_l2;
        assert!(last.is_finite(), "training diverged");
        assert!(
            last < 0.25 * first,
            "train rel-L2 fell only from {first} to {last}"
        );
    }
}

//! 2D Poisson FNO training entry point (design §7).
//!
//! The datasets come from `load_poisson_uniform`: the Chebyshev fields transferred to the
//! uniform grid s = n − 1, so the model and the loop are the Darcy ones.

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

/// Trains an FNO on the 2D Poisson dataset. `R = 4` (batch + 2 spatial +
/// channel), `RM1 = 3`.
///
/// `y_train` was encoded during loading, `y_test` was not (CONVENTIONS §10) - so
/// training decodes both sides and evaluation decodes only the prediction.
pub fn train_poisson<T: HostFloat>(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::{
        chebyshev::cgl_nodes,
        data::loaders::{
            base_dataset::DatasetConfig,
            poisson::{PoissonConfig, load_poisson_uniform},
        },
    };
    use ndarray::{Array1, Array3, ArrayD};
    use ndarray_npy::NpzWriter;
    use std::fs::{self, File};
    use std::path::{Path, PathBuf};

    const N: usize = 9;
    const N_TRAIN: usize = 4;
    const N_TEST: usize = 2;

    /// Writes `count` samples of `f` = c_i · p(x, y) and `u` = −2 f on the n = 9 CGL
    /// grid to `dir/name`, as the generator lays them out. c_i differs per sample so the
    /// pointwise std is nonzero.
    fn write_split(dir: &Path, name: &str, count: usize, offset: f64) -> PathBuf {
        let x: Array1<f64> = cgl_nodes(N);
        let f: ArrayD<f64> = Array3::from_shape_fn((count, N, N), |(i, a, b)| {
            (1.0 + i as f64 + offset) * (1.0 + x[a] - 2.0 * x[a].powi(3)) * (x[b] * x[b] - 0.3)
        })
        .into_dyn();
        let path = dir.join(name);
        let mut npz = NpzWriter::new(File::create(&path).unwrap());
        npz.add_array("f", &f).unwrap();
        npz.add_array("u", &(&f * -2.0)).unwrap();
        npz.add_array("x", &x).unwrap();
        npz.add_array("y", &x).unwrap();
        npz.finish().unwrap();
        path
    }

    #[test]
    fn train_poisson_smoke() {
        let dir = std::env::temp_dir().join(format!(
            "sciml_rs_train_poisson_smoke_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let train_path = write_split(&dir, "train.npz", N_TRAIN, 0.0);
        let test_path = write_split(&dir, "test.npz", N_TEST, 0.5);

        let cfg = PoissonConfig::new(
            DatasetConfig {
                n_train: N_TRAIN,
                n_test: N_TEST,
            },
            N,
        );
        let (train, test, normalizers) =
            load_poisson_uniform::<f32>(&train_path, &test_path, &cfg).unwrap();

        let model_cfg = FNOConfig::new(vec![2, 2], 1, 1)
            .with_hidden_channels(4)
            .with_n_layers(2);
        let train_cfg = TrainingConfig::new()
            .with_epochs(2)
            .with_batch_size(2)
            .with_test_batch_size(2);
        let (_, metrics) = train_poisson(
            train,
            test,
            &normalizers.y,
            &model_cfg,
            &train_cfg,
            &Device::default().autodiff(),
        );

        assert_eq!(metrics.len(), 2);
        for m in &metrics {
            assert!(
                m.train_mse.is_finite() && m.train_l2.is_finite() && m.test_l2.is_finite(),
                "{m:?}"
            );
        }
        let _ = fs::remove_dir_all(&dir);
    }
}

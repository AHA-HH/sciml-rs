//! Batched inference with a saved run's model.
//!
//! `runs::RunDir::load_fno` rebuilds the model on any device; [`predict_host`] runs it over
//! a host batch of 2D inputs and returns the outputs on the host as f64, ready for
//! decoding and the errors.

use burn::tensor::{Device, Tensor, TensorData};
use ndarray::{Array3, ArrayViewD, Ix4, s};

use crate::neural_operators::models::fno::FNO;

/// The outputs of `model` on `inputs`, forwarded in batches of `batch_size` in sample
/// order on `device` and copied back to the host as f64.
///
/// - `inputs`: `[N, s_x, s_y, C]`, channels-last, as the model takes them (normalized);
///   dynamic rank, as `OperatorDataset::inputs` holds them.
///
/// Returns `[N, s_x, s_y]`: the model's single output channel, still normalized.
///
/// # Panics
/// If `batch_size` is 0, `inputs` is not rank 4, or the model's output is not f32 with
/// one channel.
pub fn predict_host(
    model: &FNO<4>,
    inputs: ArrayViewD<f32>,
    batch_size: usize,
    device: &Device,
) -> Array3<f64> {
    assert!(batch_size > 0, "predict_host: batch_size must be > 0");
    let inputs = inputs
        .into_dimensionality::<Ix4>()
        .expect("predict_host: inputs must be [N, s_x, s_y, C]");
    let (n, s_x, s_y, _) = inputs.dim();
    let mut pred = Array3::<f64>::zeros((n, s_x, s_y));
    for start in (0..n).step_by(batch_size) {
        let end = (start + batch_size).min(n);
        let batch = inputs.slice(s![start..end, .., .., ..]);
        let data: Vec<f32> = batch.iter().copied().collect();
        let x = Tensor::<4>::from_data(TensorData::new(data, batch.shape().to_vec()), device);
        let out = model
            .forward(x)
            .into_data()
            .try_to_vec::<f32>()
            .expect("predict_host: f32 model output");
        let out = Array3::from_shape_vec((end - start, s_x, s_y), out)
            .expect("predict_host: output must be [B, s_x, s_y, 1]")
            .mapv(f64::from);
        pred.slice_mut(s![start..end, .., ..]).assign(&out);
    }
    pred
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::models::fno::FNOConfig;
    use ndarray::{Array4, Axis};

    fn config() -> FNOConfig {
        FNOConfig {
            modes: vec![3, 3],
            hidden_channels: 4,
            data_channels: 1,
            out_channels: 1,
            n_layers: 1,
            spectral_init: None,
            padding: Some(2),
        }
    }

    /// `count` distinct inputs `[count, 8, 8, 1]`.
    fn inputs(count: usize) -> Array4<f32> {
        Array4::from_shape_fn((count, 8, 8, 1), |(k, i, j, _)| {
            ((k * 64 + i * 8 + j) as f32 * 0.37).sin()
        })
    }

    #[test]
    fn batches_match_one_full_forward() {
        let device = Device::default();
        let model: FNO<4> = config().init::<4>(&device);
        let x = inputs(5);
        // 5 samples in batches of 2: the last batch holds one sample
        let batched = predict_host(&model, x.view().into_dyn(), 2, &device);
        let whole = predict_host(&model, x.view().into_dyn(), 5, &device);
        assert_eq!(batched.dim(), (5, 8, 8));
        let diff = (&batched - &whole)
            .iter()
            .fold(0.0_f64, |m, d| m.max(d.abs()));
        assert!(diff < 1e-6, "max |batched - whole| = {diff:e}");
        // and each sample is independent of the others in its batch
        let one = predict_host(&model, x.slice(s![3..4, .., .., ..]).into_dyn(), 1, &device);
        let diff = (&one.index_axis(Axis(0), 0) - &whole.index_axis(Axis(0), 3))
            .iter()
            .fold(0.0_f64, |m, d| m.max(d.abs()));
        assert!(diff < 1e-6, "max |single - whole| for sample 3 = {diff:e}");
    }

    #[test]
    #[should_panic(expected = "inputs must be [N, s_x, s_y, C]")]
    fn rank_three_inputs_panic() {
        let device = Device::default();
        let model: FNO<4> = config().init::<4>(&device);
        let x = Array3::<f32>::zeros((2, 8, 8));
        predict_host(&model, x.view().into_dyn(), 2, &device);
    }
}

//! Batched inference with a saved run's model.
//!
//! `runs::RunDir::load_fno` rebuilds the model on any device; [`predict_host`] runs it over
//! a host batch of inputs of any rank and returns the outputs on the host as f64, ready
//! for decoding and the errors.

use burn::tensor::{Device, Tensor, TensorData};
use ndarray::{ArrayD, ArrayViewD, Axis, IxDyn, Slice};

use crate::neural_operators::models::fno::FNO;

/// The outputs of `model` on `inputs`, forwarded in batches of `batch_size` in sample
/// order on `device` and copied back to the host as f64.
///
/// - `inputs`: `[N, *spatial, C]` of rank `R`, channels-last, as the model takes them
///   (normalized); dynamic rank, as `OperatorDataset::inputs` holds them.
///
/// Returns `[N, *spatial]` (rank `R - 1`): the model's single output channel, still
/// normalized. Callers that want a fixed rank convert with `into_dimensionality`.
///
/// # Panics
/// If `batch_size` is 0, `inputs` is not rank `R`, or the model's output is not f32 with
/// one channel.
pub fn predict_host<const R: usize>(
    model: &FNO<R>,
    inputs: ArrayViewD<f32>,
    batch_size: usize,
    device: &Device,
) -> ArrayD<f64> {
    assert!(batch_size > 0, "predict_host: batch_size must be > 0");
    assert_eq!(
        inputs.ndim(),
        R,
        "predict_host: inputs must be [N, *spatial, C] of rank {R}"
    );
    let n = inputs.shape()[0];
    let spatial = &inputs.shape()[1..R - 1];
    let mut pred = ArrayD::<f64>::zeros(IxDyn(&[&[n], spatial].concat()));
    for start in (0..n).step_by(batch_size) {
        let end = (start + batch_size).min(n);
        let rows = Slice::from(start..end);
        let batch = inputs.slice_axis(Axis(0), rows);
        let data: Vec<f32> = batch.iter().copied().collect();
        let x = Tensor::<R>::from_data(TensorData::new(data, batch.shape().to_vec()), device);
        let out = model
            .forward(x)
            .into_data()
            .try_to_vec::<f32>()
            .expect("predict_host: f32 model output");
        let out = ArrayD::from_shape_vec(IxDyn(&[&[end - start], spatial].concat()), out)
            .expect("predict_host: output must be [B, *spatial, 1]")
            .mapv(f64::from);
        pred.slice_axis_mut(Axis(0), rows).assign(&out);
    }
    pred
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::models::fno::FNOConfig;
    use ndarray::{Array3, Array4, s};

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
        assert_eq!(batched.shape(), [5, 8, 8]);
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
    #[should_panic(expected = "inputs must be [N, *spatial, C] of rank 4")]
    fn rank_three_inputs_panic() {
        let device = Device::default();
        let model: FNO<4> = config().init::<4>(&device);
        let x = Array3::<f32>::zeros((2, 8, 8));
        predict_host(&model, x.view().into_dyn(), 2, &device);
    }

    /// The same batching on a 1D model (`FNO<3>`): inputs `[N, s, 1]`, outputs `[N, s]`.
    #[test]
    fn rank_three_batches_match_one_full_forward() {
        let device = Device::default();
        let config = FNOConfig {
            modes: vec![4],
            ..config()
        };
        let model: FNO<3> = config.init::<3>(&device);
        let x = Array3::from_shape_fn((5, 16, 1), |(k, i, _)| ((k * 16 + i) as f32 * 0.37).sin());
        let batched = predict_host(&model, x.view().into_dyn(), 2, &device);
        let whole = predict_host(&model, x.view().into_dyn(), 5, &device);
        assert_eq!(batched.shape(), [5, 16]);
        let diff = (&batched - &whole)
            .iter()
            .fold(0.0_f64, |m, d| m.max(d.abs()));
        assert!(diff < 1e-6, "max |batched - whole| = {diff:e}");
    }
}

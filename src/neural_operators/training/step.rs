//! The loss step shared by training, evaluation and the `Learner`: flatten the model output
//! and target, decode the output if a decoder is given, take the summed relative L²
//! (CONVENTIONS §11);
//! and [`evaluate`], the per-sample mean of that loss over a loader.

use burn::{data::dataloader::DataLoader, prelude::*};
use std::sync::Arc;

use crate::neural_operators::{
    data::{batcher::Batch, transforms::normalizers::FlatDecoder},
    losses::data_losses::{LpLoss, Reduction},
    models::fno::FNO,
};

/// Collapse model output (rank `IR`, trailing channel axis) and target
/// (rank `RM1 = R - 1`) to the rank-2 pair `LpLoss` requires.
///
/// For `R = 3`: `[b, s, 1]` and `[b, s]` both become `[b, s]`.
///
/// `RM1` must be `R - 1`:
///
/// ```
/// use burn::{Tensor, tensor::Device};
/// use sciml_rs::neural_operators::training::step::flatten_pair;
///
/// let device = Device::default();
/// let out = Tensor::<3>::zeros([2, 4, 1], &device);
/// let target = Tensor::<2>::zeros([2, 4], &device);
/// let (o, t) = flatten_pair::<3, 2>(out, target);
/// assert_eq!((o.dims(), t.dims()), ([2, 4], [2, 4]));
/// ```
///
/// Any other pairing is rejected at compile time:
///
/// ```compile_fail,E0080
/// use burn::{Tensor, tensor::Device};
/// use sciml_rs::neural_operators::training::step::flatten_pair;
///
/// let device = Device::default();
/// let out = Tensor::<3>::zeros([2, 4, 1], &device);
/// let target = Tensor::<3>::zeros([2, 4, 1], &device);
/// let _ = flatten_pair::<3, 3>(out, target);
/// ```
///
/// # Panics
///
/// If the output has more than one channel.
pub fn flatten_pair<const R: usize, const RM1: usize>(
    out: Tensor<R>,
    target: Tensor<RM1>,
) -> (Tensor<2>, Tensor<2>) {
    const {
        assert!(
            RM1 + 1 == R,
            "target rank must be output rank minus one (no channel axis)"
        )
    };
    assert_eq!(
        out.dims()[R - 1],
        1,
        "flatten_pair requires out_channels == 1"
    );

    let dims = target.dims(); // [b, ...spatial]
    let b = dims[0];
    let n_points: usize = dims[1..].iter().product();
    (out.reshape([b, n_points]), target.reshape([b, n_points]))
}

/// One batch through [`forward_loss`]: the loss and the tensors it was computed from.
pub(crate) struct StepLoss {
    /// Relative L2 summed over the batch, `[1]`.
    pub l2_sum: Tensor<1>,
    /// Model output, decoded if a decoder was given, `[b, n_points]`.
    pub out: Tensor<2>,
    /// Target in physical units, `[b, n_points]`.
    pub target: Tensor<2>,
}

/// One forward pass and its loss: the step shared by
/// [`train_epoch`](crate::neural_operators::training::trainer::train_epoch), [`evaluate`]
/// and the Burgers `Learner` ([`learner`](crate::neural_operators::training::learner)), so
/// the loss is defined once.
///
/// Runs `model` on `batch.inputs` (`[b, ...spatial, data_channels]`), collapses output
/// and `batch.targets` (`[b, ...spatial]`) with [`flatten_pair`], decodes the output with
/// `decoder` if given (targets are physical, CONVENTIONS §10), then takes the loss
/// (`LpLoss::new(R - 2, 2, Reduction::Sum)`).
///
/// # Panics
///
/// If the output has more than one channel ([`flatten_pair`]).
pub(crate) fn forward_loss<const R: usize, const RM1: usize>(
    model: &FNO<R>,
    batch: Batch<R, RM1>,
    decoder: Option<&FlatDecoder>,
) -> StepLoss {
    let out = model.forward(batch.inputs);
    let (out, target) = flatten_pair::<R, RM1>(out, batch.targets);
    let out = match decoder {
        Some(d) => d.decode(out),
        None => out,
    };
    let l2_sum = LpLoss::new(R - 2, 2, Reduction::Sum).rel(out.clone(), target.clone());
    StepLoss {
        l2_sum,
        out,
        target,
    }
}

/// Mean relative L² per sample of `model` over every batch of `loader`: the test loss
/// that training reports as `test_l2` (`LpLoss::new(R - 2, 2, Sum)` summed over the
/// batches, divided by the loader's sample count).
///
/// - `loader` yields [`Batch`]es with inputs `[B, *spatial, C]` and targets `[B, *spatial]`,
///   e.g. from `device_loader` on a plain (non-autodiff) device.
/// - `decoder` decodes the output with the target normaliser, on the loader's device;
///   `None` for data without one (Burgers).
///
/// # Panics
/// If `loader` yields no batches.
pub fn evaluate<const R: usize, const RM1: usize>(
    model: &FNO<R>,
    loader: &Arc<dyn DataLoader<Batch<R, RM1>>>,
    decoder: Option<&FlatDecoder>,
) -> f32 {
    let l2_sum = loader
        .iter()
        .map(|batch| {
            forward_loss(
                model,
                batch.expect("dataset error during evaluation"),
                decoder,
            )
            .l2_sum
        })
        .reduce(|sum, l2| sum + l2)
        .expect("evaluation loader produced no batches");
    l2_sum.into_scalar::<f32>() / loader.num_items() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::{
        data::{
            dataset::OperatorDataset, device_batcher::device_loader,
            transforms::normalizers::UnitGaussianNormalizer,
        },
        models::fno::FNOConfig,
        training::fixtures::{tiny_dataset, tiny_dataset_2d, tiny_model_cfg},
    };

    #[test]
    fn flatten_pair_collapses_to_rank_two() {
        let device = Device::default();
        let out = Tensor::<4>::zeros([2, 5, 6, 1], &device);
        let target = Tensor::<3>::zeros([2, 5, 6], &device);
        let (o, t) = flatten_pair::<4, 3>(out, target);
        assert_eq!(o.dims(), [2, 30]);
        assert_eq!(t.dims(), [2, 30]);
    }

    /// `forward_loss` against a hand-written oracle: forward, `flatten_pair`, the decoder,
    /// then `LpLoss::new(R - 2, 2, Sum).rel`, bit for bit. The decoder scales and shifts
    /// the prediction only, so a loss taken before decoding, a decoded target, or tensors
    /// returned without decoding fail.
    fn assert_forward_loss_matches_oracle<const R: usize, const RM1: usize>(
        model: &FNO<R>,
        data: &OperatorDataset<f64>,
        device: &Device,
    ) {
        let batch = device_loader::<R, RM1, _>(data, device, data.len(), None)
            .iter()
            .next()
            .expect("one batch")
            .expect("dataset error");

        let (o, t) =
            flatten_pair::<R, RM1>(model.forward(batch.inputs.clone()), batch.targets.clone());
        let decoder = FlatDecoder::new(&UnitGaussianNormalizer::fit(data.targets()), device);
        let o = decoder.decode(o);
        let expected = LpLoss::new(R - 2, 2, Reduction::Sum).rel(o.clone(), t.clone());

        let step = forward_loss(model, batch, Some(&decoder));

        assert_eq!(step.l2_sum.dims(), [1]);
        assert_eq!(
            step.l2_sum.into_scalar::<f32>().to_bits(),
            expected.into_scalar::<f32>().to_bits()
        );
        assert_eq!(step.out.into_data(), o.into_data());
        assert_eq!(step.target.into_data(), t.into_data());
    }

    #[test]
    fn forward_loss_is_flatten_pair_then_summed_relative_l2() {
        let device = Device::default();

        let model_1d = tiny_model_cfg().init::<3>(&device);
        assert_forward_loss_matches_oracle::<3, 2>(&model_1d, &tiny_dataset(3), &device);

        let model_2d = FNOConfig::new(vec![2, 2], 1, 1)
            .with_hidden_channels(4)
            .with_n_layers(1)
            .init::<4>(&device);
        assert_forward_loss_matches_oracle::<4, 3>(&model_2d, &tiny_dataset_2d(3), &device);
    }

    #[test]
    #[should_panic(expected = "out_channels == 1")]
    fn flatten_pair_rejects_multichannel_output() {
        let device = Device::default();
        let out = Tensor::<3>::zeros([2, 5, 2], &device);
        let target = Tensor::<2>::zeros([2, 5], &device);
        let _ = flatten_pair::<3, 2>(out, target);
    }

    // --- Phase 4s T13: evaluate is the per-sample mean of the summed loss ---

    #[test]
    fn evaluate_is_the_per_sample_mean_of_the_summed_loss() {
        let device = Device::default();
        let model: FNO<3> = tiny_model_cfg().init::<3>(&device);
        let n_test = 5;
        let data = tiny_dataset(n_test);
        // batches of 2 (the last holds one sample) against the whole split in one batch
        let batched = device_loader::<3, 2, _>(&data, &device, 2, None);
        let whole = device_loader::<3, 2, _>(&data, &device, n_test, None)
            .iter()
            .next()
            .expect("one batch")
            .expect("dataset error");
        let sum = forward_loss(&model, whole, None)
            .l2_sum
            .into_scalar::<f32>();
        let mean = evaluate(&model, &batched, None);
        assert!(
            (mean - sum / n_test as f32).abs() <= 1e-6 * mean.abs(),
            "{mean} != {sum} / {n_test}"
        );
    }
}

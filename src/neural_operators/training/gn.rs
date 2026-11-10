//! Gauss–Newton training of the FNO (Phase 10 T3; design §13.17): [`FnoResidual`], the
//! FNO's least-squares residual, and [`train_epoch_gn`], one epoch of
//! [`GaussNewton`] steps, the counterpart of
//! [`train_epoch`](crate::neural_operators::training::trainer::train_epoch).

use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Instant;

use burn::{data::dataloader::DataLoader, prelude::*};

use crate::neural_operators::{
    data::{batcher::Batch, transforms::normalizers::FlatDecoder},
    models::fno::FNO,
    optim::gauss_newton::{GaussNewton, LeastSquares},
    runs::{GnPasses, GnStepRecord},
    training::{
        step::{StepLoss, flatten_pair, forward_loss},
        trainer::{EpochMeans, accumulate},
    },
};

/// The FNO's residual for Gauss–Newton, residual (A) of decision G1 (CONVENTIONS §11):
/// per sample rᵢ = (ûᵢ − uᵢ) / (‖uᵢ‖ √B) on the batch's B samples, so that ½‖r‖² =
/// (1/2B) Σᵢ eᵢ², eᵢ = ‖ûᵢ − uᵢ‖ / ‖uᵢ‖ the relative L² the baseline loss sums.
///
/// ûᵢ is the model output decoded with `decoder` (if given) and uᵢ the target in physical
/// units, both flattened as the training loss flattens them; the norms are Euclidean with equal
/// weights over the learning grid, as `LpLoss::new(2, _).rel`. `R` is the model rank and
/// `RM1` the target rank, as for [`Batch`].
pub struct FnoResidual<'a, const R: usize, const RM1: usize> {
    decoder: Option<&'a FlatDecoder>,
    batch: PhantomData<Batch<R, RM1>>,
}

impl<'a, const R: usize, const RM1: usize> FnoResidual<'a, R, RM1> {
    /// The residual with `decoder` decoding the model output with the target normaliser,
    /// on the batch's device; `None` for data in physical units.
    pub fn new(decoder: Option<&'a FlatDecoder>) -> Self {
        Self {
            decoder,
            batch: PhantomData,
        }
    }

    /// The per-sample weights wᵢ = 1 / (‖uᵢ‖ √B) of `target` (`[B, n_points]`), as
    /// `[B, 1]`: the one place a reweighted residual (G1's arm C) would change.
    fn weights(&self, target: Tensor<2>) -> Tensor<2> {
        let b = target.dims()[0] as f64;
        target.powf_scalar(2.0).sum_dim(1).sqrt().recip() / b.sqrt()
    }
}

impl<const R: usize, const RM1: usize> LeastSquares<FNO<R>> for FnoResidual<'_, R, RM1> {
    type Batch = Batch<R, RM1>;

    /// The weighted residual, `[B · n_points]`, sample-major.
    ///
    /// # Panics
    /// If the model has more than one output channel ([`flatten_pair`]).
    fn residual(&self, model: &FNO<R>, batch: &Batch<R, RM1>) -> Tensor<1> {
        let out = model.forward(batch.inputs.clone());
        let (out, target) = flatten_pair::<R, RM1>(out, batch.targets.clone());
        let out = match self.decoder {
            Some(d) => d.decode(out),
            None => out,
        };
        let [b, n] = out.dims();
        let w = self.weights(target.clone());
        ((out - target) * w).reshape([b * n])
    }
}

/// The forward and backward passes of a Gauss–Newton run's training, from its
/// `gn_steps.csv` sums: the optimiser's, plus the one forward per step that
/// [`train_epoch_gn`] spends measuring the batch's loss for `metrics.csv`.
pub fn training_passes(p: &GnPasses) -> (usize, usize) {
    (p.forward + p.steps, p.backward)
}

/// One epoch of Gauss–Newton: per batch of `loader`, one [`GaussNewton::step`] on
/// [`FnoResidual`], its row passed to `on_step`. The batch's gradient, curvature and ρ
/// all come from that batch (G3), so a larger Gauss–Newton batch is the loader's batch
/// size (`TrainingConfig::batch_size`), with no second batching.
///
/// The returned [`EpochMeans`] are those [`train_epoch`](super::trainer::train_epoch)
/// reports, at each batch's incoming parameters: one extra forward per batch (without
/// autodiff) measures the baseline loss, and `last_lr` is the damping λ the next step
/// will use (there is no learning rate). `epoch` labels the rows.
///
/// `model` and `loader` are on the autodiff device, and `decoder` too.
///
/// # Panics
/// If the loader yields no batches or a failed one; as [`GaussNewton::step`].
pub fn train_epoch_gn<const R: usize, const RM1: usize>(
    mut model: FNO<R>,
    loader: &Arc<dyn DataLoader<Batch<R, RM1>>>,
    gn: &mut GaussNewton,
    decoder: Option<&FlatDecoder>,
    epoch: usize,
    on_step: &mut dyn FnMut(&GnStepRecord),
) -> (FNO<R>, EpochMeans) {
    let residual = FnoResidual::<R, RM1>::new(decoder);
    let metric_decoder = decoder.map(FlatDecoder::inner);
    let (mut mse_sum, mut l2_sum, mut n_batches) = (None, None, 0_usize);

    for batch in loader.iter() {
        let batch = batch.expect("dataset error during training");

        let plain = Batch {
            inputs: batch.inputs.clone().inner(),
            targets: batch.targets.clone().inner(),
        };
        let StepLoss {
            l2_sum: l2,
            out,
            target,
        } = forward_loss(&model.valid(), plain, metric_decoder.as_ref());
        mse_sum = accumulate(mse_sum, (out - target).powf_scalar(2.0).mean());
        l2_sum = accumulate(l2_sum, l2);
        n_batches += 1;

        let start = Instant::now();
        let (next, step) = gn.step(model, &residual, &batch);
        model = next;
        on_step(&GnStepRecord {
            epoch,
            step,
            wall_time_s: start.elapsed().as_secs_f64(),
        });
    }

    let total = |sum: Option<Tensor<1>>| {
        sum.expect("training loader produced no batches")
            .into_scalar::<f32>()
    };
    (
        model,
        EpochMeans {
            train_mse: total(mse_sum) / n_batches as f32,
            train_l2: total(l2_sum) / loader.num_items() as f32,
            last_lr: gn.lambda(),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::{
        data::{device_batcher::device_loader, transforms::normalizers::UnitGaussianNormalizer},
        losses::data_losses::{LpLoss, Reduction},
        models::fno::FNOConfig,
        training::fixtures::tiny_dataset_2d,
    };

    /// ½‖r‖² over a batch is (1/2B) Σᵢ eᵢ², eᵢ the per-sample relative L² of `LpLoss`
    /// with `Reduction::None`, to round-off, with a decoder.
    #[test]
    fn half_squared_residual_is_the_mean_squared_relative_error() {
        let device = Device::default().autodiff();
        let data = tiny_dataset_2d(5);
        let model = FNOConfig::new(vec![2, 2], 1, 1)
            .with_hidden_channels(4)
            .with_n_layers(1)
            .init::<4>(&device);
        let decoder = FlatDecoder::new(&UnitGaussianNormalizer::fit(data.targets()), &device);
        let batch = device_loader::<4, 3, _>(&data, &device, 5, None)
            .iter()
            .next()
            .expect("one batch")
            .expect("dataset error");

        let r = FnoResidual::<4, 3>::new(Some(&decoder)).residual(&model, &batch);
        assert_eq!(r.dims(), [5 * 16]);
        let half: f64 = 0.5 * r.inner().powf_scalar(2.0).sum().into_scalar::<f64>();

        let (out, target) = flatten_pair::<4, 3>(model.forward(batch.inputs), batch.targets);
        let e = LpLoss::new(2, Reduction::None).rel(decoder.decode(out), target);
        let expected = e.inner().powf_scalar(2.0).sum().into_scalar::<f64>() / (2.0 * 5.0);
        assert!(
            (half - expected).abs() <= 1e-5 * expected,
            "{half} != {expected}"
        );
    }
}

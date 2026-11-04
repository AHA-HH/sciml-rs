//! Data-parallel training of one run on several devices (Phase 9 T2): one model replica
//! per device, each batch split into one shard per replica, the shard gradients summed
//! across the devices with Burn's `all_reduce`, then the same optimiser step on every
//! replica. The batch, the learning-rate schedule and the number of steps are those of a
//! one-device run.
//!
//! [`training_loop`](super::trainer::training_loop) takes this path when
//! [`TrainingComponents::replicas`](super::trainer::TrainingComponents::replicas) is not
//! empty, and [`train_epoch`](super::trainer::train_epoch) otherwise.
//!
//! - **Sum, not mean.** The loss is the relative L² summed over the batch (CONVENTIONS
//!   §11), so the full-batch gradient is the sum of the shard gradients, whatever the
//!   shard sizes. A mean would scale it by 1/D, which Adam's L2 weight decay and epsilon
//!   do not forgive.
//! - **Shards** are contiguous rows in replica order, of sizes differing by at most one
//!   (`shard`): a partial last batch is split the same way, and a replica whose shard is
//!   empty contributes zero gradients, so every device still joins each collective.
//! - **Data:** each replica holds its own copy of the training split ([`replicas`]),
//!   shuffled with the run's seed, so every device draws the single-device batch order and
//!   no sample crosses between devices during training. Cost: one more copy of the split
//!   per extra device.
//! - **Optimiser on every replica:** all replicas receive the same summed gradient and
//!   apply the same (elementwise) Adam step, so they stay identical without broadcasting
//!   the weights after each step; the cost is one Adam state per device. Evaluation and
//!   checkpoints use the first replica.
//!
//! Collectives: cubecl implements `all_reduce` on cuda only (NCCL); flex, metal and wgpu
//! panic. Replicas that all name one device (tests of this logic on flex) are summed on
//! that device without a collective.

use burn::{
    data::dataloader::DataLoader,
    module::{Module, ModuleVisitor, Param},
    optim::{
        GradientsParams, ModuleOptimizer,
        lr_scheduler::module_lr_scheduler::{ModuleLearningRate, ModuleLrScheduler},
    },
    prelude::*,
    tensor::distributed::{ReduceOperation, all_reduce},
};
use std::{iter::once, ops::Range, sync::Arc};

use crate::neural_operators::{
    data::{
        batcher::Batch, dataset::OperatorDataset, device_batcher::device_loader,
        host_float::HostFloat, transforms::normalizers::FlatDecoder,
    },
    models::fno::FNO,
    training::{
        config::TrainingConfig,
        step::{StepLoss, forward_loss},
        trainer::{EpochMeans, accumulate},
    },
};

/// A device beyond the first in a data-parallel run, with its copy of the training split.
pub struct Replica<const R: usize, const RM1: usize> {
    /// The replica's (autodiff) device.
    pub device: Device,
    /// Shuffled training batches on `device`, in the first device's order.
    pub train_loader: Arc<dyn DataLoader<Batch<R, RM1>>>,
}

/// One [`Replica`] per device of `devices` (the devices after the first): each uploads
/// `train_data` once (inputs `[n, ...spatial, 1]`, targets `[n, ...spatial]`) and
/// batches it with `train_cfg`'s batch size and seed, as
/// [`build_training_components`](super::trainer::build_training_components) does on the
/// first device.
///
/// # Panics
///
/// If `train_data` is not of rank `R` (inputs) and `RM1` (targets).
pub fn replicas<const R: usize, const RM1: usize, T: HostFloat>(
    train_data: &OperatorDataset<T>,
    devices: &[Device],
    train_cfg: &TrainingConfig,
) -> Vec<Replica<R, RM1>> {
    devices
        .iter()
        .map(|device| Replica {
            device: device.clone(),
            train_loader: device_loader::<R, RM1, T>(
                train_data,
                device,
                train_cfg.batch_size,
                Some(train_cfg.seed),
            ),
        })
        .collect()
}

/// The working state of the replicas after the first: model, optimiser, decoder and loader.
pub(crate) struct Peers<const R: usize, const RM1: usize> {
    models: Vec<FNO<R>>,
    optims: Vec<ModuleOptimizer>,
    decoders: Vec<Option<FlatDecoder>>,
    loaders: Vec<Arc<dyn DataLoader<Batch<R, RM1>>>>,
}

impl<const R: usize, const RM1: usize> Peers<R, RM1> {
    /// Copies of the first replica's `model`, `optim` and `decoder` on each replica's device.
    pub(crate) fn new(
        model: &FNO<R>,
        optim: &ModuleOptimizer,
        decoder: Option<&FlatDecoder>,
        replicas: &[Replica<R, RM1>],
    ) -> Self {
        Self {
            models: replicas
                .iter()
                .map(|r| model.clone().fork(&r.device))
                .collect(),
            optims: vec![optim.clone(); replicas.len()],
            decoders: replicas
                .iter()
                .map(|r| decoder.map(|d| d.to_device(&r.device)))
                .collect(),
            loaders: replicas.iter().map(|r| r.train_loader.clone()).collect(),
        }
    }
}

/// The rows of a batch of `b` that replica `d` of `n` trains on: contiguous, in replica
/// order, `b / n` rows plus one for each of the first `b % n` replicas (empty when
/// `b < n` and `d >= b`).
pub(crate) fn shard(b: usize, n: usize, d: usize) -> Range<usize> {
    let (base, extra) = (b / n, b % n);
    let start = d * base + d.min(extra);
    start..start + base + usize::from(d < extra)
}

/// Forward and backward of `model` on `rows` of `batch` (of `b` samples): the gradients,
/// and the shard's share of the batch MSE (its MSE times its share of the rows) and its
/// summed relative L², both `[1]`; no losses for an empty shard.
fn shard_grads<const R: usize, const RM1: usize>(
    model: &FNO<R>,
    batch: Batch<R, RM1>,
    rows: Range<usize>,
    decoder: Option<&FlatDecoder>,
) -> (GradientsParams, Option<(Tensor<1>, Tensor<1>)>) {
    if rows.is_empty() {
        return (GradientsParams::new(), None);
    }
    let b = batch.targets.dims()[0];
    let n = rows.len();
    let shard = Batch {
        inputs: batch.inputs.narrow(0, rows.start, n),
        targets: batch.targets.narrow(0, rows.start, n),
    };
    let StepLoss {
        l2_sum,
        out,
        target,
    } = forward_loss(model, shard, decoder);
    let mse = (out.inner() - target.inner())
        .powf_scalar(2.0)
        .mean()
        .mul_scalar(n as f64 / b as f64);
    let l2 = l2_sum.clone().inner();
    let grads = GradientsParams::from_grads(l2_sum.backward(), model);
    (grads, Some((mse, l2)))
}

/// Replaces each parameter's gradient in `grads` (one per device of `devices`, in order)
/// with its sum over the devices.
struct SumGrads<'a> {
    grads: &'a mut [GradientsParams],
    devices: &'a [Device],
}

impl ModuleVisitor for SumGrads<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let shards: Vec<Option<Tensor<D>>> = self
            .grads
            .iter_mut()
            .map(|g| g.remove::<D>(param.id))
            .collect();
        if shards.iter().all(Option::is_none) {
            return; // no gradient anywhere: the optimiser skips it, as on one device
        }
        let shape = param.val().shape();
        let shards: Vec<Tensor<D>> = shards
            .into_iter()
            .zip(self.devices)
            .map(|(g, device)| g.unwrap_or_else(|| Tensor::zeros(shape.clone(), device)))
            .collect();
        for (g, sum) in self.grads.iter_mut().zip(sum_across(shards, self.devices)) {
            g.register(param.id, sum);
        }
    }
}

/// The sum of `shards` (one per device of `devices`) on every device.
fn sum_across<const D: usize>(shards: Vec<Tensor<D>>, devices: &[Device]) -> Vec<Tensor<D>> {
    if devices.iter().all(|d| *d == devices[0]) {
        // one device standing in for several: no collective (NCCL refuses a repeated device)
        let sum = shards.into_iter().reduce(|a, b| a + b).expect("no devices");
        return vec![sum; devices.len()];
    }
    // every device's collective is issued before any is waited on
    let pending: Vec<_> = shards
        .into_iter()
        .map(|t| all_reduce(t, ReduceOperation::Sum, devices.to_vec()))
        .collect();
    pending.into_iter().map(|c| c.resolve()).collect()
}

/// The shards' gradients for one batch per device (`batches[d]` on device d, all the same
/// batch), summed over the devices: one [`GradientsParams`] per device, every one the
/// full-batch gradient, and each device's losses ([`shard_grads`]).
fn summed_grads<const R: usize, const RM1: usize>(
    models: &[&FNO<R>],
    batches: Vec<Batch<R, RM1>>,
    decoders: &[Option<&FlatDecoder>],
) -> (Vec<GradientsParams>, Vec<Option<(Tensor<1>, Tensor<1>)>>) {
    let n = models.len();
    let b = batches[0].targets.dims()[0];
    let devices: Vec<Device> = batches
        .iter()
        .map(|batch| batch.targets.device().inner())
        .collect();
    let (mut grads, losses): (Vec<_>, Vec<_>) = batches
        .into_iter()
        .enumerate()
        .map(|(d, batch)| shard_grads(models[d], batch, shard(b, n, d), decoders[d]))
        .unzip();
    models[0].visit(&mut SumGrads {
        grads: &mut grads,
        devices: &devices,
    });
    (grads, losses)
}

/// [`train_epoch`](super::trainer::train_epoch) on the first replica (`model`, `loader`,
/// `optim`, `decoder`) and `peers` together: each batch split across the replicas, the
/// gradients summed over them, one scheduler step and the same optimiser step on every
/// replica. Returns the first replica's model and the epoch's [`EpochMeans`], equal to
/// a one-device epoch's up to round-off.
///
/// # Panics
///
/// As [`train_epoch`](super::trainer::train_epoch); if a collective is not supported on
/// the devices (any backend but cuda, unless all replicas share one device).
pub(crate) fn train_epoch_parallel<const R: usize, const RM1: usize>(
    mut model: FNO<R>,
    loader: &Arc<dyn DataLoader<Batch<R, RM1>>>,
    optim: &mut ModuleOptimizer,
    scheduler: &mut ModuleLrScheduler,
    decoder: Option<&FlatDecoder>,
    peers: &mut Peers<R, RM1>,
) -> (FNO<R>, EpochMeans) {
    let mut iters: Vec<_> = once(loader)
        .chain(&peers.loaders)
        .map(|l| l.iter())
        .collect();
    let decoders: Vec<Option<&FlatDecoder>> = once(decoder)
        .chain(peers.decoders.iter().map(Option::as_ref))
        .collect();
    let mut sums: Vec<(Option<Tensor<1>>, Option<Tensor<1>>)> = vec![(None, None); iters.len()];
    let mut n_batches = 0_usize;
    let mut last_lr = ModuleLearningRate::from(0.0_f64);

    while let Some(batches) = iters
        .iter_mut()
        .map(|it| it.next())
        .collect::<Option<Vec<_>>>()
    {
        let batches = batches
            .into_iter()
            .map(|b| b.expect("dataset error during training"))
            .collect();
        let models: Vec<&FNO<R>> = once(&model).chain(&peers.models).collect();
        let (grads, losses) = summed_grads(&models, batches, &decoders);
        for (sum, loss) in sums.iter_mut().zip(losses) {
            if let Some((mse, l2)) = loss {
                sum.0 = accumulate(sum.0.take(), mse);
                sum.1 = accumulate(sum.1.take(), l2);
            }
        }
        n_batches += 1;

        last_lr = scheduler.step();
        let mut grads = grads.into_iter();
        model = optim.step(
            last_lr.clone(),
            model,
            grads.next().expect("one per replica"),
        );
        peers.models = peers
            .models
            .drain(..)
            .zip(&mut peers.optims)
            .zip(grads)
            .map(|((m, o), g)| o.step(last_lr.clone(), m, g))
            .collect();
    }
    assert!(n_batches > 0, "training loader produced no batches");

    let total = |t: Option<Tensor<1>>| t.map_or(0.0, |t| t.into_scalar::<f32>());
    let (mse_sum, l2_sum) = sums.into_iter().fold((0.0, 0.0), |(m, l), (mse, l2)| {
        (m + total(mse), l + total(l2))
    });
    (
        model,
        EpochMeans {
            train_mse: mse_sum / n_batches as f32,
            train_l2: l2_sum / loader.num_items() as f32,
            last_lr: last_lr.base(),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::training::fixtures::{tiny_dataset, tiny_model_cfg};

    #[test]
    fn shards_cover_the_batch_in_order() {
        let rows = |b, n| (0..n).map(|d| shard(b, n, d)).collect::<Vec<_>>();
        assert_eq!(rows(20, 2), [0..10, 10..20]);
        assert_eq!(rows(7, 2), [0..4, 4..7]);
        assert_eq!(rows(7, 3), [0..3, 3..5, 5..7]);
        assert_eq!(rows(1, 2), [0..1, 1..1]);
    }

    /// The brief's gradient-equivalence test: the summed gradients of two shards, one per
    /// GPU and reduced by `all_reduce`, equal the whole batch's gradient on one GPU within
    /// round-off. Needs two cuda devices: cubecl has no collectives on other backends.
    #[test]
    #[cfg_attr(
        not(feature = "cuda"),
        ignore = "needs two GPUs with collectives (cuda)"
    )]
    fn all_reduced_shard_gradients_equal_the_batch_gradient() {
        #[cfg(feature = "cuda")]
        let devices = Device::enumerate(burn::tensor::DeviceType::Cuda).into_vec();
        #[cfg(not(feature = "cuda"))]
        let devices = vec![Device::default(); 2]; // reaches the collective only if distinct
        assert!(
            devices.len() >= 2,
            "needs two devices, found {}",
            devices.len()
        );
        let devices: Vec<Device> = devices[..2].iter().map(|d| d.clone().autodiff()).collect();

        let model: FNO<3> = tiny_model_cfg().init(&devices[0]);
        let data = tiny_dataset(5); // odd: shards of 3 and 2
        let batch_on = |device: &Device| {
            let loader = device_loader::<3, 2, f64>(&data, device, 5, None);
            loader.iter().next().unwrap().unwrap()
        };

        let whole = forward_loss(&model, batch_on(&devices[0]), None).l2_sum;
        let whole = GradientsParams::from_grads(whole.backward(), &model);

        let peer = model.clone().fork(&devices[1]);
        let batches = devices.iter().map(batch_on).collect();
        let (grads, _) = summed_grads(&[&model, &peer], batches, &[None, None]);

        struct Compare<'a>(&'a GradientsParams, &'a [GradientsParams], usize);
        impl ModuleVisitor for Compare<'_> {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
                let values = |g: &GradientsParams| -> Vec<f32> {
                    g.get::<D>(param.id)
                        .unwrap()
                        .into_data()
                        .iter::<f32>()
                        .collect()
                };
                let want = values(self.0);
                // round-off relative to the tensor's largest entry: the sums run in another order
                let tol = 1e-5 * want.iter().fold(0.0_f32, |m, y| m.max(y.abs()));
                for g in self.1 {
                    for (x, y) in values(g).iter().zip(&want) {
                        assert!((x - y).abs() <= tol, "{x} != {y} (tolerance {tol})");
                    }
                }
                self.2 += 1;
            }
        }
        let mut compare = Compare(&whole, &grads, 0);
        model.visit(&mut compare);
        assert!(compare.2 > 0, "no parameters compared");
    }
}

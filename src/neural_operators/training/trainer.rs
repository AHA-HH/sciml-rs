//! Generic FNO training loop

use burn::{
    config::Config,
    data::{dataloader::DataLoader, dataset::Dataset},
    lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig,
    module::Module,
    optim::{
        AdamConfig, GradientsParams, ModuleOptimizer,
        decay::WeightDecayConfig,
        lr_scheduler::module_lr_scheduler::{ModuleLearningRate, ModuleLrScheduler},
    },
    prelude::*,
};

use std::sync::Arc;

use crate::neural_operators::{
    data::{
        batcher::Batch,
        dataitem::HostFloat,
        dataset::OperatorDataset,
        device_batcher::device_loader,
        transforms::normalizers::{FlatDecoder, UnitGaussianNormalizer},
    },
    losses::data_losses::{LpLoss, Reduction},
    models::fno::{FNO, FNOConfig},
    training::metrics::EpochMetrics,
};

/// Training config
#[derive(Config, Debug)]
pub struct TrainingConfig {
    #[config(default = 100)]
    pub epochs: usize,
    #[config(default = 20)]
    pub batch_size: usize,
    #[config(default = 20)]
    pub test_batch_size: usize,
    #[config(default = 1e-3)]
    pub learning_rate: f64,
    /// f32 to match `WeightDecayConfig::penalty`, unlike the f64 types above.
    #[config(default = 1e-4)]
    pub weight_decay: f32,
    #[config(default = 1e-7)]
    pub min_lr: f64,
    #[config(default = 42)]
    pub seed: u64,
}

/// Rank-free postprocess applied after flattening: decode, or pass through.
pub type Postprocess = dyn Fn(Tensor<2>, Tensor<2>) -> (Tensor<2>, Tensor<2>);

pub fn identity(out: Tensor<2>, target: Tensor<2>) -> (Tensor<2>, Tensor<2>) {
    (out, target)
}

/// Collapse model output (rank `IR`, trailing channel axis) and target
/// (rank `RM1 = R - 1`) to the rank-2 pair `LpLoss` requires.
///
/// For `R = 3`: `[b, s, 1]` and `[b, s]` both become `[b, s]`.
///
/// `RM1` must be `R - 1`:
///
/// ```
/// use burn::{Tensor, tensor::Device};
/// use sciml_rs::neural_operators::training::trainer::flatten_pair;
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
/// use sciml_rs::neural_operators::training::trainer::flatten_pair;
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
    /// Post-processed model output, `[b, n_points]`.
    pub out: Tensor<2>,
    /// Post-processed target, `[b, n_points]`.
    pub target: Tensor<2>,
}

/// One forward pass and its loss: the step shared by [`train_epoch`], [`eval_epoch`] and
/// the Burgers `Learner` ([`learner`](crate::neural_operators::training::learner)), so
/// the loss is defined once.
///
/// Runs `model` on `batch.inputs` (`[b, ...spatial, data_channels]`), collapses output
/// and `batch.targets` (`[b, ...spatial]`) with [`flatten_pair`] and applies `post`
/// before the loss (`LpLoss::new(R - 2, 2, Reduction::Sum)`).
///
/// # Panics
///
/// If the output has more than one channel ([`flatten_pair`]).
pub(crate) fn forward_loss<const R: usize, const RM1: usize>(
    model: &FNO<R>,
    batch: Batch<R, RM1>,
    post: &Postprocess,
) -> StepLoss {
    let out = model.forward(batch.inputs);
    let (out, target) = flatten_pair::<R, RM1>(out, batch.targets);
    let (out, target) = post(out, target);
    let l2_sum = LpLoss::new(R - 2, 2, Reduction::Sum).rel(out.clone(), target.clone());
    StepLoss {
        l2_sum,
        out,
        target,
    }
}

/// Runs one evaluation pass. Returns the summed relative L2 across all
/// batches; the caller divides by the number of test samples.
pub fn eval_epoch<const R: usize, const RM1: usize>(
    model: &FNO<R>,
    loader: &Arc<dyn DataLoader<Batch<R, RM1>>>,
    post: &Postprocess,
) -> f32 {
    let mut l2_sum_tensor = None;

    for batch in loader.iter() {
        let batch = batch.expect("dataset error during evaluation");

        let l2 = forward_loss(model, batch, post).l2_sum;

        l2_sum_tensor = Some(match l2_sum_tensor {
            Some(sum) => sum + l2,
            None => l2,
        });
    }

    l2_sum_tensor
        .expect("evaluation loader produced no batches")
        .into_scalar::<f32>()
}

/// Summed losses for one training epoch. Division by the number of training
/// samples / `steps_per_epoch` happens in the caller, which owns the
/// denominators.
pub struct EpochSums {
    pub mse_sum: f32,
    pub l2_sum: f32,
    pub last_lr: f64,
}

/// Runs one training pass. Takes and returns the model - Burn's optimizer
/// consumes it on each step.
pub fn train_epoch<const R: usize, const RM1: usize>(
    mut model: FNO<R>,
    loader: &Arc<dyn DataLoader<Batch<R, RM1>>>,
    optim: &mut ModuleOptimizer,
    scheduler: &mut ModuleLrScheduler,
    post: &Postprocess,
) -> (FNO<R>, EpochSums) {
    let mut mse_sum_tensor = None;
    let mut l2_sum_tensor = None;

    let mut last_lr = ModuleLearningRate::from(0.0_f64);

    let mut iterator = loader.iter();

    loop {
        let Some(batch) = iterator.next() else {
            break;
        };

        let batch = batch.expect("dataset error during training");

        let StepLoss {
            l2_sum: l2,
            out,
            target,
        } = forward_loss(&model, batch, post);

        let mse = (out.inner() - target.inner()).powf_scalar(2.0).mean();

        let l2_metric = l2.clone().inner();

        mse_sum_tensor = Some(match mse_sum_tensor {
            Some(sum) => sum + mse,
            None => mse,
        });

        l2_sum_tensor = Some(match l2_sum_tensor {
            Some(sum) => sum + l2_metric,
            None => l2_metric,
        });

        let grads = GradientsParams::from_grads(l2.backward(), &model);

        last_lr = scheduler.step();

        model = optim.step(last_lr.clone(), model, grads);
    }

    let mse_sum = mse_sum_tensor
        .expect("training loader produced no batches")
        .into_scalar::<f32>();

    let l2_sum = l2_sum_tensor
        .expect("training loader produced no batches")
        .into_scalar::<f32>();

    (
        model,
        EpochSums {
            mse_sum,
            l2_sum,
            last_lr: last_lr.base(),
        },
    )
}

pub struct TrainingComponents<const R: usize, const RM1: usize> {
    pub model: FNO<R>,
    pub optim: ModuleOptimizer,
    pub scheduler: ModuleLrScheduler,
    pub train_loader: Arc<dyn DataLoader<Batch<R, RM1>>>,
    pub test_loader: Arc<dyn DataLoader<Batch<R, RM1>>>,
    /// Training / test sample counts: the per-sample metric divisors.
    pub n_train: usize,
    pub n_test: usize,
    /// Batches per training epoch; the cosine schedule spans
    /// `epochs * steps_per_epoch` steps.
    pub steps_per_epoch: usize,
}

/// Builds the model, optimizer, schedule and loaders.
///
/// Both splits are uploaded to `device` once and stay there for the whole
/// run (see [`device_loader`]), so they must fit in device memory.
///
/// Sample counts come from the datasets themselves, so the schedule length
/// and the metric divisors always match the data actually trained on.
///
/// # Panics
///
/// If either dataset is empty.
pub fn build_training_components<const R: usize, const RM1: usize, T: HostFloat>(
    model_cfg: &FNOConfig,
    train_cfg: &TrainingConfig,
    train_data: OperatorDataset<T>,
    test_data: OperatorDataset<T>,
    device: &Device,
) -> TrainingComponents<R, RM1> {
    let (n_train, n_test) = (train_data.len(), test_data.len());
    assert!(
        n_train > 0 && n_test > 0,
        "training needs non-empty datasets, got {n_train} train / {n_test} test samples"
    );

    device.seed(train_cfg.seed);

    // Training tensors must live on the autodiff device.
    let train_device = device.clone();

    // Evaluation tensors must live on the plain inner backend,
    // because model.valid() returns an FNO on Metal rather than Autodiff<Metal>.
    let eval_device = device.clone().inner();

    // `init` checks `modes.len() + 2 == R`.
    let model = model_cfg.init::<R>(device);

    let optim: ModuleOptimizer = AdamConfig::new()
        .with_epsilon(1e-8)
        .with_weight_decay(Some(WeightDecayConfig::new(train_cfg.weight_decay)))
        .init();

    // DataLoader yields ceil(len / batch_size) batches, the last one partial
    let steps_per_epoch = n_train.div_ceil(train_cfg.batch_size);
    let scheduler = CosineAnnealingLrSchedulerConfig::new(
        train_cfg.learning_rate,
        train_cfg.epochs * steps_per_epoch,
    )
    .with_min_lr(train_cfg.min_lr)
    .init()
    .expect("valid cosine scheduler config");

    // Each split is uploaded once and batched on the device (REVIEW.md 4.1),
    // with the same order as a DataLoader over the host dataset. The host
    // copies are dropped on return.
    let train_loader = device_loader::<R, RM1, T>(
        &train_data,
        &train_device,
        train_cfg.batch_size,
        Some(train_cfg.seed),
    );

    let test_loader =
        device_loader::<R, RM1, T>(&test_data, &eval_device, train_cfg.test_batch_size, None);

    TrainingComponents {
        model,
        optim,
        scheduler,
        train_loader,
        test_loader,
        n_train,
        n_test,
        steps_per_epoch,
    }
}

pub fn training_loop<const R: usize, const RM1: usize>(
    mut components: TrainingComponents<R, RM1>,
    train_cfg: &TrainingConfig,
    train_post: &Postprocess,
    eval_post: &Postprocess,
) -> (FNO<R>, Vec<EpochMetrics>) {
    let start = std::time::Instant::now();
    let (n_train, n_test) = (components.n_train, components.n_test);
    let steps_per_epoch = components.steps_per_epoch;
    let mut model = components.model;
    let mut metrics = Vec::with_capacity(train_cfg.epochs);

    for epoch in 0..train_cfg.epochs {
        let epoch_start = std::time::Instant::now();

        let (m, sums) = train_epoch(
            model,
            &components.train_loader,
            &mut components.optim,
            &mut components.scheduler,
            train_post,
        );
        model = m;

        let valid_model = model.valid();

        let test_l2_sum = eval_epoch(&valid_model, &components.test_loader, eval_post);

        let epoch_time = epoch_start.elapsed();

        let record = EpochMetrics {
            epoch,
            train_mse: sums.mse_sum / steps_per_epoch as f32,
            train_l2: sums.l2_sum / n_train as f32,
            test_l2: test_l2_sum / n_test as f32,
            current_lr: sums.last_lr,
        };

        println!("{record:?} | total={:.3}s", epoch_time.as_secs_f64(),);
        metrics.push(record);
    }

    println!("total training time: {:.2?}", start.elapsed());

    (model, metrics)
}

/// Trains an FNO on any dataset: builds the components and runs [`training_loop`].
///
/// `R` is the model rank (batch + spatial axes + channel) and `RM1 = R - 1` the target
/// rank: `<3, 2>` for 1D (Burgers), `<4, 3>` for 2D (Darcy, Poisson). Inputs are
/// `[n, ...spatial, 1]` and targets `[n, ...spatial]`.
///
/// `y_norm` chooses the postprocess hooks:
/// - `None`: [`identity`] on both sides, for data trained in physical units (Burgers).
/// - `Some(y)`: `y_train` was encoded with `y` during loading and `y_test` was not
///   (CONVENTIONS §10), so training decodes both prediction and target, and evaluation
///   decodes only the prediction.
///
/// # Panics
///
/// From [`build_training_components`]:
/// - If either dataset is empty.
/// - If `model_cfg.modes.len() + 2 != R` or `model_cfg.out_channels != 1`
///   (`FNOConfig::init`; the latter is the requirement of [`flatten_pair`]).
pub fn train_fno<const R: usize, const RM1: usize, T: HostFloat>(
    train_data: OperatorDataset<T>,
    test_data: OperatorDataset<T>,
    y_norm: Option<&UnitGaussianNormalizer>,
    model_cfg: &FNOConfig,
    train_cfg: &TrainingConfig,
    device: &Device,
) -> (FNO<R>, Vec<EpochMetrics>) {
    let components =
        build_training_components::<R, RM1, T>(model_cfg, train_cfg, train_data, test_data, device);

    train_built(components, y_norm, train_cfg, device)
}

/// [`train_fno`] after the build: picks the hooks for `y_norm` and runs the loop. Split out
/// so tests can give two runs the same initial model.
fn train_built<const R: usize, const RM1: usize>(
    components: TrainingComponents<R, RM1>,
    y_norm: Option<&UnitGaussianNormalizer>,
    train_cfg: &TrainingConfig,
    device: &Device,
) -> (FNO<R>, Vec<EpochMetrics>) {
    let Some(y_norm) = y_norm else {
        return training_loop(components, train_cfg, &identity, &identity);
    };

    // One decoder per device: training tensors are autodiff, evaluation
    // tensors live on the inner device (model.valid()).
    let train_decoder = FlatDecoder::new(y_norm, device);
    let eval_decoder = FlatDecoder::new(y_norm, &device.clone().inner());

    let train_post = move |out: Tensor<2>, target: Tensor<2>| {
        (train_decoder.decode(out), train_decoder.decode(target))
    };

    let eval_post = move |out: Tensor<2>, target: Tensor<2>| (eval_decoder.decode(out), target);

    training_loop(components, train_cfg, &train_post, &eval_post)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::data::batcher::OperatorBatcher;
    use crate::neural_operators::{
        chebyshev::nodes_closed_form,
        data::loaders::{
            base_dataset::DatasetConfig,
            poisson::{PoissonConfig, load_poisson_uniform},
        },
    };
    use burn::data::dataloader::DataLoaderBuilder;
    use ndarray::{Array1, Array3, ArrayD};
    use ndarray_npy::NpzWriter;
    use std::fs::{self, File};
    use std::path::{Path, PathBuf};

    #[test]
    fn flatten_pair_collapses_to_rank_two() {
        let device = Device::default();
        let out = Tensor::<4>::zeros([2, 5, 6, 1], &device);
        let target = Tensor::<3>::zeros([2, 5, 6], &device);
        let (o, t) = flatten_pair::<4, 3>(out, target);
        assert_eq!(o.dims(), [2, 30]);
        assert_eq!(t.dims(), [2, 30]);
    }

    /// `forward_loss` against a hand-written oracle: forward, `flatten_pair`, the hook,
    /// then `LpLoss::new(R - 2, 2, Sum).rel`, bit for bit. The hook doubles the
    /// prediction, so a loss taken before the hook, or tensors returned without it, fail.
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
        let o = o.mul_scalar(2.0);
        let expected = LpLoss::new(R - 2, 2, Reduction::Sum).rel(o.clone(), t.clone());

        let double = |out: Tensor<2>, target: Tensor<2>| (out.mul_scalar(2.0), target);
        let step = forward_loss(model, batch, &double);

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

    // --- REVIEW1.md N5: schedule length and divisors come from the data ---

    /// `n` Burgers-shaped samples: inputs `[n, 8, 1]`, targets `[n, 8]`,
    /// targets nonzero so the relative L2 is well defined.
    fn tiny_dataset(n: usize) -> OperatorDataset<f64> {
        let inputs = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[n, 8, 1]), |i| {
            ((i[0] * 8 + i[1]) as f64 * 0.3 + i[2] as f64).sin()
        });
        let targets = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[n, 8]), |i| {
            2.0 + ((i[0] * 8 + i[1]) as f64 * 0.7).cos()
        });
        OperatorDataset::new(inputs, targets)
    }

    fn tiny_model_cfg() -> FNOConfig {
        FNOConfig::new(vec![2], 1, 1)
            .with_hidden_channels(4)
            .with_n_layers(1)
    }

    #[test]
    fn components_take_sample_counts_from_the_datasets() {
        let train_cfg = TrainingConfig::new().with_batch_size(2);
        let c = build_training_components::<3, 2, _>(
            &tiny_model_cfg(),
            &train_cfg,
            tiny_dataset(5),
            tiny_dataset(3),
            &Device::default().autodiff(),
        );
        assert_eq!((c.n_train, c.n_test), (5, 3));
        assert_eq!(c.steps_per_epoch, 3); // ceil(5 / 2)
    }

    #[test]
    fn schedule_and_test_divisor_match_the_data() {
        let (n_train, n_test, epochs) = (5, 3, 2);
        let (lr, min_lr) = (1e-3, 1e-5);
        let train_cfg = TrainingConfig::new()
            .with_epochs(epochs)
            .with_batch_size(2)
            .with_test_batch_size(2)
            .with_learning_rate(lr)
            .with_min_lr(min_lr);
        let components = build_training_components::<3, 2, _>(
            &tiny_model_cfg(),
            &train_cfg,
            tiny_dataset(n_train),
            tiny_dataset(n_test),
            &Device::default().autodiff(),
        );
        let (model, metrics) = training_loop(components, &train_cfg, &identity, &identity);

        // Cosine over N = epochs * ceil(5 / 2) = 6 steps: step k (1-based)
        // gives min + (max - min)(1 + cos((k - 1)π / N)) / 2.
        let n = (epochs * 3) as f64;
        let expected_lr =
            min_lr + 0.5 * (lr - min_lr) * (1.0 + ((n - 1.0) / n * std::f64::consts::PI).cos());
        let last_lr = metrics.last().unwrap().current_lr;
        assert!(
            (last_lr - expected_lr).abs() < 1e-15,
            "{last_lr} != {expected_lr}"
        );

        // test_l2 is the summed relative L2 of the final model over the 3 test
        // samples, divided by 3.
        let loader = DataLoaderBuilder::new(OperatorBatcher::<3, 2>::new())
            .set_device(Device::default())
            .batch_size(2)
            .build(tiny_dataset(n_test));
        let expected = eval_epoch(&model.valid(), &loader, &identity) / n_test as f32;
        let test_l2 = metrics.last().unwrap().test_l2;
        assert!(
            (test_l2 - expected).abs() <= 1e-6 * expected.abs(),
            "{test_l2} != {expected}"
        );
    }

    // --- REVIEW.md 4.1: device-resident loaders change nothing numerically ---

    /// Full `training_loop` with the device-resident loaders against the same
    /// components with the pre-4.1 host loaders swapped in: every metric of
    /// every epoch must be exactly equal (same batches, same order, same
    /// kernels), as must the trained model's output on flex.
    ///
    /// On the fused GPU backends the output may differ in the last bits: the
    /// fuser can compile the two runs' identical op sequences into different
    /// kernels (seen on metal: one value 2 ulp apart, exact with fusion off),
    /// so there it is compared to a relative 1e-6.
    #[test]
    fn device_loaders_train_identically_to_host_loaders() {
        let (n_train, n_test) = (7, 3); // partial last batch in both splits
        let train_cfg = TrainingConfig::new()
            .with_epochs(3)
            .with_batch_size(2)
            .with_test_batch_size(2)
            .with_seed(11);
        let device = Device::default().autodiff();
        let build = || {
            build_training_components::<3, 2, _>(
                &tiny_model_cfg(),
                &train_cfg,
                tiny_dataset(n_train),
                tiny_dataset(n_test),
                &device,
            )
        };

        // One initial model for both runs: tests run in parallel and share the
        // device RNG, so a second seeded `init` isn't guaranteed to match.
        let device_run = build();
        let mut host = build();
        host.model = device_run.model.clone();

        let (new_model, new) = training_loop(device_run, &train_cfg, &identity, &identity);

        host.train_loader = DataLoaderBuilder::new(OperatorBatcher::<3, 2>::new())
            .set_device(device.clone())
            .batch_size(train_cfg.batch_size)
            .shuffle(train_cfg.seed)
            .build(tiny_dataset(n_train));
        host.test_loader = DataLoaderBuilder::new(OperatorBatcher::<3, 2>::new())
            .set_device(device.clone().inner())
            .batch_size(train_cfg.test_batch_size)
            .build(tiny_dataset(n_test));
        let (old_model, old) = training_loop(host, &train_cfg, &identity, &identity);

        assert_eq!(metric_bits(&new), metric_bits(&old));

        let probe = |m: &FNO<3>| -> Vec<f32> {
            let x = tiny_dataset(1);
            let x = Tensor::<3>::from_data(
                TensorData::new(x.inputs().iter().map(|&v| v as f32).collect(), [1, 8, 1]),
                &device.clone().inner(),
            );
            m.valid().forward(x).into_data().iter::<f32>().collect()
        };
        let (new_out, old_out) = (probe(&new_model), probe(&old_model));

        #[cfg(feature = "flex")]
        let exact = device.clone().inner() == Device::flex();
        #[cfg(not(feature = "flex"))]
        let exact = false;

        if exact {
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&new_out), bits(&old_out));
        } else {
            for (i, (n, o)) in new_out.iter().zip(&old_out).enumerate() {
                assert!((n - o).abs() <= 1e-6 * o.abs().max(1.0), "[{i}] {n} != {o}");
            }
        }
    }

    // --- Phase 4s T3: `train_fno` reproduces the per-PDE trainers it replaced ---

    /// Every metric of every epoch, as bits.
    fn metric_bits(metrics: &[EpochMetrics]) -> Vec<(u32, u32, u32, u64)> {
        metrics
            .iter()
            .map(|m| {
                (
                    m.train_mse.to_bits(),
                    m.train_l2.to_bits(),
                    m.test_l2.to_bits(),
                    m.current_lr.to_bits(),
                )
            })
            .collect()
    }

    /// `n` Darcy-shaped samples: inputs `[n, 4, 4, 1]`, targets `[n, 4, 4]`.
    fn tiny_dataset_2d(n: usize) -> OperatorDataset<f64> {
        let inputs = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[n, 4, 4, 1]), |i| {
            ((i[0] * 16 + i[1] * 4 + i[2]) as f64 * 0.3).sin()
        });
        let targets = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[n, 4, 4]), |i| {
            2.0 + ((i[0] * 16 + i[1] * 4 + i[2]) as f64 * 0.7).cos()
        });
        OperatorDataset::new(inputs, targets)
    }

    /// `None` against the old `train_burgers` body: `identity` on both sides. Both runs
    /// start from one initial model (the device RNG is shared between parallel tests).
    #[test]
    fn train_fno_none_matches_identity_path() {
        let train_cfg = TrainingConfig::new()
            .with_epochs(2)
            .with_batch_size(2)
            .with_test_batch_size(2);
        let device = Device::default().autodiff();
        let build = || {
            build_training_components::<3, 2, _>(
                &tiny_model_cfg(),
                &train_cfg,
                tiny_dataset(5),
                tiny_dataset(3),
                &device,
            )
        };
        let new = build();
        let mut old = build();
        old.model = new.model.clone();

        let (_, new) = train_built(new, None, &train_cfg, &device);
        let (_, old) = training_loop(old, &train_cfg, &identity, &identity);

        assert_eq!(metric_bits(&new), metric_bits(&old));
    }

    /// `Some(y)` against the old `train_darcy` / `train_poisson` body: decode both sides in
    /// training, the prediction only in evaluation.
    #[test]
    fn train_fno_some_matches_decoder_path() {
        let train_cfg = TrainingConfig::new()
            .with_epochs(2)
            .with_batch_size(2)
            .with_test_batch_size(2);
        let model_cfg = FNOConfig::new(vec![2, 2], 1, 1)
            .with_hidden_channels(4)
            .with_n_layers(1);
        let y = UnitGaussianNormalizer::fit(tiny_dataset_2d(5).targets());
        let device = Device::default().autodiff();
        let build = || {
            build_training_components::<4, 3, _>(
                &model_cfg,
                &train_cfg,
                tiny_dataset_2d(5),
                tiny_dataset_2d(3),
                &device,
            )
        };
        let new = build();
        let mut old = build();
        old.model = new.model.clone();

        let (_, new) = train_built(new, Some(&y), &train_cfg, &device);

        let train_decoder = FlatDecoder::new(&y, &device);
        let eval_decoder = FlatDecoder::new(&y, &device.clone().inner());
        let train_post = move |out: Tensor<2>, target: Tensor<2>| {
            (train_decoder.decode(out), train_decoder.decode(target))
        };
        let eval_post = move |out: Tensor<2>, target: Tensor<2>| (eval_decoder.decode(out), target);
        let (_, old) = training_loop(old, &train_cfg, &train_post, &eval_post);

        assert_eq!(metric_bits(&new), metric_bits(&old));
    }

    #[test]
    #[should_panic(expected = "non-empty datasets")]
    fn empty_training_set_is_rejected() {
        let _ = build_training_components::<3, 2, _>(
            &tiny_model_cfg(),
            &TrainingConfig::new(),
            tiny_dataset(0),
            tiny_dataset(3),
            &Device::default().autodiff(),
        );
    }

    #[test]
    #[should_panic(expected = "out_channels == 1")]
    fn flatten_pair_rejects_multichannel_output() {
        let device = Device::default();
        let out = Tensor::<3>::zeros([2, 5, 2], &device);
        let target = Tensor::<2>::zeros([2, 5], &device);
        let _ = flatten_pair::<3, 2>(out, target);
    }

    // --- moved from training/trainers/poisson.rs (Phase 4s T3) ---

    const POISSON_N: usize = 9;
    const POISSON_N_TRAIN: usize = 4;
    const POISSON_N_TEST: usize = 2;

    /// Writes `count` samples of `f` = c_i · p(x, y) and `u` = −2 f on the n = 9 CGL
    /// grid to `dir/name`, as the generator lays them out. c_i differs per sample so the
    /// pointwise std is nonzero.
    fn write_poisson_split(dir: &Path, name: &str, count: usize, offset: f64) -> PathBuf {
        let x: Array1<f64> = nodes_closed_form(POISSON_N);
        let f: ArrayD<f64> = Array3::from_shape_fn((count, POISSON_N, POISSON_N), |(i, a, b)| {
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
        let train_path = write_poisson_split(&dir, "train.npz", POISSON_N_TRAIN, 0.0);
        let test_path = write_poisson_split(&dir, "test.npz", POISSON_N_TEST, 0.5);

        let cfg = PoissonConfig::new(
            DatasetConfig {
                n_train: POISSON_N_TRAIN,
                n_test: POISSON_N_TEST,
            },
            POISSON_N,
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
        let (_, metrics) = train_fno::<4, 3, _>(
            train,
            test,
            Some(&normalizers.y),
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

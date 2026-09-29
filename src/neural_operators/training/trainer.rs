//! Generic FNO training loop

use burn::{
    config::Config,
    data::{
        dataloader::{DataLoader, DataLoaderBuilder},
        dataset::Dataset,
    },
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
        batcher::{Batch, OperatorBatcher},
        dataset::OperatorDataset,
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

/// Runs one evaluation pass. Returns the summed relative L2 across all
/// batches; the caller divides by the number of test samples.
pub fn eval_epoch<const R: usize, const RM1: usize>(
    model: &FNO<R>,
    loader: &Arc<dyn DataLoader<Batch<R, RM1>>>,
    loss_fn: &LpLoss,
    post: &Postprocess,
) -> f32 {
    let mut l2_sum_tensor = None;

    for batch in loader.iter() {
        let batch = batch.expect("dataset error during evaluation");

        let out = model.forward(batch.inputs);
        let (out, target) = flatten_pair::<R, RM1>(out, batch.targets);
        let (out, target) = post(out, target);

        let l2 = loss_fn.rel(out, target);

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
    loss_fn: &LpLoss,
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

        let out = model.forward(batch.inputs);

        let (out, target) = flatten_pair::<R, RM1>(out, batch.targets);

        let (out, target) = post(out, target);

        let out_metric = out.clone().inner();
        let target_metric = target.clone().inner();

        let mse = (out_metric - target_metric).powf_scalar(2.0).mean();

        let l2 = loss_fn.rel(out, target);

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
    pub loss_fn: LpLoss,
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
/// Sample counts come from the datasets themselves, so the schedule length
/// and the metric divisors always match the data actually trained on.
///
/// # Panics
///
/// If either dataset is empty.
pub fn build_training_components<const R: usize, const RM1: usize>(
    model_cfg: &FNOConfig,
    train_cfg: &TrainingConfig,
    train_data: OperatorDataset,
    test_data: OperatorDataset,
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

    let train_loader = DataLoaderBuilder::new(OperatorBatcher::<R, RM1>::new(train_device))
        .batch_size(train_cfg.batch_size)
        .shuffle(train_cfg.seed)
        .build(train_data);

    let test_loader = DataLoaderBuilder::new(OperatorBatcher::<R, RM1>::new(eval_device))
        .batch_size(train_cfg.test_batch_size)
        .build(test_data);

    TrainingComponents {
        model,
        optim,
        scheduler,
        loss_fn: LpLoss::new(R - 2, 2, Reduction::Sum),
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
            &components.loss_fn,
            train_post,
        );
        model = m;

        let valid_model = model.valid();

        let test_l2_sum = eval_epoch(
            &valid_model,
            &components.test_loader,
            &components.loss_fn,
            eval_post,
        );

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flatten_pair_collapses_to_rank_two() {
        let device = Device::default();
        let out = Tensor::<4>::zeros([2, 5, 6, 1], &device);
        let target = Tensor::<3>::zeros([2, 5, 6], &device);
        let (o, t) = flatten_pair::<4, 3>(out, target);
        assert_eq!(o.dims(), [2, 30]);
        assert_eq!(t.dims(), [2, 30]);
    }

    // --- REVIEW1.md N5: schedule length and divisors come from the data ---

    /// `n` Burgers-shaped samples: inputs `[n, 8, 2]`, targets `[n, 8]`,
    /// targets nonzero so the relative L2 is well defined.
    fn tiny_dataset(n: usize) -> OperatorDataset {
        let inputs = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[n, 8, 2]), |i| {
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
        let c = build_training_components::<3, 2>(
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
        let components = build_training_components::<3, 2>(
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
        let loader = DataLoaderBuilder::new(OperatorBatcher::<3, 2>::new(Device::default()))
            .batch_size(2)
            .build(tiny_dataset(n_test));
        let loss_fn = LpLoss::new(1, 2, Reduction::Sum);
        let expected = eval_epoch(&model.valid(), &loader, &loss_fn, &identity) / n_test as f32;
        let test_l2 = metrics.last().unwrap().test_l2;
        assert!(
            (test_l2 - expected).abs() <= 1e-6 * expected.abs(),
            "{test_l2} != {expected}"
        );
    }

    #[test]
    #[should_panic(expected = "non-empty datasets")]
    fn empty_training_set_is_rejected() {
        let _ = build_training_components::<3, 2>(
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
}

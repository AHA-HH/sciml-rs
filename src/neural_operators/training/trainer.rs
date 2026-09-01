//! Generic FNO training loop

use burn::{
    config::Config,
    data::dataloader::{DataLoader, DataLoaderBuilder},
    lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig,
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
        loaders::base_dataset::DatasetConfig,
        transforms::normalizers::UnitGaussianNormalizer,
    },
    losses::data_losses::LpLoss,
    models::fno::{FNO, FNOConfig},
    training::metrics::EpochMetrics,
};

/// Training hyperparameters, wrapping the model's own architecture config.
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
/// (rank `TR = IR - 1`) to the rank-2 pair `LpLoss` requires.
///
/// For `IR = 3`: `[b, s, 1]` and `[b, s]` both become `[b, s]`.
fn flatten_pair<const R: usize, const RM1: usize>(
    out: Tensor<R>,
    target: Tensor<RM1>,
) -> (Tensor<2>, Tensor<2>) {
    assert_eq!(
        RM1,
        R - 1,
        "target rank must be output rank minus one (no channel axis)"
    );
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

/// Rank-independent decode: (x * (std + eps)) + mean, on flattened
/// [batch, n_points] tensors. Tensor-native so autodiff traces through it
/// into the model, unlike UnitGaussiannormalizer::decode's ndarray version.
pub fn decode_flat(x: Tensor<2>, mean: &Tensor<1>, std: &Tensor<1>, eps: f64) -> Tensor<2> {
    x * (std.clone().unsqueeze::<2>() + eps) + mean.clone().unsqueeze::<2>()
}

/// Converts a fitted UnitGaussiannormalizer's mean/std into flat rank-1
/// Tensors, once, before training starts - not called per-batch.
pub fn normalizer_to_flat_tensors(
    normalizer: &UnitGaussianNormalizer,
    device: &Device,
) -> (Tensor<1>, Tensor<1>) {
    let mean_data: Vec<f64> = normalizer.mean_ref().iter().copied().collect();
    let std_data: Vec<f64> = normalizer.std_ref().iter().copied().collect();
    let n = mean_data.len();
    (
        Tensor::<1>::from_data(TensorData::new(mean_data, vec![n]), device),
        Tensor::<1>::from_data(TensorData::new(std_data, vec![n]), device),
    )
}

/// Runs one evaluation pass. Returns the summed relative L2 across all
/// batches; the caller divides by `n_test`.
pub fn eval_epoch<const R: usize, const RM1: usize>(
    model: &FNO<R>,
    loader: &Arc<dyn DataLoader<Batch<R, RM1>>>,
    loss_fn: &LpLoss,
    post: &Postprocess,
) -> f32 {
    let mut l2_sum = 0.0f32;

    for batch in loader.iter() {
        let batch = batch.expect("dataset error during evaluation");
        let out = model.forward(batch.inputs);
        let (out, target) = flatten_pair::<R, RM1>(out, batch.targets);
        let (out, target) = post(out, target);
        l2_sum += loss_fn.rel(out, target).into_scalar::<f32>();
    }

    l2_sum
}

/// Summed losses for one training epoch. Division by `n_train` /
/// `steps_per_epoch` happens in the caller, which owns the denominators.
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
    let (mut mse_sum, mut l2_sum) = (0.0f32, 0.0f32);
    let mut last_lr = ModuleLearningRate::from(0.0_f64);

    for batch in loader.iter() {
        let batch = batch.expect("dataset error during training");

        let out = model.forward(batch.inputs);
        let (out, target) = flatten_pair::<R, RM1>(out, batch.targets);
        let (out, target) = post(out, target);

        mse_sum += (out.clone() - target.clone())
            .powf_scalar(2.0)
            .mean()
            .into_scalar::<f32>();

        let l2 = loss_fn.rel(out, target);
        l2_sum += l2.clone().into_scalar::<f32>(); // logged before the step

        let grads = GradientsParams::from_grads(l2.backward(), &model);
        last_lr = scheduler.step();
        model = optim.step(last_lr.clone(), model, grads);
    }

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
}

pub fn build_training_components<const R: usize, const RM1: usize>(
    model_cfg: &FNOConfig,
    train_cfg: &TrainingConfig,
    data_cfg: &DatasetConfig,
    train_data: OperatorDataset,
    test_data: OperatorDataset,
    device: &Device,
) -> TrainingComponents<R, RM1> {
    device.seed(train_cfg.seed);

    assert_eq!(
        model_cfg.modes.len() + 2,
        R,
        "FNO<{R}> needs {} modes, config has {}",
        R - 2,
        model_cfg.modes.len()
    );

    let model = model_cfg.init::<R>(device);

    let optim: ModuleOptimizer = AdamConfig::new()
        .with_epsilon(1e-8)
        .with_weight_decay(Some(WeightDecayConfig::new(train_cfg.weight_decay)))
        .init();

    let steps_per_epoch = data_cfg.n_train.div_ceil(train_cfg.batch_size);
    let scheduler = CosineAnnealingLrSchedulerConfig::new(
        train_cfg.learning_rate,
        train_cfg.epochs * steps_per_epoch,
    )
    .with_min_lr(train_cfg.min_lr)
    .init()
    .expect("valid cosine scheduler config");

    let train_loader = DataLoaderBuilder::new(OperatorBatcher::<R, RM1>::new(device.clone()))
        .batch_size(train_cfg.batch_size)
        .shuffle(train_cfg.seed)
        .build(train_data);

    let test_loader = DataLoaderBuilder::new(OperatorBatcher::<R, RM1>::new(device.clone()))
        .batch_size(train_cfg.test_batch_size)
        .build(test_data);

    TrainingComponents {
        model,
        optim,
        scheduler,
        loss_fn: LpLoss::new(R - 2, 2, false, true),
        train_loader,
        test_loader,
    }
}

pub fn training_loop<const R: usize, const RM1: usize>(
    mut components: TrainingComponents<R, RM1>,
    train_cfg: &TrainingConfig,
    data_cfg: &DatasetConfig,
    train_post: &Postprocess,
    eval_post: &Postprocess,
) -> (FNO<R>, Vec<EpochMetrics>) {
    let start = std::time::Instant::now();
    let steps_per_epoch = data_cfg.n_train.div_ceil(train_cfg.batch_size);
    let mut model = components.model;
    let mut metrics = Vec::with_capacity(train_cfg.epochs);

    for epoch in 0..train_cfg.epochs {
        let (m, sums) = train_epoch(
            model,
            &components.train_loader,
            &mut components.optim,
            &mut components.scheduler,
            &components.loss_fn,
            train_post,
        );
        model = m;

        let test_l2_sum = eval_epoch(
            &model,
            &components.test_loader,
            &components.loss_fn,
            eval_post,
        );

        let record = EpochMetrics {
            epoch,
            train_mse: sums.mse_sum / steps_per_epoch as f32,
            train_l2: sums.l2_sum / data_cfg.n_train as f32,
            test_l2: test_l2_sum / data_cfg.n_test as f32,
            current_lr: sums.last_lr,
        };

        println!("{record:?}");
        metrics.push(record);
    }

    println!("total training time: {:.2?}", start.elapsed());

    (model, metrics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::data::transforms::normalizers::Normalizer;

    #[test]
    fn tensor_reshape_matches_ndarray_ordering() {
        let device = Device::default();

        // 0..12 in a [3, 4] ndarray - row-major, so element (i,j) = i*4 + j.
        let arr = ndarray::Array2::from_shape_fn((3, 4), |(i, j)| (i * 4 + j) as f64);
        let flat_nd: Vec<f64> = arr.iter().copied().collect();

        // Same values as a [1, 3, 4] tensor, reshaped to [1, 12].
        let t = Tensor::<3>::from_data(TensorData::new(flat_nd.clone(), vec![1, 3, 4]), &device);
        let flat_t: Vec<f64> = t.reshape([1, 12]).into_data().iter::<f64>().collect();

        assert_eq!(
            flat_nd, flat_t,
            "ndarray and Tensor flatten in different orders"
        );
    }

    #[test]
    fn decode_flat_matches_ndarray_decode() {
        let device = Device::default();

        // Non-square spatial dims - a transpose bug is invisible on square shapes.
        let (n, s1, s2) = (2, 3, 4);
        let raw = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[n, s1, s2]), |idx| {
            (idx[0] * 100 + idx[1] * 10 + idx[2]) as f64
        });

        let normalizer = UnitGaussianNormalizer::fit(&raw);
        let encoded = normalizer.encode(raw.clone());

        // Path A: ndarray decode, the reference implementation.
        let decoded_nd = normalizer.decode(encoded.clone());

        // Path B: Tensor-native decode on the flattened pair, as training uses.
        let (mean, std) = normalizer_to_flat_tensors(&normalizer, &device);
        let flat: Vec<f64> = encoded.iter().copied().collect();
        let t = Tensor::<2>::from_data(TensorData::new(flat, vec![n, s1 * s2]), &device);
        let decoded_flat = decode_flat(t, &mean, &std, normalizer.eps_val());

        let a: Vec<f64> = decoded_nd.iter().copied().collect();
        let b: Vec<f64> = decoded_flat.into_data().iter::<f64>().collect();

        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < 1e-4, "element {i}: ndarray {x} != flat {y}");
        }
    }
}

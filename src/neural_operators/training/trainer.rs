// //! Generic FNO training loop

// use burn::{
//     config::Config,
//     data::dataloader::{DataLoader, DataLoaderBuilder},
//     lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig,
//     module::AutodiffModule,
//     optim::{
//         AdamConfig, GradientsParams, ModuleOptimizer,
//         decay::WeightDecayConfig,
//         lr_scheduler::module_lr_scheduler::{ModuleLearningRate, ModuleLrScheduler},
//     },
//     prelude::*,
// };

// use std::sync::Arc;
// use std::time::{Duration, Instant};

// use crate::neural_operators::{
//     data::{
//         batcher::{Batch, OperatorBatcher},
//         dataset::OperatorDataset,
//         loaders::base_dataset::DatasetConfig,
//     },
//     losses::data_losses::LpLoss,
//     models::fno::{FNO, FNOConfig},
//     training::metrics::EpochMetrics,
// };

// /// Training config
// #[derive(Config, Debug)]
// pub struct TrainingConfig {
//     #[config(default = 100)]
//     pub epochs: usize,
//     #[config(default = 20)]
//     pub batch_size: usize,
//     #[config(default = 20)]
//     pub test_batch_size: usize,
//     #[config(default = 1e-3)]
//     pub learning_rate: f64,
//     /// f32 to match `WeightDecayConfig::penalty`, unlike the f64 types above.
//     #[config(default = 1e-4)]
//     pub weight_decay: f32,
//     #[config(default = 1e-7)]
//     pub min_lr: f64,
//     #[config(default = 42)]
//     pub seed: u64,
// }

// /// Rank-free postprocess applied after flattening: decode, or pass through.
// pub type Postprocess = dyn Fn(Tensor<2>, Tensor<2>) -> (Tensor<2>, Tensor<2>);

// pub fn identity(out: Tensor<2>, target: Tensor<2>) -> (Tensor<2>, Tensor<2>) {
//     (out, target)
// }

// /// Collapse model output (rank `IR`, trailing channel axis) and target
// /// (rank `RM1 = R - 1`) to the rank-2 pair `LpLoss` requires.
// ///
// /// For `R = 3`: `[b, s, 1]` and `[b, s]` both become `[b, s]`.
// pub fn flatten_pair<const R: usize, const RM1: usize>(
//     out: Tensor<R>,
//     target: Tensor<RM1>,
// ) -> (Tensor<2>, Tensor<2>) {
//     assert_eq!(
//         RM1,
//         R - 1,
//         "target rank must be output rank minus one (no channel axis)"
//     );
//     assert_eq!(
//         out.dims()[R - 1],
//         1,
//         "flatten_pair requires out_channels == 1"
//     );

//     let dims = target.dims(); // [b, ...spatial]
//     let b = dims[0];
//     let n_points: usize = dims[1..].iter().product();
//     (out.reshape([b, n_points]), target.reshape([b, n_points]))
// }

// /// Runs one evaluation pass. Returns the summed relative L2 across all
// /// batches; the caller divides by `n_test`.
// pub fn eval_epoch<const R: usize, const RM1: usize>(
//     model: &FNO<R>,
//     loader: &Arc<dyn DataLoader<Batch<R, RM1>>>,
//     loss_fn: &LpLoss,
//     post: &Postprocess,
// ) -> f32 {
//     let mut l2_sum_tensor = None;

//     for batch in loader.iter() {
//         let batch = batch.expect("dataset error during evaluation");

//         let out = model.forward(batch.inputs);
//         let (out, target) = flatten_pair::<R, RM1>(out, batch.targets);
//         let (out, target) = post(out, target);

//         let l2 = loss_fn.rel(out, target);

//         l2_sum_tensor = Some(match l2_sum_tensor {
//             Some(sum) => sum + l2,
//             None => l2,
//         });
//     }

//     l2_sum_tensor
//         .expect("evaluation loader produced no batches")
//         .into_scalar::<f32>()
// }

// /// Summed losses for one training epoch. Division by `n_train` /
// /// `steps_per_epoch` happens in the caller, which owns the denominators.
// pub struct EpochSums {
//     pub mse_sum: f32,
//     pub l2_sum: f32,
//     pub last_lr: f64,
// }

// // /// Runs one training pass. Takes and returns the model - Burn's optimizer
// // /// consumes it on each step.
// pub fn train_epoch<const R: usize, const RM1: usize>(
//     mut model: FNO<R>,
//     loader: &Arc<dyn DataLoader<Batch<R, RM1>>>,
//     optim: &mut ModuleOptimizer,
//     scheduler: &mut ModuleLrScheduler,
//     loss_fn: &LpLoss,
//     post: &Postprocess,
// ) -> (FNO<R>, EpochSums) {
//     let mut mse_sum_tensor = None;
//     let mut l2_sum_tensor = None;

//     let mut last_lr = ModuleLearningRate::from(0.0_f64);

//     let mut batch_time = Duration::ZERO;
//     let mut forward_time = Duration::ZERO;
//     let mut loss_time = Duration::ZERO;
//     let mut backward_time = Duration::ZERO;
//     let mut optimizer_time = Duration::ZERO;

//     let mut iterator = loader.iter();
//     let mut n_batches = 0usize;

//     loop {
//         // ============================================
//         // Batch loading / construction
//         // ============================================

//         let batch_start = Instant::now();

//         let Some(batch) = iterator.next() else {
//             break;
//         };

//         let batch = batch.expect("dataset error during training");

//         batch_time += batch_start.elapsed();

//         // ============================================
//         // Forward
//         // ============================================

//         let forward_start = Instant::now();

//         let out = model.forward(batch.inputs);

//         let (out, target) = flatten_pair::<R, RM1>(out, batch.targets);

//         let (out, target) = post(out, target);

//         forward_time += forward_start.elapsed();

//         // ============================================
//         // Loss + metrics
//         // ============================================

//         let loss_start = Instant::now();

//         let out_metric = out.clone().inner();
//         let target_metric = target.clone().inner();

//         let mse = (out_metric - target_metric).powf_scalar(2.0).mean();

//         let l2 = loss_fn.rel(out, target);

//         let l2_metric = l2.clone().inner();

//         mse_sum_tensor = Some(match mse_sum_tensor {
//             Some(sum) => sum + mse,
//             None => mse,
//         });

//         l2_sum_tensor = Some(match l2_sum_tensor {
//             Some(sum) => sum + l2_metric,
//             None => l2_metric,
//         });

//         loss_time += loss_start.elapsed();

//         // ============================================
//         // Backward
//         // ============================================

//         let backward_start = Instant::now();

//         let grads = GradientsParams::from_grads(l2.backward(), &model);

//         backward_time += backward_start.elapsed();

//         // ============================================
//         // Optimizer
//         // ============================================

//         let optimizer_start = Instant::now();

//         last_lr = scheduler.step();

//         model = optim.step(last_lr.clone(), model, grads);

//         optimizer_time += optimizer_start.elapsed();

//         n_batches += 1;
//     }

//     // ================================================
//     // End-of-epoch synchronization / metric read
//     // ================================================

//     let metric_read_start = Instant::now();

//     let mse_sum = mse_sum_tensor
//         .expect("training loader produced no batches")
//         .into_scalar::<f32>();

//     let l2_sum = l2_sum_tensor
//         .expect("training loader produced no batches")
//         .into_scalar::<f32>();

//     let metric_read_time = metric_read_start.elapsed();

//     let n = n_batches as f64;

//     println!(
//         concat!(
//             "train profile: ",
//             "batch={:.3} ms | ",
//             "forward={:.3} ms | ",
//             "loss={:.3} ms | ",
//             "backward={:.3} ms | ",
//             "optimizer={:.3} ms | ",
//             "metric_read={:.3} ms"
//         ),
//         batch_time.as_secs_f64() * 1000.0 / n,
//         forward_time.as_secs_f64() * 1000.0 / n,
//         loss_time.as_secs_f64() * 1000.0 / n,
//         backward_time.as_secs_f64() * 1000.0 / n,
//         optimizer_time.as_secs_f64() * 1000.0 / n,
//         metric_read_time.as_secs_f64() * 1000.0,
//     );

//     (
//         model,
//         EpochSums {
//             mse_sum,
//             l2_sum,
//             last_lr: last_lr.base(),
//         },
//     )
// }

// pub struct TrainingComponents<const R: usize, const RM1: usize> {
//     pub model: FNO<R>,
//     pub optim: ModuleOptimizer,
//     pub scheduler: ModuleLrScheduler,
//     pub loss_fn: LpLoss,
//     pub train_loader: Arc<dyn DataLoader<Batch<R, RM1>>>,
//     pub test_loader: Arc<dyn DataLoader<Batch<R, RM1>>>,
// }

// pub fn build_training_components<const R: usize, const RM1: usize>(
//     model_cfg: &FNOConfig,
//     train_cfg: &TrainingConfig,
//     data_cfg: &DatasetConfig,
//     train_data: OperatorDataset,
//     test_data: OperatorDataset,
//     device: &Device,
// ) -> TrainingComponents<R, RM1> {
//     device.seed(train_cfg.seed);

//     // Training tensors must live on the autodiff device.
//     let train_device = device.clone();

//     // Evaluation tensors must live on the plain inner backend,
//     // because model.valid() returns an FNO on Metal rather than Autodiff<Metal>.
//     let eval_device = device.clone().inner();

//     assert_eq!(
//         model_cfg.modes.len() + 2,
//         R,
//         "FNO<{R}> needs {} modes, config has {}",
//         R - 2,
//         model_cfg.modes.len()
//     );

//     let model = model_cfg.init::<R>(device);

//     let optim: ModuleOptimizer = AdamConfig::new()
//         .with_epsilon(1e-8)
//         .with_weight_decay(Some(WeightDecayConfig::new(train_cfg.weight_decay)))
//         .init();

//     let steps_per_epoch = data_cfg.n_train.div_ceil(train_cfg.batch_size);
//     let scheduler = CosineAnnealingLrSchedulerConfig::new(
//         train_cfg.learning_rate,
//         train_cfg.epochs * steps_per_epoch,
//     )
//     .with_min_lr(train_cfg.min_lr)
//     .init()
//     .expect("valid cosine scheduler config");

//     // let train_loader = DataLoaderBuilder::new(OperatorBatcher::<R, RM1>::new(device.clone()))
//     let train_loader = DataLoaderBuilder::new(OperatorBatcher::<R, RM1>::new(train_device))
//         .batch_size(train_cfg.batch_size)
//         .shuffle(train_cfg.seed)
//         .build(train_data);

//     // let test_loader = DataLoaderBuilder::new(OperatorBatcher::<R, RM1>::new(device.clone()))
//     let test_loader = DataLoaderBuilder::new(OperatorBatcher::<R, RM1>::new(eval_device))
//         .batch_size(train_cfg.test_batch_size)
//         .build(test_data);

//     TrainingComponents {
//         model,
//         optim,
//         scheduler,
//         loss_fn: LpLoss::new(R - 2, 2, false, true),
//         train_loader,
//         test_loader,
//     }
// }

// pub fn training_loop<const R: usize, const RM1: usize>(
//     mut components: TrainingComponents<R, RM1>,
//     train_cfg: &TrainingConfig,
//     data_cfg: &DatasetConfig,
//     train_post: &Postprocess,
//     eval_post: &Postprocess,
// ) -> (FNO<R>, Vec<EpochMetrics>) {
//     let start = std::time::Instant::now();
//     let steps_per_epoch = data_cfg.n_train.div_ceil(train_cfg.batch_size);
//     let mut model = components.model;
//     let mut metrics = Vec::with_capacity(train_cfg.epochs);

//     for epoch in 0..train_cfg.epochs {
//         let epoch_start = std::time::Instant::now();

//         let train_start = std::time::Instant::now();

//         let (m, sums) = train_epoch(
//             model,
//             &components.train_loader,
//             &mut components.optim,
//             &mut components.scheduler,
//             &components.loss_fn,
//             train_post,
//         );
//         model = m;

//         let train_time = train_start.elapsed();

//         let eval_start = std::time::Instant::now();

//         let valid_model = model.valid();

//         let test_l2_sum = eval_epoch(
//             &valid_model,
//             &components.test_loader,
//             &components.loss_fn,
//             eval_post,
//         );

//         let eval_time = eval_start.elapsed();

//         let epoch_time = epoch_start.elapsed();

//         let record = EpochMetrics {
//             epoch,
//             train_mse: sums.mse_sum / steps_per_epoch as f32,
//             train_l2: sums.l2_sum / data_cfg.n_train as f32,
//             test_l2: test_l2_sum / data_cfg.n_test as f32,
//             current_lr: sums.last_lr,
//         };

//         println!(
//             "{record:?} | train={:.3}s eval={:.3}s total={:.3}s",
//             train_time.as_secs_f64(),
//             eval_time.as_secs_f64(),
//             epoch_time.as_secs_f64(),
//         );
//         metrics.push(record);
//     }

//     println!("total training time: {:.2?}", start.elapsed());

//     (model, metrics)
// }

//! Generic FNO training loop

use burn::{
    config::Config,
    data::dataloader::{DataLoader, DataLoaderBuilder},
    lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig,
    module::AutodiffModule,
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
    },
    losses::data_losses::LpLoss,
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
pub fn flatten_pair<const R: usize, const RM1: usize>(
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

/// Runs one evaluation pass. Returns the summed relative L2 across all
/// batches; the caller divides by `n_test`.
// pub fn eval_epoch<const R: usize, const RM1: usize>(
//     model: &FNO<R>,
//     loader: &Arc<dyn DataLoader<Batch<R, RM1>>>,
//     loss_fn: &LpLoss,
//     post: &Postprocess,
// ) -> f32 {
//     let mut l2_sum = 0.0f32;

//     for batch in loader.iter() {
//         let batch = batch.expect("dataset error during evaluation");
//         let out = model.forward(batch.inputs);
//         let (out, target) = flatten_pair::<R, RM1>(out, batch.targets);
//         let (out, target) = post(out, target);
//         l2_sum += loss_fn.rel(out, target).into_scalar::<f32>();
//     }

//     l2_sum
// }
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

/// Summed losses for one training epoch. Division by `n_train` /
/// `steps_per_epoch` happens in the caller, which owns the denominators.
pub struct EpochSums {
    pub mse_sum: f32,
    pub l2_sum: f32,
    pub last_lr: f64,
}

// /// Runs one training pass. Takes and returns the model - Burn's optimizer
// /// consumes it on each step.
// pub fn train_epoch<const R: usize, const RM1: usize>(
//     mut model: FNO<R>,
//     loader: &Arc<dyn DataLoader<Batch<R, RM1>>>,
//     optim: &mut ModuleOptimizer,
//     scheduler: &mut ModuleLrScheduler,
//     loss_fn: &LpLoss,
//     post: &Postprocess,
// ) -> (FNO<R>, EpochSums) {
//     let (mut mse_sum, mut l2_sum) = (0.0f32, 0.0f32);
//     let mut last_lr = ModuleLearningRate::from(0.0_f64);

//     for batch in loader.iter() {
//         let batch = batch.expect("dataset error during training");

//         let out = model.forward(batch.inputs);
//         let (out, target) = flatten_pair::<R, RM1>(out, batch.targets);
//         let (out, target) = post(out, target);

//         mse_sum += (out.clone() - target.clone())
//             .powf_scalar(2.0)
//             .mean()
//             .into_scalar::<f32>();

//         let l2 = loss_fn.rel(out, target);
//         l2_sum += l2.clone().into_scalar::<f32>(); // logged before the step

//         let grads = GradientsParams::from_grads(l2.backward(), &model);
//         last_lr = scheduler.step();
//         model = optim.step(last_lr.clone(), model, grads);
//     }

//     (
//         model,
//         EpochSums {
//             mse_sum,
//             l2_sum,
//             last_lr: last_lr.base(),
//         },
//     )
// }

pub fn train_epoch<const R: usize, const RM1: usize>(
    mut model: FNO<R>,
    loader: &Arc<dyn DataLoader<Batch<R, RM1>>>,
    optim: &mut ModuleOptimizer,
    scheduler: &mut ModuleLrScheduler,
    loss_fn: &LpLoss,
    post: &Postprocess,
) -> (FNO<R>, EpochSums) {
    // Keep metric sums as tensors on the GPU during the epoch.
    // The concrete tensor type will be inferred from the first batch.
    let mut mse_sum_tensor = None;
    let mut l2_sum_tensor = None;

    let mut last_lr = ModuleLearningRate::from(0.0_f64);

    for batch in loader.iter() {
        let batch = batch.expect("dataset error during training");

        // Forward pass on the autodiff backend.
        let out = model.forward(batch.inputs);

        let (out, target) = flatten_pair::<R, RM1>(out, batch.targets);
        let (out, target) = post(out, target);

        // ------------------------------------------------------------
        // Metrics
        // ------------------------------------------------------------

        // MSE is only a reporting metric, so don't build an autodiff
        // graph for its subtraction / square / reduction.
        let out_metric = out.clone().inner();
        let target_metric = target.clone().inner();

        let mse = (out_metric - target_metric)
            .powf_scalar(2.0)
            .mean();

        // L2 is the actual training loss, so calculate it using the
        // autodiff tensors.
        let l2 = loss_fn.rel(out, target);

        // Make a non-autodiff version purely for metric accumulation.
        // This stays on the GPU; it does NOT copy the scalar to the CPU.
        let l2_metric = l2.clone().inner();

        // Accumulate MSE on the GPU.
        mse_sum_tensor = Some(match mse_sum_tensor {
            Some(sum) => sum + mse,
            None => mse,
        });

        // Accumulate L2 on the GPU.
        l2_sum_tensor = Some(match l2_sum_tensor {
            Some(sum) => sum + l2_metric,
            None => l2_metric,
        });

        // ------------------------------------------------------------
        // Backpropagation
        // ------------------------------------------------------------

        // Backward uses the original autodiff L2 tensor.
        let grads = GradientsParams::from_grads(l2.backward(), &model);

        last_lr = scheduler.step();

        model = optim.step(last_lr.clone(), model, grads);
    }

    // ------------------------------------------------------------
    // GPU -> CPU synchronization only once per metric per epoch.
    // ------------------------------------------------------------

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

    // Training tensors must live on the autodiff device.
    let train_device = device.clone();

    // Evaluation tensors must live on the plain inner backend,
    // because model.valid() returns an FNO on Metal rather than Autodiff<Metal>.
    let eval_device = device.clone().inner();

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

    // let train_loader = DataLoaderBuilder::new(OperatorBatcher::<R, RM1>::new(device.clone()))
    let train_loader = DataLoaderBuilder::new(OperatorBatcher::<R, RM1>::new(train_device))
        .batch_size(train_cfg.batch_size)
        .shuffle(train_cfg.seed)
        .build(train_data);

    // let test_loader = DataLoaderBuilder::new(OperatorBatcher::<R, RM1>::new(device.clone()))
    let test_loader = DataLoaderBuilder::new(OperatorBatcher::<R, RM1>::new(eval_device))
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

// pub fn training_loop<const R: usize, const RM1: usize>(
//     mut components: TrainingComponents<R, RM1>,
//     train_cfg: &TrainingConfig,
//     data_cfg: &DatasetConfig,
//     train_post: &Postprocess,
//     eval_post: &Postprocess,
// ) -> (FNO<R>, Vec<EpochMetrics>) {
//     let start = std::time::Instant::now();
//     let steps_per_epoch = data_cfg.n_train.div_ceil(train_cfg.batch_size);
//     let mut model = components.model;
//     let mut metrics = Vec::with_capacity(train_cfg.epochs);

//     for epoch in 0..train_cfg.epochs {
//         let (m, sums) = train_epoch(
//             model,
//             &components.train_loader,
//             &mut components.optim,
//             &mut components.scheduler,
//             &components.loss_fn,
//             train_post,
//         );
//         model = m;

//         let test_l2_sum = eval_epoch(
//             &model,
//             &components.test_loader,
//             &components.loss_fn,
//             eval_post,
//         );

//         let record = EpochMetrics {
//             epoch,
//             train_mse: sums.mse_sum / steps_per_epoch as f32,
//             train_l2: sums.l2_sum / data_cfg.n_train as f32,
//             test_l2: test_l2_sum / data_cfg.n_test as f32,
//             current_lr: sums.last_lr,
//         };

//         println!("{record:?}");
//         metrics.push(record);
//     }

//     println!("total training time: {:.2?}", start.elapsed());

//     (model, metrics)
// }

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

        let valid_model = model.valid();

        let test_l2_sum = eval_epoch(
            &valid_model,
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

    #[test]
    fn flatten_pair_collapses_to_rank_two() {
        let device = Device::default();
        let out = Tensor::<4>::zeros([2, 5, 6, 1], &device);
        let target = Tensor::<3>::zeros([2, 5, 6], &device);
        let (o, t) = flatten_pair::<4, 3>(out, target);
        assert_eq!(o.dims(), [2, 30]);
        assert_eq!(t.dims(), [2, 30]);
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

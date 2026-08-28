//! Training loop implementation for FNO
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    lr_scheduler::{LrScheduler, cosine::CosineAnnealingLrSchedulerConfig},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
};

use crate::neural_operators::{
    data::{
        pipeline::{OperatorBatcherNd, OperatorDataset},
        preprocess::UnitGaussianNormaliser,
    },
    model::{fno_nd::{FNOnd, FNOndConfig}},
    utils::LpLoss,
};

#[derive(Config, Debug)]
pub struct FNOConfig {
    pub modes: Vec<usize>,          
    #[config(default = 64)]
    pub width: usize,
    #[config(default = 100)]
    pub epochs: usize,
    #[config(default = 4)]
    pub batch_size: usize,
    #[config(default = 1e-3)]
    pub learning_rate: f64,
    #[config(default = 1e-4)]
    pub weight_decay: f32,
    #[config(default = 1e-7)]
    pub min_lr: f64,
    #[config(default = 1024)]
    pub n_train: usize,
    #[config(default = 100)]
    pub n_test: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1)]
    pub data_channels: usize,       
    #[config(default = 1)]
    pub out_channels: usize,        
    #[config(default = 4)]
    pub n_layers: usize,
}

/// Per epoch metrics
#[derive(Debug, Clone)]
pub struct EpochMetricsNd {
    pub epoch: usize,
    pub train_mse: f32,
    pub train_l2: f32,
    pub test_l2: f32,
    pub current_lr: f64,
}

/// Training loops
pub fn train_burgers_fno(
    train_data: OperatorDataset,
    test_data: OperatorDataset,
    config: &FNOConfig,
    device: &Device,
) -> (FNOnd<3>, Vec<EpochMetricsNd>) {
    // Model
    let model_config = FNOndConfig {
        modes: config.modes.clone(),
        width: config.width,
        data_channels: config.data_channels,
        out_channels: config.out_channels,
        n_layers: config.n_layers,
    };
    let mut model: FNOnd<3> = model_config.init(device);

    // Optimiser Adam with weight decay
    let optim_config = AdamConfig::new()
        .with_epsilon(1e-8)
        .with_weight_decay(Some(
            burn::optim::decay::WeightDecayConfig::new(config.weight_decay),
    ));
    let mut optim = optim_config.init();

    // LR scheduler cosine annealing as OneCycleLR stopgap
    let steps_per_epoch = (config.n_train + config.batch_size - 1) / config.batch_size;
    let total_steps = config.epochs * steps_per_epoch;
    
    let mut scheduler = CosineAnnealingLrSchedulerConfig::new(config.learning_rate, total_steps)
        .with_min_lr(config.min_lr)
        .init()
        .expect("valid cosine scheduler config");

    let loss_fn = LpLoss::new(1, 2, false, true);

    // Data loaders
    let train_loader = DataLoaderBuilder::new(OperatorBatcherNd::new(device.clone()))
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .build(train_data);

    let test_loader = DataLoaderBuilder::new(OperatorBatcherNd::new(device.clone()))
        .batch_size(config.batch_size)
        .build(test_data);

    let mut metrics: Vec<EpochMetricsNd> = Vec::with_capacity(config.epochs);

    for epoch in 0..config.epochs {
        // Training
        let mut train_mse_accum = 0.0_f32;
        let mut train_l2_accum = 0.0_f32;
        let mut batch_count = 0;
        let mut last_lr = 0.0_f64;

        for batch in train_loader.iter() {
            let out = model.forward(batch.inputs.clone());
            let [b, s, _] = out.dims();
            let out_flat = out.clone().reshape([b, s]);
            let y_flat = batch.targets.clone();

            let mse = (out_flat.clone() - y_flat.clone()).powf_scalar(2.0).mean();
            let l2 = loss_fn.rel(out_flat.clone(), y_flat.clone());

            train_mse_accum += mse.into_scalar();
            train_l2_accum += l2.clone().into_scalar();  // log before step

            let grads = l2.backward();
            let grad_params = GradientsParams::from_grads(grads, &model);
            let lr = scheduler.step();
            model = optim.step(lr, model, grad_params);
            last_lr = lr;

            batch_count += 1;
        }

        // Evaluation
        let mut test_l2_accum = 0.0_f32;

        for batch in test_loader.iter() {
            let out = model.forward(batch.inputs);
            let [b, s, _] = out.dims();
            let out_flat = out.reshape([b, s]);
            let y_flat = batch.targets;

            let test_l2 = loss_fn.rel(out_flat, y_flat);
            test_l2_accum += test_l2.into_scalar();
        }

        let m = EpochMetricsNd {
            epoch,
            train_mse: train_mse_accum / steps_per_epoch as f32,
            train_l2: train_l2_accum / config.n_train as f32,
            test_l2: test_l2_accum / config.n_test as f32,
            current_lr: last_lr as f64,
        };

        println!("{:?}", m);
        metrics.push(m);
    }

    (model, metrics)
}

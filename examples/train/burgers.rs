//! FNO on 1D Burgers
//!
//! We train a Fourier Neural Operator (FNO) on a Burgers dataset.
//! Run with: cargo run --release --example train_burgers
//! (CPU by default; add `--features metal` or `--features cuda` for a GPU -
//! see "Backends" in the README).
//!
//! This example demonstrates the complete workflow of training a neural operator:
//! 1. Loading and preprocessing the Burgers dataset
//! 2. Creating an FNO model architecture
//! 3. Setting up training components (optimizer, scheduler, losses)
//! 4. Training the model
//! 5. Writing metrics, configs and trained weights to a local run directory
//!
//! The FNO's key advantage is its resolution invariance - it can make predictions
//! at different resolutions without retraining.
//!
//! Environment variables (optimisers CO1.1; unset, the run is as before):
//! - `RECIPE=petscml`: PETScML's Burgers setting (Zampini et al. 2024, §4.1.1, D10): the
//!   first 1,000 samples for training, subsample rate 64 (s = 128), GELU, and the batch mean
//!   of the squared relative L² as the objective.
//! - `OPTIMIZER=adamw`: AdamW with the learning rate halved every 100 epochs, PETScML's
//!   reference (D11); `adam` (the default) is Adam with coupled weight decay and the cosine.
//! - `WEIGHT_DECAY` and `SUBSAMPLE` override those literals.
//!
//! PETScML's reference A1, seed 42:
//! RECIPE=petscml OPTIMIZER=adamw cargo run --release --features metal --example train_burgers
//!
//! The run also writes `costs.json` (load and training time, steps and passes).

// Import the necessary modules for training a Fourier Neural Operator
use burn::{module::Module, tensor::Device};
use sciml_rs::neural_operators::data::catalogue::BurgersFiles;
use sciml_rs::neural_operators::{
    data::loaders::{
        base_dataset::DatasetConfig,
        burgers::{BurgersConfig, load_burgers_uniform},
    },
    models::fno::{Activation, FNOConfig},
    runs::{RunCosts, RunDir},
    training::{
        config::{LrSchedule, OptimizerConfig, TrainLoss, TrainingConfig},
        trainer::train_fno,
    },
    utils::backend::{backend_features, git_commit},
};
use std::fmt::Display;
use std::path::Path;
use std::str::FromStr;
use std::time::Instant;

/// The environment variable `name` parsed as `T`, or `None` when it is unset.
///
/// # Panics
/// If it is set and does not parse.
fn env_var<T: FromStr>(name: &str) -> Option<T>
where
    T::Err: Display,
{
    let v = std::env::var(name).ok()?;
    Some(
        v.parse()
            .unwrap_or_else(|e| panic!("{name} must be a number, got {v:?}: {e}")),
    )
}

fn main() {
    // Set the device for the underlying compute backend
    // .autodiff() wraps the backend to enable automatic differentiation for the tensors
    let device = Device::default().autodiff();
    println!("device: {device:?}");

    // PETScML's setting (RECIPE=petscml) or today's
    let petscml = match std::env::var("RECIPE").as_deref() {
        Err(_) => false,
        Ok("petscml") => true,
        Ok(other) => panic!("RECIPE must be petscml, got {other:?}"),
    };
    let adamw = match std::env::var("OPTIMIZER").as_deref() {
        Err(_) | Ok("adam") => false,
        Ok("adamw") => true,
        Ok(other) => panic!("OPTIMIZER must be adam or adamw, got {other:?}"),
    };

    // Set the training test split for the dataset configuration
    let data_cfg = DatasetConfig {
        // PETScML and Lu et al. 2022 train on 1,000
        n_train: if petscml { 1000 } else { 1024 },
        n_test: 100,
    };

    // Set subsample rate `r` which computes the number of grid points `s`
    // so each sample is a function `a(x)` is evaluated at `s` evenly spaced locations on [0,1]
    // paired with the solution `u(x)` at the same `s` locations
    // For this example it is set to 32 -> s = 2^13 / 32 = 256 grid points
    // PETScML (after Lu et al. 2022) subsample at 64: s = 128
    let subsample = env_var("SUBSAMPLE").unwrap_or(if petscml { 64 } else { 32 });
    let dataset_cfg = BurgersConfig::new(data_cfg, subsample);

    // Automatically finds the path for the Burgers dataset as long as the file is in
    // the datasets folder at the repository root
    let data_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("datasets")
        .join(BurgersFiles::DATA);
    assert!(
        data_path.exists(),
        "Burgers dataset not found at {}\nSee datasets/README.md for the download link.",
        data_path.display()
    );

    // Load and preprocess the Burgers dataset for training and testing using the dataset config values
    // The dataset contains fields initial condition `a` and solution at t=1 `u`
    let load_start = Instant::now();
    let (train_data, test_data) = load_burgers_uniform::<f32>(&data_path, &dataset_cfg)
        .unwrap_or_else(|e| panic!("could not load Burgers data: {e}"));
    let load_time_s = load_start.elapsed().as_secs_f64();
    // what device_loader uploads: inputs [n, s, 1] and targets [n, s], f32
    let s = train_data.inputs().shape()[1];
    let upload_bytes = |n: usize| n * s * 2 * size_of::<f32>();
    let (n_train, n_test) = (train_data.len(), test_data.len());
    println!(
        "burgers: {} train / {} test at s={}",
        train_data.inputs().shape()[0],
        test_data.inputs().shape()[0],
        train_data.inputs().shape()[1]
    );

    // Set the FNO model, the number of modes sets the dimension of the problem
    // `modes` has one entry per spatial dimension, its length fixes the tensor rank
    // at compile time, for this example: 1 entry -> 1D FNO
    let model_cfg = FNOConfig {
        // How many Fourier modes to keep on the spatial axis
        modes: vec![16],
        // Channel width flowing through each spectral layer
        hidden_channels: 64,
        // Input channels in the dataset itself
        data_channels: 1,
        // Only 1 is supported - see flatten_pair
        out_channels: 1,
        // Number of spectral convolution layers
        n_layers: 4,
        // None keeps the Li et al. U(0, 1/(I·O)) spectral init; the zero-mean
        // alternatives are SpectralInit::Normal and SpectralInit::SymmetricUniform
        spectral_init: None,
        // No domain padding: Burgers is periodic, and Li et al.'s fourier_1d.py
        // does not pad either
        padding: None,
        // PETScML's network uses GELU; ReLU (None) otherwise
        activation: petscml.then_some(Activation::Gelu),
        // 2.0.0's block options (library CM1.3b); None keeps today's block
        fno_skip: None,
        channel_mlp: None,
        norm: None,
        preactivation: None,
        lifting_channel_ratio: None,
        projection_channel_ratio: None,
        positional_embedding: None,
    };

    // Set the training configuration, defaults not overridden: seed = 42, test_batch_size = 20
    // Use the Adam optimizer with weight decay for regularization
    // Use the Cosine Annealing learning rate scheduler from learning rate down to minimum learning rate
    let weight_decay: f32 = env_var("WEIGHT_DECAY").unwrap_or(1e-4);
    assert!(
        weight_decay.is_finite() && weight_decay >= 0.0,
        "WEIGHT_DECAY must be a finite number ≥ 0, got {weight_decay}"
    );
    let mut train_cfg = TrainingConfig::new()
        .with_epochs(500)
        .with_batch_size(20)
        .with_learning_rate(1e-3)
        .with_weight_decay(weight_decay)
        .with_min_lr(1e-5);
    if adamw {
        train_cfg = train_cfg
            .with_optimizer(Some(OptimizerConfig::AdamW))
            .with_lr_schedule(Some(LrSchedule::Step { every_epochs: 100 }));
    }
    if petscml {
        // PETScML's eq. 6
        train_cfg = train_cfg.with_loss(Some(TrainLoss::SquaredRelL2));
    }
    // .with_seed(1);

    // Writes configs, metrics and weights to runs/burgers_fno_<timestamp>/. Created
    // before training: train_fno appends each epoch's row of metrics.csv to it
    let run = RunDir::create(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("runs"),
        "burgers_fno",
    )
    .unwrap_or_else(|e| panic!("could not create the run directory: {e}"));

    // No normaliser: train and evaluate in physical units
    // Use the L2 loss for training and evaluation
    let train_start = Instant::now();
    let (model, metrics) = train_fno::<3, 2, _>(
        train_data, test_data, None, &model_cfg, &train_cfg, &device, &run,
    );
    let training_time_s = train_start.elapsed().as_secs_f64();

    // every step one forward and one backward (Adam and AdamW take no budget)
    let steps = train_cfg.total_steps(n_train);
    let costs = RunCosts {
        dataset_train_wall_time_s: None,
        dataset_test_wall_time_s: None,
        load_time_s,
        training_time_s,
        inference_time_s: None,
        host_to_device_bytes_train: upload_bytes(n_train),
        host_to_device_bytes_test: upload_bytes(n_test),
        // no backend reports it; never estimated (design §13.9)
        peak_gpu_memory: "unavailable".to_string(),
        backend_features: backend_features().into_iter().map(String::from).collect(),
        device: format!("{device:?}"),
        git_commit: git_commit(),
        num_params: model.num_params(),
        optimisation_steps: Some(steps),
        resumes: Some(0),
        forward_passes: Some(steps),
        backward_passes: Some(steps),
    };

    run.save_configs(&model_cfg, &train_cfg, &dataset_cfg)
        .expect("could not save configs");
    run.save_plot(&metrics);
    run.save_weights(&model)
        .unwrap_or_else(|e| panic!("could not save model weights: {e}"));
    run.save_costs(&costs).expect("could not save costs.json");

    println!("run written to {}", run.path().display());
}

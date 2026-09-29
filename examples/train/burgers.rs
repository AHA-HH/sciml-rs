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

// Import the necessary modules for training a Fourier Neural Operator
use burn::{
    store::{BurnpackStore, ModuleSnapshot},
    tensor::Device,
};
use sciml_rs::neural_operators::{
    data::loaders::{
        base_dataset::DatasetConfig,
        burgers::{BurgersConfig, load_burgers_uniform},
    },
    metrics::io::write_run_artifacts,
    models::fno::FNOConfig,
    training::{trainer::TrainingConfig, trainers::burgers::train_burgers},
};
use std::path::Path;

fn main() {
    // Set the device for the underlying compute backend
    // .autodiff() wraps the backend to enable automatic differentiation for the tensors
    let device = Device::default().autodiff();
    println!("device: {device:?}");

    // Set the training test split for the dataset configuration
    let data_cfg = DatasetConfig {
        n_train: 1024,
        n_test: 100,
    };

    // Set subsample rate `r` which computes the number of grid points `s`
    // so each sample is a function `a(x)` is evaluated at `s` evenly spaced locations on [0,1]
    // paired with the solution `u(x)` at the same `s` locations
    // For this example it is set to 32 -> s = 2^13 / 32 = 256 grid points
    let dataset_cfg = BurgersConfig::new(data_cfg, 32);

    // Automatically finds the path for the Burgers dataset as long as the file is in
    // the datasets folder at the repository root
    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/burgers_data_R10.mat");
    assert!(
        data_path.exists(),
        "Burgers dataset not found at {}\nSee datasets/README.md for the download link.",
        data_path.display()
    );

    // Load and preprocess the Burgers dataset for training and testing using the dataset config values
    // The dataset contains fields initial condition `a` and solution at t=1 `u`
    let (train_data, test_data) = load_burgers_uniform(&data_path, &dataset_cfg)
        .unwrap_or_else(|e| panic!("could not load Burgers data: {e}"));

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
    };

    // Set the training configuration, defaults not overridden: seed = 42, test_batch_size = 20
    // Use the Adam optimizer with weight decay for regularization
    // Use the Cosine Annealing learning rate scheduler from learning rate down to minimum learning rate
    let train_cfg = TrainingConfig::new()
        .with_epochs(500)
        .with_batch_size(20)
        .with_learning_rate(1e-3)
        .with_weight_decay(1e-4)
        .with_min_lr(1e-5);
    // .with_seed(1);

    // Training wrapper function that handles the building of training components and the training loop
    // Use the L2 loss for training and evaluation
    let (model, metrics) = train_burgers(train_data, test_data, &model_cfg, &train_cfg, &device);

    // Writes metrics and configs files to runs/burgers_fno_<timestamp>/
    let dir = write_run_artifacts(
        "burgers_fno",
        &metrics,
        &model_cfg,
        &train_cfg,
        &dataset_cfg,
    )
    .unwrap_or_else(|e| panic!("could not write run artifacts: {e}"));

    // Writes model weights file to the same directory
    let mut store = BurnpackStore::from_file(dir.join("model_weights"));
    model
        .save_into(&mut store)
        .expect("could not save model weights");

    println!("run written to {}", dir.display());
}

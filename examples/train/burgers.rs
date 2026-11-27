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
//! Environment variables (optimisers CO1.1 and CO1.2; unset, the run is as before):
//! - `RECIPE=petscml`: PETScML's Burgers setting (Zampini et al. 2024, §4.1.1, D10): the
//!   first 1,000 samples for training, subsample rate 64 (s = 128), GELU, and the batch mean
//!   of the squared relative L² as the Adam path's objective.
//! - `OPTIMIZER=adamw`: AdamW with the learning rate halved every `EPOCHS / 5` epochs (every
//!   100 at 500), PETScML's reference (D11); `adam` (the default) is Adam with coupled weight
//!   decay and the cosine; `gauss-newton` is O1's Gauss–Newton arm (D12: full batch, CG cap
//!   100, the Hutchinson diagonal, μ = 1e-4), the run name gaining `_gn`.
//! - `WEIGHT_DECAY` and `SUBSAMPLE` override those literals; `GN_WEIGHT_DECAY` (μ; 0 for
//!   G0), `GN_BATCH_SIZE` (200 for G0-200), `GN_CG_MAX_ITERS` and `GN_PRECONDITIONER`
//!   Gauss–Newton's (`examples/train/common.rs`).
//! - `SEED` (default 42, `_seed<n>` in the run name), `EPOCHS` (default 500) and, with
//!   Gauss–Newton, `TIME_BUDGET_S` (stop after the first epoch that ends at or over it).
//!
//! PETScML's reference A1, seed 42, and O1's G0 at a budget:
//! RECIPE=petscml OPTIMIZER=adamw cargo run --release --features metal --example train_burgers
//! RECIPE=petscml OPTIMIZER=gauss-newton GN_WEIGHT_DECAY=0 TIME_BUDGET_S=600 \
//!     cargo run --release --features metal --example train_burgers
//!
//! A run checkpoints every 25 epochs and after the last; `--resume` continues it from there
//! with its saved configs (the variables above are then ignored):
//! cargo run --release --example train_burgers -- --resume runs/<run_dir>
//!
//! The run also writes `costs.json` (load and training time, steps and passes) and, with
//! Gauss–Newton, `gn_steps.csv`.

mod common;

// Import the necessary modules for training a Fourier Neural Operator
use burn::{module::Module, tensor::Device};
use common::{CHECKPOINT_EVERY, GnLiterals, Literals, Measured};
use sciml_rs::neural_operators::data::catalogue::BurgersFiles;
use sciml_rs::neural_operators::{
    data::loaders::{
        base_dataset::DatasetConfig,
        burgers::{BurgersConfig, load_burgers_uniform},
    },
    models::fno::{Activation, FNO, FNOConfig},
    runs::RunDir,
    training::{
        config::{TrainLoss, TrainingConfig},
        trainer::{build_components, train_components},
    },
};
use std::path::Path;
use std::time::Instant;

/// A new run from the literals below and the environment: creates its directory and saves
/// its configs there before training, so `--resume` can read them.
fn new_run() -> (RunDir, FNOConfig, TrainingConfig, BurgersConfig) {
    // PETScML's setting (RECIPE=petscml) or today's
    let petscml = match std::env::var("RECIPE").as_deref() {
        Err(_) => false,
        Ok("petscml") => true,
        Ok(other) => panic!("RECIPE must be petscml, got {other:?}"),
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
    let subsample = common::env_var("SUBSAMPLE").unwrap_or(if petscml { 64 } else { 32 });
    let dataset_cfg = BurgersConfig::new(data_cfg.clone(), subsample);

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
        conv_bias: None,
        separable: None,
        max_modes: None,
        tanh_stabilizer: None,
    };

    // Set the training configuration, test_batch_size = 20 by default
    // Adam with weight decay for regularization and the Cosine Annealing learning rate
    // scheduler from learning rate down to minimum learning rate, unless OPTIMIZER says
    // otherwise
    let default_seed = TrainingConfig::new().seed;
    let seed = common::env_var("SEED").unwrap_or(default_seed);
    let lit = Literals {
        epochs: common::env_var("EPOCHS").unwrap_or(500),
        batch_size: 20,
        learning_rate: 1e-3,
        weight_decay: 1e-4,
        min_lr: 1e-5,
        // O1's Gauss-Newton arm (D12); the other GaussNewtonConfig settings keep their
        // defaults
        gn: GnLiterals {
            // the full batch
            batch_size: data_cfg.n_train,
            cg_max_iters: 100,
            lambda_init: 1.0,
            // G1; G0 is GN_WEIGHT_DECAY=0
            weight_decay: 1e-4,
            hutchinson: true,
            hutchinson_probes: 10,
            adjoint_check_every: 10,
        },
        hybrid_adam_epochs: None,
        lbfgs: None,
    };
    let mut train_cfg = common::training_config(&["adam", "adamw", "gauss-newton"], seed, &lit);
    if petscml {
        // PETScML's eq. 6
        train_cfg = train_cfg.with_loss(Some(TrainLoss::SquaredRelL2));
    }

    // Writes configs, metrics, checkpoints and weights to runs/<name>_<timestamp>/.
    // Created before training: train_components appends each epoch's row of metrics.csv
    // to it
    let seed_tag = if seed == default_seed {
        String::new()
    } else {
        format!("_seed{seed}")
    };
    let gn_tag = if train_cfg.gauss_newton().is_some() {
        "_gn"
    } else {
        ""
    };
    let run = RunDir::create(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("runs"),
        &format!("burgers_fno{seed_tag}{gn_tag}"),
    )
    .unwrap_or_else(|e| panic!("could not create the run directory: {e}"));
    run.save_configs(&model_cfg, &train_cfg, &dataset_cfg)
        .expect("could not save configs");
    (run, model_cfg, train_cfg, dataset_cfg)
}

fn main() {
    // the commit at the start: the checkout may move during the run (CO0.4)
    let git_commit = sciml_rs::neural_operators::utils::backend::git_commit();
    // Set the device for the underlying compute backend
    // .autodiff() wraps the backend to enable automatic differentiation for the tensors
    let device = Device::default().autodiff();
    println!("device: {device:?}");

    let (run, model_cfg, train_cfg, dataset_cfg, resume) =
        common::session::<3, BurgersConfig>(&device, new_run);

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

    // No normaliser: train and evaluate in physical units
    // Use the L2 loss for training and evaluation
    // After a resume the recorded time adds the earlier sessions' up to the checkpoint
    let (prior_time_s, resumes) = resume
        .as_ref()
        .map_or((0.0, 0), |ck| (ck.state.training_time_s, ck.state.resumes));
    let train_start = Instant::now();
    let components =
        build_components::<FNO<3>, 3, 2, _>(&model_cfg, &train_cfg, train_data, test_data, &device);
    let (model, _) = train_components(
        components,
        None,
        &train_cfg,
        &device,
        &run,
        Some(CHECKPOINT_EVERY),
        resume,
        None,
    );
    let training_time_s = prior_time_s + train_start.elapsed().as_secs_f64();

    let costs = common::run_costs(
        &run,
        &train_cfg,
        n_train,
        Measured {
            load_time_s,
            training_time_s,
            host_to_device_bytes_train: upload_bytes(n_train),
            host_to_device_bytes_test: upload_bytes(n_test),
            device: format!("{device:?}"),
            num_params: model.num_params(),
            resumes,
            git_commit,
        },
    );

    // every epoch's metrics, the earlier sessions' included
    let metrics = run.metrics().expect("could not read metrics.csv");
    run.save_plot(&metrics);
    run.save_weights(&model)
        .unwrap_or_else(|e| panic!("could not save model weights: {e}"));
    run.save_costs(&costs).expect("could not save costs.json");

    println!("run written to {}", run.path().display());
}

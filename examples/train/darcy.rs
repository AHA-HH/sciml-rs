//! FNO on 2D Darcy flow
//!
//! Run with: cargo run --release --example train_darcy
//! (CPU by default; add `--features metal` or `--features cuda` for a GPU -
//! see "Backends" in the README).
//!
//! Structurally identical to `examples/train/burgers.rs` - see that file for the
//! fully annotated version. The differences: `modes` has two entries instead of
//! one and the targets are normalized which the trainer has to undo before
//! computing the loss.
//!
//! Environment variables (optimisers CO1.2, for O1's Darcy cells; unset, the run is as
//! before): `OPTIMIZER` (`adam`, `adamw` or `gauss-newton`), `WEIGHT_DECAY`, the `GN_*`
//! variables, `SEED`, `EPOCHS` and `TIME_BUDGET_S`, as in `train_burgers`; Gauss–Newton
//! defaults to O1's arm (D12: full batch, CG cap 100, the Hutchinson diagonal, μ = 1e-4).
//! A run checkpoints every 25 epochs and after the last, and resumes with
//! `-- --resume runs/<run_dir>`; it writes `costs.json` and, with Gauss–Newton,
//! `gn_steps.csv`.

mod common;

use burn::{module::Module, tensor::Device};
use common::{GnLiterals, Literals, Measured};
use sciml_rs::neural_operators::data::catalogue::DarcyFiles;
use sciml_rs::neural_operators::{
    data::dataset::OperatorDataset,
    data::loaders::{
        base_dataset::DatasetConfig,
        darcy::{DarcyConfig, load_darcy_uniform},
    },
    models::fno::FNOConfig,
    optim::gauss_newton::Preconditioner,
    runs::RunDir,
    training::{
        config::TrainingConfig,
        trainer::{build_training_components, train_components},
    },
};
use std::path::Path;
use std::time::Instant;

/// Epochs between checkpoints; the last epoch is always checkpointed (Phase 9 P2).
const CHECKPOINT_EVERY: usize = 25;

/// A new run from the literals below and the environment: creates its directory and saves
/// its configs there before training, so `--resume` can read them.
fn new_run() -> (RunDir, FNOConfig, TrainingConfig, DarcyConfig) {
    let data_cfg = DatasetConfig {
        n_train: 1000,
        n_test: 100,
    };

    // subsample rate is 5 -> s = (421-1)/5 + 1 = 85 grid points per axis,
    // the paper's setting. Non-power-of-two sizes are fine: the FFT falls
    // back to Bluestein's algorithm.
    let dataset_cfg = DarcyConfig::new(data_cfg.clone(), 5);

    let model_cfg = FNOConfig {
        // Two entries so a 2D FNO model. Same code path as Burgers' vec![16]
        modes: vec![12, 12],
        hidden_channels: 32,
        data_channels: 1,
        out_channels: 1,
        n_layers: 4,
        // None keeps the Li et al. U(0, 1/(I·O)) spectral init; the zero-mean
        // alternatives are SpectralInit::Normal and SpectralInit::SymmetricUniform
        spectral_init: None,
        // Darcy is non-periodic, so pad the domain with 9 zero cells at the end
        // of each axis before the spectral layers (cropped after), as Li et al.'s
        // fourier_2d.py does. The padded 94 x 94 grid is not a power of two
        padding: Some(9),
        activation: None,
    };

    let default_seed = TrainingConfig::new().seed;
    let seed = common::env_var("SEED").unwrap_or(default_seed);
    let lit = Literals {
        epochs: common::env_var("EPOCHS").unwrap_or(500),
        batch_size: 20,
        learning_rate: 1e-3,
        weight_decay: 1e-4,
        min_lr: 1e-5,
        // O1's Gauss-Newton arm (D12), as train_burgers
        gn: GnLiterals {
            // the full batch
            batch_size: data_cfg.n_train,
            cg_max_iters: 100,
            lambda_init: 1.0,
            weight_decay: 1e-4,
            preconditioner: Preconditioner::HutchinsonDiagonal { probes: 10 },
            hutchinson_probes: 10,
            adjoint_check_every: 10,
        },
    };
    let train_cfg = common::training_config(&["adam", "adamw", "gauss-newton"], seed, &lit);

    // Writes configs, metrics, checkpoints, normalizers and weights to
    // runs/<name>_<timestamp>/. Created before training: train_components appends each
    // epoch's row of metrics.csv to it
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
        &format!("darcy_fno{seed_tag}{gn_tag}"),
    )
    .unwrap_or_else(|e| panic!("could not create the run directory: {e}"));
    run.save_configs(&model_cfg, &train_cfg, &dataset_cfg)
        .expect("could not save configs");
    (run, model_cfg, train_cfg, dataset_cfg)
}

fn main() {
    let device = Device::default().autodiff();
    println!("device: {device:?}");

    let (run, model_cfg, train_cfg, dataset_cfg, resume) = match common::resume_dir() {
        Some(dir) => {
            let (run, model_cfg, train_cfg, dataset_cfg, ck) =
                common::reopen::<4, DarcyConfig>(&dir, &device);
            (run, model_cfg, train_cfg, dataset_cfg, Some(ck))
        }
        None => {
            let (run, model_cfg, train_cfg, dataset_cfg) = new_run();
            (run, model_cfg, train_cfg, dataset_cfg, None)
        }
    };

    let datasets = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets");
    let train_path = datasets.join(DarcyFiles::TRAIN);
    let test_path = datasets.join(DarcyFiles::TEST);
    for p in [&train_path, &test_path] {
        assert!(
            p.exists(),
            "Darcy dataset not found at {}\nSee datasets/README.md",
            p.display()
        );
    }

    let load_start = Instant::now();
    let (train_data, test_data, normalizers) =
        load_darcy_uniform::<f32>(&train_path, &test_path, &dataset_cfg)
            .unwrap_or_else(|e| panic!("could not load Darcy data: {e}"));
    let load_time_s = load_start.elapsed().as_secs_f64();
    // what device_loader uploads: inputs and targets as stored
    let bytes =
        |d: &OperatorDataset<f32>| (d.inputs().len() + d.targets().len()) * size_of::<f32>();
    let (train_bytes, test_bytes) = (bytes(&train_data), bytes(&test_data));
    let n_train = train_data.len();
    println!(
        "darcy: {} train / {} test at s={}",
        train_data.inputs().shape()[0],
        test_data.inputs().shape()[0],
        train_data.inputs().shape()[1]
    );

    // y_normalizer is fitted on y_train during loading. The trainer uses it
    // to decode both prediction and target during training, but only the
    // prediction during evaluation - y_test was never encoded
    // After a resume the recorded time adds the earlier sessions' up to the checkpoint
    let (prior_time_s, resumes) = resume
        .as_ref()
        .map_or((0.0, 0), |ck| (ck.state.training_time_s, ck.state.resumes));
    let train_start = Instant::now();
    let components = build_training_components::<4, 3, _>(
        &model_cfg, &train_cfg, train_data, test_data, &device,
    );
    let (model, _) = train_components(
        components,
        Some(&normalizers.y),
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
            host_to_device_bytes: (train_bytes, test_bytes),
            device: format!("{device:?}"),
            num_params: model.num_params(),
            resumes,
        },
    );

    // every epoch's metrics, the earlier sessions' included
    let metrics = run.metrics().expect("could not read metrics.csv");
    run.save_plot(&metrics);
    run.save_normalizers(&normalizers)
        .expect("could not save normalizers");
    run.save_weights(&model)
        .unwrap_or_else(|e| panic!("could not save model weights: {e}"));
    run.save_costs(&costs).expect("could not save costs.json");

    println!("run written to {}", run.path().display());
}

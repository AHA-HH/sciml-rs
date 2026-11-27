//! FNO on 2D Poisson on the Chebyshev grid (n nodes per axis, s on the model grid, by
//! default s = n − 1); stage 2 (n = 65) by default
//!
//! Run with: cargo run --release --example train_poisson [-- [<padding>] [<n>] [<s>] [<seed>] [<N> <subset_seed>]]
//! (CPU by default; add `--features metal` or `--features cuda` for a GPU -
//! see "Backends" in the README).
//!
//! An optional first argument overrides `PADDING`, so the padding sweep of
//! docs/poisson2d/phase3/T2-train-poisson.md needs no edits:
//! cargo run --release --example train_poisson -- 8
//! An optional second argument picks the stage by n, one of `catalogue::STAGES`
//! (design §5.3):
//! cargo run --release --example train_poisson -- 4 129
//! An optional third argument sets the learning grid s (CONVENTIONS §12), from the same
//! dataset files; the modes follow it (design §12, decision 30, D2):
//! cargo run --release --example train_poisson -- 0 129 32
//! An optional fourth argument sets `TrainingConfig::seed` (default 42; weight
//! initialisation and the shuffle), named in the run when not 42:
//! cargo run --release --example train_poisson -- 0 129 32 7
//! Optional fifth and sixth arguments train on N samples of the train split, the seeded
//! nested subset `subset_seed` (`SubsetConfig`, design §13.5; Phase 7), with the
//! normalizers fitted on those N; both are given together, and the run name carries them:
//! cargo run --release --example train_poisson -- 0 129 64 42 25 0
//!
//! A run writes a checkpoint every `CHECKPOINT_EVERY` epochs and after the last (Phase 9
//! T1). `--resume` continues an interrupted run from it, with the configs the run saved
//! (not the literals below), truncating `metrics.csv` to the checkpoint's epoch:
//! cargo run --release --example train_poisson -- --resume runs/<run_dir>
//!
//! `TRAIN_DEVICES`, a comma-separated list of cuda device indices, trains one run
//! data-parallel on those devices (Phase 9 T2): each batch of `BATCH_SIZE` is split across
//! them, with the schedule and the number of steps of a one-device run. Unset, the run
//! trains on the one default device, as before. New runs and `--resume` both take it:
//! TRAIN_DEVICES=0,1 cargo run --release --no-default-features --features cuda \
//!     --example train_poisson -- 0 129 64
//!
//! `OPTIMIZER=gauss-newton` trains a new run with damped Gauss–Newton (Phase 10 T3) with
//! the `GN_*` literals below instead of Adam, on one device, and the run name gains `_gn`;
//! unset or `adam`, the run trains with Adam as before. `--resume` takes the optimiser from
//! the run's saved config, not from `OPTIMIZER`. A Gauss–Newton run also writes
//! `gn_steps.csv`, one row per step:
//! OPTIMIZER=gauss-newton cargo run --release --no-default-features --features cuda \
//!     --example train_poisson -- 0 129 64 42 100 0
//!
//! `OPTIMIZER=lbfgs` (optimisers CO2.1) trains with Burn's L-BFGS on Gauss–Newton's
//! objective Φ, one step per epoch on the full batch of N, with the `LBFGS_*` literals below
//! (`LBFGS_HISTORY`, `LBFGS_MAX_ITER`, `LBFGS_MAX_EVAL`, `LBFGS_WEIGHT_DECAY` override them), on one
//! device; the run name gains `_lbfgs`, `TIME_BUDGET_S` stops it, `EPOCHS` caps it, and it
//! writes `lbfgs_steps.csv`, one row per step.
//!
//! `OPTIMIZER=hybrid` (Phase 10 T5's Hybrid arm) is Gauss–Newton whose first
//! `HYBRID_ADAM_EPOCHS` epochs train with Adam (`TrainingConfig::adam_epochs`), the
//! cosine schedule spanning those epochs; the run name gains `_hybrid`. Everything below
//! for Gauss–Newton applies to it too.
//!
//! `GN_PRECONDITIONER=hutchinson` picks the Hutchinson diagonal (`GN_HUTCHINSON_PROBES`
//! probes) over the literal `GN_HUTCHINSON`, `none` keeps it (T4's choice, T5's GN20-D).
//! `GN_WEIGHT_DECAY` and `GN_CG_MAX_ITERS` override the literals of those names (Phase 10
//! follow-ups: μ = 0, a higher CG cap). `WEIGHT_DECAY` overrides Adam's literal of that
//! name (Phase 10 follow-up: Adam at μ = 0, the control for GN at μ = 0).
//!
//! `EPOCHS` overrides the literal of that name for an Adam run, the cosine schedule
//! spanning the new count: Adam at a longer budget is Adam given more epochs, since a time
//! budget would stop it partway down its schedule (optimisers CO0.1). Saved in
//! `train_cfg.json`, so `--resume` keeps it:
//! EPOCHS=3142 cargo run --release --no-default-features --features cuda \
//!     --example train_poisson -- 0 129 64 43
//!
//! With Gauss–Newton, `GN_BATCH_SIZE` overrides the literal of that name (T4's full batch
//! is the subset's N), and `TIME_BUDGET_S` stops the run after the first epoch that ends
//! at or over that many seconds of training (`TrainingConfig::time_budget_s`; G4's equal
//! wall-clock, the `EPOCHS` literal staying the cap). Both are saved in `train_cfg.json`, so
//! `--resume` keeps them. Phase 10 T4, against Phase 7's Adam run of N = 100, subset 0,
//! seed 42 (`training_time_s` 132.5):
//! OPTIMIZER=gauss-newton GN_BATCH_SIZE=100 TIME_BUDGET_S=132.5 cargo run --release \
//!     --no-default-features --features cuda --example train_poisson -- 0 129 64 42 100 0
//!
//! Built with `--features mlflow` and run with `MLFLOW_EXPERIMENT_NAME` set, the run is
//! also logged live to MLflow (`MLFLOW_TRACKING_URI`, default `http://127.0.0.1:5001`;
//! Phase 8 T1): the configs as it starts, each epoch's metrics, and at the end the costs,
//! artifacts and `FINISHED`. Best-effort: a failed request turns it off with a warning,
//! and training goes on. A `--resume` continues the same MLflow run, so epochs trained
//! again after the checkpoint appear twice in its curves; `mlflow_export --replace` gives
//! the clean record:
//! MLFLOW_EXPERIMENT_NAME=scratch cargo run --release --features mlflow \
//!     --example train_poisson -- 0 129 64
//!
//! Besides the configs, metrics, normalizers and weights, the run records `costs.json`
//! (`RunCosts`, design §13.9): the dataset's generation time from its sidecars, the load,
//! transfer and training times, host-to-device bytes, the parameter count, the
//! optimisation steps, the forward and backward passes, the resumes, backend, device and
//! commit. `data_cfg.json` records n,
//! K, s, d, both split files with their generating commits and, with N, the subset's seed,
//! population and index hash.
//!
//! Structurally identical to `examples/train/darcy.rs`, with the Darcy model and
//! training hyperparameters (design §7). The loader transfers every sample from the
//! CGL grid to the uniform grid (CONVENTIONS §12); the targets are normalized, which
//! the trainer undoes before computing the loss. Needs the chosen stage's dataset: see
//! datasets/README.md.

mod common;

use burn::{
    config::Config,
    module::Module,
    optim::{LBFGSConfig, LineSearchFn},
    tensor::Device,
};
use common::{CHECKPOINT_EVERY, GnLiterals, Literals, Measured};
use sciml_rs::neural_operators::{
    chebyshev::TRANSFER_DEGREE,
    data::{
        catalogue::{PoissonSet, stage},
        dataset::OperatorDataset,
        loaders::{
            base_dataset::DatasetConfig,
            poisson::{
                DatasetSource, PoissonConfig, SidecarRecord, SplitSource, SubsetConfig,
                load_poisson_uniform,
            },
        },
    },
    models::fno::{FNO, FNOConfig},
    runs::{RunCosts, RunDir},
    training::{
        config::{LbfgsArm, TrainingConfig},
        data_parallel::replicas,
        trainer::{build_components, train_components},
    },
};
use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

/// The stage used when no n is given: stage 2.
const DEFAULT_N: usize = 65;
const N_TRAIN: usize = 1000;
/// The whole test split of each stage.
const N_TEST: usize = 200;

/// Fourier modes per axis at large s; m = min(MAX_MODES, s/2) (design §12, decision 30, D2).
const MAX_MODES: usize = 12;
const WIDTH: usize = 32;
const N_LAYERS: usize = 4;
const EPOCHS: usize = 500;
const BATCH_SIZE: usize = 20;
const LEARNING_RATE: f64 = 1e-3;
const WEIGHT_DECAY: f32 = 1e-4;
const MIN_LR: f64 = 1e-5;
/// Zero cells appended to each spatial axis (CONVENTIONS §3). Chosen by the full-length
/// sweep over {0, 4, 8, 16}: p = 8 scored lowest and p = 4 lies within 2% of it, so the
/// smaller p wins (design §12, decision 16; Phase 4 T2).
const PADDING: usize = 4;

// Gauss–Newton (`OPTIMIZER=gauss-newton`, Phase 10 T3); the other `GaussNewtonConfig`
// settings keep their defaults
/// Samples per Gauss–Newton step: gradient, curvature and ρ on one batch (G3: 20 or 200;
/// T4's full batch is the subset's N). Replaces `BATCH_SIZE` for a Gauss–Newton run.
const GN_BATCH_SIZE: usize = 20;
/// The most CG iterations per step.
const GN_CG_MAX_ITERS: usize = 50;
/// The damping λ of the first step.
const GN_LAMBDA_INIT: f64 = 1.0;
/// Weight decay μ as a Tikhonov term, Adam's for parity (G5).
const GN_WEIGHT_DECAY: f64 = 1e-4;
/// Whether CG uses the Hutchinson diagonal (T4 picked between it and none; none here);
/// `GN_PRECONDITIONER` overrides it.
const GN_HUTCHINSON: bool = false;
/// Probes per Hutchinson estimate (`GN_PRECONDITIONER=hutchinson`), re-estimated every
/// `GaussNewtonConfig::precond_every` steps.
const GN_HUTCHINSON_PROBES: usize = 10;
/// A hybrid run's Adam epochs: half of `EPOCHS`, so half of the Adam reruns' training time
/// (the budget, G4) at the same batch, since an Adam epoch costs the same in either run.
const HYBRID_ADAM_EPOCHS: usize = EPOCHS / 2;
/// The adjoint check, every this many steps (`gn_steps.csv`'s `adjoint_error`).
const GN_ADJOINT_CHECK_EVERY: usize = 10;

// L-BFGS (`OPTIMIZER=lbfgs`, optimisers CO2.1): full batch (D4), strong Wolfe, lr 1 (D5's
// recommendation; history and max_iter until CO2.2's pilot); `LBFGS_*` override these
/// Curvature pairs kept.
const LBFGS_HISTORY: usize = 100;
/// Inner iterations per step (per epoch).
const LBFGS_MAX_ITER: usize = 20;
/// Closure evaluations per step, the line search's included; `None` is Burn's
/// ⌊1.25 `LBFGS_MAX_ITER`⌋, which leaves a strong Wolfe search at `max_iter` 1 or 2 too few
/// to move (CO2.1's smoke run).
const LBFGS_MAX_EVAL: Option<usize> = None;
/// μ in the closure's Φ + ½μ‖θ‖² (D3, D13: 0 or 1e-4).
const LBFGS_WEIGHT_DECAY: f64 = 0.0;

/// The `i`-th command-line argument parsed as `T`, or `default` when it is absent.
///
/// # Panics
/// If the argument does not parse, naming it `name`.
fn arg<T: FromStr>(i: usize, name: &str, default: T) -> T
where
    T::Err: Display,
{
    match std::env::args().nth(i) {
        Some(arg) => arg
            .parse()
            .unwrap_or_else(|e| panic!("{name} argument must be an integer, got {arg:?}: {e}")),
        None => default,
    }
}

/// The training devices, autodiff: the cuda devices `TRAIN_DEVICES` lists (Phase 9 T2),
/// or the one default device when it is unset.
///
/// # Panics
/// If an index does not parse, or if `TRAIN_DEVICES` is set without the cuda feature.
fn train_devices() -> Vec<Device> {
    let Ok(list) = std::env::var("TRAIN_DEVICES") else {
        return vec![Device::default().autodiff()];
    };
    #[cfg(feature = "cuda")]
    {
        list.split(',')
            .map(|i| {
                let i: usize = i.trim().parse().unwrap_or_else(|e| {
                    panic!("TRAIN_DEVICES must list cuda indices, got {list:?}: {e}")
                });
                Device::cuda(i).autodiff()
            })
            .collect()
    }
    // cubecl implements the all_reduce data-parallel training needs on cuda only
    #[cfg(not(feature = "cuda"))]
    panic!("TRAIN_DEVICES={list} needs the cuda feature")
}

/// `datasets/poisson/` in the repository.
fn datasets() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/poisson")
}

/// The sidecar of one split of `set`.
///
/// # Panics
/// If it can't be read.
fn sidecar(set: &PoissonSet, split: &str) -> SidecarRecord {
    let path = datasets().join(set.json(split));
    SidecarRecord::load(&path)
        .unwrap_or_else(|e| panic!("could not read the sidecar {}: {e}", path.display()))
}

/// A new run from the command-line arguments and the literals above: creates its
/// directory and saves its configs there before training, so `--resume` can read them.
fn new_run(device: &Device) -> (RunDir, FNOConfig, TrainingConfig, PoissonConfig) {
    let padding = arg(1, "padding", PADDING);
    let n = arg(2, "n", DEFAULT_N);
    let set = stage(n).unwrap_or_else(|e| panic!("{e}"));
    let k = set.k;
    let s = arg(3, "s", n - 1);
    let default_seed = TrainingConfig::new().seed;
    let seed = arg(4, "seed", default_seed);
    // N and the subset seed come as a pair: a default seed would hide which subset a run
    // used
    let (n_train, subset_seed) = match (std::env::args().nth(5), std::env::args().nth(6)) {
        (None, _) => (N_TRAIN, None),
        (Some(_), None) => panic!("N needs a subset_seed argument after it"),
        (Some(_), Some(_)) => (arg(5, "N", N_TRAIN), Some(arg(6, "subset_seed", 0u64))),
    };
    let modes = MAX_MODES.min(s / 2);

    println!(
        "device: {device:?}, padding: {padding}, n: {n}, s: {s}, modes: {modes}, K: {k}, \
         seed: {seed}, N: {n_train}, subset seed: {subset_seed:?}"
    );

    let (train_sidecar, test_sidecar) = (sidecar(&set, "train"), sidecar(&set, "test"));
    let split = |split: &str, sidecar: &SidecarRecord| {
        SplitSource::new(set.npz(split), sidecar.git_commit.clone())
    };

    let subset = subset_seed.map(|seed| SubsetConfig::draw(seed, train_sidecar.count, n_train));
    let dataset_cfg = PoissonConfig::new(
        DatasetConfig {
            n_train,
            n_test: N_TEST,
        },
        n,
    )
    .with_s(Some(s))
    .with_source(Some(DatasetSource::new(
        k,
        TRANSFER_DEGREE,
        split("train", &train_sidecar),
        split("test", &test_sidecar),
    )))
    .with_subset(subset);

    let model_cfg = FNOConfig::new(vec![modes; 2], 1, 1)
        .with_hidden_channels(WIDTH)
        .with_n_layers(N_LAYERS)
        // None keeps the Li et al. U(0, 1/(I·O)) spectral init
        .with_spectral_init(None)
        // Poisson is non-periodic: pad each axis with zero cells before the
        // spectral layers (cropped after). Some(0) means no padding
        .with_padding(Some(padding));

    let lit = Literals {
        epochs: common::env_var("EPOCHS").unwrap_or(EPOCHS),
        batch_size: BATCH_SIZE,
        learning_rate: LEARNING_RATE,
        weight_decay: WEIGHT_DECAY,
        min_lr: MIN_LR,
        gn: GnLiterals {
            batch_size: GN_BATCH_SIZE,
            cg_max_iters: GN_CG_MAX_ITERS,
            lambda_init: GN_LAMBDA_INIT,
            weight_decay: GN_WEIGHT_DECAY,
            hutchinson: GN_HUTCHINSON,
            hutchinson_probes: GN_HUTCHINSON_PROBES,
            adjoint_check_every: GN_ADJOINT_CHECK_EVERY,
        },
        hybrid_adam_epochs: Some(HYBRID_ADAM_EPOCHS),
        lbfgs: Some((
            LbfgsArm::new(
                LBFGSConfig::new()
                    .with_history_size(LBFGS_HISTORY)
                    .with_max_iter(LBFGS_MAX_ITER)
                    .with_max_eval(LBFGS_MAX_EVAL)
                    .with_line_search_fn(LineSearchFn::StrongWolfe),
            )
            .with_weight_decay(LBFGS_WEIGHT_DECAY),
            n_train,
        )),
    };
    let allowed = ["adam", "gauss-newton", "hybrid", "lbfgs"];
    let train_cfg = common::training_config(&allowed, seed, &lit);
    let gn = train_cfg.gauss_newton().is_some();
    // Adam's long budget is more epochs, its cosine schedule spanning them; Gauss-Newton
    // stops by TIME_BUDGET_S, and a hybrid run's Adam half is fixed at HYBRID_ADAM_EPOCHS
    assert!(
        !gn || lit.epochs == EPOCHS,
        "EPOCHS is for Adam runs; Gauss-Newton runs take TIME_BUDGET_S"
    );

    // Stage 2 at s = n − 1 keeps its run name; other stages carry n, other s carry n and s,
    // then a subset its N and seed, and a seed other than the default is appended
    let grid = match (n, s == n - 1) {
        (DEFAULT_N, true) => String::new(),
        (_, true) => format!("_n{n}"),
        (_, false) => format!("_n{n}_s{s}"),
    };
    let subset_tag = match subset_seed {
        Some(sub) => format!("_N{n_train}_sub{sub}"),
        None => String::new(),
    };
    let seed_tag = if seed == default_seed {
        String::new()
    } else {
        format!("_seed{seed}")
    };
    let gn_tag = match (gn, train_cfg.adam_epochs) {
        (_, Some(_)) => "_hybrid",
        (true, None) => "_gn",
        (false, None) if train_cfg.lbfgs().is_some() => "_lbfgs",
        (false, None) => "",
    };
    let name = format!("poisson_fno{grid}_p{padding}{subset_tag}{seed_tag}{gn_tag}");
    // Writes configs, metrics, checkpoints, normalizers and weights to
    // runs/<name>_<timestamp>/. Created before training: train_components appends each
    // epoch's row of metrics.csv to it
    let run = RunDir::create(&Path::new(env!("CARGO_MANIFEST_DIR")).join("runs"), &name)
        .unwrap_or_else(|e| panic!("could not create the run directory: {e}"));
    run.save_configs(&model_cfg, &train_cfg, &dataset_cfg)
        .expect("could not save configs");
    (run, model_cfg, train_cfg, dataset_cfg)
}

fn main() {
    // the commit at the start: the checkout may move during the run (CO0.4)
    let git_commit = sciml_rs::neural_operators::utils::backend::git_commit();
    let devices = train_devices();
    let device = devices[0].clone();
    let (run, model_cfg, train_cfg, dataset_cfg, resume) =
        common::session::<4, PoissonConfig>(&device, || new_run(&device));
    assert!(
        (train_cfg.gauss_newton().is_none() && train_cfg.lbfgs().is_none()) || devices.len() == 1,
        "Gauss-Newton and L-BFGS train on one device: unset TRAIN_DEVICES"
    );
    let set = stage(dataset_cfg.n).unwrap_or_else(|e| panic!("{e}"));

    let train_path = datasets().join(set.npz("train"));
    let test_path = datasets().join(set.npz("test"));
    for p in [&train_path, &test_path] {
        assert!(
            p.exists(),
            "Poisson dataset not found at {}\nSee datasets/README.md",
            p.display()
        );
    }
    let (train_sidecar, test_sidecar) = (sidecar(&set, "train"), sidecar(&set, "test"));

    let load_start = Instant::now();
    let (train_data, test_data, normalizers) =
        load_poisson_uniform::<f32>(&train_path, &test_path, &dataset_cfg)
            .unwrap_or_else(|e| panic!("could not load Poisson data: {e}"));
    let load_time_s = load_start.elapsed().as_secs_f64();
    // what device_loader uploads: inputs and targets as stored
    let bytes = |d: &OperatorDataset<f32>| {
        (d.inputs().len() + d.targets().len()) * std::mem::size_of::<f32>()
    };
    // one copy of the training split per device
    let (train_bytes, test_bytes) = (bytes(&train_data) * devices.len(), bytes(&test_data));
    let n_train = train_data.inputs().shape()[0];
    println!(
        "poisson: {} train / {} test, n={} -> s={}",
        n_train,
        test_data.inputs().shape()[0],
        dataset_cfg.n,
        train_data.inputs().shape()[1]
    );

    // y_normalizer is fitted on y_train during loading. The trainer uses it
    // to decode both prediction and target during training, but only the
    // prediction during evaluation - y_test was never encoded
    // The recorded training_time_s: it includes building the model, optimiser and loaders,
    // so it is slightly above the "total training time" the trainer prints from its own
    // clock; after a resume it adds the earlier sessions' time up to the checkpoint
    let (prior_time_s, resumes) = resume
        .as_ref()
        .map_or((0.0, 0), |ck| (ck.state.training_time_s, ck.state.resumes));
    // opt-in live MLflow logging (Phase 8 T1, Q7): only built with `--features mlflow`,
    // only on with MLFLOW_EXPERIMENT_NAME set; started before the training clock
    #[cfg(feature = "mlflow")]
    let mut live = sciml_rs::neural_operators::tracking::MlflowObserver::from_env(&run);
    #[cfg(feature = "mlflow")]
    let mut live_epoch = |m: &sciml_rs::neural_operators::runs::EpochMetrics, _: &_| {
        if let Some(o) = live.as_mut() {
            o.on_epoch(m);
        }
    };
    #[cfg(feature = "mlflow")]
    let also: Option<
        &mut sciml_rs::neural_operators::training::trainer::EpochObserver<'_, FNO<4>>,
    > = Some(&mut live_epoch);
    #[cfg(not(feature = "mlflow"))]
    let also = None;
    let train_start = Instant::now();
    let replicas = replicas::<4, 3, _>(&train_data, &devices[1..], &train_cfg);
    let mut components =
        build_components::<FNO<4>, 4, 3, _>(&model_cfg, &train_cfg, train_data, test_data, &device);
    components.replicas = replicas;
    let (model, _) = train_components(
        components,
        Some(&normalizers.y),
        &train_cfg,
        &device,
        &run,
        Some(CHECKPOINT_EVERY),
        resume,
        also,
    );
    let training_time_s = prior_time_s + train_start.elapsed().as_secs_f64();
    let device_name = match devices.len() {
        1 => format!("{device:?}"),
        _ => format!("{devices:?}"),
    };
    let measured = Measured {
        load_time_s,
        training_time_s,
        host_to_device_bytes_train: train_bytes,
        host_to_device_bytes_test: test_bytes,
        device: device_name,
        num_params: model.num_params(),
        resumes,
        git_commit,
    };
    let costs = RunCosts {
        dataset_train_wall_time_s: Some(train_sidecar.wall_time_s),
        dataset_test_wall_time_s: Some(test_sidecar.wall_time_s),
        ..common::run_costs(&run, &train_cfg, n_train, measured)
    };

    // every epoch's metrics, the earlier sessions' included
    let metrics = run.metrics().expect("could not read metrics.csv");
    run.save_plot(&metrics);
    run.save_normalizers(&normalizers)
        .expect("could not save normalizers");
    run.save_weights(&model)
        .unwrap_or_else(|e| panic!("could not save model weights: {e}"));
    run.save_costs(&costs).expect("could not save costs.json");
    #[cfg(feature = "mlflow")]
    if let Some(o) = live.as_mut() {
        o.finish();
    }

    println!("run written to {}", run.path().display());
}

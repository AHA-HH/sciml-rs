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
//! `OPTIMIZER=hybrid` (Phase 10 T5's Hybrid arm) is Gauss–Newton whose first
//! `HYBRID_ADAM_EPOCHS` epochs train with Adam (`TrainingConfig::adam_epochs`), the
//! cosine schedule spanning those epochs; the run name gains `_hybrid`. Everything below
//! for Gauss–Newton applies to it too.
//!
//! `GN_PRECONDITIONER=hutchinson` picks the Hutchinson diagonal (`GN_HUTCHINSON_PROBES`
//! probes) over `GN_PRECONDITIONER`'s literal, `none` keeps it (T4's choice, T5's GN20-D).
//! `GN_WEIGHT_DECAY` and `GN_CG_MAX_ITERS` override the literals of those names (Phase 10
//! follow-ups: μ = 0, a higher CG cap). `WEIGHT_DECAY` overrides Adam's literal of that
//! name (Phase 10 follow-up: Adam at μ = 0, the control for GN at μ = 0).
//!
//! With Gauss–Newton, `GN_BATCH_SIZE` overrides the literal of that name (T4's full batch
//! is the subset's N), and `TIME_BUDGET_S` stops the run after the first epoch that ends
//! at or over that many seconds of training (`TrainingConfig::time_budget_s`; G4's equal
//! wall-clock, `EPOCHS` staying the cap). Both are saved in `train_cfg.json`, so
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

use burn::{config::Config, module::Module, tensor::Device};
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
    models::fno::FNOConfig,
    optim::gauss_newton::{GaussNewtonConfig, Preconditioner},
    runs::{Checkpoint, RunCosts, RunDir},
    training::{
        config::{OptimizerConfig, TrainingConfig},
        data_parallel::replicas,
        gn::training_passes,
        trainer::{build_training_components, train_components},
    },
    utils::backend::{backend_features, git_commit},
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
/// Epochs between checkpoints; the last epoch is always checkpointed (Phase 9 P2).
const CHECKPOINT_EVERY: usize = 25;

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
/// The CG preconditioner (T4 picks between this and the Hutchinson diagonal);
/// `GN_PRECONDITIONER` overrides it.
const GN_PRECONDITIONER: Preconditioner = Preconditioner::None;
/// Probes per Hutchinson estimate (`GN_PRECONDITIONER=hutchinson`), re-estimated every
/// `GaussNewtonConfig::precond_every` steps.
const GN_HUTCHINSON_PROBES: usize = 10;
/// A hybrid run's Adam epochs: half of `EPOCHS`, so half of the Adam reruns' training time
/// (the budget, G4) at the same batch, since an Adam epoch costs the same in either run.
const HYBRID_ADAM_EPOCHS: usize = EPOCHS / 2;
/// The adjoint check, every this many steps (`gn_steps.csv`'s `adjoint_error`).
const GN_ADJOINT_CHECK_EVERY: usize = 10;

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

/// The optimiser `OPTIMIZER` asks for and a hybrid run's Adam epochs: `(None, None)`
/// (Adam) when it is unset or `adam`, the `GN_*` literals for `gauss-newton`, and those
/// with `HYBRID_ADAM_EPOCHS` for `hybrid`; the probe seed is `seed`.
///
/// `GN_WEIGHT_DECAY` and `GN_CG_MAX_ITERS` override those literals.
///
/// # Panics
/// For any other value, any `GN_PRECONDITIONER` but `none` and `hutchinson`, a
/// `GN_WEIGHT_DECAY` that is negative or not finite, or a `GN_CG_MAX_ITERS` of 0.
fn optimizer(seed: u64) -> (Option<OptimizerConfig>, Option<usize>) {
    let adam_epochs = match std::env::var("OPTIMIZER").as_deref() {
        Err(_) | Ok("adam") => return (None, None),
        Ok("gauss-newton") => None,
        Ok("hybrid") => Some(HYBRID_ADAM_EPOCHS),
        Ok(other) => panic!("OPTIMIZER must be adam, gauss-newton or hybrid, got {other:?}"),
    };
    let preconditioner = match std::env::var("GN_PRECONDITIONER").as_deref() {
        Err(_) => GN_PRECONDITIONER,
        Ok("none") => Preconditioner::None,
        Ok("hutchinson") => Preconditioner::HutchinsonDiagonal {
            probes: GN_HUTCHINSON_PROBES,
        },
        Ok(other) => panic!("GN_PRECONDITIONER must be none or hutchinson, got {other:?}"),
    };
    let cg_max_iters = env_var("GN_CG_MAX_ITERS").unwrap_or(GN_CG_MAX_ITERS);
    assert!(cg_max_iters > 0, "GN_CG_MAX_ITERS must be positive");
    let weight_decay = env_var("GN_WEIGHT_DECAY").unwrap_or(GN_WEIGHT_DECAY);
    assert!(
        weight_decay.is_finite() && weight_decay >= 0.0,
        "GN_WEIGHT_DECAY must be a finite number ≥ 0, got {weight_decay}"
    );
    let gn = GaussNewtonConfig::new()
        .with_cg_max_iters(cg_max_iters)
        .with_lambda_init(GN_LAMBDA_INIT)
        .with_weight_decay(weight_decay)
        .with_preconditioner(preconditioner)
        .with_adjoint_check_every(GN_ADJOINT_CHECK_EVERY)
        .with_seed(seed);
    (Some(OptimizerConfig::GaussNewton(gn)), adam_epochs)
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

    let model_cfg = FNOConfig {
        modes: vec![modes; 2],
        hidden_channels: WIDTH,
        data_channels: 1,
        out_channels: 1,
        n_layers: N_LAYERS,
        // None keeps the Li et al. U(0, 1/(I·O)) spectral init
        spectral_init: None,
        // Poisson is non-periodic: pad each axis with zero cells before the
        // spectral layers (cropped after). Some(0) means no padding
        padding: Some(padding),
    };

    let (optimizer, adam_epochs) = optimizer(seed);
    let gn = optimizer.is_some();
    let time_budget_s = env_var::<f64>("TIME_BUDGET_S");
    // Adam's cosine schedule spans EPOCHS, so a budget would cut it short
    assert!(
        gn || time_budget_s.is_none(),
        "TIME_BUDGET_S is for OPTIMIZER=gauss-newton runs"
    );
    // NaN would never stop the run
    assert!(
        time_budget_s.is_none_or(|b| b.is_finite() && b > 0.0),
        "TIME_BUDGET_S must be a positive number of seconds, got {time_budget_s:?}"
    );
    let batch_size = if gn {
        env_var("GN_BATCH_SIZE").unwrap_or(GN_BATCH_SIZE)
    } else {
        BATCH_SIZE
    };
    assert!(batch_size > 0, "GN_BATCH_SIZE must be positive");
    // the Adam half must match the Adam reruns' steps for HYBRID_ADAM_EPOCHS to be half
    // their time
    assert!(
        adam_epochs.is_none() || batch_size == BATCH_SIZE,
        "OPTIMIZER=hybrid trains at BATCH_SIZE = {BATCH_SIZE}, got GN_BATCH_SIZE = {batch_size}"
    );
    let weight_decay = env_var("WEIGHT_DECAY").unwrap_or(WEIGHT_DECAY);
    assert!(
        weight_decay.is_finite() && weight_decay >= 0.0,
        "WEIGHT_DECAY must be a finite number ≥ 0, got {weight_decay}"
    );
    let train_cfg = TrainingConfig::new()
        .with_epochs(EPOCHS)
        .with_time_budget_s(time_budget_s)
        .with_batch_size(batch_size)
        .with_learning_rate(LEARNING_RATE)
        .with_weight_decay(weight_decay)
        .with_min_lr(MIN_LR)
        .with_seed(seed)
        .with_optimizer(optimizer)
        .with_adam_epochs(adam_epochs);

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
    let gn_tag = match (gn, adam_epochs) {
        (_, Some(_)) => "_hybrid",
        (true, None) => "_gn",
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

/// Reopens the run at `dir` with its saved configs and checkpoint (on `device`), and
/// truncates its `metrics.csv` (and `gn_steps.csv`) to the checkpoint's epoch. The
/// checkpoint counts this resume.
fn reopen(
    dir: &str,
    device: &Device,
) -> (
    RunDir,
    FNOConfig,
    TrainingConfig,
    PoissonConfig,
    Checkpoint<4>,
) {
    let run = RunDir::open(dir).unwrap_or_else(|e| panic!("could not open the run: {e}"));
    let model_cfg = run.model_config().unwrap_or_else(|e| panic!("{e}"));
    let train_cfg = run.training_config().unwrap_or_else(|e| panic!("{e}"));
    let dataset_cfg: PoissonConfig = run.data_config().unwrap_or_else(|e| panic!("{e}"));
    let mut ck = run
        .checkpoint::<4>(device)
        .unwrap_or_else(|e| panic!("could not read the checkpoint: {e}"))
        .unwrap_or_else(|| panic!("{dir} has no checkpoint to resume from"));
    ck.state.resumes += 1;
    run.truncate_metrics(ck.state.next_epoch)
        .unwrap_or_else(|e| panic!("could not truncate metrics.csv: {e}"));
    run.truncate_gn_steps(ck.state.next_epoch)
        .unwrap_or_else(|e| panic!("could not truncate gn_steps.csv: {e}"));
    println!(
        "device: {device:?}, resuming {dir} at epoch {} of {} (resume {})",
        ck.state.next_epoch, train_cfg.epochs, ck.state.resumes
    );
    (run, model_cfg, train_cfg, dataset_cfg, ck)
}

fn main() {
    let devices = train_devices();
    let device = devices[0].clone();
    let (run, model_cfg, train_cfg, dataset_cfg, resume) = match std::env::args().nth(1) {
        Some(flag) if flag == "--resume" => {
            let dir = std::env::args()
                .nth(2)
                .expect("--resume needs a run directory");
            let (run, model_cfg, train_cfg, dataset_cfg, ck) = reopen(&dir, &device);
            (run, model_cfg, train_cfg, dataset_cfg, Some(ck))
        }
        _ => {
            let (run, model_cfg, train_cfg, dataset_cfg) = new_run(&device);
            (run, model_cfg, train_cfg, dataset_cfg, None)
        }
    };
    assert!(
        train_cfg.gauss_newton().is_none() || devices.len() == 1,
        "Gauss-Newton trains on one device: unset TRAIN_DEVICES"
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
    let also: Option<&mut sciml_rs::neural_operators::training::trainer::EpochObserver<'_, 4>> =
        Some(&mut live_epoch);
    #[cfg(not(feature = "mlflow"))]
    let also = None;
    let train_start = Instant::now();
    let replicas = replicas::<4, 3, _>(&train_data, &devices[1..], &train_cfg);
    let mut components = build_training_components::<4, 3, _>(
        &model_cfg, &train_cfg, train_data, test_data, &device,
    );
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
    // Adam: every planned step (it takes no budget), one forward and one backward each;
    // Gauss-Newton: the steps taken, from gn_steps.csv, every session's (a budget may stop
    // it before EPOCHS), plus a hybrid run's Adam steps, from the epochs metrics.csv holds
    let (steps, forward_passes, backward_passes) = match train_cfg.gauss_newton() {
        None => {
            let steps = train_cfg.total_steps(n_train);
            (steps, steps, steps)
        }
        Some(_) => {
            let epochs = run.metrics().expect("could not read metrics.csv").len();
            let adam_epochs = (0..epochs).filter(|&e| train_cfg.adam_epoch(e)).count();
            let adam = adam_epochs * train_cfg.steps_per_epoch(n_train);
            let p = run.gn_passes().expect("could not read gn_steps.csv");
            // a hybrid run stopped by its budget within the Adam epochs has no GN steps
            assert!(
                p.is_some() || adam > 0,
                "a Gauss-Newton run writes gn_steps.csv"
            );
            let (forward, backward) = p.as_ref().map_or((0, 0), training_passes);
            let gn_steps = p.map_or(0, |p| p.steps);
            (gn_steps + adam, forward + adam, backward + adam)
        }
    };

    let costs = RunCosts {
        dataset_train_wall_time_s: Some(train_sidecar.wall_time_s),
        dataset_test_wall_time_s: Some(test_sidecar.wall_time_s),
        load_time_s,
        training_time_s,
        inference_time_s: None,
        host_to_device_bytes_train: train_bytes,
        host_to_device_bytes_test: test_bytes,
        // no backend reports it; never estimated (design §13.9)
        peak_gpu_memory: "unavailable".to_string(),
        backend_features: backend_features().into_iter().map(String::from).collect(),
        device: match devices.len() {
            1 => format!("{device:?}"),
            _ => format!("{devices:?}"),
        },
        git_commit: git_commit(),
        num_params: model.num_params(),
        optimisation_steps: Some(steps),
        resumes: Some(resumes),
        forward_passes: Some(forward_passes),
        backward_passes: Some(backward_passes),
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

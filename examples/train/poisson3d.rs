//! FNO on 3D Poisson on the Chebyshev grid (3D CL4.1): `FNO<5>` on the 3D splits of
//! `generate_poisson3d`, n nodes per axis, s on the model grid (default s = n − 1); n = 65
//! by default.
//!
//! Run with: cargo run --release --example train_poisson3d [-- [<padding>] [<n>] [<s>] [<seed>]]
//! (CPU by default; add `--features metal` or, with `--no-default-features`,
//! `--features cuda` for a GPU). n is one of `catalogue::GENERATED_3D` (33 for tests and
//! Mac runs, 65 for training, D5, D8); the modes are min(12, s/2) per axis (D4). `EPOCHS`
//! overrides the epochs, for a short run (CL4.1's, on the Mac):
//! EPOCHS=2 cargo run --release --features metal --example train_poisson3d -- 4 33
//!
//! A run writes a checkpoint every `CHECKPOINT_EVERY` epochs and after the last; `--resume`
//! continues an interrupted run from it with the configs the run saved, truncating
//! `metrics.csv` to the checkpoint's epoch:
//! cargo run --release --example train_poisson3d -- --resume runs/<run_dir>
//!
//! `train_poisson` for 3D (proposal §6: a separate example, `train_poisson` unchanged),
//! with its Adam path and literals and D4's model and batch. Not here (proposal §1.3, or
//! not needed for 3D yet): Gauss–Newton, `TRAIN_DEVICES`, training subsets, MLflow. The
//! loader transfers every sample from the CGL grid to the uniform grid along each axis
//! (CONVENTIONS §12); `data_cfg.json` is the 2D `PoissonConfig`, with the 3D split files.
//! The run records `costs.json` as `train_poisson` does. `predict_poisson3d` evaluates the
//! run. Needs the chosen set's dataset: see datasets/README.md.

use burn::{config::Config, module::Module, tensor::Device};
use sciml_rs::neural_operators::{
    chebyshev::TRANSFER_DEGREE,
    data::{
        catalogue::{GENERATED_3D, Poisson3dSet},
        dataset::OperatorDataset,
        loaders::{
            base_dataset::DatasetConfig,
            poisson::{
                DatasetSource, PoissonConfig, SidecarRecord, SplitSource, load_poisson3d_uniform,
            },
        },
    },
    models::fno::{FNO, FNOConfig},
    runs::{OperatorCheckpoint, RunCosts, RunDir},
    training::{
        config::TrainingConfig,
        trainer::{build_components, train_components},
    },
    utils::backend::{backend_features, git_commit},
};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The set used when no n is given: the training set (D5).
const DEFAULT_N: usize = 65;
const N_TRAIN: usize = 1000;
/// The whole test split.
const N_TEST: usize = 200;

// D4: width 32, 4 layers, modes 12³, batch 20 at s = 32 and 64; the rest as train_poisson
/// Fourier modes per axis at large s; m = min(MAX_MODES, s/2).
const MAX_MODES: usize = 12;
const WIDTH: usize = 32;
const N_LAYERS: usize = 4;
const EPOCHS: usize = 500;
const BATCH_SIZE: usize = 20;
const LEARNING_RATE: f64 = 1e-3;
const WEIGHT_DECAY: f32 = 1e-4;
const MIN_LR: f64 = 1e-5;
/// Zero cells appended to each spatial axis (CONVENTIONS §3), 2D Phase 4's choice.
const PADDING: usize = 4;
/// Epochs between checkpoints; the last epoch is always checkpointed.
const CHECKPOINT_EVERY: usize = 25;

/// The `i`-th command-line argument as a number, or `default` when it is absent.
///
/// # Panics
/// If the argument does not parse, naming it `name`.
fn arg<T: std::str::FromStr>(i: usize, name: &str, default: T) -> T
where
    T::Err: std::fmt::Display,
{
    match std::env::args().nth(i) {
        Some(arg) => arg
            .parse()
            .unwrap_or_else(|e| panic!("{name} argument must be an integer, got {arg:?}: {e}")),
        None => default,
    }
}

/// `datasets/poisson/` in the repository.
fn datasets() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/poisson")
}

/// The 3D set with `n` CGL nodes per axis.
///
/// # Panics
/// If `GENERATED_3D` has none.
fn set(n: usize) -> Poisson3dSet {
    GENERATED_3D
        .into_iter()
        .find(|set| set.n == n)
        .unwrap_or_else(|| {
            panic!("no 3D Poisson set with n = {n}; expected one of {GENERATED_3D:?}")
        })
}

/// The sidecar of one split of `set`.
///
/// # Panics
/// If it can't be read.
fn sidecar(set: &Poisson3dSet, split: &str) -> SidecarRecord {
    let path = datasets().join(set.json(split));
    SidecarRecord::load(&path)
        .unwrap_or_else(|e| panic!("could not read the sidecar {}: {e}", path.display()))
}

/// A new run from the command-line arguments and the literals above: creates its
/// directory and saves its configs there before training, so `--resume` can read them.
fn new_run(device: &Device) -> (RunDir, FNOConfig, TrainingConfig, PoissonConfig) {
    let padding = arg(1, "padding", PADDING);
    let n = arg(2, "n", DEFAULT_N);
    let set = set(n);
    let s = arg(3, "s", n - 1);
    let default_seed = TrainingConfig::new().seed;
    let seed = arg(4, "seed", default_seed);
    let epochs = std::env::var("EPOCHS").map_or(EPOCHS, |v| {
        v.parse()
            .unwrap_or_else(|e| panic!("EPOCHS must be an integer, got {v:?}: {e}"))
    });
    assert!(epochs > 0, "EPOCHS must be positive");
    let modes = MAX_MODES.min(s / 2);

    println!(
        "device: {device:?}, padding: {padding}, n: {n}, s: {s}, modes: {modes}³, K: {}, \
         seed: {seed}, epochs: {epochs}",
        set.k
    );

    let (train_sidecar, test_sidecar) = (sidecar(&set, "train"), sidecar(&set, "test"));
    let split = |split: &str, sidecar: &SidecarRecord| {
        SplitSource::new(set.npz(split), sidecar.git_commit.clone())
    };
    let dataset_cfg = PoissonConfig::new(
        DatasetConfig {
            n_train: N_TRAIN,
            n_test: N_TEST,
        },
        n,
    )
    .with_s(Some(s))
    .with_source(Some(DatasetSource::new(
        set.k,
        TRANSFER_DEGREE,
        split("train", &train_sidecar),
        split("test", &test_sidecar),
    )));

    let model_cfg = FNOConfig {
        modes: vec![modes; 3],
        hidden_channels: WIDTH,
        data_channels: 1,
        out_channels: 1,
        n_layers: N_LAYERS,
        spectral_init: None,
        padding: Some(padding),
        activation: None,
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
    let train_cfg = TrainingConfig::new()
        .with_epochs(epochs)
        .with_batch_size(BATCH_SIZE)
        .with_learning_rate(LEARNING_RATE)
        .with_weight_decay(WEIGHT_DECAY)
        .with_min_lr(MIN_LR)
        .with_seed(seed);

    let seed_tag = if seed == default_seed {
        String::new()
    } else {
        format!("_seed{seed}")
    };
    let name = format!("poisson3d_fno_n{n}_s{s}_p{padding}{seed_tag}");
    let run = RunDir::create(&Path::new(env!("CARGO_MANIFEST_DIR")).join("runs"), &name)
        .unwrap_or_else(|e| panic!("could not create the run directory: {e}"));
    run.save_configs(&model_cfg, &train_cfg, &dataset_cfg)
        .expect("could not save configs");
    (run, model_cfg, train_cfg, dataset_cfg)
}

/// Reopens the run at `dir` with its saved configs and checkpoint (on `device`), and
/// truncates its `metrics.csv` to the checkpoint's epoch. The checkpoint counts this
/// resume.
fn reopen(
    dir: &str,
    device: &Device,
) -> (
    RunDir,
    FNOConfig,
    TrainingConfig,
    PoissonConfig,
    OperatorCheckpoint<FNO<5>>,
) {
    let run = RunDir::open(dir).unwrap_or_else(|e| panic!("could not open the run: {e}"));
    let model_cfg = run.model_config().unwrap_or_else(|e| panic!("{e}"));
    let train_cfg = run.training_config().unwrap_or_else(|e| panic!("{e}"));
    let dataset_cfg: PoissonConfig = run.data_config().unwrap_or_else(|e| panic!("{e}"));
    let mut ck = run
        .checkpoint_of::<FNO<5>>(device)
        .unwrap_or_else(|e| panic!("could not read the checkpoint: {e}"))
        .unwrap_or_else(|| panic!("{dir} has no checkpoint to resume from"));
    ck.state.resumes += 1;
    run.truncate_metrics(ck.state.next_epoch)
        .unwrap_or_else(|e| panic!("could not truncate metrics.csv: {e}"));
    println!(
        "device: {device:?}, resuming {dir} at epoch {} of {} (resume {})",
        ck.state.next_epoch, train_cfg.epochs, ck.state.resumes
    );
    (run, model_cfg, train_cfg, dataset_cfg, ck)
}

fn main() {
    let device = Device::default().autodiff();
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
    let set = set(dataset_cfg.n);

    let train_path = datasets().join(set.npz("train"));
    let test_path = datasets().join(set.npz("test"));
    for p in [&train_path, &test_path] {
        assert!(
            p.exists(),
            "3D Poisson dataset not found at {}\nSee datasets/README.md",
            p.display()
        );
    }
    let (train_sidecar, test_sidecar) = (sidecar(&set, "train"), sidecar(&set, "test"));

    let load_start = Instant::now();
    let (train_data, test_data, normalizers) =
        load_poisson3d_uniform::<f32>(&train_path, &test_path, &dataset_cfg)
            .unwrap_or_else(|e| panic!("could not load the 3D Poisson data: {e}"));
    let load_time_s = load_start.elapsed().as_secs_f64();
    // what device_loader uploads: inputs and targets as stored
    let bytes = |d: &OperatorDataset<f32>| {
        (d.inputs().len() + d.targets().len()) * std::mem::size_of::<f32>()
    };
    let (train_bytes, test_bytes) = (bytes(&train_data), bytes(&test_data));
    let n_train = train_data.inputs().shape()[0];
    println!(
        "poisson3d: {} train / {} test, n={} -> s={}",
        n_train,
        test_data.inputs().shape()[0],
        dataset_cfg.n,
        train_data.inputs().shape()[1]
    );

    // as train_poisson: the y normalizer decodes predictions and targets during training,
    // predictions only during evaluation; the time includes building the components, and
    // after a resume the earlier sessions' time up to the checkpoint
    let (prior_time_s, resumes) = resume
        .as_ref()
        .map_or((0.0, 0), |ck| (ck.state.training_time_s, ck.state.resumes));
    let train_start = Instant::now();
    let components =
        build_components::<FNO<5>, 5, 4, _>(&model_cfg, &train_cfg, train_data, test_data, &device);
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
    // Adam: every planned step, one forward and one backward each
    let steps = train_cfg.total_steps(n_train);

    let costs = RunCosts {
        dataset_train_wall_time_s: Some(train_sidecar.wall_time_s),
        dataset_test_wall_time_s: Some(test_sidecar.wall_time_s),
        load_time_s,
        training_time_s,
        inference_time_s: None,
        host_to_device_bytes_train: train_bytes,
        host_to_device_bytes_test: test_bytes,
        peak_gpu_memory: "unavailable".to_string(),
        backend_features: backend_features().into_iter().map(String::from).collect(),
        device: format!("{device:?}"),
        git_commit: git_commit(),
        num_params: model.num_params(),
        optimisation_steps: Some(steps),
        resumes: Some(resumes),
        forward_passes: Some(steps),
        backward_passes: Some(steps),
    };

    let metrics = run.metrics().expect("could not read metrics.csv");
    run.save_plot(&metrics);
    run.save_normalizers(&normalizers)
        .expect("could not save normalizers");
    run.save_weights(&model)
        .unwrap_or_else(|e| panic!("could not save model weights: {e}"));
    run.save_costs(&costs).expect("could not save costs.json");

    println!("run written to {}", run.path().display());
}

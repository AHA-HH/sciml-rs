//! Optimisers CO0.3 (docs/optimisers/tasks/CO0.3.md): where the time of one Adam step and
//! one epoch goes, on CO0.1's Adam setting, without changing any library code.
//!
//! Run with: cargo run --release --example adam_profile -- <out.md>
//! (`--no-default-features --features cuda` on the VM). Needs stage 3 (n = 129) in
//! `datasets/poisson/`.
//!
//! Its version at `bc26896` also builds at `4541f02`, the last commit on the Burn fork; this
//! one differs from it only in calling `build_components` (library CM0.6), which the fork
//! lacks, for `build_training_components`. It builds its configs with builders, never
//! struct literals, so fields added since the fork keep their defaults. From a freshly
//! initialised model (a step's cost does not depend on the weights):
//! 1. [`WARMUP_EPOCHS`] epochs of the library's `train_epoch`, untimed.
//! 2. [`EPOCHS`] epochs of `train_epoch`, each synced at its ends, each followed by
//!    `evaluate` on the test split and a `save_checkpoint`, both synced: the real costs.
//! 3. [`EPOCHS`] epochs of a copy of `train_epoch`'s loop ([`inline_epoch`]), the host's
//!    time around each phase without syncing: what the host spends issuing it.
//! 4. The same with [`Device::sync`] before and after each phase: what each phase takes to
//!    finish on the device.
//!
//! Writes the tables to the first argument. Timings are of this machine and backend only.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use burn::{
    data::dataloader::DataLoader,
    module::Module,
    optim::{
        GradientsParams, ModuleOptimizer, lr_scheduler::module_lr_scheduler::ModuleLrScheduler,
    },
    prelude::*,
};
use sciml_rs::neural_operators::{
    data::{
        batcher::Batch,
        catalogue::stage,
        loaders::{
            base_dataset::DatasetConfig,
            poisson::{PoissonConfig, load_poisson_uniform},
        },
        transforms::normalizers::FlatDecoder,
    },
    losses::data_losses::{LpLoss, Reduction},
    models::fno::{FNO, FNOConfig},
    runs::{CheckpointState, RunDir},
    training::{
        config::TrainingConfig,
        step::{evaluate, flatten_pair},
        trainer::{build_components, train_epoch},
    },
    utils::backend::{backend_features, git_commit},
};

// CO0.1's Adam arm (Phase 5 settings): stage 3, s = 64, p = 0
const N: usize = 129;
const S: usize = 64;
const PADDING: usize = 0;
const MODES: usize = 12;
const WIDTH: usize = 32;
const N_LAYERS: usize = 4;
const N_TRAIN: usize = 1000;
const N_TEST: usize = 200;
const BATCH_SIZE: usize = 20;
const LEARNING_RATE: f64 = 1e-3;
const WEIGHT_DECAY: f32 = 1e-4;
const MIN_LR: f64 = 1e-5;
/// The epochs the cosine schedule spans (CO0.1's 500).
const SCHEDULE_EPOCHS: usize = 500;
const SEED: u64 = 42;
/// `train_poisson` checkpoints every this many epochs; a checkpoint's cost per epoch is
/// its time over this.
const CHECKPOINT_EVERY: usize = 25;

/// Untimed epochs before any timed one (autotune, the allocator).
const WARMUP_EPOCHS: usize = 3;
/// Timed epochs of each kind.
const EPOCHS: usize = 3;
/// The inline loop's unsynced epoch is reported as a faithful copy of `train_epoch` when
/// within this fraction of it.
const FAITHFUL: f64 = 0.10;

/// The step's phases in `train_epoch`'s order, then the epoch's closing host reads.
const PHASES: [&str; N_PHASES] = [
    "loader (`next`)",
    "forward and loss",
    "metrics (MSE, accumulators)",
    "backward (`backward`, `from_grads`)",
    "scheduler",
    "optimiser update",
    "epoch's host reads (2 per epoch)",
];
/// [`PHASES`]'s length, the size of every array of phase times.
const N_PHASES: usize = 7;
/// The phases that run once per step (the last runs once per epoch).
const STEP_PHASES: usize = 6;

/// The optimiser, schedule and loaders of one run.
struct Parts {
    optim: ModuleOptimizer,
    scheduler: ModuleLrScheduler,
    train: Arc<dyn DataLoader<Batch<4, 3>>>,
    test: Arc<dyn DataLoader<Batch<4, 3>>>,
}

/// `f`'s result, its wall time added to `t`; with `sync`, that device synced before and
/// after.
fn time<T>(sync: Option<&Device>, t: &mut f64, f: impl FnOnce() -> T) -> T {
    if let Some(d) = sync {
        d.sync().expect("device sync");
    }
    let start = Instant::now();
    let out = f();
    if let Some(d) = sync {
        d.sync().expect("device sync");
    }
    *t += start.elapsed().as_secs_f64();
    out
}

/// `sum + x`, starting from the first `x` (the trainer's `accumulate`, crate-private).
fn accumulate(sum: Option<Tensor<1>>, x: Tensor<1>) -> Option<Tensor<1>> {
    Some(match sum {
        Some(sum) => sum + x,
        None => x,
    })
}

/// One epoch of a copy of `train_epoch`'s loop (`trainer.rs`, the summed relative L² as
/// the objective; keep them in step), each phase timed into `t` ([`PHASES`]); returns the
/// model and the number of steps.
fn inline_epoch(
    mut model: FNO<4>,
    parts: &mut Parts,
    decoder: &FlatDecoder,
    sync: Option<&Device>,
    t: &mut [f64; N_PHASES],
) -> (FNO<4>, usize) {
    let (mut mse_sum, mut l2_sum, mut steps) = (None, None, 0);
    let mut iterator = parts.train.iter();
    while let Some(batch) = time(sync, &mut t[0], || iterator.next()) {
        let batch = batch.expect("dataset error during training");
        let (l2, out, target) = time(sync, &mut t[1], || {
            let out = model.forward(batch.inputs);
            let (out, target) = flatten_pair::<4, 3>(out, batch.targets);
            let out = decoder.decode(out);
            let l2 = LpLoss::new(2, Reduction::Sum).rel(out.clone(), target.clone());
            (l2, out, target)
        });
        time(sync, &mut t[2], || {
            let mse = (out.inner() - target.inner()).powf_scalar(2.0).mean();
            mse_sum = accumulate(mse_sum.take(), mse);
            l2_sum = accumulate(l2_sum.take(), l2.clone().inner());
        });
        let grads = time(sync, &mut t[3], || {
            GradientsParams::from_grads(l2.backward(), &model)
        });
        let lr = time(sync, &mut t[4], || parts.scheduler.step());
        model = time(sync, &mut t[5], || parts.optim.step(lr, model, grads));
        steps += 1;
    }
    time(sync, &mut t[6], || {
        let reads = [mse_sum, l2_sum].map(|s| s.expect("no batches").into_scalar::<f32>());
        assert!(reads.iter().all(|r| r.is_finite()), "loss not finite");
    });
    (model, steps)
}

/// The library's costs per epoch, in seconds: `train_epoch`, `evaluate`, `save_checkpoint`.
struct Library {
    epoch: f64,
    evaluate: f64,
    checkpoint: f64,
}

/// One phase's row: name, host and device seconds per step (per epoch for the closing
/// reads), device seconds per epoch, and its share of the device epoch.
#[derive(Debug, PartialEq)]
struct Row {
    name: &'static str,
    host: f64,
    device: f64,
    device_epoch: f64,
    share: f64,
}

/// The rows of [`PHASES`] from the phase totals of `epochs` epochs of `steps` steps each,
/// unsynced (`host`) and synced (`device`), and the device epoch they sum to.
fn rows(
    host: &[f64; N_PHASES],
    device: &[f64; N_PHASES],
    epochs: usize,
    steps: usize,
) -> (Vec<Row>, f64) {
    let device_epoch: f64 = device.iter().sum::<f64>() / epochs as f64;
    let rows = PHASES
        .iter()
        .enumerate()
        .map(|(i, &name)| {
            let per = (epochs * if i < STEP_PHASES { steps } else { 1 }) as f64;
            let epoch = device[i] / epochs as f64;
            Row {
                name,
                host: host[i] / per,
                device: device[i] / per,
                device_epoch: epoch,
                share: epoch / device_epoch,
            }
        })
        .collect();
    (rows, device_epoch)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [out] = args.as_slice() else {
        panic!("usage: adam_profile <out.md>");
    };
    let out = PathBuf::from(out);
    create_output(&out).unwrap_or_else(|e| panic!("create {}: {e}", out.display()));
    let device = Device::default().autodiff();

    let set = stage(N).unwrap_or_else(|e| panic!("{e}"));
    let data_cfg = PoissonConfig::new(
        DatasetConfig {
            n_train: N_TRAIN,
            n_test: N_TEST,
        },
        N,
    )
    .with_s(Some(S));
    let datasets = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/poisson");
    let (train, test, normalizers) = load_poisson_uniform::<f32>(
        datasets.join(set.npz("train")),
        datasets.join(set.npz("test")),
        &data_cfg,
    )
    .unwrap_or_else(|e| panic!("load stage {N}: {e}"));
    let model_cfg = FNOConfig::new(vec![MODES; 2], 1, 1)
        .with_hidden_channels(WIDTH)
        .with_n_layers(N_LAYERS)
        .with_padding(Some(PADDING));
    let train_cfg = TrainingConfig::new()
        .with_epochs(SCHEDULE_EPOCHS)
        .with_batch_size(BATCH_SIZE)
        .with_learning_rate(LEARNING_RATE)
        .with_weight_decay(WEIGHT_DECAY)
        .with_min_lr(MIN_LR)
        .with_seed(SEED);
    let c = build_components::<FNO<4>, 4, 3, f32>(&model_cfg, &train_cfg, train, test, &device);
    let mut model = c.model;
    let mut parts = Parts {
        optim: c.optim,
        scheduler: c.scheduler,
        train: c.train_loader,
        test: c.test_loader,
    };
    let decoder = FlatDecoder::new(&normalizers.y, &device);
    let eval_decoder = decoder.inner();
    println!(
        "n {N}, s {S}, p {PADDING}, batch {BATCH_SIZE}, backend {:?}",
        backend_features()
    );

    let library_epoch = |model: FNO<4>, parts: &mut Parts| {
        train_epoch(
            model,
            &parts.train,
            &mut parts.optim,
            &mut parts.scheduler,
            Some(&decoder),
        )
        .0
    };
    for e in 0..WARMUP_EPOCHS {
        let start = Instant::now();
        model = library_epoch(model, &mut parts);
        println!("warm-up epoch {e}: {:.3} s", start.elapsed().as_secs_f64());
    }

    let root = std::env::temp_dir().join(format!("adam_profile_{}", std::process::id()));
    let run = RunDir::create(&root, "adam_profile").expect("scratch run directory");
    let (mut t_epoch, mut t_evaluate, mut t_checkpoint) = (0.0, 0.0, 0.0);
    for e in 0..EPOCHS {
        model = time(Some(&device), &mut t_epoch, || {
            library_epoch(model, &mut parts)
        });
        let test_l2 = time(Some(&device), &mut t_evaluate, || {
            evaluate(&model.valid(), &parts.test, Some(&eval_decoder))
        });
        let state = CheckpointState {
            next_epoch: WARMUP_EPOCHS + e + 1,
            resumes: 0,
            training_time_s: 0.0,
        };
        time(Some(&device), &mut t_checkpoint, || {
            run.save_checkpoint(&state, &model, &parts.optim, &parts.scheduler)
                .expect("save checkpoint")
        });
        println!("library epoch {e}: test L² {test_l2:.4e}");
    }
    std::fs::remove_dir_all(&root).expect("remove the scratch run directory");
    let k = EPOCHS as f64;
    let lib = Library {
        epoch: t_epoch / k,
        evaluate: t_evaluate / k,
        checkpoint: t_checkpoint / k,
    };

    let (mut host, mut dev) = ([0.0; N_PHASES], [0.0; N_PHASES]);
    let (mut host_epoch, mut steps) = (0.0, 0);
    for _ in 0..EPOCHS {
        device.sync().expect("device sync");
        let start = Instant::now();
        (model, steps) = inline_epoch(model, &mut parts, &decoder, None, &mut host);
        device.sync().expect("device sync");
        host_epoch += start.elapsed().as_secs_f64();
    }
    let host_epoch = host_epoch / EPOCHS as f64;
    // no epoch timer here: the per-phase syncs already bracket all of it
    for _ in 0..EPOCHS {
        (model, _) = inline_epoch(model, &mut parts, &decoder, Some(&device), &mut dev);
    }

    let report = write_report(&host, &dev, steps, host_epoch, &lib);
    print!("{report}");
    std::fs::write(&out, report).unwrap_or_else(|e| panic!("write {}: {e}", out.display()));
    println!("wrote {}", out.display());
}

/// Creates `out` (empty) and its parent directories, so that a path that cannot be written
/// fails before the measurement (as `gn_profile`).
fn create_output(out: &Path) -> std::io::Result<()> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(out, "")
}

/// The report: the library's epoch, the phases unsynced and synced, the reconciliation.
fn write_report(
    host: &[f64; N_PHASES],
    dev: &[f64; N_PHASES],
    steps: usize,
    host_epoch: f64,
    lib: &Library,
) -> String {
    let (rows, device_epoch) = rows(host, dev, EPOCHS, steps);
    let mut r = String::new();
    writeln!(
        r,
        "# Adam step profile\n\nn {N}, s {S}, p {PADDING}, modes [{MODES}, {MODES}], width \
         {WIDTH}, {N_LAYERS} layers, N {N_TRAIN}, batch {BATCH_SIZE} ({steps} steps per \
         epoch), {N_TEST} test samples; backend {:?}, commit {}. {WARMUP_EPOCHS} warm-up \
         epochs, then {EPOCHS} epochs of each kind; means per epoch or step.\n",
        backend_features(),
        git_commit()
    )
    .unwrap();
    writeln!(
        r,
        "## The library's epoch\n\n| part | s per epoch |\n| --- | --- |\n\
         | `train_epoch` (synced at its ends) | {:.4} |\n| `evaluate` (test split) | {:.4} |\n\
         | **epoch as `metrics.csv` times it** (train + evaluate) | **{:.4}** |\n\
         | `save_checkpoint` (one) | {:.4} |\n\
         | checkpoint per epoch (one every {CHECKPOINT_EVERY}) | {:.4} |\n",
        lib.epoch,
        lib.evaluate,
        lib.epoch + lib.evaluate,
        lib.checkpoint,
        lib.checkpoint / CHECKPOINT_EVERY as f64
    )
    .unwrap();
    writeln!(
        r,
        "## The step by phase\n\nHost: no sync, the time the host spends issuing the phase. \
         Device: synced before and after, the time until the device has finished it.\n\n\
         | phase | host (s per step) | device (s per step) | device (s per epoch) | share |\n\
         | --- | --- | --- | --- | --- |"
    )
    .unwrap();
    for row in &rows {
        writeln!(
            r,
            "| {} | {:.5} | {:.5} | {:.4} | {:.1} % |",
            row.name,
            row.host,
            row.device,
            row.device_epoch,
            100.0 * row.share
        )
        .unwrap();
    }
    let host_sum: f64 = host.iter().sum::<f64>() / EPOCHS as f64;
    let copy_gap = (host_epoch - lib.epoch) / lib.epoch;
    let faithful = if copy_gap.abs() <= FAITHFUL {
        format!("the copy is faithful, within {:.0} %", 100.0 * FAITHFUL)
    } else {
        format!(
            "**above {:.0} %: the copy is not faithful; the phase table does not describe \
             `train_epoch`**",
            100.0 * FAITHFUL
        )
    };
    writeln!(
        r,
        "| **sum** | | | **{device_epoch:.4}** | 100 % |\n\n\
         ## Reconciliation\n\n\
         - Inline loop, unsynced, epoch (synced at its ends): {host_epoch:.4} s, against \
         `train_epoch`'s {:.4} s ({:+.1} %; {faithful}).\n\
         - Host time issuing the phases, summed: {host_sum:.4} s per epoch ({:.0} % of the \
         unsynced epoch; near 100 % means the host, not the device, sets the pace).\n\
         - Synced phases, summed: {device_epoch:.4} s per epoch ({:+.1} % against \
         `train_epoch`; the syncs' own cost and the lost overlap).",
        lib.epoch,
        100.0 * copy_gap,
        100.0 * host_sum / host_epoch,
        100.0 * (device_epoch - lib.epoch) / lib.epoch
    )
    .unwrap();
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Per-step means for the step phases, per-epoch for the closing reads; shares of the
    /// device epoch.
    #[test]
    fn rows_by_hand() {
        // 2 epochs of 10 steps
        let host = [2.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let device = [4.0, 8.0, 2.0, 2.0, 0.0, 2.0, 2.0];
        let (rows, epoch) = rows(&host, &device, 2, 10);
        assert_eq!(epoch, 10.0);
        assert_eq!(
            rows[1],
            Row {
                name: PHASES[1],
                host: 0.2,
                device: 0.4,
                device_epoch: 4.0,
                share: 0.4
            }
        );
        // the closing reads: per epoch, not per step
        assert_eq!(
            (rows[6].host, rows[6].device, rows[6].share),
            (0.5, 1.0, 0.1)
        );
        assert!((rows.iter().map(|r| r.share).sum::<f64>() - 1.0).abs() < 1e-12);
    }

    /// The report's missing parent directories are created; a parent that is a file fails.
    #[test]
    fn create_output_makes_missing_parents() {
        let root = std::env::temp_dir().join(format!("adam_profile_t_{}", std::process::id()));
        let out = root.join("a/b/CO0.3.md");
        create_output(&out).expect("create");
        assert!(out.is_file());
        assert!(create_output(&out.join("below_a_file.md")).is_err());
        std::fs::remove_dir_all(&root).expect("clean up");
    }
}

//! What `train_burgers`, `train_darcy` and `train_poisson` share (optimisers CO1.2, from
//! CO1.1's code-quality review, F5): the training configuration from the example's
//! literals and the environment, reopening a run to resume it, and `costs.json`. Each
//! includes it as `mod common;`; it is not an example of its own.
//!
//! Environment variables (unset, the example's literals):
//! - `OPTIMIZER`: `adam` (the default, Adam with coupled weight decay and the cosine),
//!   `adamw` (AdamW with the learning rate halved every `epochs / 5` epochs: every 100 at
//!   500, D11), `gauss-newton`, `hybrid` (Gauss–Newton after Adam epochs, Phase 10 T5), or
//!   `lbfgs` (L-BFGS on the full batch, optimisers CO2.1), or `soap` (SOAP on Adam's path,
//!   the cosine, optimisers CO3.3); each example names those it takes.
//! - `WEIGHT_DECAY`: Adam's, AdamW's and SOAP's (decoupled) weight decay.
//! - `LEARNING_RATE`: the initial learning rate of Adam, AdamW and SOAP.
//! - `SOAP_FREQUENCY` (steps between basis refreshes) and `SOAP_LOG_FIRST_REFRESH=1` (print
//!   the factors' eigenvalue ratios at the first refresh): SOAP's settings.
//! - `TIME_BUDGET_S`: with Gauss–Newton, L-BFGS or SOAP, stop after the first epoch that ends at
//!   or over that many seconds of training (`TrainingConfig::time_budget_s`).
//! - `LBFGS_HISTORY`, `LBFGS_MAX_ITER`, `LBFGS_MAX_EVAL` (closure evaluations per step,
//!   line search included) and `LBFGS_WEIGHT_DECAY` (μ): L-BFGS's settings.
//! - `GN_BATCH_SIZE`, `GN_CG_MAX_ITERS`, `GN_WEIGHT_DECAY` (μ), and `GN_PRECONDITIONER`
//!   (`none` or `hutchinson`): Gauss–Newton's settings.

use burn::{config::Config, tensor::Device};
use sciml_rs::neural_operators::{
    models::fno::{FNO, FNOConfig},
    optim::gauss_newton::{GaussNewtonConfig, Preconditioner},
    optim::soap::SoapConfig,
    runs::{OperatorCheckpoint, RunCosts, RunDir},
    training::{
        config::{LbfgsArm, LrSchedule, OptimizerConfig, TrainingConfig},
        gn::training_passes,
    },
    utils::backend::backend_features,
};
use std::fmt::Display;
use std::str::FromStr;

/// The environment variable `name` parsed as `T`, or `None` when it is unset.
///
/// # Panics
/// If it is set and does not parse.
pub fn env_var<T: FromStr>(name: &str) -> Option<T>
where
    T::Err: Display,
{
    let v = std::env::var(name).ok()?;
    Some(
        v.parse()
            .unwrap_or_else(|e| panic!("{name} has an invalid value {v:?}: {e}")),
    )
}

/// An example's Gauss–Newton literals, before `GN_*` overrides them.
pub struct GnLiterals {
    /// Samples per step (`GN_BATCH_SIZE`); gradient, curvature and ρ on one batch (G3).
    pub batch_size: usize,
    /// The most CG iterations per step (`GN_CG_MAX_ITERS`).
    pub cg_max_iters: usize,
    /// The damping λ of the first step.
    pub lambda_init: f64,
    /// Weight decay μ as a Tikhonov term (`GN_WEIGHT_DECAY`).
    pub weight_decay: f64,
    /// Whether CG is preconditioned with the Hutchinson diagonal (`GN_PRECONDITIONER`
    /// overrides: `hutchinson` or `none`).
    pub hutchinson: bool,
    /// Probes per Hutchinson estimate.
    pub hutchinson_probes: usize,
    /// The adjoint check, every this many steps (`gn_steps.csv`'s `adjoint_error`).
    pub adjoint_check_every: usize,
}

/// An example's training literals, before the environment overrides them.
pub struct Literals {
    /// Epochs, at least 1; under AdamW, a multiple of 5.
    pub epochs: usize,
    /// Adam's and AdamW's batch size.
    pub batch_size: usize,
    /// Adam's, AdamW's and SOAP's initial learning rate (`LEARNING_RATE`).
    pub learning_rate: f64,
    /// Adam's, AdamW's and SOAP's weight decay (`WEIGHT_DECAY`).
    pub weight_decay: f32,
    /// The cosine's final learning rate.
    pub min_lr: f64,
    /// Gauss–Newton's literals.
    pub gn: GnLiterals,
    /// A hybrid run's Adam epochs (`OPTIMIZER=hybrid`, Phase 10 T5); `None` for an example
    /// that takes no `hybrid`.
    pub hybrid_adam_epochs: Option<usize>,
    /// The L-BFGS arm (`OPTIMIZER=lbfgs`, optimisers CO2.1) and its batch size, the training
    /// set's (D4); `None` for an example that takes no `lbfgs`.
    pub lbfgs: Option<(LbfgsArm, usize)>,
    /// The SOAP arm (`OPTIMIZER=soap`, optimisers CO3.3), its weight decay replaced by
    /// `WEIGHT_DECAY`'s; `None` for an example that takes no `soap`.
    pub soap: Option<SoapConfig>,
}

/// The training configuration `OPTIMIZER`, one of `allowed` (unset is `adam`), asks for,
/// from `lit` and the environment (see the module docs), with seed `seed` (the weights,
/// the shuffle and Gauss–Newton's probes). `hybrid` is Gauss–Newton after
/// `lit.hybrid_adam_epochs` Adam epochs, at Adam's batch size.
///
/// # Panics
/// For an `OPTIMIZER` not in `allowed`, a negative or non-finite `WEIGHT_DECAY` or
/// `GN_WEIGHT_DECAY`, a `LEARNING_RATE` not positive and finite, a `TIME_BUDGET_S` without
/// Gauss–Newton, L-BFGS or SOAP or not positive and finite, a
/// batch size or `GN_CG_MAX_ITERS` of 0, a `GN_PRECONDITIONER` but `none` and `hutchinson`,
/// `lit.epochs` of 0, AdamW with `lit.epochs` not a multiple of 5, `hybrid` without
/// `lit.hybrid_adam_epochs` or with a `GN_BATCH_SIZE` other than `lit.batch_size`, `lbfgs`
/// without `lit.lbfgs`, an `LBFGS_*` of 0 or a negative or non-finite
/// `LBFGS_WEIGHT_DECAY`, `soap` without `lit.soap`, or a `SOAP_FREQUENCY` of 0.
pub fn training_config(allowed: &[&str], seed: u64, lit: &Literals) -> TrainingConfig {
    let name = std::env::var("OPTIMIZER").unwrap_or_else(|_| "adam".to_string());
    assert!(
        allowed.contains(&name.as_str()),
        "OPTIMIZER must be one of {}, got {name:?}",
        allowed.join(", ")
    );
    assert!(lit.epochs > 0, "EPOCHS must be positive");
    let (optimizer, lr_schedule, adam_epochs) = match name.as_str() {
        "adam" => (None, None, None),
        "adamw" => {
            assert!(
                lit.epochs.is_multiple_of(5),
                "OPTIMIZER=adamw halves the rate every epochs / 5: epochs must be a multiple \
                 of 5, got {}",
                lit.epochs
            );
            let every_epochs = lit.epochs / 5;
            (
                Some(OptimizerConfig::AdamW),
                Some(LrSchedule::Step { every_epochs }),
                None,
            )
        }
        "gauss-newton" => (
            Some(OptimizerConfig::GaussNewton(gauss_newton(seed, &lit.gn))),
            None,
            None,
        ),
        "hybrid" => (
            Some(OptimizerConfig::GaussNewton(gauss_newton(seed, &lit.gn))),
            None,
            Some(
                lit.hybrid_adam_epochs
                    .expect("OPTIMIZER=hybrid needs the example's hybrid_adam_epochs"),
            ),
        ),
        "lbfgs" => (Some(OptimizerConfig::Lbfgs(lbfgs(lit))), None, None),
        "soap" => (Some(OptimizerConfig::Soap(soap(lit))), None, None),
        other => unreachable!("{other:?} is in `allowed` but names no optimiser"),
    };
    let gn = matches!(optimizer, Some(OptimizerConfig::GaussNewton(_)));
    let is_lbfgs = matches!(optimizer, Some(OptimizerConfig::Lbfgs(_)));
    let is_soap = matches!(optimizer, Some(OptimizerConfig::Soap(_)));

    let time_budget_s = env_var::<f64>("TIME_BUDGET_S");
    // Adam's cosine schedule spans the epochs, so a budget would cut it short; SOAP runs
    // at Adam's equal time (D6), its cosine cut short if it is slower
    assert!(
        gn || is_lbfgs || is_soap || time_budget_s.is_none(),
        "TIME_BUDGET_S is for OPTIMIZER=gauss-newton, lbfgs and soap runs"
    );
    // NaN would never stop the run
    assert!(
        time_budget_s.is_none_or(|b| b.is_finite() && b > 0.0),
        "TIME_BUDGET_S must be a positive number of seconds, got {time_budget_s:?}"
    );
    let batch_size = if gn {
        env_var("GN_BATCH_SIZE").unwrap_or(lit.gn.batch_size)
    } else if let Some((_, full)) = lit.lbfgs.as_ref().filter(|_| is_lbfgs) {
        *full
    } else {
        lit.batch_size
    };
    assert!(batch_size > 0, "GN_BATCH_SIZE must be positive");
    // the Adam half must match the Adam reruns' steps for hybrid_adam_epochs to be half
    // their time
    assert!(
        adam_epochs.is_none() || batch_size == lit.batch_size,
        "OPTIMIZER=hybrid trains at the batch size {}, got GN_BATCH_SIZE = {batch_size}",
        lit.batch_size
    );
    let weight_decay = env_var("WEIGHT_DECAY").unwrap_or(lit.weight_decay);
    assert!(
        weight_decay.is_finite() && weight_decay >= 0.0,
        "WEIGHT_DECAY must be a finite number ≥ 0, got {weight_decay}"
    );
    let learning_rate = env_var("LEARNING_RATE").unwrap_or(lit.learning_rate);
    assert!(
        learning_rate.is_finite() && learning_rate > 0.0,
        "LEARNING_RATE must be a positive number, got {learning_rate}"
    );
    let optimizer = match optimizer {
        Some(OptimizerConfig::Soap(soap)) => {
            Some(OptimizerConfig::Soap(soap.with_weight_decay(weight_decay)))
        }
        other => other,
    };
    TrainingConfig::new()
        .with_epochs(lit.epochs)
        .with_time_budget_s(time_budget_s)
        .with_batch_size(batch_size)
        .with_learning_rate(learning_rate)
        .with_weight_decay(weight_decay)
        .with_min_lr(lit.min_lr)
        .with_seed(seed)
        .with_optimizer(optimizer)
        .with_lr_schedule(lr_schedule)
        .with_adam_epochs(adam_epochs)
}

/// The L-BFGS arm of `lit` with the `LBFGS_*` variables.
fn lbfgs(lit: &Literals) -> LbfgsArm {
    let (arm, _) = lit
        .lbfgs
        .clone()
        .expect("OPTIMIZER=lbfgs needs the example's L-BFGS literals");
    let history_size = env_var("LBFGS_HISTORY").unwrap_or(arm.lbfgs.history_size);
    let max_iter = env_var("LBFGS_MAX_ITER").unwrap_or(arm.lbfgs.max_iter);
    // unset and without a literal, Burn's ⌊1.25 max_iter⌋
    let max_eval = env_var("LBFGS_MAX_EVAL").or(arm.lbfgs.max_eval);
    assert!(
        history_size > 0 && max_iter > 0 && max_eval != Some(0),
        "LBFGS_HISTORY, LBFGS_MAX_ITER and LBFGS_MAX_EVAL must be positive"
    );
    let weight_decay = env_var("LBFGS_WEIGHT_DECAY").unwrap_or(arm.weight_decay);
    assert!(
        weight_decay.is_finite() && weight_decay >= 0.0,
        "LBFGS_WEIGHT_DECAY must be a finite number ≥ 0, got {weight_decay}"
    );
    LbfgsArm {
        lbfgs: arm
            .lbfgs
            .with_history_size(history_size)
            .with_max_iter(max_iter)
            .with_max_eval(max_eval),
        weight_decay,
        ..arm
    }
}

/// The SOAP arm of `lit` with the `SOAP_*` variables (its weight decay set by the caller).
fn soap(lit: &Literals) -> SoapConfig {
    let soap = lit
        .soap
        .clone()
        .expect("OPTIMIZER=soap needs the example's SOAP literals");
    let frequency = env_var("SOAP_FREQUENCY").unwrap_or(soap.precondition_frequency);
    assert!(frequency > 0, "SOAP_FREQUENCY must be positive");
    let log = env_var::<u8>("SOAP_LOG_FIRST_REFRESH").map_or(soap.log_first_refresh, |v| v == 1);
    soap.with_precondition_frequency(frequency)
        .with_log_first_refresh(log)
}

/// Gauss–Newton from `lit` and the `GN_*` variables, its probes seeded with `seed`.
fn gauss_newton(seed: u64, lit: &GnLiterals) -> GaussNewtonConfig {
    let preconditioner = match std::env::var("GN_PRECONDITIONER").as_deref() {
        Err(_) => lit.hutchinson,
        Ok("none") => false,
        Ok("hutchinson") => true,
        Ok(other) => panic!("GN_PRECONDITIONER must be none or hutchinson, got {other:?}"),
    };
    let preconditioner = if preconditioner {
        Preconditioner::HutchinsonDiagonal {
            probes: lit.hutchinson_probes,
        }
    } else {
        Preconditioner::None
    };
    let cg_max_iters = env_var("GN_CG_MAX_ITERS").unwrap_or(lit.cg_max_iters);
    assert!(cg_max_iters > 0, "GN_CG_MAX_ITERS must be positive");
    let weight_decay = env_var("GN_WEIGHT_DECAY").unwrap_or(lit.weight_decay);
    assert!(
        weight_decay.is_finite() && weight_decay >= 0.0,
        "GN_WEIGHT_DECAY must be a finite number ≥ 0, got {weight_decay}"
    );
    GaussNewtonConfig::new()
        .with_cg_max_iters(cg_max_iters)
        .with_lambda_init(lit.lambda_init)
        .with_weight_decay(weight_decay)
        .with_preconditioner(preconditioner)
        .with_adjoint_check_every(lit.adjoint_check_every)
        .with_seed(seed)
}

/// Epochs between checkpoints; the last epoch is always checkpointed (Phase 9 P2).
pub const CHECKPOINT_EVERY: usize = 25;

/// A run's directory, model, training and data (`C`) configs, and the checkpoint of a
/// rank-`R` FNO it resumes from (`None` for a new run).
pub type Session<const R: usize, C> = (
    RunDir,
    FNOConfig,
    TrainingConfig,
    C,
    Option<OperatorCheckpoint<FNO<R>>>,
);

/// The run to train: with `--resume <dir>` as the first arguments, that run reopened from
/// its checkpoint on `device` ([`reopen`]); otherwise `new_run()`, a new run whose configs
/// it has saved.
///
/// # Panics
/// As [`resume_dir`] and [`reopen`].
pub fn session<const R: usize, C: Config>(
    device: &Device,
    new_run: impl FnOnce() -> (RunDir, FNOConfig, TrainingConfig, C),
) -> Session<R, C> {
    match resume_dir() {
        Some(dir) => {
            let (run, model_cfg, train_cfg, data_cfg, ck) = reopen::<R, C>(&dir, device);
            (run, model_cfg, train_cfg, data_cfg, Some(ck))
        }
        None => {
            let (run, model_cfg, train_cfg, data_cfg) = new_run();
            (run, model_cfg, train_cfg, data_cfg, None)
        }
    }
}

/// The run directory given after `--resume` as the first argument, or `None` for a new
/// run when the first argument is anything else or absent.
///
/// # Panics
/// If `--resume` has no directory after it.
fn resume_dir() -> Option<String> {
    match std::env::args().nth(1).as_deref() {
        Some("--resume") => Some(
            std::env::args()
                .nth(2)
                .expect("--resume needs a run directory"),
        ),
        _ => None,
    }
}

/// Reopens the run at `dir` with its saved configs (data config `C`) and its checkpoint
/// of a rank-`R` FNO (on `device`), and truncates its `metrics.csv` (and `gn_steps.csv`)
/// to the checkpoint's epoch. The checkpoint counts this resume.
///
/// # Panics
/// If the run, a config or the checkpoint cannot be read, or the run has no checkpoint.
fn reopen<const R: usize, C: Config>(
    dir: &str,
    device: &Device,
) -> (
    RunDir,
    FNOConfig,
    TrainingConfig,
    C,
    OperatorCheckpoint<FNO<R>>,
) {
    let run = RunDir::open(dir).unwrap_or_else(|e| panic!("could not open the run: {e}"));
    let model_cfg = run.model_config().unwrap_or_else(|e| panic!("{e}"));
    let train_cfg = run.training_config().unwrap_or_else(|e| panic!("{e}"));
    let dataset_cfg: C = run.data_config().unwrap_or_else(|e| panic!("{e}"));
    let mut ck = run
        .checkpoint_of::<FNO<R>>(device)
        .unwrap_or_else(|e| panic!("could not read the checkpoint: {e}"))
        .unwrap_or_else(|| panic!("{dir} has no checkpoint to resume from"));
    ck.state.resumes += 1;
    run.truncate_metrics(ck.state.next_epoch)
        .unwrap_or_else(|e| panic!("could not truncate metrics.csv: {e}"));
    run.truncate_gn_steps(ck.state.next_epoch)
        .unwrap_or_else(|e| panic!("could not truncate gn_steps.csv: {e}"));
    run.truncate_lbfgs_steps(ck.state.next_epoch)
        .unwrap_or_else(|e| panic!("could not truncate lbfgs_steps.csv: {e}"));
    println!(
        "device: {device:?}, resuming {dir} at epoch {} of {} (resume {})",
        ck.state.next_epoch, train_cfg.epochs, ck.state.resumes
    );
    (run, model_cfg, train_cfg, dataset_cfg, ck)
}

/// What an example measured, for [`run_costs`].
pub struct Measured {
    /// Reading the splits and preparing them.
    pub load_time_s: f64,
    /// Training, every session's up to the last checkpoint and this one's.
    pub training_time_s: f64,
    /// Bytes uploaded for the train split (every device's copy).
    pub host_to_device_bytes_train: usize,
    /// Bytes uploaded for the test split.
    pub host_to_device_bytes_test: usize,
    /// The training device or devices.
    pub device: String,
    /// The model's parameters.
    pub num_params: usize,
    /// Times the run was resumed.
    pub resumes: usize,
    /// The commit the session started from, read before training.
    pub git_commit: String,
}

/// `costs.json` of a run trained on `n_train` samples, without dataset sidecars: `m` and
/// the steps and passes. Adam: every planned step (it takes no budget), one forward and
/// one backward each. Gauss–Newton: the steps taken, from `gn_steps.csv`, every session's
/// (a budget may stop it before the last epoch), plus a hybrid run's Adam steps, from the
/// epochs `metrics.csv` holds. L-BFGS: one step per epoch, from `lbfgs_steps.csv`, its
/// closure evaluations as passes (D7) plus one metrics forward per step, as Gauss–Newton's.
/// SOAP: every step of the epochs `metrics.csv` holds (a budget may stop it), one forward
/// and one backward each (D7).
///
/// # Panics
/// If a Gauss–Newton or SOAP run's `metrics.csv` or `gn_steps.csv` cannot be read, or it has no
/// `gn_steps.csv` and no Adam epochs; if an L-BFGS run's `lbfgs_steps.csv` is missing or
/// cannot be read.
pub fn run_costs(
    run: &RunDir,
    train_cfg: &TrainingConfig,
    n_train: usize,
    m: Measured,
) -> RunCosts {
    let lbfgs = train_cfg.lbfgs().map(|_| {
        run.lbfgs_passes()
            .expect("could not read lbfgs_steps.csv")
            .expect("an L-BFGS run writes lbfgs_steps.csv")
    });
    let (steps, forward_passes, backward_passes) = match train_cfg.gauss_newton() {
        None => match lbfgs {
            Some(p) => {
                let (forward, backward) = training_passes(&p);
                (p.steps, forward, backward)
            }
            None if train_cfg.soap().is_some() => {
                let epochs = run.metrics().expect("could not read metrics.csv").len();
                let steps = epochs * train_cfg.steps_per_epoch(n_train);
                (steps, steps, steps)
            }
            None => {
                let steps = train_cfg.total_steps(n_train);
                (steps, steps, steps)
            }
        },
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
    RunCosts {
        dataset_train_wall_time_s: None,
        dataset_test_wall_time_s: None,
        load_time_s: m.load_time_s,
        training_time_s: m.training_time_s,
        inference_time_s: None,
        host_to_device_bytes_train: m.host_to_device_bytes_train,
        host_to_device_bytes_test: m.host_to_device_bytes_test,
        // no backend reports it; never estimated (design §13.9)
        peak_gpu_memory: "unavailable".to_string(),
        backend_features: backend_features().into_iter().map(String::from).collect(),
        device: m.device,
        git_commit: m.git_commit,
        num_params: m.num_params,
        optimisation_steps: Some(steps),
        resumes: Some(m.resumes),
        forward_passes: Some(forward_passes),
        backward_passes: Some(backward_passes),
    }
}

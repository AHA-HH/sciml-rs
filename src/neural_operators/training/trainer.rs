//! The training loop, generic over the operator ([`GridOperator`], library CM0.3): the
//! components, one training epoch, the loop with its per-epoch observer, and [`train`],
//! the entry point every PDE trains through (`train::<FNO<R>, …>` for the FNO).

use burn::{
    data::dataloader::DataLoader,
    lr_scheduler::{cosine::CosineAnnealingLrSchedulerConfig, step::StepLrSchedulerConfig},
    optim::{
        AdamConfig, AdamWConfig, GradientsParams, LBFGS, ModuleOptimizer,
        decay::WeightDecayConfig,
        lr_scheduler::module_lr_scheduler::{ModuleLearningRate, ModuleLrScheduler},
    },
    prelude::*,
};

use std::sync::Arc;
use std::time::Instant;

use crate::neural_operators::{
    data::{
        batcher::Batch,
        dataset::OperatorDataset,
        device_batcher::device_loader,
        host_float::HostFloat,
        transforms::normalizers::{FlatDecoder, UnitGaussianNormalizer},
    },
    models::{fno::FNO, operator::GridOperator},
    optim::gauss_newton::GaussNewton,
    runs::{
        CheckpointState, EpochMetrics, GnStepRecord, LbfgsStepRecord, OperatorCheckpoint, RunDir,
    },
    training::{
        config::{LrSchedule, OptimizerConfig, TrainingConfig},
        data_parallel::{Peers, Replica, train_epoch_parallel},
        gn::train_epoch_gn,
        lbfgs::train_epoch_lbfgs,
        step::{StepLoss, evaluate, forward_loss, squared_rel_l2},
    },
};

/// One training epoch's losses, averaged as `EpochMetrics` reports them.
pub struct EpochMeans {
    /// MSE averaged over the epoch's batches (each batch's MSE is already a mean).
    pub train_mse: f32,
    /// Relative L² averaged over the epoch's samples.
    pub train_l2: f32,
    /// Learning rate after the epoch's final step. For a Gauss–Newton run (Phase 10 T3), which has no learning rate, the damping λ the
    /// next step will use.
    pub last_lr: f64,
}

/// `sum + x`, starting from the first `x`.
pub(crate) fn accumulate(sum: Option<Tensor<1>>, x: Tensor<1>) -> Option<Tensor<1>> {
    Some(match sum {
        Some(sum) => sum + x,
        None => x,
    })
}

/// Runs one training pass and returns its [`EpochMeans`]: the MSE divided by the batch
/// count, the relative L² by `loader.num_items()`. Takes and returns the model - Burn's
/// optimizer consumes it on each step.
///
/// `decoder` (on the loader's autodiff device) decodes the model output before the loss;
/// targets are in physical units (CONVENTIONS §10). `None` trains in physical units.
///
/// # Panics
///
/// If the loader yields no batches or a failed one, or if the model has more than one
/// output channel ([`flatten_pair`](crate::neural_operators::training::step::flatten_pair)).
pub fn train_epoch<const R: usize, const RM1: usize, M: GridOperator<R>>(
    model: M,
    loader: &Arc<dyn DataLoader<Batch<R, RM1>>>,
    optim: &mut ModuleOptimizer,
    scheduler: &mut ModuleLrScheduler,
    decoder: Option<&FlatDecoder>,
) -> (M, EpochMeans) {
    train_epoch_with_loss(model, loader, optim, scheduler, decoder, false)
}

/// [`train_epoch`], back-propagating the batch mean of the squared relative L² (PETScML's
/// eq. 6, optimisers CO1.1) instead of the summed relative L² when `squared_loss`; the
/// means it reports are the same either way.
fn train_epoch_with_loss<const R: usize, const RM1: usize, M: GridOperator<R>>(
    mut model: M,
    loader: &Arc<dyn DataLoader<Batch<R, RM1>>>,
    optim: &mut ModuleOptimizer,
    scheduler: &mut ModuleLrScheduler,
    decoder: Option<&FlatDecoder>,
    squared_loss: bool,
) -> (M, EpochMeans) {
    let mut mse_sum_tensor = None;
    let mut l2_sum_tensor = None;
    let mut n_batches = 0_usize;

    let mut last_lr = ModuleLearningRate::from(0.0_f64);

    let mut iterator = loader.iter();

    loop {
        let Some(batch) = iterator.next() else {
            break;
        };

        let batch = batch.expect("dataset error during training");

        let StepLoss {
            l2_sum: l2,
            out,
            target,
        } = forward_loss(&model, batch, decoder);

        let objective = if squared_loss {
            squared_rel_l2(out.clone(), target.clone())
        } else {
            l2.clone()
        };

        let mse = (out.inner() - target.inner()).powf_scalar(2.0).mean();

        let l2_metric = l2.clone().inner();

        mse_sum_tensor = accumulate(mse_sum_tensor, mse);
        l2_sum_tensor = accumulate(l2_sum_tensor, l2_metric);
        n_batches += 1;

        let grads = GradientsParams::from_grads(objective.backward(), &model);

        last_lr = scheduler.step();

        model = optim.step(last_lr.clone(), model, grads);
    }

    let mse_sum = mse_sum_tensor
        .expect("training loader produced no batches")
        .into_scalar::<f32>();

    let l2_sum = l2_sum_tensor
        .expect("training loader produced no batches")
        .into_scalar::<f32>();

    (
        model,
        EpochMeans {
            train_mse: mse_sum / n_batches as f32,
            train_l2: l2_sum / loader.num_items() as f32,
            last_lr: last_lr.base(),
        },
    )
}

/// Everything [`training_loop`] needs, built by [`build_components`]; `M` is the model,
/// the FNO by default.
pub struct TrainingComponents<const R: usize, const RM1: usize, M = FNO<R>> {
    /// The freshly initialised model, on the autodiff device.
    pub model: M,
    /// Adam with coupled weight decay, or AdamW or SOAP when `TrainingConfig::optimizer`
    /// asks for it.
    pub optim: ModuleOptimizer,
    /// Cosine annealing over [`TrainingConfig::schedule_steps`] steps, or the step schedule
    /// of [`TrainingConfig::lr_schedule`]; stepped once per batch.
    pub scheduler: ModuleLrScheduler,
    /// Shuffled training batches on the autodiff device.
    pub train_loader: Arc<dyn DataLoader<Batch<R, RM1>>>,
    /// Unshuffled test batches on the inner (non-autodiff) device.
    pub test_loader: Arc<dyn DataLoader<Batch<R, RM1>>>,
    /// The devices after the first of a data-parallel run, with their copies of the
    /// training split ([`replicas`](crate::neural_operators::training::data_parallel::replicas));
    /// empty, the default, for one device.
    pub replicas: Vec<Replica<R, RM1>>,
    /// The Gauss–Newton optimiser when `TrainingConfig::optimizer` asks for it (Phase 10
    /// T3); `None` trains with `optim` and `scheduler`, which a Gauss–Newton run leaves
    /// unused.
    pub gn: Option<GaussNewton>,
    /// Burn's L-BFGS when `TrainingConfig::optimizer` asks for it (optimisers CO2.1); like
    /// `gn`, it leaves `optim` and `scheduler` unused.
    pub lbfgs: Option<LBFGS>,
}

/// Builds the model (`M::init` from `model_cfg`, after seeding `device` with
/// `train_cfg.seed`), optimizer, schedule and loaders.
///
/// Both splits are uploaded to `device` once and stay there for the whole
/// run (see [`device_loader`]), so they must fit in device memory.
///
/// The schedule length comes from the training set's size, and the metric divisors from
/// the loaders ([`train_epoch`], [`evaluate`]), so both match the data actually trained on.
///
/// # Panics
///
/// If either dataset is empty, as `M::init`, if `train_cfg.adam_epochs` is set without
/// Gauss–Newton or outside 1..`epochs`, or if a step schedule's `every_epochs` is 0.
pub fn build_components<M: GridOperator<R>, const R: usize, const RM1: usize, T: HostFloat>(
    model_cfg: &M::Config,
    train_cfg: &TrainingConfig,
    train_data: OperatorDataset<T>,
    test_data: OperatorDataset<T>,
    device: &Device,
) -> TrainingComponents<R, RM1, M> {
    let (n_train, n_test) = (train_data.len(), test_data.len());
    assert!(
        n_train > 0 && n_test > 0,
        "training needs non-empty datasets, got {n_train} train / {n_test} test samples"
    );

    // a hybrid run trains both optimisers; 0 would also leave the cosine with no steps
    assert!(
        train_cfg.adam_epochs.is_none_or(|a| {
            train_cfg.gauss_newton().is_some() && 0 < a && a < train_cfg.epochs
        }),
        "adam_epochs needs Gauss-Newton and 0 < adam_epochs < epochs"
    );

    device.seed(train_cfg.seed);

    // Training tensors must live on the autodiff device.
    let train_device = device.clone();

    // Evaluation tensors must live on the plain inner backend,
    // because model.valid() returns an FNO on Metal rather than Autodiff<Metal>.
    let eval_device = device.clone().inner();

    let model = M::init(model_cfg, device);

    let optim: ModuleOptimizer = match train_cfg.optimizer {
        // ε as Optax's and our Adam's (Burn's default is 1e-5); the decay is lr-scaled
        Some(OptimizerConfig::AdamW) => AdamWConfig::new()
            .with_epsilon(1e-8)
            .with_weight_decay(train_cfg.weight_decay)
            .init(),
        Some(OptimizerConfig::Soap(ref soap)) => soap.init(),
        // a Gauss-Newton or L-BFGS run builds Adam too: a hybrid run's epochs, or unused
        Some(
            OptimizerConfig::Adam | OptimizerConfig::GaussNewton(_) | OptimizerConfig::Lbfgs(_),
        )
        | None => AdamConfig::new()
            .with_epsilon(1e-8)
            .with_weight_decay(Some(WeightDecayConfig::new(train_cfg.weight_decay)))
            .init(),
    };

    let scheduler = match train_cfg.lr_schedule {
        // stepped once per batch: halve after every `every_epochs` epochs' steps
        Some(LrSchedule::Step { every_epochs }) => StepLrSchedulerConfig::new(
            train_cfg.learning_rate,
            every_epochs * train_cfg.steps_per_epoch(n_train),
        )
        .with_gamma(0.5)
        .init()
        .expect("every_epochs must be at least 1"),
        None => CosineAnnealingLrSchedulerConfig::new(
            train_cfg.learning_rate,
            train_cfg.schedule_steps(n_train),
        )
        .with_min_lr(train_cfg.min_lr)
        .init()
        .expect("valid cosine scheduler config"),
    };

    // Each split is uploaded once and batched on the device (commit `40fcfd4`),
    // with the same order as a DataLoader over the host dataset. The host
    // copies are dropped on return.
    let train_loader = device_loader::<R, RM1, T>(
        &train_data,
        &train_device,
        train_cfg.batch_size,
        Some(train_cfg.seed),
    );

    let test_loader =
        device_loader::<R, RM1, T>(&test_data, &eval_device, train_cfg.test_batch_size, None);

    TrainingComponents {
        model,
        optim,
        scheduler,
        train_loader,
        test_loader,
        replicas: Vec::new(),
        gn: train_cfg
            .gauss_newton()
            .map(|cfg| GaussNewton::new(cfg.clone())),
        lbfgs: train_cfg.lbfgs().map(|arm| arm.lbfgs.init()),
    }
}

/// Per-epoch observer for [`training_loop`]: the epoch's metrics, including its wall time
/// (training and evaluation, `epoch_time_s`), and the model after the epoch.
///
/// A plain closure rather than a trait: call sites pass `&mut |m, model| ...`, and two
/// observers combine as two calls in one closure (see [`train_components`]). The model is
/// for checkpointing (Phase 5); nothing reads it yet. `M` is the model.
pub type EpochObserver<'a, M> = dyn FnMut(&EpochMetrics, &M) + 'a;

/// The default observer: prints the epoch's metrics, wall time included, to stdout, one
/// line per epoch.
pub fn print_epoch<M>(m: &EpochMetrics, _model: &M) {
    println!("{}", epoch_line(m));
}

fn epoch_line(m: &EpochMetrics) -> String {
    format!("{m:?}")
}

/// Runs epochs of [`train_epoch`] then [`evaluate`] up to `train_cfg.epochs`, calling
/// `on_epoch` after each, and returns the trained model and the metrics of the epochs it
/// ran. With `train_cfg.time_budget_s`, it stops early after the first epoch that ends
/// with the training time (this call's, plus the checkpoint's carried time) at or over the
/// budget, and checkpoints that epoch as the last.
///
/// `decoder`, on the training (autodiff) device, decodes the model output in both training
/// and evaluation ([`FlatDecoder::inner`] for the latter); `None` for data in physical units.
///
/// `checkpoints`, if given as `(run, k)`, writes the run's checkpoint
/// ([`RunDir::save_checkpoint`]) after every k-th epoch and after the last, once
/// `on_epoch` has seen the epoch; a failed write stops training (the atomic write leaves
/// the previous checkpoint intact, so the run resumes from there).
/// The loop saves the checkpoint itself rather than handing the optimiser and scheduler
/// to the observer: they are the loop's working state, and only the loop knows the epoch
/// count and the training time carried over from earlier sessions.
///
/// `resume` starts from a checkpoint ([`RunDir::checkpoint_of`]) instead of epoch 0: its
/// model replaces `components.model`, its records are loaded into the optimiser and
/// scheduler, and the training loader's shuffle is replayed for the epochs already done
/// (each `iter()` draws one epoch's order), so every epoch sees the order of an
/// uninterrupted run. `components` must be built from the same configs and data as the
/// checkpointed run. On flex the resumed run then matches the uninterrupted one bit for
/// bit.
///
/// With `components.gn`, each epoch is [`train_epoch_gn`] instead (Phase 10 T3), and the
/// checkpoints carry the Gauss–Newton state; `resume` must then hold one. A hybrid run's
/// first `train_cfg.adam_epochs` epochs are [`train_epoch`] (Phase 10 T5), its checkpoints
/// carrying both optimisers' states.
///
/// With `components.replicas`, each epoch is data-parallel
/// ([`data_parallel`](crate::neural_operators::training::data_parallel)): the replicas
/// start as copies of the (possibly resumed) model and optimiser, their loaders replay the
/// same shuffle, and evaluation, the observer and checkpoints see the first replica.
///
/// `training_loop` drops each Gauss–Newton step's row; [`train_components`] writes them to
/// `gn_steps.csv`.
///
/// Prints nothing itself: pass [`print_epoch`] (or a closure calling it) for the console
/// line.
///
/// # Panics
///
/// As [`train_epoch`] and [`evaluate`]; if `k` is 0; if a checkpoint cannot be written;
/// for Gauss–Newton with replicas (data-parallel Gauss–Newton is out of scope), or a
/// `resume` whose optimiser state does not match the components'.
pub fn training_loop<const R: usize, const RM1: usize, M: GridOperator<R>>(
    components: TrainingComponents<R, RM1, M>,
    train_cfg: &TrainingConfig,
    decoder: Option<&FlatDecoder>,
    on_epoch: &mut EpochObserver<'_, M>,
    checkpoints: Option<(&RunDir, usize)>,
    resume: Option<OperatorCheckpoint<M>>,
) -> (M, Vec<EpochMetrics>) {
    run_loop(
        components,
        train_cfg,
        decoder,
        on_epoch,
        &mut |_| {},
        checkpoints,
        resume,
    )
}

/// [`training_loop`] with `on_gn_step`, called after each Gauss–Newton step with its row
/// (never for Adam).
fn run_loop<const R: usize, const RM1: usize, M: GridOperator<R>>(
    components: TrainingComponents<R, RM1, M>,
    train_cfg: &TrainingConfig,
    decoder: Option<&FlatDecoder>,
    on_epoch: &mut EpochObserver<'_, M>,
    on_gn_step: &mut dyn FnMut(&GnStepRecord),
    checkpoints: Option<(&RunDir, usize)>,
    resume: Option<OperatorCheckpoint<M>>,
) -> (M, Vec<EpochMetrics>) {
    let sinks = StepSinks {
        gn: on_gn_step,
        lbfgs: &mut |_| {},
    };
    run_loop_with(
        components,
        train_cfg,
        decoder,
        on_epoch,
        sinks,
        checkpoints,
        resume,
    )
}

/// Where a run's per-step rows go: Gauss–Newton's (`gn_steps.csv`) and L-BFGS's
/// (`lbfgs_steps.csv`, optimisers CO2.1).
struct StepSinks<'a> {
    gn: &'a mut dyn FnMut(&GnStepRecord),
    lbfgs: &'a mut dyn FnMut(&LbfgsStepRecord),
}

/// [`run_loop`] with both step sinks: with `components.lbfgs` each epoch is
/// [`train_epoch_lbfgs`] (one device, no hybrid), its checkpoints carrying the L-BFGS
/// record, which `resume` must then hold.
fn run_loop_with<const R: usize, const RM1: usize, M: GridOperator<R>>(
    mut components: TrainingComponents<R, RM1, M>,
    train_cfg: &TrainingConfig,
    decoder: Option<&FlatDecoder>,
    on_epoch: &mut EpochObserver<'_, M>,
    sinks: StepSinks<'_>,
    checkpoints: Option<(&RunDir, usize)>,
    resume: Option<OperatorCheckpoint<M>>,
) -> (M, Vec<EpochMetrics>) {
    let StepSinks {
        gn: on_gn_step,
        lbfgs: on_lbfgs_step,
    } = sinks;
    assert!(
        components.gn.is_none() || components.replicas.is_empty(),
        "Gauss-Newton trains on one device: data-parallel Gauss-Newton is out of scope"
    );
    assert!(
        components.lbfgs.is_none() || components.replicas.is_empty(),
        "L-BFGS trains on one device: data-parallel L-BFGS is out of scope"
    );
    assert!(
        train_cfg.soap().is_none() || components.replicas.is_empty(),
        "SOAP trains on one device: data-parallel SOAP is out of scope"
    );
    assert!(
        !train_cfg.squared_loss() || components.replicas.is_empty(),
        "the squared loss trains on one device (optimisers CO1.1)"
    );
    // evaluation tensors live on the inner device (model.valid())
    let eval_decoder = decoder.map(FlatDecoder::inner);
    let mut model = components.model;
    let mut metrics = Vec::with_capacity(train_cfg.epochs);
    let mut state = CheckpointState::default();
    if let Some(ck) = resume {
        model = ck.model;
        components.optim = components.optim.load_record(ck.optim);
        components.scheduler = components.scheduler.load_record(ck.scheduler);
        components.gn = match (components.gn, ck.gn) {
            (None, None) => None,
            (Some(gn), Some(state)) => Some(GaussNewton::from_state(gn.config().clone(), state)),
            (Some(_), None) => panic!("a Gauss-Newton run's checkpoint has no Gauss-Newton state"),
            (None, Some(_)) => panic!("an Adam run's checkpoint has a Gauss-Newton state"),
        };
        components.lbfgs = match (components.lbfgs, ck.lbfgs) {
            (None, None) => None,
            (Some(lbfgs), Some(record)) => Some(lbfgs.load_record(record)),
            (Some(_), None) => panic!("an L-BFGS run's checkpoint has no L-BFGS state"),
            (None, Some(_)) => panic!("the checkpoint has an L-BFGS state the run does not"),
        };
        let loaders = std::iter::once(&components.train_loader)
            .chain(components.replicas.iter().map(|r| &r.train_loader));
        for loader in loaders {
            for _ in 0..ck.state.next_epoch {
                drop(loader.iter());
            }
        }
        state = ck.state;
    }
    let mut peers = (!components.replicas.is_empty())
        .then(|| Peers::new(&model, &components.optim, decoder, &components.replicas));
    let carried_time_s = state.training_time_s;
    let start = Instant::now();
    // a run resumed from the checkpoint its budget stopped it at has nothing left to train
    let first_epoch = if state.next_epoch > 0 && train_cfg.budget_spent(carried_time_s) {
        train_cfg.epochs
    } else {
        state.next_epoch
    };

    for epoch in first_epoch..train_cfg.epochs {
        let epoch_start = Instant::now();

        let gn = components
            .gn
            .as_mut()
            .filter(|_| !train_cfg.adam_epoch(epoch));
        let (m, means) = if let Some(lbfgs) = components.lbfgs.as_mut() {
            // an L-BFGS run has neither Gauss-Newton nor replicas
            let arm = train_cfg
                .lbfgs()
                .expect("L-BFGS components come from an L-BFGS config");
            train_epoch_lbfgs(
                model,
                &components.train_loader,
                lbfgs,
                arm,
                decoder,
                epoch,
                on_lbfgs_step,
            )
        } else {
            match (gn, peers.as_mut()) {
                (Some(gn), None) => train_epoch_gn(
                    model,
                    &components.train_loader,
                    gn,
                    decoder,
                    epoch,
                    on_gn_step,
                ),
                (None, None) => train_epoch_with_loss(
                    model,
                    &components.train_loader,
                    &mut components.optim,
                    &mut components.scheduler,
                    decoder,
                    train_cfg.squared_loss(),
                ),
                (Some(_), Some(_)) => unreachable!("Gauss-Newton with replicas is rejected above"),
                (None, Some(peers)) => train_epoch_parallel(
                    model,
                    &components.train_loader,
                    &mut components.optim,
                    &mut components.scheduler,
                    decoder,
                    peers,
                ),
            }
        };
        model = m;

        let test_l2 = evaluate(
            &model.valid(),
            &components.test_loader,
            eval_decoder.as_ref(),
        );

        let epoch_time = epoch_start.elapsed();

        let record = EpochMetrics {
            epoch,
            train_mse: means.train_mse,
            train_l2: means.train_l2,
            test_l2,
            current_lr: means.last_lr,
            epoch_time_s: Some(epoch_time.as_secs_f64()),
        };

        on_epoch(&record, &model);
        metrics.push(record);

        let next_epoch = epoch + 1;
        // one clock read: the stop decision and the checkpoint's time agree
        let trained_s = carried_time_s + start.elapsed().as_secs_f64();
        let last = next_epoch == train_cfg.epochs || train_cfg.budget_spent(trained_s);
        if let Some((run, every)) = checkpoints
            && (next_epoch % every == 0 || last)
        {
            state.next_epoch = next_epoch;
            state.training_time_s = trained_s;
            let saved = match &components.lbfgs {
                Some(lbfgs) => run.save_checkpoint_lbfgs(
                    &state,
                    &model,
                    &components.optim,
                    &components.scheduler,
                    lbfgs,
                ),
                None => run.save_checkpoint_gn(
                    &state,
                    &model,
                    &components.optim,
                    &components.scheduler,
                    components.gn.as_ref().map(GaussNewton::state),
                ),
            };
            if let Err(e) = saved {
                panic!("could not save the checkpoint after epoch {epoch}: {e}");
            }
        }
        if last {
            break;
        }
    }

    (model, metrics)
}

/// Trains an operator `M` on any dataset: builds the components ([`build_components`])
/// and runs [`training_loop`].
///
/// After each epoch it prints the console line ([`print_epoch`]) and appends the epoch's
/// row to `run`'s `metrics.csv` ([`RunDir::append_metrics`]), so the run directory must
/// exist before training. A failed append warns on stderr and training goes on; that row
/// is then missing from the file but not from the returned metrics. Prints the total
/// training time at the end.
///
/// `R` is the model rank (batch + spatial axes + channel) and `RM1 = R - 1` the target
/// rank: `<3, 2>` for 1D (Burgers), `<4, 3>` for 2D (Darcy, Poisson). Inputs are
/// `[n, ...spatial, 1]` and targets `[n, ...spatial]`.
///
/// Both target splits are in physical units (CONVENTIONS §10). `y_norm` is the target
/// normaliser, if any:
/// - `None`: the output is compared as is, for data trained in physical units (Burgers).
/// - `Some(y)`: the output is decoded with `y` before the loss, in training and evaluation.
///
/// # Panics
///
/// - From [`build_components`]: if either dataset is empty, or as `M::init`.
/// - At the first step, if the model has more than one output channel
///   ([`flatten_pair`](crate::neural_operators::training::step::flatten_pair)).
pub fn train<M: GridOperator<R>, const R: usize, const RM1: usize, T: HostFloat>(
    train_data: OperatorDataset<T>,
    test_data: OperatorDataset<T>,
    y_norm: Option<&UnitGaussianNormalizer>,
    model_cfg: &M::Config,
    train_cfg: &TrainingConfig,
    device: &Device,
    run: &RunDir,
) -> (M, Vec<EpochMetrics>) {
    let components =
        build_components::<M, R, RM1, T>(model_cfg, train_cfg, train_data, test_data, device);
    train_components(components, y_norm, train_cfg, device, run, None, None, None)
}

/// [`train`] after [`build_components`], with checkpoints and resume: prints
/// each epoch's line, appends its `metrics.csv` row to `run`, and runs [`training_loop`].
///
/// `checkpoint_every`, if `Some(k)`, writes `run`'s checkpoint after every k-th epoch and
/// after the last; `resume` continues from a checkpoint read with [`RunDir::checkpoint_of`]
/// (see [`training_loop`]). The caller truncates `metrics.csv` to the checkpoint first
/// ([`RunDir::truncate_metrics`], and [`RunDir::truncate_gn_steps`] for Gauss–Newton).
/// A Gauss–Newton run appends each step's row to `gn_steps.csv`
/// ([`RunDir::append_gn_step`]); a failed append warns, as for `metrics.csv`. `also`, if
/// given, is called after each epoch's line
/// and row (live MLflow logging in `train_poisson`, Phase 8 T1). Returns the metrics of
/// the epochs this call ran.
///
/// # Panics
///
/// As [`training_loop`]; at the first step, if the model has more than one output channel.
// one more than clippy's 7: the optional observer (Phase 8 T1) joins the run's own
// arguments; a struct for them would change every caller for one optional hook
#[allow(clippy::too_many_arguments)]
pub fn train_components<const R: usize, const RM1: usize, M: GridOperator<R>>(
    components: TrainingComponents<R, RM1, M>,
    y_norm: Option<&UnitGaussianNormalizer>,
    train_cfg: &TrainingConfig,
    device: &Device,
    run: &RunDir,
    checkpoint_every: Option<usize>,
    resume: Option<OperatorCheckpoint<M>>,
    mut also: Option<&mut EpochObserver<'_, M>>,
) -> (M, Vec<EpochMetrics>) {
    let start = Instant::now();
    let mut on_epoch = |m: &EpochMetrics, model: &M| {
        print_epoch(m, model);
        if let Err(e) = run.append_metrics(m) {
            eprintln!(
                "warning: could not append epoch {} to metrics.csv ({e})",
                m.epoch
            );
        }
        if let Some(also) = also.as_mut() {
            also(m, model);
        }
    };
    let mut on_gn_step = |r: &GnStepRecord| {
        if let Err(e) = run.append_gn_step(r) {
            eprintln!(
                "warning: could not append step {} to gn_steps.csv ({e})",
                r.step.step
            );
        }
    };
    let mut on_lbfgs_step = |r: &LbfgsStepRecord| {
        if let Err(e) = run.append_lbfgs_step(r) {
            eprintln!(
                "warning: could not append epoch {} to lbfgs_steps.csv ({e})",
                r.epoch
            );
        }
    };
    let decoder = y_norm.map(|y| FlatDecoder::new(y, device));
    let sinks = StepSinks {
        gn: &mut on_gn_step,
        lbfgs: &mut on_lbfgs_step,
    };
    let trained = run_loop_with(
        components,
        train_cfg,
        decoder.as_ref(),
        &mut on_epoch,
        sinks,
        checkpoint_every.map(|k| (run, k)),
        resume,
    );
    println!("total training time: {:.2?}", start.elapsed());
    trained
}

/// [`train`] after the build, for the FNO: builds the decoder for `y_norm` and runs the loop. Split
/// out so tests can give two runs the same initial model.
#[cfg(test)]
fn train_built<const R: usize, const RM1: usize>(
    components: TrainingComponents<R, RM1>,
    y_norm: Option<&UnitGaussianNormalizer>,
    train_cfg: &TrainingConfig,
    device: &Device,
    on_epoch: &mut EpochObserver<'_, FNO<R>>,
) -> (FNO<R>, Vec<EpochMetrics>) {
    let decoder = y_norm.map(|y| FlatDecoder::new(y, device));
    training_loop(
        components,
        train_cfg,
        decoder.as_ref(),
        on_epoch,
        None,
        None,
    )
}

#[cfg(test)]
mod tests;

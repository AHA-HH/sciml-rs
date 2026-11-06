//! Generic FNO training loop: the components, one training epoch, the loop with its
//! per-epoch observer, and [`train_fno`], the one entry point every PDE trains through.

use burn::{
    data::dataloader::DataLoader,
    lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig,
    module::Module,
    optim::{
        AdamConfig, GradientsParams, ModuleOptimizer,
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
    models::fno::{FNO, FNOConfig},
    runs::{Checkpoint, CheckpointState, EpochMetrics, RunDir},
    training::{
        config::TrainingConfig,
        data_parallel::{Peers, Replica, train_epoch_parallel},
        step::{StepLoss, evaluate, forward_loss},
    },
};

/// One training epoch's losses, averaged as `EpochMetrics` reports them.
pub struct EpochMeans {
    /// MSE averaged over the epoch's batches (each batch's MSE is already a mean).
    pub train_mse: f32,
    /// Relative L² averaged over the epoch's samples.
    pub train_l2: f32,
    /// Learning rate after the epoch's final step.
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
pub fn train_epoch<const R: usize, const RM1: usize>(
    mut model: FNO<R>,
    loader: &Arc<dyn DataLoader<Batch<R, RM1>>>,
    optim: &mut ModuleOptimizer,
    scheduler: &mut ModuleLrScheduler,
    decoder: Option<&FlatDecoder>,
) -> (FNO<R>, EpochMeans) {
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

        let mse = (out.inner() - target.inner()).powf_scalar(2.0).mean();

        let l2_metric = l2.clone().inner();

        mse_sum_tensor = accumulate(mse_sum_tensor, mse);
        l2_sum_tensor = accumulate(l2_sum_tensor, l2_metric);
        n_batches += 1;

        let grads = GradientsParams::from_grads(l2.backward(), &model);

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

/// Everything [`training_loop`] needs, built by [`build_training_components`].
pub struct TrainingComponents<const R: usize, const RM1: usize> {
    /// The freshly initialised model, on the autodiff device.
    pub model: FNO<R>,
    /// Adam with weight decay.
    pub optim: ModuleOptimizer,
    /// Cosine annealing over [`TrainingConfig::total_steps`] steps, stepped once per batch.
    pub scheduler: ModuleLrScheduler,
    /// Shuffled training batches on the autodiff device.
    pub train_loader: Arc<dyn DataLoader<Batch<R, RM1>>>,
    /// Unshuffled test batches on the inner (non-autodiff) device.
    pub test_loader: Arc<dyn DataLoader<Batch<R, RM1>>>,
    /// The devices after the first of a data-parallel run, with their copies of the
    /// training split ([`replicas`](crate::neural_operators::training::data_parallel::replicas));
    /// empty, the default, for one device.
    pub replicas: Vec<Replica<R, RM1>>,
}

/// Builds the model, optimizer, schedule and loaders.
///
/// Both splits are uploaded to `device` once and stay there for the whole
/// run (see [`device_loader`]), so they must fit in device memory.
///
/// The schedule length comes from the training set's size, and the metric divisors from
/// the loaders ([`train_epoch`], [`evaluate`]), so both match the data actually trained on.
///
/// # Panics
///
/// If either dataset is empty, or if `model_cfg.modes.len() + 2 != R` (`FNOConfig::init`).
pub fn build_training_components<const R: usize, const RM1: usize, T: HostFloat>(
    model_cfg: &FNOConfig,
    train_cfg: &TrainingConfig,
    train_data: OperatorDataset<T>,
    test_data: OperatorDataset<T>,
    device: &Device,
) -> TrainingComponents<R, RM1> {
    let (n_train, n_test) = (train_data.len(), test_data.len());
    assert!(
        n_train > 0 && n_test > 0,
        "training needs non-empty datasets, got {n_train} train / {n_test} test samples"
    );

    device.seed(train_cfg.seed);

    // Training tensors must live on the autodiff device.
    let train_device = device.clone();

    // Evaluation tensors must live on the plain inner backend,
    // because model.valid() returns an FNO on Metal rather than Autodiff<Metal>.
    let eval_device = device.clone().inner();

    // `init` checks `modes.len() + 2 == R`.
    let model = model_cfg.init::<R>(device);

    let optim: ModuleOptimizer = AdamConfig::new()
        .with_epsilon(1e-8)
        .with_weight_decay(Some(WeightDecayConfig::new(train_cfg.weight_decay)))
        .init();

    let scheduler = CosineAnnealingLrSchedulerConfig::new(
        train_cfg.learning_rate,
        train_cfg.total_steps(n_train),
    )
    .with_min_lr(train_cfg.min_lr)
    .init()
    .expect("valid cosine scheduler config");

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
    }
}

/// Per-epoch observer for [`training_loop`]: the epoch's metrics, including its wall time
/// (training and evaluation, `epoch_time_s`), and the model after the epoch.
///
/// A plain closure rather than a trait: call sites pass `&mut |m, model| ...`, and two
/// observers combine as two calls in one closure (see [`train_fno`]). The model is for
/// checkpointing (Phase 5); nothing reads it yet.
pub type EpochObserver<'a, const R: usize> = dyn FnMut(&EpochMetrics, &FNO<R>) + 'a;

/// The default observer: prints the epoch's metrics, wall time included, to stdout, one
/// line per epoch.
pub fn print_epoch<const R: usize>(m: &EpochMetrics, _model: &FNO<R>) {
    println!("{}", epoch_line(m));
}

fn epoch_line(m: &EpochMetrics) -> String {
    format!("{m:?}")
}

/// Runs epochs of [`train_epoch`] then [`evaluate`] up to `train_cfg.epochs`, calling
/// `on_epoch` after each, and returns the trained model and the metrics of the epochs it
/// ran.
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
/// `resume` starts from a checkpoint ([`RunDir::checkpoint`]) instead of epoch 0: its
/// model replaces `components.model`, its records are loaded into the optimiser and
/// scheduler, and the training loader's shuffle is replayed for the epochs already done
/// (each `iter()` draws one epoch's order), so every epoch sees the order of an
/// uninterrupted run. `components` must be built from the same configs and data as the
/// checkpointed run. On flex the resumed run then matches the uninterrupted one bit for
/// bit.
///
/// With `components.replicas`, each epoch is data-parallel
/// ([`data_parallel`](crate::neural_operators::training::data_parallel)): the replicas
/// start as copies of the (possibly resumed) model and optimiser, their loaders replay the
/// same shuffle, and evaluation, the observer and checkpoints see the first replica.
///
/// Prints nothing itself: pass [`print_epoch`] (or a closure calling it) for the console
/// line.
///
/// # Panics
///
/// As [`train_epoch`] and [`evaluate`]; if `k` is 0; if a checkpoint cannot be written.
pub fn training_loop<const R: usize, const RM1: usize>(
    mut components: TrainingComponents<R, RM1>,
    train_cfg: &TrainingConfig,
    decoder: Option<&FlatDecoder>,
    on_epoch: &mut EpochObserver<'_, R>,
    checkpoints: Option<(&RunDir, usize)>,
    resume: Option<Checkpoint<R>>,
) -> (FNO<R>, Vec<EpochMetrics>) {
    // evaluation tensors live on the inner device (model.valid())
    let eval_decoder = decoder.map(FlatDecoder::inner);
    let mut model = components.model;
    let mut metrics = Vec::with_capacity(train_cfg.epochs);
    let mut state = CheckpointState::default();
    if let Some(ck) = resume {
        model = ck.model;
        components.optim = components.optim.load_record(ck.optim);
        components.scheduler = components.scheduler.load_record(ck.scheduler);
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

    for epoch in state.next_epoch..train_cfg.epochs {
        let epoch_start = Instant::now();

        let (m, means) = match peers.as_mut() {
            None => train_epoch(
                model,
                &components.train_loader,
                &mut components.optim,
                &mut components.scheduler,
                decoder,
            ),
            Some(peers) => train_epoch_parallel(
                model,
                &components.train_loader,
                &mut components.optim,
                &mut components.scheduler,
                decoder,
                peers,
            ),
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

        if let Some((run, every)) = checkpoints {
            let next_epoch = epoch + 1;
            if next_epoch % every == 0 || next_epoch == train_cfg.epochs {
                state.next_epoch = next_epoch;
                state.training_time_s = carried_time_s + start.elapsed().as_secs_f64();
                if let Err(e) =
                    run.save_checkpoint(&state, &model, &components.optim, &components.scheduler)
                {
                    panic!("could not save the checkpoint after epoch {epoch}: {e}");
                }
            }
        }
    }

    (model, metrics)
}

/// Trains an FNO on any dataset: builds the components and runs [`training_loop`].
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
/// - From [`build_training_components`]: if either dataset is empty, or if
///   `model_cfg.modes.len() + 2 != R` (`FNOConfig::init`).
/// - At the first step, if `model_cfg.out_channels != 1`
///   ([`flatten_pair`](crate::neural_operators::training::step::flatten_pair)).
pub fn train_fno<const R: usize, const RM1: usize, T: HostFloat>(
    train_data: OperatorDataset<T>,
    test_data: OperatorDataset<T>,
    y_norm: Option<&UnitGaussianNormalizer>,
    model_cfg: &FNOConfig,
    train_cfg: &TrainingConfig,
    device: &Device,
    run: &RunDir,
) -> (FNO<R>, Vec<EpochMetrics>) {
    let components =
        build_training_components::<R, RM1, T>(model_cfg, train_cfg, train_data, test_data, device);
    train_components(components, y_norm, train_cfg, device, run, None, None, None)
}

/// [`train_fno`] after [`build_training_components`], with checkpoints and resume: prints
/// each epoch's line, appends its `metrics.csv` row to `run`, and runs [`training_loop`].
///
/// `checkpoint_every`, if `Some(k)`, writes `run`'s checkpoint after every k-th epoch and
/// after the last; `resume` continues from a checkpoint read with [`RunDir::checkpoint`]
/// (see [`training_loop`]). The caller truncates `metrics.csv` to the checkpoint first
/// ([`RunDir::truncate_metrics`]). `also`, if given, is called after each epoch's line
/// and row (live MLflow logging in `train_poisson`, Phase 8 T1). Returns the metrics of
/// the epochs this call ran.
///
/// # Panics
///
/// As [`training_loop`]; at the first step, if the model has more than one output channel.
// one more than clippy's 7: the optional observer (Phase 8 T1) joins the run's own
// arguments; a struct for them would change every caller for one optional hook
#[allow(clippy::too_many_arguments)]
pub fn train_components<const R: usize, const RM1: usize>(
    components: TrainingComponents<R, RM1>,
    y_norm: Option<&UnitGaussianNormalizer>,
    train_cfg: &TrainingConfig,
    device: &Device,
    run: &RunDir,
    checkpoint_every: Option<usize>,
    resume: Option<Checkpoint<R>>,
    mut also: Option<&mut EpochObserver<'_, R>>,
) -> (FNO<R>, Vec<EpochMetrics>) {
    let start = Instant::now();
    let mut on_epoch = |m: &EpochMetrics, model: &FNO<R>| {
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
    let decoder = y_norm.map(|y| FlatDecoder::new(y, device));
    let trained = training_loop(
        components,
        train_cfg,
        decoder.as_ref(),
        &mut on_epoch,
        checkpoint_every.map(|k| (run, k)),
        resume,
    );
    println!("total training time: {:.2?}", start.elapsed());
    trained
}

/// [`train_fno`] after the build: builds the decoder for `y_norm` and runs the loop. Split
/// out so tests can give two runs the same initial model.
#[cfg(test)]
fn train_built<const R: usize, const RM1: usize>(
    components: TrainingComponents<R, RM1>,
    y_norm: Option<&UnitGaussianNormalizer>,
    train_cfg: &TrainingConfig,
    device: &Device,
    on_epoch: &mut EpochObserver<'_, R>,
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
mod tests {
    use super::*;
    use crate::neural_operators::data::host_oracle::{HostItems, OperatorBatcher};
    use crate::neural_operators::training::fixtures::{
        tiny_dataset, tiny_dataset_2d, tiny_model_cfg,
    };
    use crate::neural_operators::{
        chebyshev::nodes,
        data::loaders::{
            base_dataset::DatasetConfig,
            poisson::{PoissonConfig, load_poisson_uniform},
        },
        losses::data_losses::{LpLoss, Reduction},
        training::{data_parallel, step::flatten_pair},
    };
    use burn::data::dataloader::DataLoaderBuilder;
    use burn::store::ModuleSnapshot;
    use ndarray::{Array1, Array3, ArrayD};
    use ndarray_npy::NpzWriter;
    use std::fs::{self, File};
    use std::path::{Path, PathBuf};

    // --- merge `04ec800`: schedule length and divisors come from the data ---

    #[test]
    fn components_take_sample_counts_from_the_datasets() {
        let train_cfg = TrainingConfig::new().with_batch_size(2);
        let c = build_training_components::<3, 2, _>(
            &tiny_model_cfg(),
            &train_cfg,
            tiny_dataset(5),
            tiny_dataset(3),
            &Device::default().autodiff(),
        );
        assert_eq!(
            (c.train_loader.num_items(), c.test_loader.num_items()),
            (5, 3)
        );
    }

    #[test]
    fn schedule_and_test_divisor_match_the_data() {
        let (n_train, n_test, epochs) = (5, 3, 2);
        let (lr, min_lr) = (1e-3, 1e-5);
        let train_cfg = TrainingConfig::new()
            .with_epochs(epochs)
            .with_batch_size(2)
            .with_test_batch_size(2)
            .with_learning_rate(lr)
            .with_min_lr(min_lr);
        let components = build_training_components::<3, 2, _>(
            &tiny_model_cfg(),
            &train_cfg,
            tiny_dataset(n_train),
            tiny_dataset(n_test),
            &Device::default().autodiff(),
        );
        let (model, metrics) =
            training_loop(components, &train_cfg, None, &mut |_, _| {}, None, None);

        // Cosine over N = epochs * ceil(5 / 2) = 6 steps: step k (1-based)
        // gives min + (max - min)(1 + cos((k - 1)π / N)) / 2.
        let n = (epochs * 3) as f64;
        let expected_lr =
            min_lr + 0.5 * (lr - min_lr) * (1.0 + ((n - 1.0) / n * std::f64::consts::PI).cos());
        let last_lr = metrics.last().unwrap().current_lr;
        assert!(
            (last_lr - expected_lr).abs() < 1e-15,
            "{last_lr} != {expected_lr}"
        );

        // test_l2 is the summed relative L2 of the final model over the 3 test
        // samples, divided by 3.
        let loader = DataLoaderBuilder::new(OperatorBatcher::<3, 2>::new())
            .set_device(Device::default())
            .batch_size(2)
            .build(HostItems(tiny_dataset(n_test)));
        let expected = evaluate(&model.valid(), &loader, None);
        let test_l2 = metrics.last().unwrap().test_l2;
        assert!(
            (test_l2 - expected).abs() <= 1e-6 * expected.abs(),
            "{test_l2} != {expected}"
        );
    }

    // --- commit `40fcfd4`: device-resident loaders change nothing numerically ---

    /// Full `training_loop` with the device-resident loaders against the same
    /// components with the host loaders from before commit `40fcfd4` swapped in:
    /// every metric of every epoch must be exactly equal (same batches, same
    /// order, same kernels), as must the trained model's output on flex.
    ///
    /// On the fused GPU backends the output may differ in the last bits: the
    /// fuser can compile the two runs' identical op sequences into different
    /// kernels (seen on metal: one value 2 ulp apart, exact with fusion off),
    /// so there it is compared to a relative 1e-6.
    #[test]
    fn device_loaders_train_identically_to_host_loaders() {
        let (n_train, n_test) = (7, 3); // partial last batch in both splits
        let train_cfg = TrainingConfig::new()
            .with_epochs(3)
            .with_batch_size(2)
            .with_test_batch_size(2)
            .with_seed(11);
        let device = Device::default().autodiff();
        let build = || {
            build_training_components::<3, 2, _>(
                &tiny_model_cfg(),
                &train_cfg,
                tiny_dataset(n_train),
                tiny_dataset(n_test),
                &device,
            )
        };

        // One initial model for both runs: tests run in parallel and share the
        // device RNG, so a second seeded `init` isn't guaranteed to match.
        let device_run = build();
        let mut host = build();
        host.model = device_run.model.clone();

        let (new_model, new) =
            training_loop(device_run, &train_cfg, None, &mut |_, _| {}, None, None);

        host.train_loader = DataLoaderBuilder::new(OperatorBatcher::<3, 2>::new())
            .set_device(device.clone())
            .batch_size(train_cfg.batch_size)
            .shuffle(train_cfg.seed)
            .build(HostItems(tiny_dataset(n_train)));
        host.test_loader = DataLoaderBuilder::new(OperatorBatcher::<3, 2>::new())
            .set_device(device.clone().inner())
            .batch_size(train_cfg.test_batch_size)
            .build(HostItems(tiny_dataset(n_test)));
        let (old_model, old) = training_loop(host, &train_cfg, None, &mut |_, _| {}, None, None);

        assert_same_metrics(&new, &old);

        let probe = |m: &FNO<3>| -> Vec<f32> {
            let x = tiny_dataset(1);
            let x = Tensor::<3>::from_data(
                TensorData::new(x.inputs().iter().map(|&v| v as f32).collect(), [1, 8, 1]),
                &device.clone().inner(),
            );
            m.valid().forward(x).into_data().iter::<f32>().collect()
        };
        let (new_out, old_out) = (probe(&new_model), probe(&old_model));

        #[cfg(feature = "flex")]
        let exact = device.clone().inner() == Device::flex();
        #[cfg(not(feature = "flex"))]
        let exact = false;

        if exact {
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&new_out), bits(&old_out));
        } else {
            for (i, (n, o)) in new_out.iter().zip(&old_out).enumerate() {
                assert!((n - o).abs() <= 1e-6 * o.abs().max(1.0), "[{i}] {n} != {o}");
            }
        }
    }

    // --- Phase 4s T3: `train_fno` reproduces the per-PDE trainers it replaced ---

    /// Every metric of every epoch, as bits.
    fn metric_bits(metrics: &[EpochMetrics]) -> Vec<(u32, u32, u32, u64)> {
        metrics
            .iter()
            .map(|m| {
                (
                    m.train_mse.to_bits(),
                    m.train_l2.to_bits(),
                    m.test_l2.to_bits(),
                    m.current_lr.to_bits(),
                )
            })
            .collect()
    }

    /// `None` against the old `train_burgers` body: no decoding on either side. Both runs
    /// start from one initial model (the device RNG is shared between parallel tests).
    #[test]
    fn train_fno_none_matches_identity_path() {
        let train_cfg = TrainingConfig::new()
            .with_epochs(2)
            .with_batch_size(2)
            .with_test_batch_size(2);
        let device = Device::default().autodiff();
        let build = || {
            build_training_components::<3, 2, _>(
                &tiny_model_cfg(),
                &train_cfg,
                tiny_dataset(5),
                tiny_dataset(3),
                &device,
            )
        };
        let new = build();
        let mut old = build();
        old.model = new.model.clone();

        let (_, new) = train_built(new, None, &train_cfg, &device, &mut |_, _| {});
        let (_, old) = training_loop(old, &train_cfg, None, &mut |_, _| {}, None, None);

        assert_same_metrics(&new, &old);
    }

    // --- Phase 4s T17: y_train stored in physical units ---

    /// One epoch of `train_fno`'s `Some(y)` path on a physical `y_train` (only the
    /// prediction decoded) against the old asymmetric path, inlined here as the oracle:
    /// `y_train` encoded at load, prediction and target both decoded in training, the
    /// prediction only in evaluation. Same initial model and batches, so the metrics differ
    /// only by f32 round-off (the target is no longer encoded and decoded).
    #[test]
    fn train_loss_matches_old_asymmetric_path() {
        let (n_train, n_test) = (5, 3);
        let train_cfg = TrainingConfig::new()
            .with_epochs(1)
            .with_batch_size(2)
            .with_test_batch_size(2);
        let model_cfg = FNOConfig::new(vec![2, 2], 1, 1)
            .with_hidden_channels(4)
            .with_n_layers(1);
        let physical = tiny_dataset_2d(n_train);
        let y = UnitGaussianNormalizer::fit(physical.targets());
        let encoded = || {
            OperatorDataset::new(
                physical.inputs().clone(),
                y.encode(physical.targets().clone()),
            )
        };
        let device = Device::default().autodiff();
        let build = |train| {
            build_training_components::<4, 3, _>(
                &model_cfg,
                &train_cfg,
                train,
                tiny_dataset_2d(n_test),
                &device,
            )
        };
        let mut old = build(encoded());
        let new = build(physical);
        old.model = new.model.clone();

        let (_, new) = train_built(new, Some(&y), &train_cfg, &device, &mut |_, _| {});

        // The old path: decode both sides in training.
        let loss = LpLoss::new(2, Reduction::Sum);
        let decoder = FlatDecoder::new(&y, &device);
        let mut model = old.model;
        let (mut mse_sum, mut l2_sum, mut n_batches) = (0.0_f32, 0.0_f32, 0);
        for batch in old.train_loader.iter() {
            let batch = batch.expect("dataset error");
            let (out, target) = flatten_pair::<4, 3>(model.forward(batch.inputs), batch.targets);
            let (out, target) = (decoder.decode(out), decoder.decode(target));
            let l2 = loss.rel(out.clone(), target.clone());
            mse_sum += (out.inner() - target.inner())
                .powf_scalar(2.0)
                .mean()
                .into_scalar::<f32>();
            l2_sum += l2.clone().inner().into_scalar::<f32>();
            n_batches += 1;
            let grads = GradientsParams::from_grads(l2.backward(), &model);
            let lr = old.scheduler.step();
            model = old.optim.step(lr, model, grads);
        }
        // ... and the prediction only in evaluation.
        let eval_decoder = FlatDecoder::new(&y, &device.clone().inner());
        let valid = model.valid();
        let mut test_sum = 0.0_f32;
        for batch in old.test_loader.iter() {
            let batch = batch.expect("dataset error");
            let (out, target) = flatten_pair::<4, 3>(valid.forward(batch.inputs), batch.targets);
            test_sum += loss
                .rel(eval_decoder.decode(out), target)
                .into_scalar::<f32>();
        }

        let close = |name: &str, new: f32, old: f32| {
            assert!(
                (new - old).abs() <= 1e-6 * old.abs(),
                "{name}: {new} != {old}"
            );
        };
        let m = &new[0];
        close("train_mse", m.train_mse, mse_sum / n_batches as f32);
        close("train_l2", m.train_l2, l2_sum / n_train as f32);
        close("test_l2", m.test_l2, test_sum / n_test as f32);
    }

    #[test]
    #[should_panic(expected = "non-empty datasets")]
    fn empty_training_set_is_rejected() {
        let _ = build_training_components::<3, 2, _>(
            &tiny_model_cfg(),
            &TrainingConfig::new(),
            tiny_dataset(0),
            tiny_dataset(3),
            &Device::default().autodiff(),
        );
    }

    // --- moved from training/trainers/poisson.rs (Phase 4s T3) ---

    const POISSON_N: usize = 9;
    const POISSON_N_TRAIN: usize = 4;
    const POISSON_N_TEST: usize = 2;

    /// Writes `count` samples of `f` = c_i · p(x, y) and `u` = −2 f on the n = 9 CGL
    /// grid to `dir/name`, as the generator lays them out. c_i differs per sample so the
    /// pointwise std is nonzero.
    fn write_poisson_split(dir: &Path, name: &str, count: usize, offset: f64) -> PathBuf {
        let x: Array1<f64> = nodes(POISSON_N);
        let f: ArrayD<f64> = Array3::from_shape_fn((count, POISSON_N, POISSON_N), |(i, a, b)| {
            (1.0 + i as f64 + offset) * (1.0 + x[a] - 2.0 * x[a].powi(3)) * (x[b] * x[b] - 0.3)
        })
        .into_dyn();
        let path = dir.join(name);
        let mut npz = NpzWriter::new(File::create(&path).unwrap());
        npz.add_array("f", &f).unwrap();
        npz.add_array("u", &(&f * -2.0)).unwrap();
        npz.add_array("x", &x).unwrap();
        npz.add_array("y", &x).unwrap();
        npz.finish().unwrap();
        path
    }

    #[test]
    fn train_poisson_smoke() {
        let dir = std::env::temp_dir().join(format!(
            "sciml_rs_train_poisson_smoke_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let train_path = write_poisson_split(&dir, "train.npz", POISSON_N_TRAIN, 0.0);
        let test_path = write_poisson_split(&dir, "test.npz", POISSON_N_TEST, 0.5);

        let cfg = PoissonConfig::new(
            DatasetConfig {
                n_train: POISSON_N_TRAIN,
                n_test: POISSON_N_TEST,
            },
            POISSON_N,
        );
        let (train, test, normalizers) =
            load_poisson_uniform::<f32>(&train_path, &test_path, &cfg).unwrap();

        let model_cfg = FNOConfig::new(vec![2, 2], 1, 1)
            .with_hidden_channels(4)
            .with_n_layers(2);
        let train_cfg = TrainingConfig::new()
            .with_epochs(2)
            .with_batch_size(2)
            .with_test_batch_size(2);
        let (_, metrics) = train_fno::<4, 3, _>(
            train,
            test,
            Some(&normalizers.y),
            &model_cfg,
            &train_cfg,
            &Device::default().autodiff(),
            &RunDir::create(&dir, "smoke").unwrap(),
        );

        let finite = |metrics: &[EpochMetrics]| {
            for m in metrics {
                assert!(
                    m.train_mse.is_finite() && m.train_l2.is_finite() && m.test_l2.is_finite(),
                    "{m:?}"
                );
            }
        };
        assert_eq!(metrics.len(), 2);
        finite(&metrics);

        // Phase 9 T1: stop after epoch 1 (its checkpoint written; epoch 2 killed after its
        // metrics row) and resume to epoch 2 through the checkpoint, as train_poisson does
        let device = Device::default().autodiff();
        let run = RunDir::create(&dir, "smoke_resume").unwrap();
        run.save_configs(&model_cfg, &train_cfg, &cfg).unwrap();
        let build = || {
            let (train, test, _) =
                load_poisson_uniform::<f32>(&train_path, &test_path, &cfg).unwrap();
            build_training_components::<4, 3, _>(&model_cfg, &train_cfg, train, test, &device)
        };
        let decoder = FlatDecoder::new(&normalizers.y, &device);
        interrupt_at(build(), &train_cfg, Some(&decoder), &run, 1, 1);
        let run = RunDir::open(run.path()).unwrap();
        let ck = run
            .checkpoint::<4>(&device)
            .unwrap()
            .expect("checkpoint after epoch 1");
        assert_eq!(ck.state.next_epoch, 1);
        run.truncate_metrics(ck.state.next_epoch).unwrap();
        let (_, resumed) = train_components(
            build(),
            Some(&normalizers.y),
            &train_cfg,
            &device,
            &run,
            Some(1),
            Some(ck),
            None,
        );
        assert_eq!(resumed.len(), 1);
        let rows = run.metrics().unwrap();
        assert_eq!(rows.iter().map(|m| m.epoch).collect::<Vec<_>>(), [0, 1]);
        finite(&rows);
        let last = run.checkpoint::<4>(&device).unwrap().unwrap();
        assert_eq!(last.state.next_epoch, 2);
        let _ = fs::remove_dir_all(&dir);
    }

    // --- Phase 9 T1: checkpoint and resume ---

    /// Runs `components` as a checkpointed run (every `every` epochs) that is killed in
    /// epoch `stop`, after its `metrics.csv` row and before its checkpoint: the observer
    /// panics there.
    fn interrupt_at<const R: usize, const RM1: usize>(
        components: TrainingComponents<R, RM1>,
        train_cfg: &TrainingConfig,
        decoder: Option<&FlatDecoder>,
        run: &RunDir,
        every: usize,
        stop: usize,
    ) {
        let killed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            training_loop(
                components,
                train_cfg,
                decoder,
                &mut |m, _| {
                    run.append_metrics(m).unwrap();
                    if m.epoch == stop {
                        panic!("simulated interruption in epoch {stop}");
                    }
                },
                Some((run, every)),
                None,
            )
        }));
        assert!(killed.is_err(), "the run was not interrupted");
    }

    /// Every parameter's values, by name.
    fn weights<const R: usize>(model: &FNO<R>) -> Vec<(String, Vec<f32>)> {
        model
            .collect(None, None, false)
            .iter()
            .map(|t| {
                let data = TensorData::from_bytes(t.to_bytes().unwrap(), t.shape.clone(), t.dtype);
                (t.name.clone(), data.iter::<f32>().collect())
            })
            .collect()
    }

    /// Whether runs on the default device are reproducible bit for bit: flex only. The GPU
    /// backends' reductions may sum in a different order from run to run (Phase 9 P3).
    fn exact_backend() -> bool {
        #[cfg(feature = "flex")]
        let exact = Device::default() == Device::flex();
        #[cfg(not(feature = "flex"))]
        let exact = false;
        exact
    }

    /// The losses of two runs' epochs: bit for bit on flex, to a relative 1e-5 on the GPU
    /// backends, as [`assert_same_weights`]. The learning rate is computed on the host
    /// from the scheduler, so it is compared bit for bit everywhere.
    fn assert_same_metrics(a: &[EpochMetrics], b: &[EpochMetrics]) {
        if exact_backend() {
            assert_eq!(metric_bits(a), metric_bits(b));
            return;
        }
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert_eq!(a.epoch, b.epoch);
            assert_eq!(
                a.current_lr.to_bits(),
                b.current_lr.to_bits(),
                "epoch {}",
                a.epoch
            );
            for (name, x, y) in [
                ("train_mse", a.train_mse, b.train_mse),
                ("train_l2", a.train_l2, b.train_l2),
                ("test_l2", a.test_l2, b.test_l2),
            ] {
                assert!(
                    (x - y).abs() <= 1e-5 * y.abs().max(1.0),
                    "epoch {} {name}: {x} != {y}",
                    a.epoch
                );
            }
        }
    }

    /// Bit for bit on flex; on the fused GPU backends to a relative 1e-5 (see
    /// `device_loaders_train_identically_to_host_loaders`).
    fn assert_same_weights<const R: usize>(a: &FNO<R>, b: &FNO<R>) {
        let exact = exact_backend();
        let (a, b) = (weights(a), weights(b));
        assert_eq!(a.len(), b.len());
        for ((name, a), (_, b)) in a.iter().zip(&b) {
            if exact {
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(a), bits(b), "{name}");
            } else {
                for (x, y) in a.iter().zip(b) {
                    assert!(
                        (x - y).abs() <= 1e-5 * y.abs().max(1.0),
                        "{name}: {x} != {y}"
                    );
                }
            }
        }
    }

    /// The brief's resume test: 6 epochs, checkpoints every 2, killed in the third epoch
    /// and resumed from the checkpoint after the second. Every `metrics.csv` row (but
    /// `epoch_time_s`) and the final weights equal the uninterrupted run's, bit for bit on
    /// flex.
    #[test]
    fn resumed_run_matches_uninterrupted_run() {
        let root = std::env::temp_dir().join(format!("sciml_rs_resume_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let train_cfg = TrainingConfig::new()
            .with_epochs(6)
            .with_batch_size(2)
            .with_test_batch_size(2)
            .with_seed(5);
        let device = Device::default().autodiff();
        let build = || {
            build_training_components::<3, 2, _>(
                &tiny_model_cfg(),
                &train_cfg,
                tiny_dataset(5), // partial last batch
                tiny_dataset(3),
                &device,
            )
        };
        // one initial model for both runs (the device RNG is shared between parallel tests)
        let whole = build();
        let mut first = build();
        first.model = whole.model.clone();

        let straight = RunDir::create(&root, "straight").unwrap();
        let (straight_model, _) = train_components(
            whole,
            None,
            &train_cfg,
            &device,
            &straight,
            Some(2),
            None,
            None,
        );

        let run = RunDir::create(&root, "resumed").unwrap();
        // the checkpoint reader needs model_cfg.json; the data config is unused here
        run.save_configs(&tiny_model_cfg(), &train_cfg, &train_cfg)
            .unwrap();
        interrupt_at(first, &train_cfg, None, &run, 2, 2);
        assert_eq!(run.metrics().unwrap().len(), 3);
        let ck = run.checkpoint::<3>(&device).unwrap().unwrap();
        assert_eq!(ck.state.next_epoch, 2);
        run.truncate_metrics(ck.state.next_epoch).unwrap();
        let (resumed_model, _) = train_components(
            build(),
            None,
            &train_cfg,
            &device,
            &run,
            Some(2),
            Some(ck),
            None,
        );

        let (resumed, uninterrupted) = (run.metrics().unwrap(), straight.metrics().unwrap());
        let epochs = |m: &[EpochMetrics]| m.iter().map(|m| m.epoch).collect::<Vec<_>>();
        assert_eq!(epochs(&resumed), [0, 1, 2, 3, 4, 5]);
        assert_same_metrics(&resumed, &uninterrupted);
        assert_same_weights(&resumed_model, &straight_model);
        let last = run.checkpoint::<3>(&device).unwrap().unwrap();
        assert_eq!(last.state.next_epoch, 6);
        let _ = fs::remove_dir_all(&root);
    }

    /// A checkpoint carries Adam's moments and the cosine schedule's position: one more
    /// epoch from the reloaded state equals one from the live state, and differs without
    /// the moments.
    #[test]
    fn checkpoint_round_trips_adam_moments_and_cosine_position() {
        let root = std::env::temp_dir().join(format!(
            "sciml_rs_checkpoint_round_trip_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let train_cfg = TrainingConfig::new().with_epochs(4).with_batch_size(2);
        let device = Device::default().autodiff();
        let build = || {
            build_training_components::<3, 2, _>(
                &tiny_model_cfg(),
                &train_cfg,
                tiny_dataset(5),
                tiny_dataset(3),
                &device,
            )
        };
        let loader = device_loader::<3, 2, _>(&tiny_dataset(5), &device, 2, None);
        let mut live = build();
        let (model, first) = train_epoch(
            live.model,
            &loader,
            &mut live.optim,
            &mut live.scheduler,
            None,
        );

        let run = RunDir::create(&root, "ck").unwrap();
        run.save_configs(&tiny_model_cfg(), &train_cfg, &train_cfg)
            .unwrap();
        let state = CheckpointState {
            next_epoch: 1,
            resumes: 3,
            training_time_s: 1.5,
        };
        run.save_checkpoint(&state, &model, &live.optim, &live.scheduler)
            .unwrap();
        let ck = run.checkpoint::<3>(&device).unwrap().unwrap();
        assert_eq!(ck.state, state);
        assert!(!ck.optim.is_empty(), "no Adam moments in the checkpoint");
        let mut loaded = build();
        loaded.optim = loaded.optim.load_record(ck.optim);
        loaded.scheduler = loaded.scheduler.load_record(ck.scheduler.clone());

        let (a, ma) = train_epoch(
            model.clone(),
            &loader,
            &mut live.optim,
            &mut live.scheduler,
            None,
        );
        let (b, mb) = train_epoch(
            ck.model,
            &loader,
            &mut loaded.optim,
            &mut loaded.scheduler,
            None,
        );
        assert_eq!(ma.last_lr.to_bits(), mb.last_lr.to_bits());
        assert_ne!(ma.last_lr, first.last_lr, "the schedule did not move on");
        assert_same_weights(&a, &b);

        // the same schedule position without the moments steps differently
        let mut fresh = build();
        fresh.scheduler = fresh.scheduler.load_record(ck.scheduler);
        let (c, _) = train_epoch(model, &loader, &mut fresh.optim, &mut fresh.scheduler, None);
        assert_ne!(weights(&c), weights(&a));
        let _ = fs::remove_dir_all(&root);
    }

    // --- Phase 4s T12: the epoch observer ---

    /// `train_fno` writes one `metrics.csv` row per epoch through the observer, and the
    /// rows read back as exactly the metrics it returns.
    #[test]
    fn metrics_csv_holds_every_epoch() {
        let root =
            std::env::temp_dir().join(format!("sciml_rs_metrics_csv_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let train_cfg = TrainingConfig::new()
            .with_epochs(3)
            .with_batch_size(2)
            .with_test_batch_size(2);
        let run = RunDir::create(&root, "live").unwrap();
        let (_, metrics) = train_fno::<3, 2, _>(
            tiny_dataset(5),
            tiny_dataset(3),
            None,
            &tiny_model_cfg(),
            &train_cfg,
            &Device::default().autodiff(),
            &run,
        );
        let written = run.metrics().unwrap();
        let epochs = |m: &[EpochMetrics]| m.iter().map(|m| m.epoch).collect::<Vec<_>>();
        assert_eq!(epochs(&written), [0, 1, 2]);
        assert_eq!(metric_bits(&written), metric_bits(&metrics));
        let _ = fs::remove_dir_all(&root);
    }

    // --- Phase 9 T2: data-parallel training ---

    /// Components for `tiny_dataset(7)` (a partial last batch of one sample: the second
    /// replica's shard is empty) with one replica per device of `extra`.
    fn parallel_components(
        train_cfg: &TrainingConfig,
        device: &Device,
        extra: &[Device],
    ) -> TrainingComponents<3, 2> {
        let mut c = build_training_components::<3, 2, _>(
            &tiny_model_cfg(),
            train_cfg,
            tiny_dataset(7),
            tiny_dataset(3),
            device,
        );
        c.replicas = data_parallel::replicas(&tiny_dataset(7), extra, train_cfg);
        c
    }

    /// Two replicas on one device (summed without a collective, so this runs on flex)
    /// train as one device does, with a target decoder: every epoch's metrics and the
    /// final weights agree to round-off (the gradient sums run in another order).
    #[test]
    fn two_replicas_train_as_one_device() {
        let train_cfg = TrainingConfig::new()
            .with_epochs(3)
            .with_batch_size(2)
            .with_test_batch_size(2)
            .with_seed(9);
        let device = Device::default().autodiff();
        let one = parallel_components(&train_cfg, &device, &[]);
        let mut two = parallel_components(&train_cfg, &device, std::slice::from_ref(&device));
        two.model = one.model.clone();
        let y = UnitGaussianNormalizer::fit(tiny_dataset(7).targets());
        let decoder = FlatDecoder::new(&y, &device);

        let run = |c| training_loop(c, &train_cfg, Some(&decoder), &mut |_, _| {}, None, None);
        let ((one_model, one), (two_model, two)) = (run(one), run(two));

        // relative 1e-5, absolute 1e-5 near zero, as `assert_same_weights` on GPU
        let close = |x: f64, y: f64| (x - y).abs() <= 1e-5 * y.abs().max(1.0);
        assert_eq!(one.len(), 3);
        for (a, b) in two.iter().zip(&one) {
            assert_eq!(a.current_lr, b.current_lr);
            for (x, y) in [
                (a.train_mse, b.train_mse),
                (a.train_l2, b.train_l2),
                (a.test_l2, b.test_l2),
            ] {
                assert!(close(x.into(), y.into()), "epoch {}: {x} != {y}", a.epoch);
            }
        }
        for ((name, a), (_, b)) in weights(&two_model).iter().zip(&weights(&one_model)) {
            for (x, y) in a.iter().zip(b) {
                assert!(close((*x).into(), (*y).into()), "{name}: {x} != {y}");
            }
        }
    }

    /// T1's resume test on a two-replica run: killed in the third epoch and resumed from
    /// the first replica's checkpoint, it ends as the uninterrupted two-replica run does,
    /// bit for bit on flex.
    #[test]
    fn resumed_parallel_run_matches_uninterrupted_run() {
        let root =
            std::env::temp_dir().join(format!("sciml_rs_parallel_resume_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let train_cfg = TrainingConfig::new()
            .with_epochs(4)
            .with_batch_size(2)
            .with_test_batch_size(2)
            .with_seed(5);
        let device = Device::default().autodiff();
        let build = || parallel_components(&train_cfg, &device, std::slice::from_ref(&device));
        let whole = build();
        let mut first = build();
        first.model = whole.model.clone();

        let straight = RunDir::create(&root, "straight").unwrap();
        let (straight_model, _) = train_components(
            whole,
            None,
            &train_cfg,
            &device,
            &straight,
            Some(2),
            None,
            None,
        );

        let run = RunDir::create(&root, "resumed").unwrap();
        run.save_configs(&tiny_model_cfg(), &train_cfg, &train_cfg)
            .unwrap();
        interrupt_at(first, &train_cfg, None, &run, 2, 2);
        let ck = run.checkpoint::<3>(&device).unwrap().unwrap();
        run.truncate_metrics(ck.state.next_epoch).unwrap();
        let (resumed_model, _) = train_components(
            build(),
            None,
            &train_cfg,
            &device,
            &run,
            Some(2),
            Some(ck),
            None,
        );

        assert_same_metrics(&run.metrics().unwrap(), &straight.metrics().unwrap());
        assert_same_weights(&resumed_model, &straight_model);
        let _ = fs::remove_dir_all(&root);
    }

    /// The console line: the metrics alone, the epoch time among them (since Phase 5 T4;
    /// before, a `| total=` suffix repeated it).
    #[test]
    fn default_observer_prints_the_metrics() {
        let m = EpochMetrics {
            epoch: 4,
            train_mse: 0.5,
            train_l2: 0.25,
            test_l2: 0.125,
            current_lr: 1e-3,
            epoch_time_s: Some(1.234567),
        };
        assert_eq!(
            epoch_line(&m),
            "EpochMetrics { epoch: 4, train_mse: 0.5, train_l2: 0.25, test_l2: 0.125, \
             current_lr: 0.001, epoch_time_s: Some(1.234567) }"
        );
    }
}

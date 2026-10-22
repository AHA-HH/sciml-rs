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
use std::time::{Duration, Instant};

use crate::neural_operators::{
    data::{
        batcher::Batch,
        dataset::OperatorDataset,
        device_batcher::device_loader,
        host_float::HostFloat,
        transforms::normalizers::{FlatDecoder, UnitGaussianNormalizer},
    },
    models::fno::{FNO, FNOConfig},
    runs::{EpochMetrics, RunDir},
    training::{
        config::TrainingConfig,
        step::{Postprocess, StepLoss, evaluate, forward_loss, identity},
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
fn accumulate(sum: Option<Tensor<1>>, x: Tensor<1>) -> Option<Tensor<1>> {
    Some(match sum {
        Some(sum) => sum + x,
        None => x,
    })
}

/// Runs one training pass and returns its [`EpochMeans`]: the MSE divided by the batch
/// count, the relative L² by `loader.num_items()`. Takes and returns the model - Burn's
/// optimizer consumes it on each step.
pub fn train_epoch<const R: usize, const RM1: usize>(
    mut model: FNO<R>,
    loader: &Arc<dyn DataLoader<Batch<R, RM1>>>,
    optim: &mut ModuleOptimizer,
    scheduler: &mut ModuleLrScheduler,
    post: &Postprocess,
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
        } = forward_loss(&model, batch, post);

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

pub struct TrainingComponents<const R: usize, const RM1: usize> {
    pub model: FNO<R>,
    pub optim: ModuleOptimizer,
    pub scheduler: ModuleLrScheduler,
    pub train_loader: Arc<dyn DataLoader<Batch<R, RM1>>>,
    pub test_loader: Arc<dyn DataLoader<Batch<R, RM1>>>,
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
/// If either dataset is empty.
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

    // DataLoader yields ceil(len / batch_size) batches, the last one partial
    let steps_per_epoch = n_train.div_ceil(train_cfg.batch_size);
    let scheduler = CosineAnnealingLrSchedulerConfig::new(
        train_cfg.learning_rate,
        train_cfg.epochs * steps_per_epoch,
    )
    .with_min_lr(train_cfg.min_lr)
    .init()
    .expect("valid cosine scheduler config");

    // Each split is uploaded once and batched on the device (REVIEW.md 4.1),
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
    }
}

/// Per-epoch observer for [`training_loop`]: the epoch's metrics, its wall time (training
/// and evaluation) and the model after the epoch.
///
/// A plain closure rather than a trait: call sites pass `&mut |m, t, model| ...`, and two
/// observers combine as two calls in one closure (see [`train_fno`]). The model is for
/// checkpointing (Phase 5); nothing reads it yet.
pub type EpochObserver<'a, const R: usize> = dyn FnMut(&EpochMetrics, Duration, &FNO<R>) + 'a;

/// The default observer: prints the epoch's metrics and wall time to stdout, one line per
/// epoch.
pub fn print_epoch<const R: usize>(m: &EpochMetrics, epoch_time: Duration, _model: &FNO<R>) {
    println!("{}", epoch_line(m, epoch_time));
}

fn epoch_line(m: &EpochMetrics, epoch_time: Duration) -> String {
    format!("{m:?} | total={:.3}s", epoch_time.as_secs_f64())
}

/// Runs `train_cfg.epochs` epochs of [`train_epoch`] then [`evaluate`], calling `on_epoch`
/// after each, and returns the trained model and every epoch's metrics.
///
/// Prints nothing itself: pass [`print_epoch`] (or a closure calling it) for the console
/// line.
pub fn training_loop<const R: usize, const RM1: usize>(
    mut components: TrainingComponents<R, RM1>,
    train_cfg: &TrainingConfig,
    train_post: &Postprocess,
    eval_post: &Postprocess,
    on_epoch: &mut EpochObserver<'_, R>,
) -> (FNO<R>, Vec<EpochMetrics>) {
    let mut model = components.model;
    let mut metrics = Vec::with_capacity(train_cfg.epochs);

    for epoch in 0..train_cfg.epochs {
        let epoch_start = Instant::now();

        let (m, means) = train_epoch(
            model,
            &components.train_loader,
            &mut components.optim,
            &mut components.scheduler,
            train_post,
        );
        model = m;

        let test_l2 = evaluate(&model.valid(), &components.test_loader, eval_post);

        let epoch_time = epoch_start.elapsed();

        let record = EpochMetrics {
            epoch,
            train_mse: means.train_mse,
            train_l2: means.train_l2,
            test_l2,
            current_lr: means.last_lr,
        };

        on_epoch(&record, epoch_time, &model);
        metrics.push(record);
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
/// `y_norm` chooses the postprocess hooks:
/// - `None`: [`identity`] on both sides, for data trained in physical units (Burgers).
/// - `Some(y)`: `y_train` was encoded with `y` during loading and `y_test` was not
///   (CONVENTIONS §10), so training decodes both prediction and target, and evaluation
///   decodes only the prediction.
///
/// # Panics
///
/// From [`build_training_components`]:
/// - If either dataset is empty.
/// - If `model_cfg.modes.len() + 2 != R` or `model_cfg.out_channels != 1`
///   (`FNOConfig::init`; the latter is the requirement of
///   [`flatten_pair`](crate::neural_operators::training::step::flatten_pair)).
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

    let start = Instant::now();
    let mut on_epoch = |m: &EpochMetrics, epoch_time: Duration, model: &FNO<R>| {
        print_epoch(m, epoch_time, model);
        if let Err(e) = run.append_metrics(m) {
            eprintln!(
                "warning: could not append epoch {} to metrics.csv ({e})",
                m.epoch
            );
        }
    };
    let trained = train_built(components, y_norm, train_cfg, device, &mut on_epoch);
    println!("total training time: {:.2?}", start.elapsed());
    trained
}

/// [`train_fno`] after the build: picks the hooks for `y_norm` and runs the loop. Split out
/// so tests can give two runs the same initial model.
fn train_built<const R: usize, const RM1: usize>(
    components: TrainingComponents<R, RM1>,
    y_norm: Option<&UnitGaussianNormalizer>,
    train_cfg: &TrainingConfig,
    device: &Device,
    on_epoch: &mut EpochObserver<'_, R>,
) -> (FNO<R>, Vec<EpochMetrics>) {
    let Some(y_norm) = y_norm else {
        return training_loop(components, train_cfg, &identity, &identity, on_epoch);
    };

    // One decoder per device: training tensors are autodiff, evaluation
    // tensors live on the inner device (model.valid()).
    let train_decoder = FlatDecoder::new(y_norm, device);
    let eval_decoder = FlatDecoder::new(y_norm, &device.clone().inner());

    let train_post = move |out: Tensor<2>, target: Tensor<2>| {
        (train_decoder.decode(out), train_decoder.decode(target))
    };

    let eval_post = move |out: Tensor<2>, target: Tensor<2>| (eval_decoder.decode(out), target);

    training_loop(components, train_cfg, &train_post, &eval_post, on_epoch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::data::host_oracle::{HostItems, OperatorBatcher};
    use crate::neural_operators::training::fixtures::{
        tiny_dataset, tiny_dataset_2d, tiny_model_cfg,
    };
    use crate::neural_operators::{
        chebyshev::nodes_closed_form,
        data::loaders::{
            base_dataset::DatasetConfig,
            poisson::{PoissonConfig, load_poisson_uniform},
        },
    };
    use burn::data::dataloader::DataLoaderBuilder;
    use ndarray::{Array1, Array3, ArrayD};
    use ndarray_npy::NpzWriter;
    use std::fs::{self, File};
    use std::path::{Path, PathBuf};

    // --- REVIEW1.md N5: schedule length and divisors come from the data ---

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
        let (model, metrics) = training_loop(
            components,
            &train_cfg,
            &identity,
            &identity,
            &mut |_, _, _| {},
        );

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
        let expected = evaluate(&model.valid(), &loader, &identity);
        let test_l2 = metrics.last().unwrap().test_l2;
        assert!(
            (test_l2 - expected).abs() <= 1e-6 * expected.abs(),
            "{test_l2} != {expected}"
        );
    }

    // --- REVIEW.md 4.1: device-resident loaders change nothing numerically ---

    /// Full `training_loop` with the device-resident loaders against the same
    /// components with the pre-4.1 host loaders swapped in: every metric of
    /// every epoch must be exactly equal (same batches, same order, same
    /// kernels), as must the trained model's output on flex.
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

        let (new_model, new) = training_loop(
            device_run,
            &train_cfg,
            &identity,
            &identity,
            &mut |_, _, _| {},
        );

        host.train_loader = DataLoaderBuilder::new(OperatorBatcher::<3, 2>::new())
            .set_device(device.clone())
            .batch_size(train_cfg.batch_size)
            .shuffle(train_cfg.seed)
            .build(HostItems(tiny_dataset(n_train)));
        host.test_loader = DataLoaderBuilder::new(OperatorBatcher::<3, 2>::new())
            .set_device(device.clone().inner())
            .batch_size(train_cfg.test_batch_size)
            .build(HostItems(tiny_dataset(n_test)));
        let (old_model, old) =
            training_loop(host, &train_cfg, &identity, &identity, &mut |_, _, _| {});

        assert_eq!(metric_bits(&new), metric_bits(&old));

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

    /// `None` against the old `train_burgers` body: `identity` on both sides. Both runs
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

        let (_, new) = train_built(new, None, &train_cfg, &device, &mut |_, _, _| {});
        let (_, old) = training_loop(old, &train_cfg, &identity, &identity, &mut |_, _, _| {});

        assert_eq!(metric_bits(&new), metric_bits(&old));
    }

    /// `Some(y)` against the old `train_darcy` / `train_poisson` body: decode both sides in
    /// training, the prediction only in evaluation.
    #[test]
    fn train_fno_some_matches_decoder_path() {
        let train_cfg = TrainingConfig::new()
            .with_epochs(2)
            .with_batch_size(2)
            .with_test_batch_size(2);
        let model_cfg = FNOConfig::new(vec![2, 2], 1, 1)
            .with_hidden_channels(4)
            .with_n_layers(1);
        let y = UnitGaussianNormalizer::fit(tiny_dataset_2d(5).targets());
        let device = Device::default().autodiff();
        let build = || {
            build_training_components::<4, 3, _>(
                &model_cfg,
                &train_cfg,
                tiny_dataset_2d(5),
                tiny_dataset_2d(3),
                &device,
            )
        };
        let new = build();
        let mut old = build();
        old.model = new.model.clone();

        let (_, new) = train_built(new, Some(&y), &train_cfg, &device, &mut |_, _, _| {});

        let train_decoder = FlatDecoder::new(&y, &device);
        let eval_decoder = FlatDecoder::new(&y, &device.clone().inner());
        let train_post = move |out: Tensor<2>, target: Tensor<2>| {
            (train_decoder.decode(out), train_decoder.decode(target))
        };
        let eval_post = move |out: Tensor<2>, target: Tensor<2>| (eval_decoder.decode(out), target);
        let (_, old) = training_loop(old, &train_cfg, &train_post, &eval_post, &mut |_, _, _| {});

        assert_eq!(metric_bits(&new), metric_bits(&old));
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
        let x: Array1<f64> = nodes_closed_form(POISSON_N);
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

        assert_eq!(metrics.len(), 2);
        for m in &metrics {
            assert!(
                m.train_mse.is_finite() && m.train_l2.is_finite() && m.test_l2.is_finite(),
                "{m:?}"
            );
        }
        let _ = fs::remove_dir_all(&dir);
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

    /// The line `training_loop` printed before the observer existed.
    #[test]
    fn default_observer_prints_same_line() {
        let m = EpochMetrics {
            epoch: 4,
            train_mse: 0.5,
            train_l2: 0.25,
            test_l2: 0.125,
            current_lr: 1e-3,
        };
        let t = Duration::from_micros(1_234_567);
        assert_eq!(
            epoch_line(&m, t),
            format!("{m:?} | total={:.3}s", t.as_secs_f64(),)
        );
        assert_eq!(
            epoch_line(&m, t),
            "EpochMetrics { epoch: 4, train_mse: 0.5, train_l2: 0.25, test_l2: 0.125, \
             current_lr: 0.001 } | total=1.235s"
        );
    }
}

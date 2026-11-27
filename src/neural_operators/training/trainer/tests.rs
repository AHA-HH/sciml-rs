use super::*;
use crate::neural_operators::data::host_oracle::{HostItems, OperatorBatcher};
use crate::neural_operators::models::fno::FNOConfig;
use crate::neural_operators::training::config::{LrSchedule, OptimizerConfig, TrainLoss};
use crate::neural_operators::training::fixtures::{tiny_dataset, tiny_dataset_2d, tiny_model_cfg};
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
use burn::module::Module;
use burn::store::ModuleSnapshot;
use ndarray::{Array1, Array3, ArrayD};
use ndarray_npy::NpzWriter;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

// --- merge `04ec800`: schedule length and divisors come from the data ---

#[test]
fn components_take_sample_counts_from_the_datasets() {
    let train_cfg = TrainingConfig::new().with_batch_size(2);
    let c = build_components::<FNO<3>, 3, 2, _>(
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
    let components = build_components::<FNO<3>, 3, 2, _>(
        &tiny_model_cfg(),
        &train_cfg,
        tiny_dataset(n_train),
        tiny_dataset(n_test),
        &Device::default().autodiff(),
    );
    let (model, metrics) = training_loop(components, &train_cfg, None, &mut |_, _| {}, None, None);

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
        build_components::<FNO<3>, 3, 2, _>(
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

    let (new_model, new) = training_loop(device_run, &train_cfg, None, &mut |_, _| {}, None, None);

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

// --- Phase 4s T3: `train` reproduces the per-PDE trainers it replaced ---

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
        build_components::<FNO<3>, 3, 2, _>(
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

/// One epoch of `train`'s `Some(y)` path on a physical `y_train` (only the
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
        build_components::<FNO<4>, 4, 3, _>(
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
    let _ = build_components::<FNO<3>, 3, 2, _>(
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
    let (_, metrics) = super::train::<FNO<4>, 4, 3, _>(
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
        let (train, test, _) = load_poisson_uniform::<f32>(&train_path, &test_path, &cfg).unwrap();
        build_components::<FNO<4>, 4, 3, _>(&model_cfg, &train_cfg, train, test, &device)
    };
    let decoder = FlatDecoder::new(&normalizers.y, &device);
    interrupt_at(build(), &train_cfg, Some(&decoder), &run, 1, 1);
    let run = RunDir::open(run.path()).unwrap();
    let ck = run
        .checkpoint_of::<FNO<4>>(&device)
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
    let last = run.checkpoint_of::<FNO<4>>(&device).unwrap().unwrap();
    assert_eq!(last.state.next_epoch, 2);
    let _ = fs::remove_dir_all(&dir);
}

// --- Phase 10 T3: Gauss-Newton in the trainer ---

/// A `train_cfg.json` from before Phase 10 T3 (no `optimizer`) builds Adam components
/// and trains as the same settings given in code, from one initial model.
#[test]
fn old_training_config_trains_as_adam() {
    let old = r#"{"epochs": 2, "batch_size": 2, "test_batch_size": 2,
            "learning_rate": 0.001, "weight_decay": 0.0001, "min_lr": 1e-7, "seed": 42}"#;
    let loaded = TrainingConfig::load_binary(old.as_bytes()).unwrap();
    let given = TrainingConfig::new()
        .with_epochs(2)
        .with_batch_size(2)
        .with_test_batch_size(2);
    let device = Device::default().autodiff();
    let build = |cfg: &TrainingConfig| {
        build_components::<FNO<3>, 3, 2, _>(
            &tiny_model_cfg(),
            cfg,
            tiny_dataset(5),
            tiny_dataset(3),
            &device,
        )
    };
    let a = build(&loaded);
    let mut b = build(&given);
    b.model = a.model.clone();
    assert!(a.gn.is_none(), "an old config must train with Adam");
    let (_, a) = train_built(a, None, &loaded, &device, &mut |_, _| {});
    let (_, b) = train_built(b, None, &given, &device, &mut |_, _| {});
    assert_same_metrics(&a, &b);
}

/// A Gauss-Newton run of `train_poisson_smoke`'s fixture writes `metrics.csv` and one
/// `gn_steps.csv` row per step, and checkpoints its state; stopped in its second epoch
/// and resumed from the first epoch's checkpoint, it ends as the uninterrupted run
/// does, bit for bit on flex.
#[test]
fn gauss_newton_smoke_run_resumes_exactly() {
    let (_, rows, _) = gn_smoke_run_resumes_exactly(None, "gn_smoke");
    assert_eq!(rows.len(), 7, "a header and six steps");
}

/// A hybrid run (Phase 10 T5) trains its first `adam_epochs` with Adam, as an Adam run of
/// that many epochs does, then Gauss-Newton; killed in its first Gauss-Newton epoch
/// and resumed from the Adam epoch's checkpoint, it ends as the uninterrupted run does.
#[test]
fn hybrid_smoke_run_switches_and_resumes_exactly() {
    let (metrics, rows, init) = gn_smoke_run_resumes_exactly(Some(1), "hybrid_smoke");
    // two Gauss-Newton epochs of two steps, labelled with epochs 1 and 2
    assert_eq!(rows.len(), 5, "a header and four steps");
    assert!(rows[1..].iter().all(|r| !r.starts_with("0,")), "{rows:?}");

    // epoch 0 is an Adam run of one epoch, from the same initial model
    let dir = std::env::temp_dir().join(format!("sciml_rs_hybrid_adam_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let (train_path, test_path, cfg, model_cfg) = gn_smoke_fixture(&dir);
    let adam_cfg = TrainingConfig::new()
        .with_epochs(1)
        .with_batch_size(2)
        .with_test_batch_size(2);
    let (train, test, normalizers) =
        load_poisson_uniform::<f32>(&train_path, &test_path, &cfg).unwrap();
    let device = Device::default().autodiff();
    let mut adam = build_components::<FNO<4>, 4, 3, _>(&model_cfg, &adam_cfg, train, test, &device);
    adam.model = init;
    let (_, adam) = train_built(
        adam,
        Some(&normalizers.y),
        &adam_cfg,
        &device,
        &mut |_, _| {},
    );
    assert_same_metrics(&metrics[..1], &adam);
    let _ = fs::remove_dir_all(&dir);
}

/// `train_poisson_smoke`'s fixture in `dir`: the train and test files, the data config and
/// the tiny model's config.
fn gn_smoke_fixture(dir: &Path) -> (PathBuf, PathBuf, PoissonConfig, FNOConfig) {
    let train_path = write_poisson_split(dir, "train.npz", POISSON_N_TRAIN, 0.0);
    let test_path = write_poisson_split(dir, "test.npz", POISSON_N_TEST, 0.5);
    let cfg = PoissonConfig::new(
        DatasetConfig {
            n_train: POISSON_N_TRAIN,
            n_test: POISSON_N_TEST,
        },
        POISSON_N,
    );
    let model_cfg = FNOConfig::new(vec![2, 2], 1, 1)
        .with_hidden_channels(4)
        .with_n_layers(2);
    (train_path, test_path, cfg, model_cfg)
}

/// A 3-epoch Gauss-Newton run (hybrid with `adam_epochs`) of `train_poisson_smoke`'s
/// fixture, uninterrupted and killed in epoch 1 then resumed from epoch 0's checkpoint:
/// both must agree, bit for bit on flex. Returns the uninterrupted run's metrics, its
/// `gn_steps.csv` lines without the wall times, and its initial model.
fn gn_smoke_run_resumes_exactly(
    adam_epochs: Option<usize>,
    tag: &str,
) -> (Vec<EpochMetrics>, Vec<String>, FNO<4>) {
    use crate::neural_operators::{
        optim::gauss_newton::GaussNewtonConfig, training::config::OptimizerConfig,
    };
    let dir = std::env::temp_dir().join(format!("sciml_rs_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let (train_path, test_path, cfg, model_cfg) = gn_smoke_fixture(&dir);
    let gn = GaussNewtonConfig::new()
        .with_cg_max_iters(5)
        .with_adjoint_check_every(2);
    let train_cfg = TrainingConfig::new()
        .with_epochs(3)
        .with_batch_size(2)
        .with_test_batch_size(2)
        .with_optimizer(Some(OptimizerConfig::GaussNewton(gn)))
        .with_adam_epochs(adam_epochs);
    let device = Device::default().autodiff();
    let (_, _, normalizers) = load_poisson_uniform::<f32>(&train_path, &test_path, &cfg).unwrap();
    let build = || {
        let (train, test, _) = load_poisson_uniform::<f32>(&train_path, &test_path, &cfg).unwrap();
        build_components::<FNO<4>, 4, 3, _>(&model_cfg, &train_cfg, train, test, &device)
    };
    let whole = build();
    let init = whole.model.clone();
    let mut first = build();
    first.model = init.clone();

    let straight = RunDir::create(&dir, "gn_straight").unwrap();
    straight.save_configs(&model_cfg, &train_cfg, &cfg).unwrap();
    let (straight_model, metrics) = train_components(
        whole,
        Some(&normalizers.y),
        &train_cfg,
        &device,
        &straight,
        Some(1),
        None,
        None,
    );
    assert_eq!(metrics.len(), 3);
    for m in &metrics {
        assert!(m.train_l2.is_finite() && m.test_l2.is_finite(), "{m:?}");
        assert!(m.current_lr > 0.0, "the damping λ is reported: {m:?}");
    }
    // the Gauss-Newton epochs, 2 batches each, one step each
    let steps = 2 * (3 - adam_epochs.unwrap_or(0));
    let passes = straight.gn_passes().unwrap().unwrap();
    assert_eq!(passes.steps, steps);
    assert!(passes.backward >= passes.steps, "{passes:?}");
    // one metrics forward per step on top of the optimiser's
    assert_eq!(
        crate::neural_operators::training::gn::training_passes(&passes),
        (passes.forward + steps, passes.backward)
    );
    let ck = straight.checkpoint_of::<FNO<4>>(&device).unwrap().unwrap();
    assert_eq!(ck.gn.expect("Gauss-Newton state").step as usize, steps);

    let run = RunDir::create(&dir, "gn_resumed").unwrap();
    run.save_configs(&model_cfg, &train_cfg, &cfg).unwrap();
    let decoder = FlatDecoder::new(&normalizers.y, &device);
    // killed in epoch 1 after its metrics row and its steps' rows, before its checkpoint
    let killed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_loop(
            first,
            &train_cfg,
            Some(&decoder),
            &mut |m, _| {
                run.append_metrics(m).unwrap();
                if m.epoch == 1 {
                    panic!("simulated interruption in epoch 1");
                }
            },
            &mut |r| run.append_gn_step(r).unwrap(),
            Some((&run, 1)),
            None,
        )
    }));
    assert!(killed.is_err(), "the run was not interrupted");
    let ck = run.checkpoint_of::<FNO<4>>(&device).unwrap().unwrap();
    assert_eq!(ck.state.next_epoch, 1);
    run.truncate_metrics(ck.state.next_epoch).unwrap();
    run.truncate_gn_steps(ck.state.next_epoch).unwrap();
    let (resumed_model, _) = train_components(
        build(),
        Some(&normalizers.y),
        &train_cfg,
        &device,
        &run,
        Some(1),
        Some(ck),
        None,
    );
    assert_same_metrics(&run.metrics().unwrap(), &straight.metrics().unwrap());
    assert_same_weights(&resumed_model, &straight_model);
    // gn_steps.csv as the uninterrupted run's, but for the wall times
    let rows = |r: &RunDir| -> Vec<String> {
        fs::read_to_string(r.path().join("gn_steps.csv"))
            .unwrap()
            .lines()
            .map(|l| l.rsplit_once(',').unwrap().0.to_string())
            .collect()
    };
    let (resumed_rows, straight_rows) = (rows(&run), rows(&straight));
    assert_eq!(resumed_rows.len(), straight_rows.len());
    if exact_backend() {
        assert_eq!(resumed_rows, straight_rows);
    }
    let _ = fs::remove_dir_all(&dir);
    (metrics, straight_rows, init)
}

/// Data-parallel Gauss-Newton is out of scope: replicas with Gauss-Newton panic.
#[test]
#[should_panic(expected = "data-parallel Gauss-Newton is out of scope")]
fn gauss_newton_with_replicas_panics() {
    use crate::neural_operators::{
        optim::gauss_newton::GaussNewtonConfig, training::config::OptimizerConfig,
    };
    let train_cfg = TrainingConfig::new()
        .with_epochs(1)
        .with_batch_size(2)
        .with_optimizer(Some(OptimizerConfig::GaussNewton(GaussNewtonConfig::new())));
    let device = Device::default().autodiff();
    let c = parallel_components(&train_cfg, &device, std::slice::from_ref(&device));
    let _ = training_loop(c, &train_cfg, None, &mut |_, _| {}, None, None);
}

/// Optimisers CO2.1: an L-BFGS run of `train_poisson_smoke`'s fixture, full batch,
/// writes `metrics.csv` and one `lbfgs_steps.csv` row per epoch, and checkpoints its
/// state (and no Gauss-Newton state); killed in epoch 1 and resumed from epoch 0's
/// checkpoint, it ends as the uninterrupted run does, bit for bit on flex.
#[test]
fn lbfgs_smoke_run_resumes_exactly() {
    use crate::neural_operators::training::config::{LbfgsArm, OptimizerConfig};
    use burn::optim::{LBFGSConfig, LineSearchFn};
    let dir = std::env::temp_dir().join(format!("sciml_rs_lbfgs_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let (train_path, test_path, cfg, model_cfg) = gn_smoke_fixture(&dir);
    let arm = LbfgsArm::new(
        LBFGSConfig::new()
            .with_max_iter(3)
            .with_history_size(5)
            .with_line_search_fn(LineSearchFn::StrongWolfe),
    );
    let train_cfg = TrainingConfig::new()
        .with_epochs(3)
        .with_batch_size(POISSON_N_TRAIN)
        .with_test_batch_size(2)
        .with_optimizer(Some(OptimizerConfig::Lbfgs(arm)));
    let device = Device::default().autodiff();
    let (_, _, normalizers) = load_poisson_uniform::<f32>(&train_path, &test_path, &cfg).unwrap();
    let build = || {
        let (train, test, _) = load_poisson_uniform::<f32>(&train_path, &test_path, &cfg).unwrap();
        build_components::<FNO<4>, 4, 3, _>(&model_cfg, &train_cfg, train, test, &device)
    };
    let whole = build();
    let mut first = build();
    first.model = whole.model.clone();

    let straight = RunDir::create(&dir, "lbfgs_straight").unwrap();
    straight.save_configs(&model_cfg, &train_cfg, &cfg).unwrap();
    let (straight_model, metrics) = train_components(
        whole,
        Some(&normalizers.y),
        &train_cfg,
        &device,
        &straight,
        Some(1),
        None,
        None,
    );
    assert_eq!(metrics.len(), 3);
    assert!(metrics.iter().all(|m| m.current_lr == 1.0), "{metrics:?}");
    assert!(metrics[2].train_l2 < metrics[0].train_l2, "{metrics:?}");
    let passes = straight.lbfgs_passes().unwrap().unwrap();
    assert_eq!(passes.steps, 3);
    assert_eq!(passes.forward, passes.backward);
    assert!(passes.forward >= 2 * 3, "{passes:?}");
    assert!(straight.gn_passes().unwrap().is_none());
    let ck = straight.checkpoint_of::<FNO<4>>(&device).unwrap().unwrap();
    assert!(ck.gn.is_none() && ck.lbfgs.is_some());

    let run = RunDir::create(&dir, "lbfgs_resumed").unwrap();
    run.save_configs(&model_cfg, &train_cfg, &cfg).unwrap();
    let decoder = FlatDecoder::new(&normalizers.y, &device);
    // killed in epoch 1 after its metrics row and its step's row, before its checkpoint
    let killed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_loop_with(
            first,
            &train_cfg,
            Some(&decoder),
            &mut |m, _| {
                run.append_metrics(m).unwrap();
                if m.epoch == 1 {
                    panic!("simulated interruption in epoch 1");
                }
            },
            StepSinks {
                gn: &mut |_| panic!("no Gauss-Newton step in an L-BFGS run"),
                lbfgs: &mut |r| run.append_lbfgs_step(r).unwrap(),
            },
            Some((&run, 1)),
            None,
        )
    }));
    assert!(killed.is_err(), "the run was not interrupted");
    let ck = run.checkpoint_of::<FNO<4>>(&device).unwrap().unwrap();
    assert_eq!(ck.state.next_epoch, 1);
    run.truncate_metrics(ck.state.next_epoch).unwrap();
    run.truncate_lbfgs_steps(ck.state.next_epoch).unwrap();
    let (resumed_model, _) = train_components(
        build(),
        Some(&normalizers.y),
        &train_cfg,
        &device,
        &run,
        Some(1),
        Some(ck),
        None,
    );
    assert_same_metrics(&run.metrics().unwrap(), &straight.metrics().unwrap());
    assert_same_weights(&resumed_model, &straight_model);
    // lbfgs_steps.csv as the uninterrupted run's, but for the wall times
    let rows = |r: &RunDir| -> Vec<String> {
        fs::read_to_string(r.path().join("lbfgs_steps.csv"))
            .unwrap()
            .lines()
            .map(|l| l.rsplit_once(',').unwrap().0.to_string())
            .collect()
    };
    assert_eq!(rows(&run).len(), 4, "a header and three steps");
    if exact_backend() {
        assert_eq!(rows(&run), rows(&straight));
    }
    let _ = fs::remove_dir_all(&dir);
}

/// Data-parallel L-BFGS is out of scope: replicas with L-BFGS panic.
#[test]
#[should_panic(expected = "data-parallel L-BFGS is out of scope")]
fn lbfgs_with_replicas_panics() {
    use crate::neural_operators::training::config::{LbfgsArm, OptimizerConfig};
    let arm = LbfgsArm::new(burn::optim::LBFGSConfig::new());
    let train_cfg = TrainingConfig::new()
        .with_epochs(1)
        .with_batch_size(2)
        .with_optimizer(Some(OptimizerConfig::Lbfgs(arm)));
    let device = Device::default().autodiff();
    let c = parallel_components(&train_cfg, &device, std::slice::from_ref(&device));
    let _ = training_loop(c, &train_cfg, None, &mut |_, _| {}, None, None);
}

/// A hybrid run must train both optimisers: `adam_epochs` equal to `epochs` panics.
#[test]
#[should_panic(expected = "0 < adam_epochs < epochs")]
fn hybrid_without_gauss_newton_epochs_panics() {
    use crate::neural_operators::{
        optim::gauss_newton::GaussNewtonConfig, training::config::OptimizerConfig,
    };
    let cfg = TrainingConfig::new()
        .with_epochs(2)
        .with_optimizer(Some(OptimizerConfig::GaussNewton(GaussNewtonConfig::new())))
        .with_adam_epochs(Some(2));
    let _ = build_components::<FNO<3>, 3, 2, _>(
        &tiny_model_cfg(),
        &cfg,
        tiny_dataset(5),
        tiny_dataset(3),
        &Device::default().autodiff(),
    );
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
        build_components::<FNO<3>, 3, 2, _>(
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
    let ck = run.checkpoint_of::<FNO<3>>(&device).unwrap().unwrap();
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
    let last = run.checkpoint_of::<FNO<3>>(&device).unwrap().unwrap();
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
        build_components::<FNO<3>, 3, 2, _>(
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
    let ck = run.checkpoint_of::<FNO<3>>(&device).unwrap().unwrap();
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

/// `train` writes one `metrics.csv` row per epoch through the observer, and the
/// rows read back as exactly the metrics it returns.
#[test]
fn metrics_csv_holds_every_epoch() {
    let root = std::env::temp_dir().join(format!("sciml_rs_metrics_csv_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let train_cfg = TrainingConfig::new()
        .with_epochs(3)
        .with_batch_size(2)
        .with_test_batch_size(2);
    let run = RunDir::create(&root, "live").unwrap();
    let (_, metrics) = train::<FNO<3>, 3, 2, _>(
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

// --- Phase 10 T4: the wall-clock budget ---

/// A budget of 0 s stops after the first epoch and checkpoints it as the last; resumed
/// from that checkpoint the run trains nothing more. A budget that is never reached
/// trains every epoch.
#[test]
fn time_budget_stops_after_the_epoch_that_reaches_it() {
    let root = std::env::temp_dir().join(format!("sciml_rs_time_budget_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let device = Device::default().autodiff();
    let cfg = |budget: f64| {
        TrainingConfig::new()
            .with_epochs(3)
            .with_batch_size(2)
            .with_test_batch_size(2)
            .with_time_budget_s(Some(budget))
    };
    let build = |train_cfg: &TrainingConfig| {
        build_components::<FNO<3>, 3, 2, _>(
            &tiny_model_cfg(),
            train_cfg,
            tiny_dataset(5),
            tiny_dataset(3),
            &device,
        )
    };
    let run = RunDir::create(&root, "budget").unwrap();
    let stopped = cfg(0.0);
    run.save_configs(&tiny_model_cfg(), &stopped, &stopped)
        .unwrap();
    let (_, metrics) = training_loop(
        build(&stopped),
        &stopped,
        None,
        &mut |_, _| {},
        Some((&run, 10)),
        None,
    );
    assert_eq!(metrics.len(), 1);
    let ck = run.checkpoint_of::<FNO<3>>(&device).unwrap().unwrap();
    assert_eq!(ck.state.next_epoch, 1);

    let (_, resumed) = training_loop(
        build(&stopped),
        &stopped,
        None,
        &mut |_, _| {},
        None,
        Some(ck),
    );
    assert!(resumed.is_empty(), "{resumed:?}");

    let ample = cfg(1e9);
    let (_, all) = training_loop(build(&ample), &ample, None, &mut |_, _| {}, None, None);
    assert_eq!(all.len(), 3);
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
    let mut c = build_components::<FNO<3>, 3, 2, _>(
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
    let ck = run.checkpoint_of::<FNO<3>>(&device).unwrap().unwrap();
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

// --- optimisers CO1.1: AdamW, the step schedule and the squared loss ---

/// `a` and `b` on 5 tiny Burgers-shaped samples (3 batches of 2 per epoch) from one
/// initial model: two builds draw different weights while tests share Flex's RNG.
fn co1_1_pair(
    a: &TrainingConfig,
    b: &TrainingConfig,
    n_train: usize,
) -> [(FNO<3>, Vec<EpochMetrics>); 2] {
    let device = Device::default().autodiff();
    let build = |cfg: &TrainingConfig| {
        build_components::<FNO<3>, 3, 2, _>(
            &tiny_model_cfg(),
            cfg,
            tiny_dataset(n_train),
            tiny_dataset(3),
            &device,
        )
    };
    let ca = build(a);
    let mut cb = build(b);
    cb.model = ca.model.clone();
    [
        training_loop(ca, a, None, &mut |_, _| {}, None, None),
        training_loop(cb, b, None, &mut |_, _| {}, None, None),
    ]
}

fn co1_1_cfg(optimizer: OptimizerConfig, weight_decay: f32) -> TrainingConfig {
    TrainingConfig::new()
        .with_epochs(3)
        .with_batch_size(2)
        .with_weight_decay(weight_decay)
        .with_optimizer(Some(optimizer))
        .with_lr_schedule(Some(LrSchedule::Step { every_epochs: 1 }))
}

/// Proposal §9 (O1): AdamW at weight decay 0 is Adam at weight decay 0 under the same
/// schedule, to f32 round-off; at weight decay 0.1 the two differ (coupled against
/// decoupled), so the test sees which optimiser was built.
#[test]
fn adamw_without_weight_decay_equals_adam() {
    let [(adamw, _), (adam, _)] = co1_1_pair(
        &co1_1_cfg(OptimizerConfig::AdamW, 0.0),
        &co1_1_cfg(OptimizerConfig::Adam, 0.0),
        5,
    );
    for ((name, a), (_, b)) in weights(&adamw).iter().zip(&weights(&adam)) {
        for (x, y) in a.iter().zip(b) {
            assert!(
                (x - y).abs() <= 1e-6 * y.abs().max(1.0),
                "{name}: {x} != {y}"
            );
        }
    }
    let [(adamw, _), (adam, _)] = co1_1_pair(
        &co1_1_cfg(OptimizerConfig::AdamW, 0.1),
        &co1_1_cfg(OptimizerConfig::Adam, 0.1),
        5,
    );
    assert!(weights(&adamw) != weights(&adam), "AdamW trained as Adam");
}

/// The step schedule halves the learning rate every `every_epochs` epochs, at the
/// step that opens the epoch: with 3 steps per epoch and `every_epochs` 2, steps 0..15
/// run at lr (0–5), lr/2 (6–11) and lr/4 (12–14), bit for bit.
#[test]
fn step_schedule_halves_at_the_stated_epochs() {
    let lr = 1e-3;
    let cfg = TrainingConfig::new()
        .with_epochs(5)
        .with_batch_size(2)
        .with_learning_rate(lr)
        .with_lr_schedule(Some(LrSchedule::Step { every_epochs: 2 }));
    let mut c = build_components::<FNO<3>, 3, 2, _>(
        &tiny_model_cfg(),
        &cfg,
        tiny_dataset(5),
        tiny_dataset(3),
        &Device::default().autodiff(),
    );
    let got: Vec<u64> = (0..15)
        .map(|_| c.scheduler.step().base().to_bits())
        .collect();
    let want: Vec<u64> = (0..15)
        .map(|i| (lr * 0.5_f64.powi(i / 6)).to_bits())
        .collect();
    assert_eq!(got, want);
}

/// The squared loss changes the objective but not the metric: one batch per epoch, so
/// epoch 0's `train_l2` is the initial model's mean relative L² on the training set,
/// as with the default loss; the trained weights differ.
#[test]
fn squared_loss_trains_differently_and_reports_the_unsquared_l2() {
    let cfg = |loss| {
        TrainingConfig::new()
            .with_epochs(3)
            .with_batch_size(4)
            .with_loss(loss)
    };
    let [(squared, sq_metrics), (summed, metrics)] =
        co1_1_pair(&cfg(Some(TrainLoss::SquaredRelL2)), &cfg(None), 4);
    assert_eq!(
        sq_metrics[0].train_l2.to_bits(),
        metrics[0].train_l2.to_bits()
    );
    assert!(
        weights(&squared) != weights(&summed),
        "the loss changed nothing"
    );
}

/// The step the trainer takes under the squared loss is one optimiser step on
/// `squared_rel_l2`'s gradient, taken by hand from the same model, batch and fresh
/// optimiser: equal weights (bit for bit on flex).
#[test]
fn squared_loss_steps_on_the_squared_objective() {
    let cfg = TrainingConfig::new()
        .with_epochs(1)
        .with_batch_size(4)
        .with_loss(Some(TrainLoss::SquaredRelL2));
    let device = Device::default().autodiff();
    let build = || {
        build_components::<FNO<3>, 3, 2, _>(
            &tiny_model_cfg(),
            &cfg,
            tiny_dataset(4),
            tiny_dataset(3),
            &device,
        )
    };
    let trained = build();
    let mut by_hand = build();
    let model = trained.model.clone();
    let (trained, _) = training_loop(trained, &cfg, None, &mut |_, _| {}, None, None);

    let batch = by_hand.train_loader.iter().next().unwrap().unwrap();
    let StepLoss { out, target, .. } = forward_loss(&model, batch, None);
    let grads = GradientsParams::from_grads(squared_rel_l2(out, target).backward(), &model);
    let lr = by_hand.scheduler.step();
    let model = by_hand.optim.step(lr, model, grads);
    assert_same_weights(&trained, &model);
}

/// The squared loss is for the one-device Adam path.
#[test]
#[should_panic(expected = "the squared loss trains on one device")]
fn squared_loss_with_replicas_panics() {
    let train_cfg = TrainingConfig::new()
        .with_epochs(1)
        .with_batch_size(2)
        .with_loss(Some(TrainLoss::SquaredRelL2));
    let device = Device::default().autodiff();
    let c = parallel_components(&train_cfg, &device, std::slice::from_ref(&device));
    let _ = training_loop(c, &train_cfg, None, &mut |_, _| {}, None, None);
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

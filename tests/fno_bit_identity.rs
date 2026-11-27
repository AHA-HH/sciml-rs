//! Constraint 1 of the library objective (CM0.3): the FNO's outputs, bit for bit.
//!
//! Each case writes its results as raw bytes to `target/bit-identity/<case>.bin`. The
//! file uses only names on `main` since optimisers CO2.1 (the generic stack of CM0.3, the
//! L-BFGS arm), so `/verify` runs this same file on a `main` worktree and on the branch
//! and `cmp`s the dumps: the trusted path is `main`'s code. Here the test only checks that every case runs and is deterministic
//! within the process (flex's bits need not agree across machines, so no golden values).
//! Ignored by default (about 4 minutes in a debug build): run it with
//! `cargo test --release --test fno_bit_identity -- --ignored`.
//! Files are compared by what they hold, not their bytes: Burnpack records each
//! parameter's random `ParamId`, so even `main` never writes the same bytes twice. Wall
//! times (`epoch_time_s`, the checkpoint's `training_time_s`) are left out.

use std::{fs, path::PathBuf};

use burn::{
    optim::LBFGSConfig,
    prelude::*,
    tensor::{Device, TensorData},
};
use ndarray::{ArrayD, IxDyn};
use sciml_rs::neural_operators::{
    data::{
        dataset::OperatorDataset,
        loaders::{base_dataset::DatasetConfig, darcy::DarcyConfig},
        transforms::normalizers::UnitGaussianNormalizer,
    },
    models::{
        fno::{FNO, FNOConfig},
        predict::predict_host,
    },
    optim::gauss_newton::{GaussNewtonConfig, params},
    runs::{EpochMetrics, RunDir},
    training::{
        config::{LbfgsArm, OptimizerConfig, TrainingConfig},
        trainer::{build_components, train_components},
    },
};

/// The Phase 5 baseline's model at n = 65 (`examples/train/poisson.rs`): s = 64, modes 12,
/// width 32, 4 layers, padding 4.
fn phase5_model() -> FNOConfig {
    FNOConfig::new(vec![12, 12], 1, 1)
        .with_hidden_channels(32)
        .with_n_layers(4)
        .with_padding(Some(4))
}

/// `n` samples on an `s`×`s` grid, deterministic (no RNG): inputs `[n, s, s, 1]`, targets
/// `[n, s, s]`, nonzero.
fn dataset(n: usize, s: usize, phase: f64) -> OperatorDataset<f64> {
    let k = |i: &IxDyn| ((i[0] * s + i[1]) * s + i[2]) as f64;
    let inputs = ArrayD::from_shape_fn(IxDyn(&[n, s, s, 1]), |i| (k(&i) * 0.37 + phase).sin());
    let targets = ArrayD::from_shape_fn(IxDyn(&[n, s, s]), |i| 2.0 + (k(&i) * 0.11 + phase).cos());
    OperatorDataset::new(inputs, targets)
}

fn out_dir() -> PathBuf {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/bit-identity");
    fs::create_dir_all(&d).unwrap();
    d
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn tensor_bytes<const D: usize>(t: Tensor<D>) -> Vec<u8> {
    f32_bytes(&t.into_data().try_to_vec::<f32>().unwrap())
}

/// The fields of `m` that do not depend on the clock.
fn metrics_bytes(m: &EpochMetrics) -> Vec<u8> {
    let mut b = (m.epoch as u64).to_le_bytes().to_vec();
    b.extend(f32_bytes(&[m.train_mse, m.train_l2, m.test_l2]));
    b.extend(m.current_lr.to_le_bytes());
    b
}

/// Every float parameter of `model`, in module order.
fn param_bytes<const R: usize>(model: &FNO<R>) -> Vec<u8> {
    tensor_bytes(params(model).to_flat())
}

/// Writes `parts` to `target/bit-identity/<case>.bin` and returns them joined.
fn dump(case: &str, parts: Vec<Vec<u8>>) -> Vec<u8> {
    let bytes = parts.concat();
    fs::write(out_dir().join(format!("{case}.bin")), &bytes).unwrap();
    bytes
}

/// A fixed Phase 5 FNO forward: seeded weights, deterministic input `[4, 64, 64, 1]`.
fn fixed_forward() -> Vec<u8> {
    let device = Device::default();
    device.seed(42);
    let model: FNO<4> = phase5_model().init::<4>(&device);
    let x: Vec<f32> = (0..4 * 64 * 64).map(|i| (i as f32 * 0.013).sin()).collect();
    let x = Tensor::<4>::from_data(TensorData::new(x, [4, 64, 64, 1]), &device);
    dump("forward", vec![tensor_bytes(model.forward(x))])
}

/// One training epoch through `build_components` and `train_components` (the path
/// `train` and `train_poisson` take) with a checkpoint, then the run read back:
/// the epoch's metrics, the trained, reloaded and checkpointed parameters, the
/// checkpoint's epoch counters, and the reloaded models'
/// predictions (`load_model`, `RunDir::checkpoint_of`, `predict_host`).
fn one_epoch(
    case: &str,
    model_cfg: &FNOConfig,
    train_cfg: &TrainingConfig,
    s: usize,
    n: (usize, usize),
) -> Vec<u8> {
    let root = std::env::temp_dir().join(format!("sciml_rs_bit_{case}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let (train, test) = (dataset(n.0, s, 0.0), dataset(n.1, s, 0.5));
    let y_norm = UnitGaussianNormalizer::fit(train.targets());
    let device = Device::default().autodiff();
    let components =
        build_components::<FNO<4>, 4, 3, _>(model_cfg, train_cfg, train, test, &device);
    let run = RunDir::create(&root, case).unwrap();
    // the data are synthetic: a Darcy-shaped data config (subsampling 1) for `data_cfg.json`
    let data_cfg = DarcyConfig::new(
        DatasetConfig {
            n_train: n.0,
            n_test: n.1,
        },
        1,
    );
    run.save_configs(model_cfg, train_cfg, &data_cfg).unwrap();
    let (model, metrics) = train_components(
        components,
        Some(&y_norm),
        train_cfg,
        &device,
        &run,
        Some(1),
        None,
        None,
    );
    run.save_weights(&model).unwrap();
    let trained = param_bytes(&model);

    let plain = Device::default();
    let inputs = dataset(2, s, 0.9).inputs().mapv(|v| v as f32);
    let loaded: FNO<4> = run.load_model::<FNO<4>>(&plain).unwrap();
    let ck = run.checkpoint_of::<FNO<4>>(&plain).unwrap().unwrap();
    let resumed = ck.model;
    let state = [ck.state.next_epoch, ck.state.resumes].map(|v| v as u64);
    // its fields, not `to_bytes`, which records the random `ParamId`s
    let gn_state = ck.gn.map_or_else(Vec::new, |g| {
        let mut b = g.lambda.to_le_bytes().to_vec();
        b.extend(g.step.to_le_bytes());
        if let Some(w) = &g.warm_start {
            b.extend(tensor_bytes(w.to_flat()));
        }
        b
    });
    let predict = |m: &FNO<4>| {
        let p = predict_host::<4>(m, inputs.view(), 1, &plain);
        f32_bytes(&p.iter().map(|&v| v as f32).collect::<Vec<_>>())
    };
    let parts = vec![
        metrics.iter().flat_map(metrics_bytes).collect(),
        trained,
        param_bytes(&loaded),
        param_bytes(&resumed),
        state.iter().flat_map(|v| v.to_le_bytes()).collect(),
        gn_state,
        predict(&loaded),
        predict(&resumed),
    ];
    let _ = fs::remove_dir_all(&root);
    dump(case, parts)
}

/// One Phase 5 epoch: the baseline model and `train_poisson`'s Adam settings (batch 20,
/// lr 1e-3, weight decay 1e-4, min lr 1e-5, seed 42), on 30 training samples at s = 64:
/// two batches, the second partial, so two scheduler steps and the epoch's accumulation.
fn phase5_epoch() -> Vec<u8> {
    let train_cfg = TrainingConfig::new()
        .with_epochs(1)
        .with_batch_size(20)
        .with_learning_rate(1e-3)
        .with_weight_decay(1e-4)
        .with_min_lr(1e-5)
        .with_seed(42);
    one_epoch("phase5_epoch", &phase5_model(), &train_cfg, 64, (30, 4))
}

/// One Gauss–Newton epoch (Phase 10 T3) on a tiny model: two steps of batch 2, CG capped
/// at 5 iterations, the adjoint check on; the checkpoint's Gauss–Newton state is dumped.
fn gauss_newton_epoch() -> Vec<u8> {
    let model_cfg = FNOConfig::new(vec![2, 2], 1, 1)
        .with_hidden_channels(4)
        .with_n_layers(2);
    let gn = GaussNewtonConfig::new()
        .with_cg_max_iters(5)
        .with_adjoint_check_every(1);
    let train_cfg = TrainingConfig::new()
        .with_epochs(1)
        .with_batch_size(2)
        .with_test_batch_size(2)
        .with_optimizer(Some(OptimizerConfig::GaussNewton(gn)));
    one_epoch("gauss_newton_epoch", &model_cfg, &train_cfg, 8, (4, 2))
}

/// One L-BFGS epoch (optimisers CO2.1) on the Gauss–Newton case's tiny model: one
/// `LBFGS::step` on the full batch of 4, history 5, at most 5 inner iterations, no line
/// search; the trained, reloaded and checkpointed parameters cover the record.
fn lbfgs_epoch() -> Vec<u8> {
    let model_cfg = FNOConfig::new(vec![2, 2], 1, 1)
        .with_hidden_channels(4)
        .with_n_layers(2);
    let arm = LbfgsArm::new(LBFGSConfig::new().with_history_size(5).with_max_iter(5));
    let train_cfg = TrainingConfig::new()
        .with_epochs(1)
        .with_batch_size(4)
        .with_test_batch_size(2)
        .with_optimizer(Some(OptimizerConfig::Lbfgs(arm)));
    one_epoch("lbfgs_epoch", &model_cfg, &train_cfg, 8, (4, 2))
}

#[test]
#[ignore = "the bit-identity dump /verify compares against main (CM0.3); run in release"]
fn fno_outputs_dumped_and_deterministic() {
    for (case, f) in [
        ("forward", fixed_forward as fn() -> Vec<u8>),
        ("phase5_epoch", phase5_epoch),
        ("gauss_newton_epoch", gauss_newton_epoch),
        ("lbfgs_epoch", lbfgs_epoch),
    ] {
        let first = f();
        assert!(!first.is_empty(), "{case}: nothing dumped");
        // `assert!`, not `assert_eq!`: a failure would print megabytes of bytes
        assert!(first == f(), "{case}: not deterministic within the process");
    }
}

//! Evaluates a saved Poisson run at resolutions it was never trained on: the resolution
//! study of design §7. Errors 3 and 4 on each test file, with the weights unchanged.
//!
//! Run with: cargo run --release --example poisson_resolution -- runs/poisson_fno_<...>
//! poisson_n65_K32_test.npz poisson_n129_K32_test.npz ... (add `--features metal` for the GPU).
//!
//! Each test file is named `poisson_n{n}_K{K}_test.npz` in `datasets/poisson/`; its train
//! split, the same name with `_train`, must sit beside it. The model loads as in
//! `predict_poisson` (model_cfg.json, model_weights.bpk). The x and y normalizers are
//! pointwise, fitted at the training s, so at each file's n they are refit by
//! `load_poisson_uniform` on that train split: the same training fields (seeds and K) on
//! the finer grid (design §12, decision 18). The weights are never refit. At the run's own
//! n the refit normalizers are the saved ones, and the mean error 3 is cross-checked
//! against the run's last test_l2. Needs the datasets: see datasets/README.md.

use std::path::{Path, PathBuf};

use burn::{
    prelude::*,
    store::{BurnpackStore, ModuleSnapshot},
    tensor::Device,
};
use ndarray::{Array3, s};
use sciml_rs::neural_operators::{
    data::{
        loaders::{
            base_dataset::HasBaseConfig,
            poisson::{PoissonConfig, load_poisson_cheb, load_poisson_uniform},
        },
        transforms::normalizers::Normalizer,
    },
    metrics::{
        io::read_metrics_csv,
        poisson::{ErrorStats, poisson_errors},
    },
    models::fno::FNOConfig,
};

const BATCH_SIZE: usize = 20;
/// Mean error 3 at the run's n against its last test_l2, relative (the run's value is f32).
const ERROR3_TOL: f64 = 1e-5;

/// The backend cargo features compiled into this build, in `Device::default()`'s order.
fn backend_features() -> Vec<&'static str> {
    [
        ("cuda", cfg!(feature = "cuda")),
        ("metal", cfg!(feature = "metal")),
        ("wgpu", cfg!(feature = "wgpu")),
        ("flex", cfg!(feature = "flex")),
    ]
    .into_iter()
    .filter(|&(_, on)| on)
    .map(|(name, _)| name)
    .collect()
}

/// (n, K) from a file name `poisson_n{n}_K{K}_test.npz`.
///
/// # Panics
/// If the name does not have that form.
fn parse_test_file(name: &str) -> (usize, usize) {
    let parse = || -> Option<(usize, usize)> {
        let rest = name.strip_prefix("poisson_n")?.strip_suffix("_test.npz")?;
        let (n, k) = rest.split_once("_K")?;
        Some((n.parse().ok()?, k.parse().ok()?))
    };
    parse().unwrap_or_else(|| panic!("{name}: expected poisson_n{{n}}_K{{K}}_test.npz"))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let usage = "usage: poisson_resolution <run_dir> <test file> [<test file> ...]";
    let dir = PathBuf::from(args.next().expect(usage));
    let test_files: Vec<String> = args.collect();
    assert!(!test_files.is_empty(), "{usage}");
    // Inference only: no autodiff, so forward passes record no backward graph.
    let device = Device::default();

    let model_cfg = FNOConfig::load(dir.join("model_cfg.json")).expect("load model config");
    let mut model = model_cfg.init::<4>(&device);
    let mut store = BurnpackStore::from_file(dir.join("model_weights.bpk"));
    model.load_from(&mut store).expect("load model weights");

    let trained_cfg = PoissonConfig::load(dir.join("data_cfg.json")).expect("load dataset config");
    let n_test = trained_cfg.base().n_test;
    let datasets = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/poisson");

    println!(
        "run {} (trained at n = {}), {n_test} test samples, device {device:?}, backend features {:?}",
        dir.display(),
        trained_cfg.n,
        backend_features()
    );
    println!(
        "{:>5} {:>5} {:>4}  {:>11} {:>11}  {:>11} {:>11}  {:>11}",
        "n", "s", "K", "e3 mean", "e3 max", "e4 mean", "e4 max", "e2 mean"
    );
    let mut cross_check_ok = true;
    for test_file in &test_files {
        let (n, k) = parse_test_file(test_file);
        let train_file = test_file.replace("_test.npz", "_train.npz");
        let cfg = PoissonConfig::new(trained_cfg.base().clone(), n);
        let s = cfg.s();

        // Refits both normalizers on the train split at this n (decision 18)
        let (_, test_data, normalizers) =
            load_poisson_uniform::<f32>(datasets.join(&train_file), datasets.join(test_file), &cfg)
                .unwrap_or_else(|e| panic!("could not load {test_file}: {e}"));
        let u = load_poisson_cheb(datasets.join(test_file), n, n_test)
            .unwrap_or_else(|e| panic!("could not load the CGL solutions of {test_file}: {e}"));

        // Forward pass in batches, in sample order; outputs are normalized [B, s, s, 1]
        let inputs = test_data.inputs();
        let mut pred = Array3::<f64>::zeros((n_test, s, s));
        for start in (0..n_test).step_by(BATCH_SIZE) {
            let end = (start + BATCH_SIZE).min(n_test);
            let batch = inputs.slice(s![start..end, .., .., ..]);
            let data: Vec<f32> = batch.iter().copied().collect();
            let x = Tensor::<4>::from_data(TensorData::new(data, batch.shape().to_vec()), &device);
            let out = model
                .forward(x)
                .into_data()
                .try_to_vec::<f32>()
                .expect("f32 model output");
            let out = Array3::from_shape_vec((end - start, s, s), out)
                .expect("output is [B, s, s, 1]")
                .mapv(f64::from);
            pred.slice_mut(s![start..end, .., ..]).assign(&out);
        }
        // Decode to physical scale in f64 with the refit y normalizer
        let pred = normalizers
            .y
            .decode(pred.into_dyn())
            .into_dimensionality::<ndarray::Ix3>()
            .expect("decoded predictions are [N, s, s]");

        let errors = poisson_errors(pred.view(), u.view());
        let e2 = ErrorStats::of(&errors.transfer);
        let e3 = ErrorStats::of(&errors.model);
        let e4 = ErrorStats::of(&errors.total);
        let trained_here = n == trained_cfg.n;
        println!(
            "{n:>5} {s:>5} {k:>4}  {:>11.4e} {:>11.4e}  {:>11.4e} {:>11.4e}  {:>11.4e}{}",
            e3.mean,
            e3.max,
            e4.mean,
            e4.max,
            e2.mean,
            if trained_here {
                "  <- trained here"
            } else {
                ""
            }
        );

        if trained_here {
            let metrics = read_metrics_csv(dir.join("metrics.csv")).expect("read metrics.csv");
            let test_l2 = f64::from(metrics.last().expect("metrics.csv has no epochs").test_l2);
            let rel = (e3.mean - test_l2).abs() / test_l2.abs();
            let ok = rel <= ERROR3_TOL;
            println!(
                "      check: mean error 3 vs metrics.csv test_l2 {test_l2:.6e}: rel diff {rel:.1e} (tol {ERROR3_TOL:e}) {}",
                if ok { "ok" } else { "FAILED" }
            );
            cross_check_ok &= ok;
        }
    }

    if !cross_check_ok {
        eprintln!("a cross-check failed; see above");
        std::process::exit(1);
    }
}

//! Evaluates a saved Poisson run at resolutions it was never trained on: the resolution
//! study of design §7. Errors 3 and 4 at each n, with the weights unchanged.
//!
//! Run with: cargo run --release --example poisson_resolution -- runs/poisson_fno_<...> 65 129 257
//! (add `--features metal` for the GPU).
//!
//! K stays at the run's own (its stage's, from data_cfg.json's n), so each n reads the
//! split pair `poisson_n{n}_K{K}_{train,test}.npz` from `datasets/poisson/`. The model
//! loads as in `predict_poisson`. The x and y normalizers are pointwise, fitted at the
//! training s, so at each n they are refit by `load_poisson_uniform` on that n's train
//! split: the same training fields (seeds and K) on the finer grid (design §12, decision
//! 18). The weights are never refit. At the run's own n the refit normalizers are the
//! saved ones, and the mean error 3 is cross-checked against the run's last test_l2.
//! Needs the datasets: see datasets/README.md.

use std::path::{Path, PathBuf};

use burn::{prelude::*, tensor::Device};
use sciml_rs::neural_operators::{
    data::loaders::poisson::{
        PoissonConfig, PoissonSet, load_poisson_cheb, load_poisson_uniform, stage,
    },
    metrics::{
        poisson::{check_error3, poisson_errors},
        stats::ErrorStats,
    },
    models::run::{load_fno, predict_host},
    utils::backend::backend_features,
};

const BATCH_SIZE: usize = 20;

fn main() {
    let mut args = std::env::args().skip(1);
    let usage = "usage: poisson_resolution <run_dir> <n> [<n> ...]";
    let dir = PathBuf::from(args.next().expect(usage));
    let ns: Vec<usize> = args
        .map(|a| {
            a.parse()
                .unwrap_or_else(|e| panic!("n must be an integer, got {a:?}: {e}"))
        })
        .collect();
    assert!(!ns.is_empty(), "{usage}");
    // Inference only: no autodiff, so forward passes record no backward graph.
    let device = Device::default();
    let model = load_fno::<4>(&dir, &device).unwrap_or_else(|e| panic!("load the model: {e}"));

    let trained_cfg = PoissonConfig::load(dir.join("data_cfg.json")).expect("load dataset config");
    let k = stage(trained_cfg.n)
        .unwrap_or_else(|e| panic!("the run's stage: {e}"))
        .k;
    let n_test = trained_cfg.base.n_test;
    let datasets = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/poisson");

    println!(
        "run {} (trained at n = {}, K = {k}), {n_test} test samples, device {device:?}, backend features {:?}",
        dir.display(),
        trained_cfg.n,
        backend_features()
    );
    println!(
        "{:>5} {:>5} {:>4}  {:>11} {:>11}  {:>11} {:>11}  {:>11}",
        "n", "s", "K", "e3 mean", "e3 max", "e4 mean", "e4 max", "e2 mean"
    );
    let mut cross_check_ok = true;
    for n in ns {
        let set = PoissonSet { n, k };
        let test_file = datasets.join(set.npz("test"));
        let cfg = PoissonConfig::new(trained_cfg.base.clone(), n);
        let s = cfg.s();

        // Refits both normalizers on the train split at this n (decision 18)
        let (_, test_data, normalizers) =
            load_poisson_uniform::<f32>(datasets.join(set.npz("train")), &test_file, &cfg)
                .unwrap_or_else(|e| panic!("could not load {}: {e}", set.stem("test")));
        let u = load_poisson_cheb(&test_file, n, n_test).unwrap_or_else(|e| {
            panic!(
                "could not load the CGL solutions of {}: {e}",
                set.stem("test")
            )
        });

        // Decode to physical scale in f64 with the refit y normalizer
        let pred = predict_host(&model, test_data.inputs().view(), BATCH_SIZE, &device);
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
            let check = check_error3(dir.join("metrics.csv"), e3.mean).expect("read metrics.csv");
            println!("      check: {check}");
            cross_check_ok &= check.passed;
        }
    }

    if !cross_check_ok {
        eprintln!("a cross-check failed; see above");
        std::process::exit(1);
    }
}

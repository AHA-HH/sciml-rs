//! Poisson datasets on the Chebyshev grid (design §5.1, §5.3)
//!
//! Run with: cargo run --release --features chebyshev --example generate_poisson
//!
//! Writes one `.npz` and one JSON sidecar per split to `datasets/poisson/` (git-ignored),
//! named `poisson_n{n}_K{K}_{split}`. Each sample is a GRF forcing f and its solver-A
//! solution u on the n × n CGL grid; every label is checked against the exact sine-series
//! solution (enforced at 1e-8 for n ≥ 65) and each split records its transfer error
//! (error 2). See `datasets/README.md` for the file layout.
//!
//! Edit `DATASETS` to choose what to generate. Stages 1 and 2 are on by default; stages 3
//! and 4 and the K = 32 evaluation sets 3e and 4e are commented out.

use sciml_rs::neural_operators::pde::poisson::dataset::{SplitSpec, generate, write_split};
use std::path::Path;
use std::process::Command;

/// Base seed of every training split: sample i uses seed TRAIN_SEED + i.
const TRAIN_SEED: u64 = 0;
/// Base seed of every test split, far from the training seeds. The same seeds are used
/// at every resolution (design §5.3).
const TEST_SEED: u64 = 1_000_000;
const N_TRAIN: usize = 1000;
const N_TEST: usize = 200;

/// (split name, n, explicit K or None for K(n), base seed, sample count).
const DATASETS: &[(&str, usize, Option<usize>, u64, usize)] = &[
    // Stage 1: n = 33, K = 16 (tests and debugging; ε_K and the label check not enforced).
    ("train", 33, None, TRAIN_SEED, N_TRAIN),
    ("test", 33, None, TEST_SEED, N_TEST),
    // Stage 2: n = 65, K = 32.
    ("train", 65, None, TRAIN_SEED, N_TRAIN),
    ("test", 65, None, TEST_SEED, N_TEST),
    // Stage 3: n = 129, K = 64.
    // ("train", 129, None, TRAIN_SEED, N_TRAIN),
    // ("test", 129, None, TEST_SEED, N_TEST),
    // Stage 4: n = 257, K = 64.
    // ("train", 257, None, TRAIN_SEED, N_TRAIN),
    // ("test", 257, None, TEST_SEED, N_TEST),
    // 3e and 4e: the stage 2 test fields (K = 32, its seeds) at n = 129 and 257.
    // ("test", 129, Some(32), TEST_SEED, N_TEST),
    // ("test", 257, Some(32), TEST_SEED, N_TEST),
];

fn main() {
    let out = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("datasets")
        .join("poisson");
    let commit = git_commit();
    println!("writing to {} (commit {commit})", out.display());

    for &(name, n, k, base_seed, count) in DATASETS {
        let spec = SplitSpec {
            n,
            k,
            base_seed,
            count,
        };
        let split = generate(&spec).unwrap_or_else(|e| panic!("{name} at n = {n}: {e}"));
        let (npz, _) = write_split(&out, name, &split, &commit)
            .unwrap_or_else(|e| panic!("writing {name} at n = {n}: {e}"));
        let mb = std::fs::metadata(&npz).map_or(f64::NAN, |m| m.len() as f64 / 1e6);
        println!(
            "n = {n:3}, K = {:2}, {name:5} N = {count:4}: label check max {:.1e} mean {:.1e} \
             ({}), error 2 max {:.1e} mean {:.1e}, {:.2} s, {mb:.1} MB",
            split.k,
            split.label_check.max,
            split.label_check.mean,
            if split.label_check_enforced {
                "enforced"
            } else {
                "recorded"
            },
            split.error2.max,
            split.error2.mean,
            split.wall_time.as_secs_f64(),
        );
    }
}

/// `git rev-parse HEAD`, or "unknown" if git is unavailable or this is not a checkout.
fn git_commit() -> String {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map_or_else(|| "unknown".to_string(), |s| s.trim().to_string())
}

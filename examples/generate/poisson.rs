//! Poisson datasets on the Chebyshev grid (design §5.1, §5.3)
//!
//! Run with: cargo run --release --features datagen --example generate_poisson
//!
//! Writes one `.npz` and one JSON sidecar per split to `datasets/poisson/` (git-ignored),
//! named `poisson_n{n}_K{K}_{split}`. Each sample is a GRF forcing f and its solver-A
//! solution u on the n × n CGL grid; every label is checked against the exact sine-series
//! solution (enforced at 1e-8 for n ≥ 65) and each split records its transfer error
//! (error 2). See `datasets/README.md` for the file layout.
//!
//! Generates each of `catalogue::SPLITS` (train, test) of every set in `catalogue::GENERATED`, in its order;
//! filter that list in `main` to choose what to generate. Stage 4 and the K = 32 sets at
//! n = 129 and 257 are used only by the resolution study (design §7): their train splits
//! exist only to refit the pointwise normalisers at the new n.

use sciml_rs::neural_operators::data::catalogue::{GENERATED, SPLITS};
use sciml_rs::neural_operators::pde::poisson::dataset::{SplitSpec, generate, write_split};
use sciml_rs::neural_operators::runs::git_commit;
use std::path::Path;

/// Base seed of every training split: sample i uses seed TRAIN_SEED + i.
const TRAIN_SEED: u64 = 0;
/// Base seed of every test split, far from the training seeds. The same seeds are used
/// at every resolution (design §5.3).
const TEST_SEED: u64 = 1_000_000;
const N_TRAIN: usize = 1000;
const N_TEST: usize = 200;

fn main() {
    let out = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("datasets")
        .join("poisson");
    let commit = git_commit();
    println!("writing to {} (commit {commit})", out.display());

    let splits = GENERATED.into_iter().flat_map(|set| {
        // (base seed, count) of each of the SPLITS, train then test
        SPLITS
            .into_iter()
            .zip([(TRAIN_SEED, N_TRAIN), (TEST_SEED, N_TEST)])
            .map(move |(name, (base_seed, count))| (name, SplitSpec::of_set(set, base_seed, count)))
    });
    for (name, spec) in splits {
        let (n, count) = (spec.n, spec.count);
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

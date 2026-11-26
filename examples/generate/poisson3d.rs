//! 3D Poisson datasets on the Chebyshev grid (3D proposal §3.3, §6; D3, D5)
//!
//! Run with: cargo run --release --features datagen --example generate_poisson3d [-- <n> ...]
//!
//! Writes one `.npz` and one JSON sidecar per split to `datasets/poisson/` (git-ignored,
//! beside the 2D files), named `poisson3d_n{n}_K{K}_{split}`. Each sample is a 3D GRF forcing
//! f and its A3 solution u on the n × n × n CGL grid; every label is checked against the
//! exact sine-series solution (enforced at 1e-8 for n ≥ 65) and each split records its
//! transfer error (error 2). See `datasets/README.md`.
//!
//! Generates each of `catalogue::SPLITS` (train, test) of every set in
//! `catalogue::GENERATED_3D` (n = 33 and 65), or only the sets whose n are given: the Mac
//! writes n = 33, the VM n = 65 (D8). The n = 65 train file is 4.4 GB (zip64).

use sciml_rs::neural_operators::data::catalogue::{GENERATED_3D, SPLITS};
use sciml_rs::neural_operators::pde::poisson::dataset::SplitSpec;
use sciml_rs::neural_operators::pde::poisson::dataset3d::{generate_3d, write_split_3d};
use sciml_rs::neural_operators::utils::backend::git_commit;
use std::path::Path;

/// Base seed of every training split: sample i uses seed TRAIN_SEED + i, as in 2D (the 3D
/// draws are on their own ChaCha8 stream, so the seeds never repeat a 2D sample).
const TRAIN_SEED: u64 = 0;
/// Base seed of every test split, far from the training seeds, the same at every n.
const TEST_SEED: u64 = 1_000_000;
const N_TRAIN: usize = 1000;
const N_TEST: usize = 200;

fn main() {
    let only: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| {
            a.parse()
                .unwrap_or_else(|_| panic!("expected n, got {a:?}"))
        })
        .collect();
    let sets: Vec<_> = GENERATED_3D
        .into_iter()
        .filter(|set| only.is_empty() || only.contains(&set.n))
        .collect();
    assert!(
        !sets.is_empty(),
        "no 3D set with n in {only:?}; expected some of {:?}",
        GENERATED_3D.map(|set| set.n)
    );
    let out = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("datasets")
        .join("poisson");
    let commit = git_commit();
    println!("writing to {} (commit {commit})", out.display());

    for set in sets {
        for (name, (base_seed, count)) in SPLITS
            .into_iter()
            .zip([(TRAIN_SEED, N_TRAIN), (TEST_SEED, N_TEST)])
        {
            let spec = SplitSpec {
                n: set.n,
                // GENERATED_3D is at K(n) (catalogue test `poisson3d_names`), so no explicit K
                k: None,
                base_seed,
                count,
            };
            let n = set.n;
            let split = generate_3d(&spec).unwrap_or_else(|e| panic!("{name} at n = {n}: {e}"));
            let (npz, _) = write_split_3d(&out, name, &split, &commit)
                .unwrap_or_else(|e| panic!("writing {name} at n = {n}: {e}"));
            let mb = std::fs::metadata(&npz).map_or(f64::NAN, |m| m.len() as f64 / 1e6);
            println!(
                "n = {n:2}, K = {:2}, {name:5} N = {count:4}: label check max {:.1e} mean {:.1e} \
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
}

//! Nested training subsets (design §13.5; Phase 7 T1): N samples of a train split chosen
//! by a seeded permutation, so that for one seed each smaller N is contained in every
//! larger one.
//!
//! The rule, part of a run's reproducibility: a Fisher–Yates shuffle of `0..population`
//! driven by `ChaCha8Rng::seed_from_u64(seed)` (rand_chacha 0.10.0), each swap index drawn
//! uniformly by rejection from `next_u64`; the first N of the permutation, sorted
//! ascending, so the loaded tensors keep file order. [`indices_hash`] fingerprints the
//! result, and the loader checks it, so a change to the rule cannot pass silently.

// Burn's `#[derive(Config)]` writes `field: field` in the `new` it generates, in an
// impl block outside the struct's attributes (burn-derive `config/analyzer_struct.rs`).
#![allow(clippy::redundant_field_names)]

use burn::config::Config;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{Rng, SeedableRng};

/// A seeded subset of a train split, saved in `data_cfg.json` (design §13.9). Its size N
/// is the config's `base.n_train`.
#[derive(Config, Debug)]
pub struct SubsetConfig {
    /// The permutation seed.
    pub seed: u64,
    /// The number of samples permuted, `0..population`: the train split's size when the
    /// subset was drawn.
    pub population: usize,
    /// [`indices_hash`] of the selected indices.
    pub hash: String,
}

impl SubsetConfig {
    /// The subset of `n` samples out of `population` for `seed`, with its hash.
    ///
    /// # Panics
    /// If `n > population`.
    pub fn draw(seed: u64, population: usize, n: usize) -> Self {
        Self::new(
            seed,
            population,
            indices_hash(&subset_indices(population, n, seed)),
        )
    }
}

/// The first `n` indices of the seeded permutation of `0..population` (module docs),
/// sorted ascending.
///
/// # Panics
/// If `n > population`.
pub fn subset_indices(population: usize, n: usize, seed: u64) -> Vec<usize> {
    assert!(n <= population, "subset of {n} out of {population} samples");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut perm: Vec<usize> = (0..population).collect();
    // Fisher–Yates: position i takes a uniform pick from the not-yet-placed tail i..
    for i in 0..population.saturating_sub(1) {
        let j = i + below(&mut rng, (population - i) as u64) as usize;
        perm.swap(i, j);
    }
    perm.truncate(n);
    perm.sort_unstable();
    perm
}

/// A uniform draw from `0..bound`, unbiased: draws at or above the largest multiple of
/// `bound` are rejected.
fn below(rng: &mut ChaCha8Rng, bound: u64) -> u64 {
    let limit = u64::MAX - u64::MAX % bound;
    loop {
        let r = rng.next_u64();
        if r < limit {
            return r % bound;
        }
    }
}

/// FNV-1a 64 over the indices as little-endian `u64`s, as 16 lowercase hex digits.
pub fn indices_hash(indices: &[usize]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in indices.iter().flat_map(|&i| (i as u64).to_le_bytes()) {
        h ^= byte as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subsets_are_nested() {
        for seed in [0, 1, 2] {
            let mut prev: Vec<usize> = Vec::new();
            for n in [25, 50, 100, 250, 500, 1000] {
                let idx = subset_indices(1000, n, seed);
                assert_eq!(idx.len(), n);
                assert!(idx.windows(2).all(|w| w[0] < w[1]), "sorted, distinct");
                assert!(prev.iter().all(|i| idx.binary_search(i).is_ok()), "n={n}");
                prev = idx;
            }
        }
    }

    #[test]
    fn subsets_are_reproducible_and_seeded() {
        assert_eq!(subset_indices(1000, 25, 0), subset_indices(1000, 25, 0));
        assert_eq!(
            SubsetConfig::draw(0, 1000, 25).hash,
            SubsetConfig::draw(0, 1000, 25).hash
        );
        assert_ne!(subset_indices(1000, 25, 0), subset_indices(1000, 25, 1));
        assert_ne!(subset_indices(1000, 25, 1), subset_indices(1000, 25, 2));
    }

    #[test]
    fn whole_population_is_every_index() {
        for seed in [0, 7] {
            assert_eq!(
                subset_indices(1000, 1000, seed),
                (0..1000).collect::<Vec<_>>()
            );
        }
        assert!(subset_indices(0, 0, 0).is_empty());
    }

    #[test]
    #[should_panic(expected = "subset of 4 out of 3 samples")]
    fn more_than_population_panics() {
        subset_indices(3, 4, 0);
    }

    /// Pins the rule: a change to the RNG, the shuffle or the hash fails here.
    #[test]
    fn subset_is_pinned() {
        assert_eq!(indices_hash(&[]), "cbf29ce484222325");
        let idx = subset_indices(1000, 25, 0);
        assert_eq!(
            idx,
            [
                18, 36, 87, 108, 132, 152, 185, 230, 266, 320, 400, 468, 469, 590, 597, 607, 652,
                712, 716, 776, 786, 791, 803, 864, 959
            ]
        );
        assert_eq!(indices_hash(&idx), "65104831155be79e");
    }
}

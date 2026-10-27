//! The dataset catalogue: the names of the dataset files under `datasets/`
//! (`datasets/README.md`), and the Poisson training stages.
//!
//! It loads nothing and sits below both the loaders and the generator
//! (`pde::poisson::dataset`), which name their files from here.

use std::fmt;

/// The Burgers dataset (FNO paper), one `.mat` file under `datasets/`.
pub struct BurgersFiles;

impl BurgersFiles {
    /// Both splits come from this one file (`load_burgers_uniform`).
    pub const DATA: &'static str = "burgers_data_R10.mat";
}

/// The Darcy dataset (FNO paper), one `.mat` file per split under `datasets/`.
pub struct DarcyFiles;

impl DarcyFiles {
    /// The train split.
    pub const TRAIN: &'static str = "piececonst_r421_N1024_smooth1.mat";
    /// The test split.
    pub const TEST: &'static str = "piececonst_r421_N1024_smooth2.mat";
}

/// One Poisson dataset on disk, under `datasets/poisson/`: `n` CGL nodes per axis and the
/// GRF truncation `k` (design §5.3). Its splits are files named by [`PoissonSet::stem`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoissonSet {
    /// CGL nodes per axis; the model grid is s = n − 1.
    pub n: usize,
    /// The GRF truncation K.
    pub k: usize,
}

impl PoissonSet {
    /// The file stem of one split, `poisson_n{n}_K{K}_{split}` (datasets/README.md), so
    /// that sets differing only in K (stage 3 and the K = 32 set 3e) get distinct files.
    pub fn stem(&self, split: &str) -> String {
        format!("poisson_n{}_K{}_{split}", self.n, self.k)
    }

    /// The `.npz` file of one split: `{stem}.npz`.
    pub fn npz(&self, split: &str) -> String {
        format!("{}.npz", self.stem(split))
    }

    /// The `.json` sidecar of one split: `{stem}.json`.
    pub fn json(&self, split: &str) -> String {
        format!("{}.json", self.stem(split))
    }
}

/// The stages a model trains on (design §5.3): K = K(n) at n = 65, 129 and 257.
pub const STAGES: [PoissonSet; 3] = [
    PoissonSet { n: 65, k: 32 },
    PoissonSet { n: 129, k: 64 },
    PoissonSet { n: 257, k: 64 },
];

/// The splits of every Poisson set, in the order `generate_poisson` writes them.
pub const SPLITS: [&str; 2] = ["train", "test"];

/// Every set `generate_poisson` writes, in its order, each as the [`SPLITS`]:
/// stage 1 (n = 33, K = 16; tests and debugging), the [`STAGES`], then 3e and 4e, the
/// K = 32 fields of stage 2 at n = 129 and 257 for the resolution study (design §7).
pub const GENERATED: [PoissonSet; 6] = [
    PoissonSet { n: 33, k: 16 },
    STAGES[0],
    STAGES[1],
    STAGES[2],
    PoissonSet { n: 129, k: 32 },
    PoissonSet { n: 257, k: 32 },
];

/// The error of [`stage`]: no stage has this n.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoStage(pub usize);

impl fmt::Display for NoStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ns: Vec<usize> = STAGES.iter().map(|set| set.n).collect();
        write!(
            f,
            "no Poisson stage with n = {}; expected one of {ns:?}",
            self.0
        )
    }
}

impl std::error::Error for NoStage {}

/// The stage in [`STAGES`] with `n` CGL nodes per axis.
///
/// # Errors
/// [`NoStage`], whose message names the stages' n, if there is none.
pub fn stage(n: usize) -> Result<PoissonSet, NoStage> {
    STAGES.into_iter().find(|set| set.n == n).ok_or(NoStage(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_names_its_split_files() {
        let set = PoissonSet { n: 129, k: 32 };
        assert_eq!(set.stem("test"), "poisson_n129_K32_test");
        assert_eq!(set.npz("train"), "poisson_n129_K32_train.npz");
        assert_eq!(set.json("train"), "poisson_n129_K32_train.json");
    }

    #[test]
    fn stage_finds_each_stage_and_rejects_other_n() {
        for set in STAGES {
            assert_eq!(stage(set.n).unwrap(), set);
        }
        let msg = stage(33).unwrap_err().to_string();
        assert!(msg.contains("[65, 129, 257]"), "{msg}");
    }
}

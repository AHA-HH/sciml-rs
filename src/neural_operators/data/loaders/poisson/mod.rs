//! Poisson dataset: config and constructors (design §5.2).
//!
//! `PoissonConfig` holds the Chebyshev size `n` on top of the shared `DatasetConfig`. The
//! files are named by [`PoissonSet`](crate::neural_operators::data::catalogue::PoissonSet)
//! in the dataset catalogue.
//! `load_poisson_uniform` (in `uniform`) builds train/test `OperatorDataset`s from the
//! `.npz` files that `generate_poisson` writes (`datasets/README.md`): forcing `f` and
//! solution `u` on the n × n CGL grid. It reads and validates them, transfers every sample
//! to the uniform grid of size s = n − 1 and then normalizes, as Darcy does. Inputs carry
//! the forcing only; the model appends the 2D grid coordinates itself (see `FNO::forward`).
//! `load_poisson_uniform_test` builds the test split alone, encoded with a run's saved
//! normalizers, so predict neither reads nor transfers the train split.
//!
//! Transfer: T_cu (`chebyshev::Transfers::to_uniform`, CONVENTIONS §12) is applied along both
//! axes at load time, in f64, before normalization; transferred fields are never written
//! to disk. T_cu is polynomial interpolation on the CGL nodes, exact to round-off
//! (≤ 1e-15) on the analytic test fields, and for n ≥ 65 within 2e-8 on GRF solutions u
//! and 3.4e-6 on GRF forcings f (Phase 0 T3, `spikes/transfers/REPORT.md` table 1). The
//! round-trip error of design §7 (error 2) is recorded per split in the dataset's `.json`
//! sidecar by the generator; this loader does not read the sidecar. `SidecarRecord` is
//! that sidecar's type, which the generator writes and the evaluation reads.
//!
//! `load_poisson_cheb_field` (in `cheb`) reads one split's raw field, [`PoissonField::F`]
//! or [`PoissonField::U`], on the CGL grid, untransferred, with the same validation: the
//! reference for the errors on the Chebyshev grid (design §7).
//!
//! Normalization (CONVENTIONS §10, as in `load_darcy_uniform`): both inputs are encoded
//! with the `x` normalizer fit on `x_train`; both target splits stay in physical units, and
//! predictions are decoded back to physical scale before the loss, not targets encoded
//! forward. The returned `normalizers.y` is what the caller needs to do that decode.

// Burn's `#[derive(Config)]` writes `field: field` in the `new` it generates, in an
// impl block outside the struct's attributes (burn-derive `config/analyzer_struct.rs`).
#![allow(clippy::redundant_field_names)]

mod cheb;
mod uniform;

pub use cheb::{PoissonField, load_poisson_cheb_field};
pub use uniform::{load_poisson_uniform, load_poisson_uniform_test};

use crate::neural_operators::data::loaders::base_dataset::DatasetConfig;
use burn::config::Config;
use burn::serde::{Deserialize, Serialize};

/// Poisson-specific config: the Chebyshev size `n` of the stored fields.
#[derive(Config, Debug)]
pub struct PoissonConfig {
    /// Shared train/test sample counts.
    pub base: DatasetConfig,
    /// Number of CGL nodes per axis in the dataset files.
    pub n: usize,
}

impl PoissonConfig {
    /// Uniform (model) grid size s = n − 1 (design §12, decision 5). Saturates at 0 for
    /// n = 0; `load_poisson_uniform` rejects n < 4.
    pub fn s(&self) -> usize {
        self.n.saturating_sub(1)
    }
}

/// A split's `.json` sidecar (datasets/README.md): written by the generator
/// (`pde::poisson::dataset::write_split`), read with [`Config::load`] by the evaluation.
///
/// The fields are declared in key order, as the files hold them sorted. Serde derives
/// with `Config` implemented by hand, not `#[derive(Config)]`, because the keys `K`,
/// `K_explicit`, `K_max` and `eps_K` need renames, which that derive does not carry into
/// its generated serde struct.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "burn::serde")]
pub struct SidecarRecord {
    /// The GRF truncation K in effect.
    #[serde(rename = "K")]
    pub k: usize,
    /// Whether K was given explicitly rather than K(n).
    #[serde(rename = "K_explicit")]
    pub k_explicit: bool,
    /// The largest production truncation, `K_MAX`.
    #[serde(rename = "K_max")]
    pub k_max: usize,
    /// α of the GRF covariance.
    pub alpha: f64,
    /// Sample i uses seed `base_seed + i`.
    pub base_seed: u64,
    /// The `CONVENTION_VERSION` the split was generated under.
    pub convention_version: u32,
    /// The number of samples in the split.
    pub count: usize,
    /// The crate version that generated the split.
    pub crate_version: String,
    /// ε_K, the fraction of the GRF's expected energy outside the truncation K.
    #[serde(rename = "eps_K")]
    pub eps_k: f64,
    /// Error 2 (transfer) over the split.
    pub error2: SidecarError2,
    /// The commit that generated the split.
    pub git_commit: String,
    /// The label check over the split.
    pub label_check: SidecarLabelCheck,
    /// CGL nodes per axis.
    pub n: usize,
    /// The RNG the forcings were drawn with.
    pub rng: String,
    /// The uniform grid size s = n − 1.
    pub s: usize,
    /// The split's name, e.g. `"train"` or `"test"`.
    pub split: String,
    /// τ of the GRF covariance.
    pub tau: f64,
    /// The generation wall time in seconds.
    pub wall_time_s: f64,
}

impl Config for SidecarRecord {}

/// The sidecar's `error2` entry, in key order.
#[derive(Config, Debug)]
pub struct SidecarError2 {
    /// The transfer degree d.
    pub d: usize,
    /// Max over the split.
    pub max: f64,
    /// Mean over the split.
    pub mean: f64,
    /// The uniform grid size s.
    pub s: usize,
}

/// The sidecar's `label_check` entry, in key order.
#[derive(Config, Debug)]
pub struct SidecarLabelCheck {
    /// Whether the tolerance was enforced (n ≥ 65).
    pub enforced: bool,
    /// Max over the split.
    pub max: f64,
    /// Mean over the split.
    pub mean: f64,
    /// The label check's tolerance.
    pub tolerance: f64,
}

/// The test fixtures shared by the `cheb` and `uniform` tests.
#[cfg(test)]
mod fixtures {
    use super::*;
    use crate::neural_operators::chebyshev::nodes;
    use ndarray::{Array1, Array3, ArrayD};
    use ndarray_npy::NpzWriter;
    use std::fs::{self, File};
    use std::path::{Path, PathBuf};

    pub(super) const N: usize = 9;
    pub(super) const N_TRAIN: usize = 3;
    pub(super) const N_TEST: usize = 2;

    pub(super) fn cfg(n_train: usize, n_test: usize) -> PoissonConfig {
        PoissonConfig::new(DatasetConfig { n_train, n_test }, N)
    }

    /// A fresh, empty directory under the system temp dir, unique to `test` and this
    /// process.
    pub(super) fn fresh_dir(test: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("sciml_rs_poisson_{test}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Writes `f`, `u`, `x`, `y` to `dir/name` as an uncompressed `.npz`, as T2 does.
    pub(super) fn write_npz(
        dir: &Path,
        name: &str,
        f: &ArrayD<f64>,
        u: &ArrayD<f64>,
        x: &Array1<f64>,
        y: &Array1<f64>,
    ) -> PathBuf {
        let path = dir.join(name);
        let mut npz = NpzWriter::new(File::create(&path).unwrap());
        npz.add_array("f", f).unwrap();
        npz.add_array("u", u).unwrap();
        npz.add_array("x", x).unwrap();
        npz.add_array("y", y).unwrap();
        npz.finish().unwrap();
        path
    }

    /// A polynomial of degree < N in each variable, reproduced exactly by T_cu.
    fn poly(x: f64, y: f64) -> f64 {
        (1.0 + x - 2.0 * x.powi(3)) * (y * y + 0.5 * y.powi(4) - 0.3)
    }

    /// `[count, xs.len(), ys.len()]` with sample i = c_i · poly(x, y), where
    /// c_i = 1 + i + offset differs per sample so the pointwise std is nonzero.
    pub(super) fn poly_field(
        count: usize,
        offset: f64,
        xs: &Array1<f64>,
        ys: &Array1<f64>,
    ) -> ArrayD<f64> {
        Array3::from_shape_fn((count, xs.len(), ys.len()), |(i, a, b)| {
            (1.0 + i as f64 + offset) * poly(xs[a], ys[b])
        })
        .into_dyn()
    }

    /// Writes a valid train/test pair of polynomial fields on the CGL grid; f and u
    /// differ by scale (u = −2 f, test samples offset) so they are told apart.
    pub(super) fn write_valid_pair(dir: &Path) -> (PathBuf, PathBuf) {
        let x = nodes(N);
        let f_tr = poly_field(N_TRAIN, 0.0, &x, &x);
        let f_te = poly_field(N_TEST, 0.5, &x, &x);
        let train = write_npz(dir, "train.npz", &f_tr, &(&f_tr * -2.0), &x, &x);
        let test = write_npz(dir, "test.npz", &f_te, &(&f_te * -2.0), &x, &x);
        (train, test)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sidecar `generate_poisson` wrote, copied from `datasets/poisson/`.
    const SIDECAR_FIXTURE: &str =
        include_str!("../../../../../tests/fixtures/poisson_n65_K32_test.json");

    #[test]
    fn sidecar_fixture_parses_and_round_trips() {
        let sidecar = SidecarRecord::load_binary(SIDECAR_FIXTURE.as_bytes()).unwrap();
        assert_eq!(
            (sidecar.n, sidecar.s, sidecar.k, sidecar.count),
            (65, 64, 32, 200)
        );
        assert_eq!(sidecar.split, "test");
        assert_eq!((sidecar.error2.d, sidecar.error2.s), (2, 64));
        assert!(sidecar.label_check.enforced);

        // Written back as the writer writes it (pretty, plus a newline), every line matches
        // the fixture: keys, nesting and layout byte for byte, and numbers to an ulp, as
        // serde_json parses floats to within an ulp without its `float_roundtrip` feature.
        let written = burn::config::config_to_json(&sidecar) + "\n";
        assert_eq!(written.lines().count(), SIDECAR_FIXTURE.lines().count());
        for (w, f) in written.lines().zip(SIDECAR_FIXTURE.lines()) {
            if w == f {
                continue;
            }
            let (wk, wv) = w.split_once(": ").expect(w);
            let (fk, fv) = f.split_once(": ").expect(f);
            assert_eq!(wk, fk);
            let num = |v: &str| v.trim_end_matches(',').parse::<f64>().expect(v);
            let (wv, fv) = (num(wv), num(fv));
            assert!((wv - fv).abs() <= 1e-15 * fv.abs(), "{w} vs {f}");
        }
    }
}

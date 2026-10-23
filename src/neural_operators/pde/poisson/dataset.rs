//! Poisson datasets on the CGL grid: GRF forcings labelled by solver A (design §5.1).
//!
//! [`generate`] builds one split: for sample i it draws the forcing f of seed
//! `base_seed + i` ([`grf`](super::grf)), solves −Δu = f with
//! [`CollocationSolver::solve_full`], and records two errors, both relative
//! Clenshaw–Curtis L² on the CGL grid (CONVENTIONS §12):
//! - the **label check** ([`label_check`]): u_A against the exact sine-series solution
//!   u_series (design §4.3). For n ≥ [`LABEL_CHECK_MIN_N`] a sample above
//!   [`LABEL_CHECK_TOL`] aborts the split; below that n it is recorded only;
//! - **error 2** (design §7): the transfer round trip T_uc(T_cu u_A) − u_A with
//!   d = [`TRANSFER_DEGREE`] and s = n − 1 (design §12, decisions 8 and 13). It is
//!   recorded only, and the transferred fields are not kept (design §5.2).
//!
//! [`write_split`] writes the split as one `.npz` and one JSON sidecar. Labels come from
//! solver A only; solver C never enters this path (design §4.2).

use ndarray::{Array1, Array3, ArrayView2, Axis};
use ndarray_npy::NpzWriter;
use std::fmt;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use super::collocation::CollocationSolver;
use super::grf::{
    ALPHA, K_MAX, RNG, TAU, draw_xi, eps_k, evaluate, forcing_coeffs, k_default, solution_coeffs,
};
use crate::CONVENTION_VERSION;
use crate::neural_operators::chebyshev::{TRANSFER_DEGREE, Transfers, cc_rel_l2_error, nodes};
use crate::neural_operators::data::catalogue::PoissonSet;
use crate::neural_operators::data::loaders::poisson::{
    SidecarError2, SidecarLabelCheck, SidecarRecord,
};
use crate::neural_operators::metrics::stats::ErrorStats;

/// Bound on the label check, relative CC-L², enforced for n ≥ [`LABEL_CHECK_MIN_N`]
/// (design §5.1; Phase 0 T4 observed at most 7.4e-10 at n = 65).
pub const LABEL_CHECK_TOL: f64 = 1e-8;

/// The smallest n at which [`LABEL_CHECK_TOL`] is enforced. Stage 1 (n = 33) records
/// its label check without enforcing it (design §5.1).
pub const LABEL_CHECK_MIN_N: usize = 65;

/// What to generate for one split (design §5.1, §5.3).
#[derive(Clone, Debug, PartialEq)]
pub struct SplitSpec {
    /// CGL nodes per axis; the grid is `[n, n]`.
    pub n: usize,
    /// An explicit truncation K, or `None` for [`k_default`]`(n)`. Used only for the K = 32
    /// evaluation sets (design §3.3, §5.3).
    pub k: Option<usize>,
    /// Sample i uses seed `base_seed + i`.
    pub base_seed: u64,
    /// The number of samples N.
    pub count: usize,
}

impl SplitSpec {
    /// One split of the catalogued `set`, its K explicit only where it isn't
    /// [`k_default`]`(n)` (the K = 32 sets 3e and 4e), as `generate_poisson` writes it.
    ///
    /// # Panics
    /// If `set.n < 3`.
    pub fn of_set(set: PoissonSet, base_seed: u64, count: usize) -> Self {
        Self {
            n: set.n,
            k: (set.k != k_default(set.n)).then_some(set.k),
            base_seed,
            count,
        }
    }

    /// The truncation K in effect: the explicit one, else [`k_default`]`(n)`.
    ///
    /// # Panics
    /// If `k` is `None` and `n < 3`.
    pub fn truncation(&self) -> usize {
        self.k.unwrap_or_else(|| k_default(self.n))
    }
}

/// One generated split (design §5.1).
#[derive(Clone, Debug)]
pub struct Split {
    /// What was generated.
    pub spec: SplitSpec,
    /// The truncation K in effect ([`SplitSpec::truncation`]).
    pub k: usize,
    /// The forcings, `[N, n, n]`, each sample stored 'ij' (CONVENTIONS §12).
    pub f: Array3<f64>,
    /// The solver-A solutions, `[N, n, n]`, 'ij', zero on the boundary.
    pub u: Array3<f64>,
    /// The ascending CGL nodes, `[n]`, on both axes; [`write_split`] stores them as both
    /// `x` and `y`.
    pub x: Array1<f64>,
    /// The label check ‖u_A − u_series‖ / ‖u_series‖ over the samples.
    pub label_check: ErrorStats,
    /// Whether [`LABEL_CHECK_TOL`] was enforced (n ≥ [`LABEL_CHECK_MIN_N`]).
    pub label_check_enforced: bool,
    /// Error 2, ‖T_uc(T_cu u_A) − u_A‖ / ‖u_A‖ over the samples.
    pub error2: ErrorStats,
    /// Wall time of [`generate`].
    pub wall_time: Duration,
}

/// A sample whose label check exceeded [`LABEL_CHECK_TOL`] (design §5.1).
#[derive(Clone, Debug, PartialEq)]
pub struct LabelCheckError {
    /// The sample index i in its split.
    pub sample: usize,
    /// Its seed, `base_seed + i`.
    pub seed: u64,
    /// CGL nodes per axis.
    pub n: usize,
    /// The relative CC-L² discrepancy ‖u_A − u_series‖ / ‖u_series‖.
    pub error: f64,
}

impl fmt::Display for LabelCheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "label check failed at n = {}: sample {} (seed {}) has relative CC-L² error \
             {:e} against the exact series, above {LABEL_CHECK_TOL:e}",
            self.n, self.sample, self.seed, self.error
        )
    }
}

impl std::error::Error for LabelCheckError {}

/// The label check of one sample (design §4.3, §5.1): the relative Clenshaw–Curtis L²
/// discrepancy ‖u_A − u_series‖ / ‖u_series‖ (CONVENTIONS §12).
///
/// `u_a` and `u_series` are `[n, n]`, 'ij'; n is read from their shape and the weights
/// are [`clencurt`](crate::neural_operators::chebyshev::clencurt)`(n)`
/// ([`cc_rel_l2_error`]). `sample` and `seed` only label the error. Returns the discrepancy; for n ≥ [`LABEL_CHECK_MIN_N`] a value above
/// [`LABEL_CHECK_TOL`] (or NaN) is an error.
///
/// # Errors
/// [`LabelCheckError`] when the check is enforced and fails.
///
/// # Panics
/// If the fields differ in shape or are not square, or either axis has fewer than 2
/// nodes.
pub fn label_check(
    u_a: ArrayView2<f64>,
    u_series: ArrayView2<f64>,
    sample: usize,
    seed: u64,
) -> Result<f64, LabelCheckError> {
    let error = cc_rel_l2_error(u_a, u_series);
    let n = u_a.nrows();
    assert_eq!(n, u_a.ncols(), "label_check: fields must be square");
    if n >= LABEL_CHECK_MIN_N && (error.is_nan() || error > LABEL_CHECK_TOL) {
        return Err(LabelCheckError {
            sample,
            seed,
            n,
            error,
        });
    }
    Ok(error)
}

/// Generates one split (design §5.1): f, u_A, the label check and error 2.
///
/// Builds one [`CollocationSolver`] and one pair of transfers for the split, then for
/// sample i draws the forcing of seed `base_seed + i` ([`draw_xi`]) at the truncation
/// [`SplitSpec::truncation`], evaluates it at the CGL nodes, solves, and checks the label
/// against the exact series.
///
/// # Errors
/// [`LabelCheckError`] for the first sample whose label check fails, when
/// n ≥ [`LABEL_CHECK_MIN_N`].
///
/// # Panics
/// If `count` is 0, `n < 3`, K is 0 or above [`K_MAX`], or `base_seed + i` overflows.
pub fn generate(spec: &SplitSpec) -> Result<Split, LabelCheckError> {
    let start = Instant::now();
    let n = spec.n;
    assert!(spec.count > 0, "generate: count must be > 0");
    assert!(n >= 3, "generate: n must be >= 3, got {n}");
    let k = spec.truncation();
    let x = nodes(n);
    let solver = CollocationSolver::new(n, n);
    let transfers = Transfers::new(n);

    let mut f = Array3::zeros((spec.count, n, n));
    let mut u = Array3::zeros((spec.count, n, n));
    let mut label = Vec::with_capacity(spec.count);
    let mut err2 = Vec::with_capacity(spec.count);
    for (i, (mut f_i, mut u_i)) in f
        .axis_iter_mut(Axis(0))
        .zip(u.axis_iter_mut(Axis(0)))
        .enumerate()
    {
        let seed = spec
            .base_seed
            .checked_add(i as u64)
            .expect("generate: base_seed + i overflows u64");
        let c = forcing_coeffs(draw_xi(seed).view(), k);
        f_i.assign(&evaluate(c.view(), x.view(), x.view()));
        u_i.assign(&solver.solve_full(f_i.view()));
        let u_series = evaluate(solution_coeffs(c.view()).view(), x.view(), x.view());
        label.push(label_check(u_i.view(), u_series.view(), i, seed)?);
        let round_trip = transfers.to_cheb(transfers.to_uniform(u_i.view()).view());
        err2.push(cc_rel_l2_error(round_trip.view(), u_i.view()));
    }

    Ok(Split {
        spec: spec.clone(),
        k,
        f,
        u,
        x,
        label_check: ErrorStats::of(&label),
        label_check_enforced: n >= LABEL_CHECK_MIN_N,
        error2: ErrorStats::of(&err2),
        wall_time: start.elapsed(),
    })
}

/// The file stem of a split, `poisson_n{n}_K{K}_{name}`: [`PoissonSet::stem`] of its n
/// and K in effect.
pub fn split_stem(split: &Split, name: &str) -> String {
    PoissonSet {
        n: split.spec.n,
        k: split.k,
    }
    .stem(name)
}

/// Writes a split to `dir` as `{stem}.npz` and `{stem}.json`, with the stem from
/// [`split_stem`] (design §5.1). Creates `dir` if needed and overwrites existing files.
///
/// The `.npz` holds `f` and `u` (`[N, n, n]`, f64, 'ij') and the CGL nodes `x` and `y`
/// (`[n]`, both [`Split::x`]), uncompressed. The JSON sidecar records n and s, K and
/// whether it was explicit, τ, α, K_MAX, the RNG, the base seed and count, ε_K, the label
/// check (max, mean, enforced, tolerance), error 2 (max, mean, d, s),
/// [`CONVENTION_VERSION`], the crate version, `git_commit` and the wall time in seconds. `name` is the split's name, e.g.
/// `"train"` or `"test"`. Returns the two paths. The sidecar is a [`SidecarRecord`].
///
/// # Errors
/// Any I/O error from creating `dir` or writing either file.
pub fn write_split(
    dir: &Path,
    name: &str,
    split: &Split,
    git_commit: &str,
) -> io::Result<(PathBuf, PathBuf)> {
    fs::create_dir_all(dir)?;
    let stem = split_stem(split, name);
    let npz_path = dir.join(format!("{stem}.npz"));
    let json_path = dir.join(format!("{stem}.json"));

    let mut npz = NpzWriter::new(File::create(&npz_path)?);
    for (field, array) in [("f", &split.f), ("u", &split.u)] {
        npz.add_array(field, array).map_err(io::Error::other)?;
    }
    // `y` repeats `x`: the file format keeps both axes (design §5.1), and the loader checks
    // both against the CGL nodes.
    for (field, array) in [("x", &split.x), ("y", &split.x)] {
        npz.add_array(field, array).map_err(io::Error::other)?;
    }
    npz.finish().map_err(io::Error::other)?;

    let n = split.spec.n;
    let sidecar = SidecarRecord {
        k: split.k,
        k_explicit: split.spec.k.is_some(),
        k_max: K_MAX,
        alpha: ALPHA,
        base_seed: split.spec.base_seed,
        convention_version: CONVENTION_VERSION,
        count: split.spec.count,
        crate_version: env!("CARGO_PKG_VERSION").to_string(),
        eps_k: eps_k(split.k),
        error2: SidecarError2 {
            d: TRANSFER_DEGREE,
            max: split.error2.max,
            mean: split.error2.mean,
            s: n - 1,
        },
        git_commit: git_commit.to_string(),
        label_check: SidecarLabelCheck {
            enforced: split.label_check_enforced,
            max: split.label_check.max,
            mean: split.label_check.mean,
            tolerance: LABEL_CHECK_TOL,
        },
        n,
        rng: RNG.to_string(),
        s: n - 1,
        split: name.to_string(),
        tau: TAU,
        wall_time_s: split.wall_time.as_secs_f64(),
    };
    let text = serde_json::to_string_pretty(&sidecar).map_err(io::Error::other)?;
    fs::write(&json_path, text + "\n")?;
    Ok((npz_path, json_path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::data::io::readers::npz::NpzFileReader;
    use ndarray::Array2;

    /// u_A and u_series for one seed, n, K.
    fn a_and_series(solver: &CollocationSolver, n: usize, k: usize, seed: u64) -> [Array2<f64>; 2] {
        let x = nodes(n);
        let c = forcing_coeffs(draw_xi(seed).view(), k);
        let f = evaluate(c.view(), x.view(), x.view());
        [
            solver.solve_full(f.view()),
            evaluate(solution_coeffs(c.view()).view(), x.view(), x.view()),
        ]
    }

    #[test]
    fn collocation_matches_series_on_grf() {
        for n in [33, 65, 129] {
            let solver = CollocationSolver::new(n, n);
            let k = k_default(n);
            let errs: Vec<f64> = (0..5)
                .map(|seed| {
                    let [ua, us] = a_and_series(&solver, n, k, seed);
                    cc_rel_l2_error(ua.view(), us.view())
                })
                .collect();
            let max = errs.iter().copied().fold(0.0_f64, f64::max);
            println!("A vs series, n = {n}, K = {k}: max relative CC-L² {max:.1e}");
            if n >= 65 {
                assert!(max <= 1e-8, "n = {n}: {max:e}");
            }
        }
    }

    #[test]
    fn label_check_aborts() {
        let n = 65;
        let solver = CollocationSolver::new(n, n);
        let [ua, us] = a_and_series(&solver, n, k_default(n), 4);
        assert!(label_check(ua.view(), us.view(), 3, 4).is_ok());

        let mut corrupted = ua.clone();
        corrupted[[32, 32]] += 1e-4 * us.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let err = label_check(corrupted.view(), us.view(), 3, 4)
            .expect_err("a corrupted label at n = 65 must fail the check");
        assert_eq!((err.sample, err.seed, err.n), (3, 4, n));
        assert!(err.error > LABEL_CHECK_TOL);
        assert!(err.to_string().contains("sample 3 (seed 4)"), "{err}");

        // Not enforced below n = 65: the same corruption is recorded, not rejected.
        let n = 33;
        let solver = CollocationSolver::new(n, n);
        let [ua, us] = a_and_series(&solver, n, k_default(n), 4);
        let mut corrupted = ua.clone();
        corrupted[[16, 16]] += 1e-4 * us.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let e = label_check(corrupted.view(), us.view(), 3, 4).unwrap();
        assert!(e > LABEL_CHECK_TOL);
    }

    #[test]
    fn files_round_trip() {
        use crate::neural_operators::data::loaders::base_dataset::DatasetConfig;
        use crate::neural_operators::data::loaders::poisson::{
            PoissonConfig, PoissonField, load_poisson_cheb_field, load_poisson_uniform,
        };
        let spec = SplitSpec {
            n: 17,
            k: None,
            base_seed: 40,
            count: 2,
        };
        let split = generate(&spec).unwrap();
        assert_eq!(split.k, 8);
        assert!(!split.label_check_enforced);

        let dir =
            std::env::temp_dir().join(format!("sciml_rs_files_round_trip_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let (npz_path, json_path) = write_split(&dir, "train", &split, "abc123").unwrap();
        assert_eq!(npz_path, dir.join("poisson_n17_K8_train.npz"));

        let mut reader = NpzFileReader::new(&npz_path).unwrap();
        for (field, written) in [("f", &split.f), ("u", &split.u)] {
            let read = reader.read_field(field).unwrap();
            assert_eq!(read.shape(), &[2, 17, 17], "{field}");
            assert_eq!(read, written.clone().into_dyn(), "{field}");
        }
        for (field, written) in [("x", &split.x), ("y", &split.x)] {
            let read = reader.read_field(field).unwrap();
            assert_eq!(read.shape(), &[17], "{field}");
            assert_eq!(read, written.clone().into_dyn(), "{field}");
        }

        let text = fs::read_to_string(&json_path).unwrap();
        // The typed record writes the bytes the untyped writer did before it (phase 4s T19)
        let untyped = serde_json::json!({
            "split": "train",
            "n": 17,
            "s": 16,
            "K": split.k,
            "K_explicit": split.spec.k.is_some(),
            "tau": TAU,
            "alpha": ALPHA,
            "K_max": K_MAX,
            "rng": RNG,
            "base_seed": split.spec.base_seed,
            "count": split.spec.count,
            "eps_K": eps_k(split.k),
            "label_check": {
                "max": split.label_check.max,
                "mean": split.label_check.mean,
                "enforced": split.label_check_enforced,
                "tolerance": LABEL_CHECK_TOL,
            },
            "error2": {
                "max": split.error2.max,
                "mean": split.error2.mean,
                "d": TRANSFER_DEGREE,
                "s": 16,
            },
            "convention_version": CONVENTION_VERSION,
            "crate_version": env!("CARGO_PKG_VERSION"),
            "git_commit": "abc123",
            "wall_time_s": split.wall_time.as_secs_f64(),
        });
        assert_eq!(text, serde_json::to_string_pretty(&untyped).unwrap() + "\n");
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["split"], "train");
        assert_eq!(v["n"], 17);
        assert_eq!(v["s"], 16);
        assert_eq!(v["K"], 8);
        assert_eq!(v["K_explicit"], false);
        assert_eq!(v["tau"], TAU);
        assert_eq!(v["alpha"], ALPHA);
        assert_eq!(v["K_max"], K_MAX);
        assert_eq!(v["rng"], RNG);
        assert_eq!(v["base_seed"], 40);
        assert_eq!(v["count"], 2);
        // serde_json writes floats exactly but parses them to within an ulp without its
        // `float_roundtrip` feature.
        let close = |field: &serde_json::Value, written: f64| {
            let read = field.as_f64().unwrap();
            assert!(
                (read - written).abs() <= 1e-15 * written.abs(),
                "{read} vs {written}"
            );
        };
        close(&v["eps_K"], eps_k(8));
        close(&v["label_check"]["max"], split.label_check.max);
        close(&v["label_check"]["mean"], split.label_check.mean);
        assert_eq!(v["label_check"]["enforced"], false);
        assert_eq!(v["label_check"]["tolerance"], LABEL_CHECK_TOL);
        close(&v["error2"]["max"], split.error2.max);
        close(&v["error2"]["mean"], split.error2.mean);
        assert_eq!(v["error2"]["d"], 2);
        assert_eq!(v["error2"]["s"], 16);
        assert_eq!(v["convention_version"], CONVENTION_VERSION);
        assert_eq!(v["crate_version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(v["git_commit"], "abc123");
        assert!(v["wall_time_s"].as_f64().unwrap() > 0.0);

        // The loaders read the written split back: raw on the CGL grid, and on the
        // uniform grid s = n − 1 beside a written test split.
        assert_eq!(
            load_poisson_cheb_field(&npz_path, PoissonField::U, 17, 2).unwrap(),
            split.u
        );
        let test_spec = SplitSpec {
            n: 17,
            k: None,
            base_seed: 100,
            count: 1,
        };
        let test_split = generate(&test_spec).unwrap();
        let (test_path, _) = write_split(&dir, "test", &test_split, "abc123").unwrap();
        let config = PoissonConfig::new(
            DatasetConfig {
                n_train: 2,
                n_test: 1,
            },
            17,
        );
        let (train, test, _) = load_poisson_uniform::<f64>(&npz_path, &test_path, &config).unwrap();
        assert_eq!(train.inputs().shape(), [2, 16, 16, 1]);
        assert_eq!(test.targets().shape(), [1, 16, 16]);

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn stages_use_the_default_truncation() {
        use crate::neural_operators::data::catalogue::STAGES;
        for set in STAGES {
            assert_eq!(set.k, k_default(set.n), "stage n = {}", set.n);
        }
    }

    /// The splits `generate_poisson` builds from the catalogue are the ones its
    /// hard-coded table listed before phase 4s T22: same order, n, explicit K and stems.
    #[test]
    fn generator_stems_unchanged() {
        use crate::neural_operators::data::catalogue::GENERATED;
        let before: [(&str, usize, Option<usize>, &str); 12] = [
            ("train", 33, None, "poisson_n33_K16_train"),
            ("test", 33, None, "poisson_n33_K16_test"),
            ("train", 65, None, "poisson_n65_K32_train"),
            ("test", 65, None, "poisson_n65_K32_test"),
            ("train", 129, None, "poisson_n129_K64_train"),
            ("test", 129, None, "poisson_n129_K64_test"),
            ("train", 257, None, "poisson_n257_K64_train"),
            ("test", 257, None, "poisson_n257_K64_test"),
            ("train", 129, Some(32), "poisson_n129_K32_train"),
            ("test", 129, Some(32), "poisson_n129_K32_test"),
            ("train", 257, Some(32), "poisson_n257_K32_train"),
            ("test", 257, Some(32), "poisson_n257_K32_test"),
        ];
        let after: Vec<_> = GENERATED
            .into_iter()
            .flat_map(|set| ["train", "test"].map(|name| (name, SplitSpec::of_set(set, 0, 1))))
            .map(|(name, spec)| {
                let set = PoissonSet {
                    n: spec.n,
                    k: spec.truncation(),
                };
                (name, spec.n, spec.k, set.stem(name))
            })
            .collect();
        let before: Vec<_> = before
            .into_iter()
            .map(|(name, n, k, stem)| (name, n, k, stem.to_string()))
            .collect();
        assert_eq!(after, before);
    }
}

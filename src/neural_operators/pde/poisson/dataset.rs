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

use ndarray::{Array1, Array3, ArrayView1, ArrayView2, Axis};
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
use crate::neural_operators::chebyshev::{
    apply, cheb_to_uniform, clenshaw_curtis, nodes, rel_l2_error, uniform_to_cheb,
};

/// Bound on the label check, relative CC-L², enforced for n ≥ [`LABEL_CHECK_MIN_N`]
/// (design §5.1; Phase 0 T4 observed at most 7.4e-10 at n = 65).
pub const LABEL_CHECK_TOL: f64 = 1e-8;

/// The smallest n at which [`LABEL_CHECK_TOL`] is enforced. Stage 1 (n = 33) records
/// its label check without enforcing it (design §5.1).
pub const LABEL_CHECK_MIN_N: usize = 65;

/// The Floater–Hormann degree d of T_uc in error 2 (design §12, decision 8).
pub const TRANSFER_DEGREE: usize = 2;

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
    /// The truncation K in effect: the explicit one, else [`k_default`]`(n)`.
    ///
    /// # Panics
    /// If `k` is `None` and `n < 3`.
    pub fn truncation(&self) -> usize {
        self.k.unwrap_or_else(|| k_default(self.n))
    }
}

/// The maximum and mean of a per-sample error over a split.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErrorStats {
    /// Largest value over the samples.
    pub max: f64,
    /// Mean over the samples.
    pub mean: f64,
}

impl ErrorStats {
    /// The statistics of `values`; NaN for both when it is empty or any value is NaN.
    fn of(values: &[f64]) -> Self {
        Self {
            // `f64::max` skips NaN, so check for it first: a NaN sample must show in the max.
            max: if values.is_empty() || values.iter().any(|v| v.is_nan()) {
                f64::NAN
            } else {
                values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            },
            mean: values.iter().sum::<f64>() / values.len() as f64,
        }
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
    /// The ascending CGL nodes on x, `[n]`.
    pub x: Array1<f64>,
    /// The ascending CGL nodes on y, `[n]`; equal to `x`.
    pub y: Array1<f64>,
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
/// `u_a` and `u_series` are `[n, n]`, 'ij'; `w: [n]` holds the Clenshaw–Curtis weights.
/// `sample` and `seed` only label the error. Returns the discrepancy; for
/// n ≥ [`LABEL_CHECK_MIN_N`] a value above [`LABEL_CHECK_TOL`] (or NaN) is an error.
///
/// # Errors
/// [`LabelCheckError`] when the check is enforced and fails.
///
/// # Panics
/// If `w` has fewer than 2 entries, or the fields are not both `[w.len(), w.len()]`.
pub fn label_check(
    u_a: ArrayView2<f64>,
    u_series: ArrayView2<f64>,
    w: ArrayView1<f64>,
    sample: usize,
    seed: u64,
) -> Result<f64, LabelCheckError> {
    let error = rel_l2_error(u_a, u_series, w, w);
    let n = w.len();
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
/// Builds one [`CollocationSolver`], one set of Clenshaw–Curtis weights and one pair of
/// transfers for the split, then for sample i draws the forcing of seed
/// `base_seed + i` ([`draw_xi`]) at the truncation [`SplitSpec::truncation`], evaluates
/// it at the CGL nodes, solves, and checks the label against the exact series.
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
    let w = clenshaw_curtis(n);
    let solver = CollocationSolver::new(n, n);
    let s = n - 1;
    let t_cu = cheb_to_uniform(n, s);
    let t_uc = uniform_to_cheb(s, n, TRANSFER_DEGREE);

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
        label.push(label_check(u_i.view(), u_series.view(), w.view(), i, seed)?);
        let round_trip = apply(
            t_uc.view(),
            t_uc.view(),
            apply(t_cu.view(), t_cu.view(), u_i.view()).view(),
        );
        err2.push(rel_l2_error(
            round_trip.view(),
            u_i.view(),
            w.view(),
            w.view(),
        ));
    }

    Ok(Split {
        spec: spec.clone(),
        k,
        f,
        u,
        y: x.clone(),
        x,
        label_check: ErrorStats::of(&label),
        label_check_enforced: n >= LABEL_CHECK_MIN_N,
        error2: ErrorStats::of(&err2),
        wall_time: start.elapsed(),
    })
}

/// The file stem of a split, `poisson_n{n}_K{K}_{name}`, so that sets differing only in
/// K (stage 3 and the K = 32 set 3e) get distinct files.
pub fn split_stem(split: &Split, name: &str) -> String {
    format!("poisson_n{}_K{}_{name}", split.spec.n, split.k)
}

/// Writes a split to `dir` as `{stem}.npz` and `{stem}.json`, with the stem from
/// [`split_stem`] (design §5.1). Creates `dir` if needed and overwrites existing files.
///
/// The `.npz` holds `f` and `u` (`[N, n, n]`, f64, 'ij') and the CGL nodes `x` and `y`
/// (`[n]`), uncompressed. The JSON sidecar records n and s, K and whether it was explicit,
/// τ, α, K_MAX, the RNG, the base seed and count, ε_K, the label check (max, mean,
/// enforced, tolerance), error 2 (max, mean, d, s), [`CONVENTION_VERSION`], the crate
/// version, `git_commit` and the wall time in seconds. `name` is the split's name, e.g.
/// `"train"` or `"test"`. Returns the two paths.
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
    for (field, array) in [("x", &split.x), ("y", &split.y)] {
        npz.add_array(field, array).map_err(io::Error::other)?;
    }
    npz.finish().map_err(io::Error::other)?;

    let n = split.spec.n;
    let sidecar = serde_json::json!({
        "split": name,
        "n": n,
        "s": n - 1,
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
            "s": n - 1,
        },
        "convention_version": CONVENTION_VERSION,
        "crate_version": env!("CARGO_PKG_VERSION"),
        "git_commit": git_commit,
        "wall_time_s": split.wall_time.as_secs_f64(),
    });
    let text = serde_json::to_string_pretty(&sidecar).map_err(io::Error::other)?;
    fs::write(&json_path, text + "\n")?;
    Ok((npz_path, json_path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::data::io::{readers::npz::NpzFileReader, traits::FieldReader};
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
    fn error_stats_propagate_nan() {
        let ok = ErrorStats::of(&[1.0, 3.0]);
        assert_eq!((ok.max, ok.mean), (3.0, 2.0));
        let bad = ErrorStats::of(&[1.0, f64::NAN, 3.0]);
        assert!(bad.max.is_nan() && bad.mean.is_nan());
        let empty = ErrorStats::of(&[]);
        assert!(empty.max.is_nan() && empty.mean.is_nan());
    }

    #[test]
    fn collocation_matches_series_on_grf() {
        for n in [33, 65, 129] {
            let solver = CollocationSolver::new(n, n);
            let w = clenshaw_curtis(n);
            let k = k_default(n);
            let errs: Vec<f64> = (0..5)
                .map(|seed| {
                    let [ua, us] = a_and_series(&solver, n, k, seed);
                    rel_l2_error(ua.view(), us.view(), w.view(), w.view())
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
        let w = clenshaw_curtis(n);
        let [ua, us] = a_and_series(&solver, n, k_default(n), 4);
        assert!(label_check(ua.view(), us.view(), w.view(), 3, 4).is_ok());

        let mut corrupted = ua.clone();
        corrupted[[32, 32]] += 1e-4 * us.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let err = label_check(corrupted.view(), us.view(), w.view(), 3, 4)
            .expect_err("a corrupted label at n = 65 must fail the check");
        assert_eq!((err.sample, err.seed, err.n), (3, 4, n));
        assert!(err.error > LABEL_CHECK_TOL);
        assert!(err.to_string().contains("sample 3 (seed 4)"), "{err}");

        // Not enforced below n = 65: the same corruption is recorded, not rejected.
        let n = 33;
        let solver = CollocationSolver::new(n, n);
        let w = clenshaw_curtis(n);
        let [ua, us] = a_and_series(&solver, n, k_default(n), 4);
        let mut corrupted = ua.clone();
        corrupted[[16, 16]] += 1e-4 * us.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let e = label_check(corrupted.view(), us.view(), w.view(), 3, 4).unwrap();
        assert!(e > LABEL_CHECK_TOL);
    }

    #[test]
    fn files_round_trip() {
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

        let reader = NpzFileReader::new(&npz_path);
        for (field, written) in [("f", &split.f), ("u", &split.u)] {
            let read = reader.read_field(field).unwrap();
            assert_eq!(read.shape(), &[2, 17, 17], "{field}");
            assert_eq!(read, written.clone().into_dyn(), "{field}");
        }
        for (field, written) in [("x", &split.x), ("y", &split.y)] {
            let read = reader.read_field(field).unwrap();
            assert_eq!(read.shape(), &[17], "{field}");
            assert_eq!(read, written.clone().into_dyn(), "{field}");
        }

        let text = fs::read_to_string(&json_path).unwrap();
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

        fs::remove_dir_all(&dir).unwrap();
    }
}

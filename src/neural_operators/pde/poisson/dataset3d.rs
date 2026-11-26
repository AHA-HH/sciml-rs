//! 3D Poisson datasets on the CGL grid (3D proposal §3.3, §5, §6; D3, D5): the 2D
//! [`dataset`](super::dataset) with a third axis.
//!
//! [`generate_3d`] builds one split: for sample i it draws the 3D forcing f of seed
//! `base_seed + i` ([`grf3d`](super::grf3d)), solves −Δu = f with
//! [`CollocationSolver3d::solve_full`] (A3), and records two relative Clenshaw–Curtis L²
//! errors on the CGL grid (CONVENTIONS §12):
//! - the **label check** ([`label_check_3d`]): u_A3 against the exact sine-series solution
//!   of proposal §3.3, with the 2D rule: for n ≥ [`LABEL_CHECK_MIN_N`] (65) a sample above
//!   [`LABEL_CHECK_TOL`] aborts the split; below that n it is recorded only (n = 33,
//!   K = 16 gives about 4e-7, as in 2D; n = 65, K = 32 about 4e-10, CL3.2's report);
//! - **error 2**: the transfer round trip T_uc(T_cu u) − u with d = [`TRANSFER_DEGREE`] and
//!   s = n − 1 along every axis, recorded only.
//!
//! [`write_split_3d`] writes the split as one `.npz` and one JSON sidecar named by
//! [`Poisson3dSet`]. The sidecar is the 2D one, written by the shared
//! `dataset::write_sidecar` with the 3D α, RNG and ε_K.
//!
//! [`TRANSFER_DEGREE`]: crate::neural_operators::chebyshev::TRANSFER_DEGREE

use ndarray::{Array1, Array4, ArrayView3, Axis};
use ndarray_npy::NpzWriter;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use super::collocation::CollocationSolver3d;
use super::dataset::{
    LABEL_CHECK_MIN_N, LABEL_CHECK_TOL, LabelCheckError, SidecarFacts, SplitSpec, write_sidecar,
};
use super::grf3d::{ALPHA, RNG, draw_xi, eps_k, evaluate, forcing_coeffs, solution_coeffs};
use crate::neural_operators::chebyshev::{Transfers, cc_rel_l2_error_nd, nodes};
use crate::neural_operators::data::catalogue::Poisson3dSet;
use crate::neural_operators::metrics::stats::ErrorStats;

/// One generated 3D split.
#[derive(Clone, Debug)]
pub struct Split3d {
    /// What was generated.
    pub spec: SplitSpec,
    /// The truncation K in effect ([`SplitSpec::truncation`]).
    pub k: usize,
    /// The forcings, `[N, n, n, n]`, each sample stored 'ijk' (CONVENTIONS §12).
    pub f: Array4<f64>,
    /// The A3 solutions, `[N, n, n, n]`, 'ijk', zero on the six faces.
    pub u: Array4<f64>,
    /// The ascending CGL nodes, `[n]`, on every axis; [`write_split_3d`] stores them as
    /// `x`, `y` and `z`.
    pub x: Array1<f64>,
    /// The label check ‖u_A3 − u_series‖ / ‖u_series‖ over the samples.
    pub label_check: ErrorStats,
    /// Whether [`LABEL_CHECK_TOL`] was enforced (n ≥ [`LABEL_CHECK_MIN_N`]).
    pub label_check_enforced: bool,
    /// Error 2, ‖T_uc(T_cu u) − u‖ / ‖u‖ over the samples.
    pub error2: ErrorStats,
    /// Wall time of [`generate_3d`].
    pub wall_time: Duration,
}

impl Split3d {
    /// The catalogue set this split belongs to: its n and the K in effect.
    fn set(&self) -> Poisson3dSet {
        Poisson3dSet {
            n: self.spec.n,
            k: self.k,
        }
    }
}

/// The 3D label check of one sample: the relative Clenshaw–Curtis L² discrepancy
/// ‖u_a − u_series‖ / ‖u_series‖ (CONVENTIONS §12), [`cc_rel_l2_error_nd`].
///
/// `u_a` and `u_series` are `[n, n, n]`, 'ijk'. `sample` and `seed` only label the error.
/// Returns the discrepancy; for n ≥ [`LABEL_CHECK_MIN_N`] a value above
/// [`LABEL_CHECK_TOL`] (or NaN) is an error.
///
/// # Errors
/// [`LabelCheckError`] when the check is enforced and fails.
///
/// # Panics
/// If the fields differ in shape or are not cubes, or an axis has fewer than 2 nodes.
pub fn label_check_3d(
    u_a: ArrayView3<f64>,
    u_series: ArrayView3<f64>,
    sample: usize,
    seed: u64,
) -> Result<f64, LabelCheckError> {
    let n = u_a.dim().0;
    assert_eq!(u_a.dim(), (n, n, n), "label_check_3d: fields must be cubes");
    let error = cc_rel_l2_error_nd(u_a, u_series);
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

/// Generates one 3D split: f, u_A3, the label check and error 2.
///
/// Builds one [`CollocationSolver3d`] and one pair of transfers for the split, then for
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
///
/// [`K_MAX`]: super::grf::K_MAX
pub fn generate_3d(spec: &SplitSpec) -> Result<Split3d, LabelCheckError> {
    let start = Instant::now();
    let n = spec.n;
    assert!(spec.count > 0, "generate_3d: count must be > 0");
    assert!(n >= 3, "generate_3d: n must be >= 3, got {n}");
    let k = spec.truncation();
    let x = nodes(n);
    let solver = CollocationSolver3d::new(n, n, n);
    let transfers = Transfers::new(n);

    let mut f = Array4::zeros((spec.count, n, n, n));
    let mut u = Array4::zeros((spec.count, n, n, n));
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
            .expect("generate_3d: base_seed + i overflows u64");
        let c = forcing_coeffs(draw_xi(seed).view(), k);
        f_i.assign(&evaluate(c.view(), x.view(), x.view(), x.view()));
        u_i.assign(&solver.solve_full(f_i.view()));
        let u_series = evaluate(
            solution_coeffs(c.view()).view(),
            x.view(),
            x.view(),
            x.view(),
        );
        label.push(label_check_3d(u_i.view(), u_series.view(), i, seed)?);
        let round_trip = transfers.to_cheb_nd(transfers.to_uniform_nd(u_i.view()).view());
        err2.push(cc_rel_l2_error_nd(round_trip.view(), u_i.view()));
    }

    Ok(Split3d {
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

/// Writes a 3D split to `dir` as `{stem}.npz` and `{stem}.json`, the stem
/// [`Poisson3dSet::stem`] of the split's n and K in effect. Creates `dir` if needed and overwrites
/// existing files.
///
/// The `.npz` holds `f` and `u` (`[N, n, n, n]`, f64, 'ijk') and the CGL nodes `x`, `y` and
/// `z` (`[n]`, each [`Split3d::x`]), uncompressed: zip64 lets the archive pass 4 GiB, but each array must stay below 4 GiB
/// (the `zip` writer refuses a larger member; at n = 65 that is N ≤ 1954, well above
/// D5's 1000). The sidecar is a
/// [`SidecarRecord`](crate::neural_operators::data::loaders::poisson::SidecarRecord) as in
/// 2D, written by `dataset::write_sidecar`, with the 3D α ([`ALPHA`]), RNG ([`RNG`]) and ε_K
/// ([`eps_k`]). `name` is the split's name, e.g. `"train"`. Returns the two paths.
///
/// # Errors
/// Any I/O error from creating `dir` or writing either file.
pub fn write_split_3d(
    dir: &Path,
    name: &str,
    split: &Split3d,
    git_commit: &str,
) -> io::Result<(PathBuf, PathBuf)> {
    fs::create_dir_all(dir)?;
    let set = split.set();
    let npz_path = dir.join(set.npz(name));
    let json_path = dir.join(set.json(name));

    let mut npz = NpzWriter::new(File::create(&npz_path)?);
    for (field, array) in [("f", &split.f), ("u", &split.u)] {
        npz.add_array(field, array).map_err(io::Error::other)?;
    }
    // y and z repeat x: the file keeps every axis, as the 2D files keep x and y.
    for field in ["x", "y", "z"] {
        npz.add_array(field, &split.x).map_err(io::Error::other)?;
    }
    npz.finish().map_err(io::Error::other)?;

    let facts = SidecarFacts {
        spec: &split.spec,
        k: split.k,
        label_check: split.label_check,
        label_check_enforced: split.label_check_enforced,
        error2: split.error2,
        wall_time: split.wall_time,
        alpha: ALPHA,
        rng: RNG,
        eps_k,
    };
    write_sidecar(&json_path, name, git_commit, &facts)?;
    Ok((npz_path, json_path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CONVENTION_VERSION;
    use crate::neural_operators::chebyshev::TRANSFER_DEGREE;
    use crate::neural_operators::data::io::readers::npz::NpzFileReader;
    use crate::neural_operators::data::loaders::poisson::SidecarRecord;
    use crate::neural_operators::pde::poisson::grf::{K_MAX, TAU, k_default};
    use ndarray::{Array3, s};

    /// u_A3 and u_series for one seed, n, K.
    fn a3_and_series(n: usize, k: usize, seed: u64) -> [Array3<f64>; 2] {
        let x = nodes(n);
        let c = forcing_coeffs(draw_xi(seed).view(), k);
        let f = evaluate(c.view(), x.view(), x.view(), x.view());
        [
            CollocationSolver3d::new(n, n, n).solve_full(f.view()),
            evaluate(
                solution_coeffs(c.view()).view(),
                x.view(),
                x.view(),
                x.view(),
            ),
        ]
    }

    fn spec(n: usize, base_seed: u64, count: usize) -> SplitSpec {
        SplitSpec {
            n,
            k: None,
            base_seed,
            count,
        }
    }

    #[test]
    fn generate_3d_matches_its_parts() {
        let split = generate_3d(&spec(17, 7, 2)).unwrap();
        assert_eq!(split.k, 8);
        assert_eq!(split.f.shape(), [2, 17, 17, 17]);
        assert!(!split.label_check_enforced);
        let x = nodes(17);
        let solver = CollocationSolver3d::new(17, 17, 17);
        for i in 0..2 {
            let c = forcing_coeffs(draw_xi(7 + i as u64).view(), 8);
            let f = evaluate(c.view(), x.view(), x.view(), x.view());
            assert_eq!(split.f.index_axis(Axis(0), i), f, "f of sample {i}");
            let u = split.u.index_axis(Axis(0), i);
            assert_eq!(u, solver.solve_full(f.view()), "u of sample {i}");
            for face in [
                u.slice(s![0, .., ..]),
                u.slice(s![16, .., ..]),
                u.slice(s![.., 0, ..]),
                u.slice(s![.., 16, ..]),
                u.slice(s![.., .., 0]),
                u.slice(s![.., .., 16]),
            ] {
                assert!(face.iter().all(|&v| v == 0.0), "u of sample {i} on a face");
            }
        }
        assert!(split.label_check.max.is_finite() && split.error2.max.is_finite());
    }

    #[test]
    fn label_check_3d_at_n65() {
        let split = generate_3d(&spec(65, 0, 2)).unwrap();
        assert_eq!(split.k, 32);
        assert!(split.label_check_enforced);
        assert!(!generate_3d(&spec(33, 0, 1)).unwrap().label_check_enforced);
        println!(
            "3D label check, n = 65, K = 32: max {:.1e}; error 2 max {:.1e}",
            split.label_check.max, split.error2.max
        );
        assert!(split.label_check.max <= 1e-8, "{}", split.label_check.max);
    }

    #[test]
    fn label_check_3d_aborts() {
        for (n, enforced) in [(65, true), (33, false)] {
            let [ua, us] = a3_and_series(n, k_default(n), 4);
            assert!(label_check_3d(ua.view(), us.view(), 3, 4).is_ok());
            let mut corrupted = ua.clone();
            let c = n / 2;
            corrupted[[c, c, c]] += 1e-4 * us.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let result = label_check_3d(corrupted.view(), us.view(), 3, 4);
            if enforced {
                let err = result.expect_err("a corrupted label at n = 65 must fail");
                assert_eq!((err.sample, err.seed, err.n), (3, 4, n));
                assert!(err.to_string().contains("sample 3 (seed 4)"), "{err}");
            } else {
                assert!(result.unwrap() > LABEL_CHECK_TOL, "recorded at n = {n}");
            }
        }
    }

    #[test]
    fn files_round_trip_3d() {
        let split = generate_3d(&spec(9, 40, 2)).unwrap();
        let dir = std::env::temp_dir().join(format!(
            "sciml_rs_files_round_trip_3d_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let (npz_path, json_path) = write_split_3d(&dir, "train", &split, "abc123").unwrap();
        assert_eq!(npz_path, dir.join("poisson3d_n9_K4_train.npz"));
        assert_eq!(json_path, dir.join("poisson3d_n9_K4_train.json"));

        let mut reader = NpzFileReader::new(&npz_path).unwrap();
        for (field, written) in [("f", &split.f), ("u", &split.u)] {
            assert_eq!(
                reader.read_field(field).unwrap(),
                written.clone().into_dyn(),
                "{field}"
            );
        }
        for field in ["x", "y", "z"] {
            assert_eq!(
                reader.read_field(field).unwrap(),
                split.x.clone().into_dyn(),
                "{field}"
            );
        }

        let record: SidecarRecord =
            serde_json::from_str(&fs::read_to_string(&json_path).unwrap()).unwrap();
        assert_eq!(record.split, "train");
        assert_eq!((record.n, record.s, record.k), (9, 8, 4));
        assert!(!record.k_explicit);
        assert_eq!((record.base_seed, record.count), (40, 2));
        assert_eq!((record.tau, record.alpha, record.k_max), (TAU, 3.0, K_MAX));
        assert_eq!(record.rng, RNG);
        // serde_json parses floats to within an ulp without `float_roundtrip`.
        let close = |read: f64, written: f64| {
            assert!(
                (read - written).abs() <= 1e-15 * written.abs(),
                "{read} vs {written}"
            )
        };
        close(record.eps_k, eps_k(4));
        close(record.label_check.max, split.label_check.max);
        close(record.label_check.mean, split.label_check.mean);
        assert!(!record.label_check.enforced);
        assert_eq!(record.label_check.tolerance, LABEL_CHECK_TOL);
        close(record.error2.max, split.error2.max);
        close(record.error2.mean, split.error2.mean);
        assert_eq!((record.error2.d, record.error2.s), (TRANSFER_DEGREE, 8));
        assert_eq!(record.convention_version, CONVENTION_VERSION);
        assert_eq!(record.git_commit, "abc123");
        assert_eq!(record.crate_version, env!("CARGO_PKG_VERSION"));
        assert!(record.wall_time_s > 0.0);
        fs::remove_dir_all(&dir).unwrap();
    }
}

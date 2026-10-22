//! Poisson dataset: config and constructor (design §5.2).
//!
//! `PoissonConfig` holds the Chebyshev size `n` on top of the shared `DatasetConfig`.
//! `PoissonSet` names a dataset's split files and `STAGES` lists the training stages.
//! `load_poisson_uniform` builds train/test `OperatorDataset`s from the `.npz` files that
//! `generate_poisson` writes (`datasets/README.md`): forcing `f` and solution `u` on the
//! n × n CGL grid. It reads and validates them, transfers every sample to the uniform grid
//! of size s = n − 1 and then normalizes, as Darcy does. Inputs carry the forcing only;
//! the model appends the 2D grid coordinates itself (see `FNO::forward`).
//! `load_poisson_uniform_test` builds the test split alone, encoded with a run's saved
//! normalizers, so predict neither reads nor transfers the train split.
//!
//! Transfer: T_cu (`chebyshev::Transfers::to_uniform`, CONVENTIONS §12) is applied along both
//! axes at load time, in f64, before normalization; transferred fields are never written
//! to disk. T_cu is polynomial interpolation on the CGL nodes, exact to round-off
//! (≤ 1e-15) on the analytic test fields, and for n ≥ 65 within 2e-8 on GRF solutions u
//! and 3.4e-6 on GRF forcings f (Phase 0 T3, `spikes/transfers/REPORT.md` table 1). The
//! round-trip error of design §7 (error 2) is recorded per split in the dataset's `.json`
//! sidecar by the generator; this loader does not read the sidecar.
//!
//! `load_poisson_cheb` reads one split's raw solutions `u` on the CGL grid, untransferred,
//! with the same validation: the reference for the errors on the Chebyshev grid (design §7).
//! `load_poisson_cheb_field` does the same for either field, `f` or `u`.
//!
//! Normalization (CONVENTIONS §10, as in `load_darcy_uniform`): both inputs are encoded
//! with the `x` normalizer fit on `x_train`; both target splits stay in physical units, and
//! predictions are decoded back to physical scale before the loss, not targets encoded
//! forward. The returned `normalizers.y` is what the caller needs to do that decode.

use crate::neural_operators::chebyshev::{TRANSFER_DEGREE, Transfers, nodes};
use crate::neural_operators::data::{
    dataset::OperatorDataset,
    host_float::HostFloat,
    io::{errors::LoadError, readers::npz::NpzFileReader},
    loaders::{
        base_dataset::DatasetConfig,
        common::{Normalizers, check_n_train, normalize_split_pair, normalize_test_split},
    },
};
use burn::config::Config;
use ndarray::{Array1, Array3, ArrayD, Axis, Ix3, s};
use std::path::Path;

/// Tolerance on the stored `x`, `y` nodes against the closed-form CGL nodes.
const NODE_TOL: f64 = 1e-14;

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

/// One Poisson dataset on disk: `n` CGL nodes per axis and the GRF truncation `k`
/// (design §5.3). Its splits are files named by [`PoissonSet::stem`].
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
}

/// The stages a model trains on (design §5.3): K = K(n) at n = 65, 129 and 257.
pub const STAGES: [PoissonSet; 3] = [
    PoissonSet { n: 65, k: 32 },
    PoissonSet { n: 129, k: 64 },
    PoissonSet { n: 257, k: 64 },
];

/// The stage in [`STAGES`] with `n` CGL nodes per axis.
///
/// # Errors
/// [`LoadError::Invalid`] naming the stages' n if there is none.
pub fn stage(n: usize) -> Result<PoissonSet, LoadError> {
    STAGES.into_iter().find(|set| set.n == n).ok_or_else(|| {
        let ns: Vec<usize> = STAGES.iter().map(|set| set.n).collect();
        LoadError::Invalid(format!(
            "no Poisson stage with n = {n}; expected one of {ns:?}"
        ))
    })
}

/// Builds the Poisson dataset on the uniform grid: inputs `[N, s, s, 1]` (the normalized
/// forcing; no grid channels, the model generates them), targets `[N, s, s]`, with
/// s = `config.s()` = n − 1.
///
/// Each file must hold `f` and `u` as `[samples, n, n]` on the CGL grid (`f[i, a, b] =
/// f(x_a, y_b)`) and `x`, `y` as the ascending CGL nodes `[n]`. The first
/// `config.base.n_train` / `config.base.n_test` samples are used.
///
/// Returns `(train_dataset, test_dataset, normalizers)`. `normalizers.y` is what the
/// caller needs to decode predictions back to physical scale; `normalizers.x` is only
/// needed for encoding inputs that didn't come from this dataset.
///
/// # Errors
/// [`LoadError::Reader`] if a file can't be read or lacks `f`/`u`/`x`/`y`;
/// [`LoadError::Invalid`] if `n_train < 2` (the normalizers' sample std is undefined for
/// one sample), `config.n <` [`Transfers::MIN_N`] (T_uc of degree [`TRANSFER_DEGREE`]
/// needs s = n − 1 > d), `f` or `u` isn't
/// `[samples, n, n]`, `x` or `y` isn't `[n]` or differs from the CGL nodes by more than
/// 1e-14, or a file has fewer samples than requested.
///
/// Reading, the transfer and normalization are computed in `f64`, and the normalizers
/// keep `f64` statistics; the datasets are stored as `T`, rounded once at the end (see
/// [`HostFloat`]).
pub fn load_poisson_uniform<T: HostFloat>(
    train_path: impl AsRef<Path>,
    test_path: impl AsRef<Path>,
    config: &PoissonConfig,
) -> Result<(OperatorDataset<T>, OperatorDataset<T>, Normalizers), LoadError> {
    check_n_train(config.base.n_train)?;
    check_transfer_n(config.n)?;

    // The transfers are built once and shared by both splits; each split's reader and
    // Chebyshev fields are freed before the next file is parsed.
    let transfers = Transfers::new(config.n);
    let (f_train, u_train) = read_uniform_split(
        train_path.as_ref(),
        "train",
        config.base.n_train,
        &transfers,
    )?;
    let (f_test, u_test) =
        read_uniform_split(test_path.as_ref(), "test", config.base.n_test, &transfers)?;
    Ok(normalize_split_pair(f_train, u_train, f_test, u_test))
}

/// Builds the Poisson test split only, on the uniform grid, encoded with `normalizers` (a
/// run's saved ones,
/// [`RunDir::normalizers`](crate::neural_operators::runs::RunDir::normalizers)) instead of
/// normalizers refit on the train split, which is neither read nor transferred: inputs
/// `[n_test, s, s, 1]`, targets `[n_test, s, s]`. With the normalizers
/// [`load_poisson_uniform`] fits, the dataset equals its test split bit for bit.
///
/// The file is validated as [`load_poisson_uniform`] validates it.
///
/// # Errors
/// [`LoadError::Reader`] if the file can't be read or lacks `f`/`u`/`x`/`y`;
/// [`LoadError::Invalid`] if `config.n <` [`Transfers::MIN_N`], `f` or `u` isn't
/// `[samples, n, n]`, `x` or `y` isn't `[n]` or differs from the CGL nodes by more than
/// 1e-14, the file has fewer than `n_test` samples, or `normalizers.x` was not fitted on
/// `[s, s]`.
pub fn load_poisson_uniform_test<T: HostFloat>(
    test_path: impl AsRef<Path>,
    config: &PoissonConfig,
    normalizers: &Normalizers,
) -> Result<OperatorDataset<T>, LoadError> {
    check_transfer_n(config.n)?;
    let transfers = Transfers::new(config.n);
    let (f_test, u_test) =
        read_uniform_split(test_path.as_ref(), "test", config.base.n_test, &transfers)?;
    normalize_test_split(f_test, u_test, normalizers)
}

/// Checks that T_uc of degree [`TRANSFER_DEGREE`] works at `n`: s = n − 1 > d.
fn check_transfer_n(n: usize) -> Result<(), LoadError> {
    if n < Transfers::MIN_N {
        return Err(LoadError::Invalid(format!(
            "n must be >= {} (T_uc of degree {TRANSFER_DEGREE} needs s = n - 1 > d), got {n}",
            Transfers::MIN_N
        )));
    }
    Ok(())
}

/// Reads one split's `f` and `u`, validated, and transfers the first `count` samples of
/// each to the uniform grid with `transfers`: `([count, s, s], [count, s, s])`. `split`
/// only labels the error messages.
fn read_uniform_split(
    path: &Path,
    split: &str,
    count: usize,
    transfers: &Transfers,
) -> Result<(ArrayD<f64>, ArrayD<f64>), LoadError> {
    let n = transfers.n();
    let mut reader = NpzFileReader::new(path)?;
    check_nodes(&mut reader, split, n, &nodes(n))?;
    let f = read_samples(&mut reader, split, "f", n, count)?;
    let f = transfer_samples(f, count, transfers);
    let u = read_samples(&mut reader, split, "u", n, count)?;
    let u = transfer_samples(u, count, transfers);
    Ok((f, u))
}

/// Reads the first `count` raw solutions `u` of one split on the n × n CGL grid,
/// untransferred: the reference field of the errors on the Chebyshev grid (design §7,
/// CONVENTIONS §12). Returns `[count, n, n]` with `u[i, a, b] = u_i(x_a, y_b)`.
///
/// The file is validated as [`load_poisson_uniform`] validates it: `x`, `y` must be the
/// ascending CGL nodes `[n]` and `u` must be `[samples, n, n]`. `f` is not read. The
/// sidecar is not read either.
///
/// # Errors
/// [`LoadError::Reader`] if the file can't be read or lacks `u`/`x`/`y`;
/// [`LoadError::Invalid`] if `n < 3`, `u` isn't `[samples, n, n]`, `x` or `y` isn't `[n]`
/// or differs from the CGL nodes by more than 1e-14, or the file has fewer than `count`
/// samples. Messages name the file in place of the split.
pub fn load_poisson_cheb(
    path: impl AsRef<Path>,
    n: usize,
    count: usize,
) -> Result<Array3<f64>, LoadError> {
    load_poisson_cheb_field(path, "u", n, count)
}

/// Reads the first `count` samples of the raw field `field` (`"f"`, the forcing, or `"u"`,
/// the solution) of one split on the n × n CGL grid, untransferred. Returns
/// `[count, n, n]` with `field[i, a, b] = field_i(x_a, y_b)` (CONVENTIONS §12).
///
/// The file is validated as [`load_poisson_cheb`] validates it: `x`, `y` must be the
/// ascending CGL nodes `[n]` and `field` must be `[samples, n, n]`. The other field and
/// the sidecar are not read.
///
/// # Errors
/// [`LoadError::Reader`] if the file can't be read or lacks `field`/`x`/`y`;
/// [`LoadError::Invalid`] if `field` is neither `"f"` nor `"u"` (checked before the file
/// is opened), `n < 3`, `field` isn't `[samples, n, n]`, `x` or `y` isn't `[n]` or differs
/// from the CGL nodes by more than 1e-14, or the file has fewer than `count` samples.
/// Messages name the file in place of the split.
pub fn load_poisson_cheb_field(
    path: impl AsRef<Path>,
    field: &str,
    n: usize,
    count: usize,
) -> Result<Array3<f64>, LoadError> {
    if !matches!(field, "f" | "u") {
        return Err(LoadError::Invalid(format!(
            "field must be 'f' or 'u', got '{field}'"
        )));
    }
    if n < 3 {
        return Err(LoadError::Invalid(format!(
            "n must be >= 3 (the uniform grid s = n - 1 needs both endpoints), got {n}"
        )));
    }
    let path = path.as_ref();
    let label = path.file_name().map_or_else(
        || path.display().to_string(),
        |f| f.to_string_lossy().into(),
    );
    let mut reader = NpzFileReader::new(path)?;
    check_nodes(&mut reader, &label, n, &nodes(n))?;
    let values = read_samples(&mut reader, &label, field, n, count)?;
    Ok(values.slice_move(s![..count, .., ..]))
}

/// Checks that the fields `x` and `y` are `[n]` and within [`NODE_TOL`] of the CGL
/// `nodes`. `split` only labels the error messages.
fn check_nodes(
    reader: &mut NpzFileReader,
    split: &str,
    n: usize,
    nodes: &Array1<f64>,
) -> Result<(), LoadError> {
    for name in ["x", "y"] {
        let field = reader.read_field(name)?;
        if field.shape() != [n] {
            return Err(LoadError::Invalid(format!(
                "{split} field '{name}' must be [{n}] (the CGL nodes), got shape {:?}",
                field.shape()
            )));
        }
        let err = field
            .iter()
            .zip(nodes)
            .map(|(a, b)| (a - b).abs())
            // unlike f64::max, keeps a NaN deviation so NaN nodes are rejected
            .fold(0.0, |m, d| if d.is_nan() || d > m { d } else { m });
        if err.is_nan() || err > NODE_TOL {
            return Err(LoadError::Invalid(format!(
                "{split} field '{name}' must hold the {n} ascending CGL nodes, \
                 max deviation {err:e}"
            )));
        }
    }
    Ok(())
}

/// Reads the field `name`, checks that it is `[samples, n, n]` with at least `count`
/// samples, and returns it whole. `split` only labels the error messages.
fn read_samples(
    reader: &mut NpzFileReader,
    split: &str,
    name: &str,
    n: usize,
    count: usize,
) -> Result<Array3<f64>, LoadError> {
    let field = reader.read_field(name)?;
    let shape = field.shape().to_vec();
    let field = match field.into_dimensionality::<Ix3>() {
        Ok(field) if field.shape()[1..] == [n, n] => field,
        _ => {
            return Err(LoadError::Invalid(format!(
                "{split} field '{name}' must be [samples, {n}, {n}], got shape {shape:?}"
            )));
        }
    };
    if count > field.shape()[0] {
        return Err(LoadError::Invalid(format!(
            "requested {count} {split} samples but '{name}' has {}",
            field.shape()[0]
        )));
    }
    Ok(field)
}

/// The first `count` samples of a `[samples, n, n]` Chebyshev field, each transferred to
/// the uniform grid as T_cu · F · T_cuᵀ ([`Transfers::to_uniform`]); returns
/// `[count, s, s]`.
///
/// The caller has checked the shape and that `count` samples exist.
fn transfer_samples(field: Array3<f64>, count: usize, transfers: &Transfers) -> ArrayD<f64> {
    let s = transfers.s();
    let mut out = Array3::zeros((count, s, s));
    for (mut dst, src) in out
        .axis_iter_mut(Axis(0))
        .zip(field.slice(s![..count, .., ..]).axis_iter(Axis(0)))
    {
        dst.assign(&transfers.to_uniform(src));
    }
    out.into_dyn()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::chebyshev::uniform_nodes;
    use crate::neural_operators::data::io::errors::ReaderError;
    use ndarray::{Array1, IxDyn};
    use ndarray_npy::NpzWriter;
    use std::fs::{self, File};
    use std::path::PathBuf;

    const N: usize = 9;
    const N_TRAIN: usize = 3;
    const N_TEST: usize = 2;

    fn cfg(n_train: usize, n_test: usize) -> PoissonConfig {
        PoissonConfig::new(DatasetConfig { n_train, n_test }, N)
    }

    /// A fresh, empty directory under the system temp dir, unique to `test` and this
    /// process.
    fn fresh_dir(test: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("sciml_rs_poisson_{test}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Writes `f`, `u`, `x`, `y` to `dir/name` as an uncompressed `.npz`, as T2 does.
    fn write_npz(
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
    fn poly_field(count: usize, offset: f64, xs: &Array1<f64>, ys: &Array1<f64>) -> ArrayD<f64> {
        Array3::from_shape_fn((count, xs.len(), ys.len()), |(i, a, b)| {
            (1.0 + i as f64 + offset) * poly(xs[a], ys[b])
        })
        .into_dyn()
    }

    /// Writes a valid train/test pair of polynomial fields on the CGL grid; f and u
    /// differ by scale (u = −2 f, test samples offset) so they are told apart.
    fn write_valid_pair(dir: &Path) -> (PathBuf, PathBuf) {
        let x = nodes(N);
        let f_tr = poly_field(N_TRAIN, 0.0, &x, &x);
        let f_te = poly_field(N_TEST, 0.5, &x, &x);
        let train = write_npz(dir, "train.npz", &f_tr, &(&f_tr * -2.0), &x, &x);
        let test = write_npz(dir, "test.npz", &f_te, &(&f_te * -2.0), &x, &x);
        (train, test)
    }

    fn rel_err(a: &ArrayD<f64>, b: &ArrayD<f64>) -> f64 {
        let diff = (a - b).mapv(|v| v * v).sum().sqrt();
        diff / b.mapv(|v| v * v).sum().sqrt()
    }

    fn expect_invalid(
        result: Result<(OperatorDataset<f64>, OperatorDataset<f64>, Normalizers), LoadError>,
        needle: &str,
    ) {
        let err = result.err().expect("loading should fail");
        assert!(matches!(err, LoadError::Invalid(_)), "{err:?}");
        assert!(err.to_string().contains(needle), "{err}");
    }

    #[test]
    fn shapes_match_uniform_grid() {
        let dir = fresh_dir("shapes");
        let (train_path, test_path) = write_valid_pair(&dir);
        let config = cfg(N_TRAIN, N_TEST);
        let s = config.s();
        assert_eq!(s, N - 1);
        let (train, test, _) =
            load_poisson_uniform::<f32>(&train_path, &test_path, &config).unwrap();
        assert_eq!(train.inputs().shape(), [N_TRAIN, s, s, 1]);
        assert_eq!(train.targets().shape(), [N_TRAIN, s, s]);
        assert_eq!(test.inputs().shape(), [N_TEST, s, s, 1]);
        assert_eq!(test.targets().shape(), [N_TEST, s, s]);
        assert!(train.inputs().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn transfer_is_applied() {
        let dir = fresh_dir("transfer");
        let (train_path, test_path) = write_valid_pair(&dir);
        let config = cfg(N_TRAIN, N_TEST);
        let (train, test, _) =
            load_poisson_uniform::<f64>(&train_path, &test_path, &config).unwrap();

        let xu = uniform_nodes(config.s());
        let u_train = poly_field(N_TRAIN, 0.0, &xu, &xu) * -2.0;
        let u_test = poly_field(N_TEST, 0.5, &xu, &xu) * -2.0;
        assert!(rel_err(train.targets(), &u_train) < 1e-12, "train");
        assert!(rel_err(test.targets(), &u_test) < 1e-12, "test");
    }

    #[test]
    fn normalization_matches_darcy() {
        let dir = fresh_dir("asymmetry");
        let (train_path, test_path) = write_valid_pair(&dir);
        let config = cfg(N_TRAIN, N_TEST);
        let s = config.s();
        let (train, test, norms) =
            load_poisson_uniform::<f64>(&train_path, &test_path, &config).unwrap();

        // The transferred fields, computed independently of the loader's file path.
        let t = Transfers::new(N);
        let x = nodes(N);
        let rank3 = |a: ArrayD<f64>| a.into_dimensionality::<Ix3>().unwrap();
        let f_tr = transfer_samples(rank3(poly_field(N_TRAIN, 0.0, &x, &x)), N_TRAIN, &t);
        let f_te = transfer_samples(rank3(poly_field(N_TEST, 0.5, &x, &x)), N_TEST, &t);
        let (u_tr, u_te) = (&f_tr * -2.0, &f_te * -2.0);

        let as_input = |a: ArrayD<f64>| {
            let n = a.shape()[0];
            a.into_shape_with_order(IxDyn(&[n, s, s, 1])).unwrap()
        };
        // x encoded with the train normalizer, both y splits raw.
        assert_eq!(*train.inputs(), as_input(norms.x.encode(f_tr)));
        assert_eq!(*test.inputs(), as_input(norms.x.encode(f_te)));
        assert_eq!(*train.targets(), u_tr);
        assert_eq!(*test.targets(), u_te);
    }

    /// Order-sensitive sum, compared bit for bit.
    fn checksum<T: Copy + Into<f64>>(a: &ArrayD<T>) -> u64 {
        let sum: f64 = a
            .iter()
            .enumerate()
            .map(|(i, &v)| v.into() * (i + 1) as f64)
            .sum();
        sum.to_bits()
    }

    // --- phase 4s T9: the loader tail's output is pinned across its refactor ---

    #[test]
    fn poisson_output_is_pinned() {
        let dir = fresh_dir("pinned");
        let (train_path, test_path) = write_valid_pair(&dir);
        let config = cfg(N_TRAIN, N_TEST);
        let (train, test, norms) =
            load_poisson_uniform::<f32>(&train_path, &test_path, &config).unwrap();

        let sums = [
            checksum(train.inputs()),
            checksum(train.targets()),
            checksum(test.inputs()),
            checksum(test.targets()),
            checksum(norms.x.mean()),
            checksum(norms.x.std()),
            checksum(norms.y.mean()),
            checksum(norms.y.std()),
        ];
        assert_eq!(
            sums,
            [
                4595258215196790784,
                13895949219414011904,
                4586251015942045696,
                13890022982133194752,
                4651835031652120632,
                4650335882944466476,
                13879710668134266936,
                4654839482571836972,
            ],
            "{sums:?}"
        );
    }

    // --- phase 4s T18: predict encodes the test split with the run's saved normalizers ---

    #[test]
    fn test_loader_with_saved_normalizers_matches_refit() {
        use crate::neural_operators::runs::RunDir;

        let dir = fresh_dir("saved_normalizers");
        let (train_path, test_path) = write_valid_pair(&dir);
        let config = cfg(N_TRAIN, N_TEST);
        let (_, refit, fitted) =
            load_poisson_uniform::<f32>(&train_path, &test_path, &config).unwrap();

        // through the run directory, as predict reads them
        let run = RunDir::create(&dir, "run").unwrap();
        run.save_normalizers(&fitted).unwrap();
        let saved = run.normalizers().unwrap().expect("normalizers were saved");
        let test = load_poisson_uniform_test::<f32>(&test_path, &config, &saved).unwrap();

        let bits = |a: &ArrayD<f32>| a.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(test.inputs().shape(), refit.inputs().shape());
        assert_eq!(test.targets().shape(), refit.targets().shape());
        assert_eq!(bits(test.inputs()), bits(refit.inputs()));
        assert_eq!(bits(test.targets()), bits(refit.targets()));

        // normalizers fitted at another s are rejected, not broadcast into a panic
        let mut other = cfg(N_TRAIN, N_TEST);
        other.n = N + 4;
        let x = nodes(other.n);
        let f = poly_field(N_TRAIN, 0.0, &x, &x);
        let train_path = write_npz(&dir, "other.npz", &f, &f, &x, &x);
        let (_, _, wrong) = load_poisson_uniform::<f32>(&train_path, &train_path, &other).unwrap();
        let err = load_poisson_uniform_test::<f32>(&test_path, &config, &wrong)
            .err()
            .expect("loading should fail");
        assert!(matches!(err, LoadError::Invalid(_)), "{err:?}");
        assert!(err.to_string().contains("fitted on [12, 12]"), "{err}");
    }

    #[test]
    fn missing_file_is_a_reader_error() {
        let err = load_poisson_uniform::<f32>("/nope/train.npz", "/nope/test.npz", &cfg(2, 1))
            .err()
            .expect("loading should fail");
        assert!(
            matches!(err, LoadError::Reader(ReaderError::FileNotFound(_))),
            "{err:?}"
        );
        assert!(err.to_string().contains("train.npz"), "{err}");
    }

    #[test]
    fn single_train_sample_is_invalid_not_nan() {
        // nonexistent paths: the check must run before any file I/O
        expect_invalid(
            load_poisson_uniform("/nope/train.npz", "/nope/test.npz", &cfg(1, 1)),
            "at least 2",
        );
    }

    #[test]
    fn n_below_four_is_invalid_not_a_panic() {
        let mut config = cfg(2, 1);
        for n in [0, 1, 2, 3] {
            config.n = n;
            expect_invalid(
                load_poisson_uniform("/nope/train.npz", "/nope/test.npz", &config),
                "n must be >= 4",
            );
        }
    }

    #[test]
    fn uniform_nodes_are_invalid() {
        let dir = fresh_dir("uniform_nodes");
        let (_, test_path) = write_valid_pair(&dir);
        let x = uniform_nodes(N);
        let f = poly_field(N_TRAIN, 0.0, &x, &x);
        let train_path = write_npz(&dir, "bad.npz", &f, &f, &x, &nodes(N));
        expect_invalid(
            load_poisson_uniform(&train_path, &test_path, &cfg(N_TRAIN, N_TEST)),
            "'x' must hold the 9 ascending CGL nodes",
        );

        // a NaN node must not pass the tolerance check
        let mut y = nodes(N);
        y[3] = f64::NAN;
        let train_path = write_npz(&dir, "nan.npz", &f, &f, &nodes(N), &y);
        expect_invalid(
            load_poisson_uniform(&train_path, &test_path, &cfg(N_TRAIN, N_TEST)),
            "'y' must hold the 9 ascending CGL nodes",
        );
    }

    #[test]
    fn wrong_shape_is_invalid() {
        let dir = fresh_dir("wrong_shape");
        let (train_path, _) = write_valid_pair(&dir);
        let x = nodes(N);

        // u not square
        let f = poly_field(N_TEST, 0.0, &x, &x);
        let u = poly_field(N_TEST, 0.0, &x, &nodes(N - 1));
        let test_path = write_npz(&dir, "bad_u.npz", &f, &u, &x, &x);
        expect_invalid(
            load_poisson_uniform(&train_path, &test_path, &cfg(N_TRAIN, N_TEST)),
            "test field 'u' must be [samples, 9, 9]",
        );

        // f of rank 2
        let f2 = f.index_axis(Axis(0), 0).to_owned();
        let test_path = write_npz(&dir, "bad_f.npz", &f2, &f, &x, &x);
        expect_invalid(
            load_poisson_uniform(&train_path, &test_path, &cfg(N_TRAIN, N_TEST)),
            "test field 'f' must be [samples, 9, 9]",
        );

        // valid files, but the config's n disagrees with them
        let (train_path, test_path) = write_valid_pair(&dir);
        let mut config = cfg(N_TRAIN, N_TEST);
        config.n = 17;
        expect_invalid(
            load_poisson_uniform(&train_path, &test_path, &config),
            "train field 'x' must be [17]",
        );
    }

    #[test]
    fn too_few_samples_is_invalid() {
        let dir = fresh_dir("too_few");
        let (train_path, test_path) = write_valid_pair(&dir);
        expect_invalid(
            load_poisson_uniform(&train_path, &test_path, &cfg(N_TRAIN + 1, N_TEST)),
            "requested 4 train samples but 'f' has 3",
        );
        expect_invalid(
            load_poisson_uniform(&train_path, &test_path, &cfg(N_TRAIN, N_TEST + 1)),
            "requested 3 test samples but 'f' has 2",
        );
    }

    #[test]
    fn cheb_loader_reads_raw_u() {
        let dir = fresh_dir("cheb_reads");
        let (train_path, test_path) = write_valid_pair(&dir);
        let x = nodes(N);

        let u = load_poisson_cheb(&test_path, N, N_TEST).unwrap();
        assert_eq!(u.shape(), [N_TEST, N, N]);
        assert_eq!(u.into_dyn(), poly_field(N_TEST, 0.5, &x, &x) * -2.0);

        // fewer than stored: the first `count` samples
        let u = load_poisson_cheb(&train_path, N, 2).unwrap();
        assert_eq!(u.shape(), [2, N, N]);
        assert_eq!(u.into_dyn(), poly_field(2, 0.0, &x, &x) * -2.0);
    }

    #[test]
    fn cheb_loader_missing_file_is_a_reader_error() {
        let err = load_poisson_cheb("/nope/test.npz", N, 1).expect_err("loading should fail");
        assert!(
            matches!(err, LoadError::Reader(ReaderError::FileNotFound(_))),
            "{err:?}"
        );
        assert!(err.to_string().contains("test.npz"), "{err}");
    }

    /// Asserts that `result` is a [`LoadError::Invalid`] whose message contains `needle`.
    fn expect_cheb_invalid(result: Result<Array3<f64>, LoadError>, needle: &str) {
        let err = result.expect_err("loading should fail");
        assert!(matches!(err, LoadError::Invalid(_)), "{err:?}");
        assert!(err.to_string().contains(needle), "{err}");
    }

    #[test]
    fn cheb_loader_non_cgl_nodes_are_invalid() {
        let dir = fresh_dir("cheb_nodes");
        let x = uniform_nodes(N);
        let f = poly_field(N_TEST, 0.0, &x, &x);
        let path = write_npz(&dir, "bad.npz", &f, &f, &x, &nodes(N));
        expect_cheb_invalid(
            load_poisson_cheb(&path, N, N_TEST),
            "bad.npz field 'x' must hold the 9 ascending CGL nodes",
        );
        expect_cheb_invalid(load_poisson_cheb(&path, 2, 1), "n must be >= 3");
    }

    #[test]
    fn cheb_loader_too_few_samples_is_invalid() {
        let dir = fresh_dir("cheb_too_few");
        let (_, test_path) = write_valid_pair(&dir);
        expect_cheb_invalid(
            load_poisson_cheb(&test_path, N, N_TEST + 1),
            "requested 3 test.npz samples but 'u' has 2",
        );
    }

    #[test]
    fn cheb_field_loader_reads_f() {
        let dir = fresh_dir("cheb_field_f");
        let (train_path, test_path) = write_valid_pair(&dir);
        let x = nodes(N);

        let f = load_poisson_cheb_field(&test_path, "f", N, N_TEST).unwrap();
        assert_eq!(f.shape(), [N_TEST, N, N]);
        assert_eq!(f.into_dyn(), poly_field(N_TEST, 0.5, &x, &x));

        // fewer than stored: the first `count` samples
        let f = load_poisson_cheb_field(&train_path, "f", N, 2).unwrap();
        assert_eq!(f.shape(), [2, N, N]);
        assert_eq!(f.into_dyn(), poly_field(2, 0.0, &x, &x));
    }

    #[test]
    fn cheb_field_loader_rejects_unknown_field() {
        // nonexistent path: the check must run before any file I/O
        expect_cheb_invalid(
            load_poisson_cheb_field("/nope/test.npz", "g", N, 1),
            "field must be 'f' or 'u', got 'g'",
        );
    }

    /// Loads the stage 2 dataset written by `generate_poisson` (n = 65, 1000 train / 200
    /// test) and reports the wall time; for the Phase 2 Results table. Run with
    /// `cargo test --release <full path> -- --exact --ignored --nocapture`.
    #[test]
    #[ignore = "needs datasets/poisson from generate_poisson"]
    fn stage2_dataset_loads() {
        let config = PoissonConfig::new(
            DatasetConfig {
                n_train: 1000,
                n_test: 200,
            },
            65,
        );
        let start = std::time::Instant::now();
        let (train, test, _) = load_poisson_uniform::<f32>(
            "datasets/poisson/poisson_n65_K32_train.npz",
            "datasets/poisson/poisson_n65_K32_test.npz",
            &config,
        )
        .unwrap();
        let elapsed = start.elapsed();
        assert_eq!(train.inputs().shape(), [1000, 64, 64, 1]);
        assert_eq!(train.targets().shape(), [1000, 64, 64]);
        assert_eq!(test.inputs().shape(), [200, 64, 64, 1]);
        assert_eq!(test.targets().shape(), [200, 64, 64]);
        println!("stage 2 load: {:.3} s", elapsed.as_secs_f64());
    }

    #[test]
    fn set_names_its_split_files() {
        let set = PoissonSet { n: 129, k: 32 };
        assert_eq!(set.stem("test"), "poisson_n129_K32_test");
        assert_eq!(set.npz("train"), "poisson_n129_K32_train.npz");
    }

    #[test]
    fn stage_finds_each_stage_and_rejects_other_n() {
        for set in STAGES {
            assert_eq!(stage(set.n).unwrap(), set);
        }
        match stage(33) {
            Err(LoadError::Invalid(msg)) => assert!(msg.contains("[65, 129, 257]"), "{msg}"),
            other => panic!("expected Invalid, got {other:?}"),
        }
    }
}

//! Poisson dataset: config and constructor (design §5.2).
//!
//! `PoissonConfig` holds the Chebyshev size `n` on top of the shared `DatasetConfig`.
//! `PoissonSet` names a dataset's split files and `STAGES` lists the training stages.
//! `load_poisson_uniform` builds train/test `OperatorDataset`s from the `.npz` files that
//! `generate_poisson` writes (`datasets/README.md`): forcing `f` and solution `u` on the
//! n × n CGL grid. It reads and validates them, transfers every sample to the uniform grid
//! of size s = n − 1 and then normalizes, as Darcy does. Inputs carry the forcing only;
//! the model appends the 2D grid coordinates itself (see `FNO::forward`).
//!
//! Transfer: T_cu (`chebyshev::cheb_to_uniform`, CONVENTIONS §12) is applied along both
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
//! Normalization asymmetry (CONVENTIONS §10, as in `load_darcy_uniform`): `x_test` is
//! encoded with the `x` normalizer fit on `x_train`, but `y_test` is left un-encoded:
//! predictions are decoded back to physical scale before comparison, not targets encoded
//! forward. The returned `normalizers.y` is what the caller needs to do that decode.

use crate::neural_operators::chebyshev::{apply, cgl_nodes, cheb_to_uniform};
use crate::neural_operators::data::{
    dataitem::HostFloat,
    dataset::OperatorDataset,
    io::{errors::LoadError, readers::npz::NpzFileReader, traits::FieldReader},
    loaders::base_dataset::{BaseDatasetConfig, DatasetConfig, HasBaseConfig},
    transforms::normalizers::{Normalizer, UnitGaussianNormalizer},
};
use burn::config::Config;
use ndarray::{Array1, Array2, Array3, ArrayD, Axis, Ix3, IxDyn, s};
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
    /// n = 0; `load_poisson_uniform` rejects n < 3.
    pub fn s(&self) -> usize {
        self.n.saturating_sub(1)
    }
}

impl HasBaseConfig for PoissonConfig {
    fn base(&self) -> &DatasetConfig {
        &self.base
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

/// The two normalizers fitted during Poisson loading. `x` encodes inputs (the forcing);
/// `y` decodes predictions (the solution) back to physical scale. Named fields because
/// both have the same type and swapping them fails silently.
pub struct PoissonNormalizers {
    /// Fitted on the transferred train forcing; encodes inputs `[N, s, s]`.
    pub x: UnitGaussianNormalizer,
    /// Fitted on the transferred train solution; decodes predictions `[N, s, s]`.
    pub y: UnitGaussianNormalizer,
}

/// Builds the Poisson dataset on the uniform grid: inputs `[N, s, s, 1]` (the normalized
/// forcing; no grid channels, the model generates them), targets `[N, s, s]`, with
/// s = `config.s()` = n − 1.
///
/// Each file must hold `f` and `u` as `[samples, n, n]` on the CGL grid (`f[i, a, b] =
/// f(x_a, y_b)`) and `x`, `y` as the ascending CGL nodes `[n]`. The first
/// `config.n_train()` / `config.n_test()` samples are used.
///
/// Returns `(train_dataset, test_dataset, normalizers)`. `normalizers.y` is what the
/// caller needs to decode predictions back to physical scale; `normalizers.x` is only
/// needed for encoding inputs that didn't come from this dataset.
///
/// # Errors
/// [`LoadError::Reader`] if a file can't be read or lacks `f`/`u`/`x`/`y`;
/// [`LoadError::Invalid`] if `n_train < 2` (the normalizers' sample std is undefined for
/// one sample), `config.n < 3` (the uniform grid needs s ≥ 2), `f` or `u` isn't
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
) -> Result<(OperatorDataset<T>, OperatorDataset<T>, PoissonNormalizers), LoadError> {
    // ddof = 1 std of a single sample is 0/0 = NaN, which eps can't guard
    if config.n_train() < 2 {
        return Err(LoadError::Invalid(format!(
            "n_train must be >= 2: a sample std needs at least 2 training samples, got {}",
            config.n_train()
        )));
    }
    let n = config.n;
    if n < 3 {
        return Err(LoadError::Invalid(format!(
            "n must be >= 3 (the uniform grid s = n - 1 needs both endpoints), got {n}"
        )));
    }

    // T_cu is built once and shared by both splits; each split's reader and Chebyshev
    // fields are freed before the next file is parsed.
    let t_cu = cheb_to_uniform(n, config.s());
    let nodes = cgl_nodes(n);
    let load_split = |path: &Path, split: &str, count: usize| -> Result<_, LoadError> {
        let reader = NpzFileReader::new(path);
        check_nodes(&reader, split, n, &nodes)?;
        let f = read_samples(&reader, split, "f", n, count)?;
        let f = transfer_samples(f, count, &t_cu);
        let u = read_samples(&reader, split, "u", n, count)?;
        let u = transfer_samples(u, count, &t_cu);
        Ok((f, u))
    };
    let (f_train, u_train) = load_split(train_path.as_ref(), "train", config.n_train())?;
    let (f_test, u_test) = load_split(test_path.as_ref(), "test", config.n_test())?;
    poisson_from_fields(f_train, u_train, f_test, u_test, n)
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
    let reader = NpzFileReader::new(path);
    check_nodes(&reader, &label, n, &cgl_nodes(n))?;
    let values = read_samples(&reader, &label, field, n, count)?
        .into_dimensionality::<Ix3>()
        .expect("read_samples checked rank 3");
    Ok(values.slice_move(s![..count, .., ..]))
}

/// Checks that the fields `x` and `y` are `[n]` and within [`NODE_TOL`] of the CGL
/// `nodes`. `split` only labels the error messages.
fn check_nodes(
    reader: &NpzFileReader,
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
    reader: &NpzFileReader,
    split: &str,
    name: &str,
    n: usize,
    count: usize,
) -> Result<ArrayD<f64>, LoadError> {
    let field = reader.read_field(name)?;
    if field.ndim() != 3 || field.shape()[1..] != [n, n] {
        return Err(LoadError::Invalid(format!(
            "{split} field '{name}' must be [samples, {n}, {n}], got shape {:?}",
            field.shape()
        )));
    }
    if count > field.shape()[0] {
        return Err(LoadError::Invalid(format!(
            "requested {count} {split} samples but '{name}' has {}",
            field.shape()[0]
        )));
    }
    Ok(field)
}

/// The first `count` samples of a `[samples, n, n]` Chebyshev field, each transferred to
/// the uniform grid as T_cu · F · T_cuᵀ. `t_cu: [s, n]`; returns `[count, s, s]`.
///
/// The caller has checked the shape and that `count` samples exist.
fn transfer_samples(field: ArrayD<f64>, count: usize, t_cu: &Array2<f64>) -> ArrayD<f64> {
    let field = field
        .into_dimensionality::<Ix3>()
        .expect("caller checked rank 3");
    let s = t_cu.nrows();
    let mut out = Array3::zeros((count, s, s));
    for (mut dst, src) in out
        .axis_iter_mut(Axis(0))
        .zip(field.slice(s![..count, .., ..]).axis_iter(Axis(0)))
    {
        dst.assign(&apply(t_cu.view(), t_cu.view(), src));
    }
    out.into_dyn()
}

/// The rest of [`load_poisson_uniform`] on transferred `[samples, s, s]` fields:
/// normalize, reshape and cast. Separate from the reader so the output can be tested
/// without `.npz` files. `n` is only reported.
fn poisson_from_fields<T: HostFloat>(
    f_train: ArrayD<f64>,
    u_train: ArrayD<f64>,
    f_test: ArrayD<f64>,
    u_test: ArrayD<f64>,
    n: usize,
) -> Result<(OperatorDataset<T>, OperatorDataset<T>, PoissonNormalizers), LoadError> {
    let s = f_train.shape()[1];
    let n_train = f_train.shape()[0];
    let n_test = f_test.shape()[0];

    // fit x-normalizer on the train forcing, encode both train and test forcings;
    // fit y-normalizer on the train solution, encode it only - the test solution stays
    // raw, decoded against at eval time via the returned y normalizer
    let x_normalizer = UnitGaussianNormalizer::fit(&f_train);
    let x_train = x_normalizer.encode(f_train);
    let x_test = x_normalizer.encode(f_test);

    let y_normalizer = UnitGaussianNormalizer::fit(&u_train);
    let y_train = y_normalizer.encode(u_train);
    // y_test intentionally NOT encoded
    let y_test = u_test;

    // add a trailing channel axis to inputs: [N, s, s] -> [N, s, s, 1]
    let x_train = x_train
        .into_shape_with_order(IxDyn(&[n_train, s, s, 1]))
        .map_err(|e| LoadError::Invalid(format!("reshape x_train: {e}")))?;
    let x_test = x_test
        .into_shape_with_order(IxDyn(&[n_test, s, s, 1]))
        .map_err(|e| LoadError::Invalid(format!("reshape x_test: {e}")))?;

    // package into OperatorDataset, casting to the host dtype T
    let train_dataset = OperatorDataset::from_f64(x_train, y_train);
    let test_dataset = OperatorDataset::from_f64(x_test, y_test);

    println!("poisson: {n_train} train / {n_test} test, n={n} -> s={s}");

    Ok((
        train_dataset,
        test_dataset,
        PoissonNormalizers {
            x: x_normalizer,
            y: y_normalizer,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::chebyshev::uniform_nodes;
    use crate::neural_operators::data::io::errors::ReaderError;
    use ndarray::Array1;
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
        let x = cgl_nodes(N);
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
        result: Result<
            (
                OperatorDataset<f64>,
                OperatorDataset<f64>,
                PoissonNormalizers,
            ),
            LoadError,
        >,
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
        let (train, test, norms) =
            load_poisson_uniform::<f64>(&train_path, &test_path, &config).unwrap();

        let xu = uniform_nodes(config.s());
        let u_train = poly_field(N_TRAIN, 0.0, &xu, &xu) * -2.0;
        let u_test = poly_field(N_TEST, 0.5, &xu, &xu) * -2.0;
        let decoded = norms.y.decode(train.targets().clone());
        assert!(rel_err(&decoded, &u_train) < 1e-12, "train");
        assert!(rel_err(test.targets(), &u_test) < 1e-12, "test");
    }

    #[test]
    fn normalizer_asymmetry_matches_darcy() {
        let dir = fresh_dir("asymmetry");
        let (train_path, test_path) = write_valid_pair(&dir);
        let config = cfg(N_TRAIN, N_TEST);
        let s = config.s();
        let (train, test, norms) =
            load_poisson_uniform::<f64>(&train_path, &test_path, &config).unwrap();

        // The transferred fields, computed independently of the loader's file path.
        let t = cheb_to_uniform(N, s);
        let x = cgl_nodes(N);
        let f_tr = transfer_samples(poly_field(N_TRAIN, 0.0, &x, &x), N_TRAIN, &t);
        let f_te = transfer_samples(poly_field(N_TEST, 0.5, &x, &x), N_TEST, &t);
        let (u_tr, u_te) = (&f_tr * -2.0, &f_te * -2.0);

        let as_input = |a: ArrayD<f64>| {
            let n = a.shape()[0];
            a.into_shape_with_order(IxDyn(&[n, s, s, 1])).unwrap()
        };
        // x encoded with the train normalizer, y_train encoded, y_test raw.
        assert_eq!(*train.inputs(), as_input(norms.x.encode(f_tr)));
        assert_eq!(*test.inputs(), as_input(norms.x.encode(f_te)));
        assert_eq!(*train.targets(), norms.y.encode(u_tr));
        assert_eq!(*test.targets(), u_te);
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
    fn n_below_three_is_invalid_not_a_panic() {
        let mut config = cfg(2, 1);
        for n in [0, 1, 2] {
            config.n = n;
            expect_invalid(
                load_poisson_uniform("/nope/train.npz", "/nope/test.npz", &config),
                "n must be >= 3",
            );
        }
    }

    #[test]
    fn uniform_nodes_are_invalid() {
        let dir = fresh_dir("uniform_nodes");
        let (_, test_path) = write_valid_pair(&dir);
        let x = uniform_nodes(N);
        let f = poly_field(N_TRAIN, 0.0, &x, &x);
        let train_path = write_npz(&dir, "bad.npz", &f, &f, &x, &cgl_nodes(N));
        expect_invalid(
            load_poisson_uniform(&train_path, &test_path, &cfg(N_TRAIN, N_TEST)),
            "'x' must hold the 9 ascending CGL nodes",
        );

        // a NaN node must not pass the tolerance check
        let mut y = cgl_nodes(N);
        y[3] = f64::NAN;
        let train_path = write_npz(&dir, "nan.npz", &f, &f, &cgl_nodes(N), &y);
        expect_invalid(
            load_poisson_uniform(&train_path, &test_path, &cfg(N_TRAIN, N_TEST)),
            "'y' must hold the 9 ascending CGL nodes",
        );
    }

    #[test]
    fn wrong_shape_is_invalid() {
        let dir = fresh_dir("wrong_shape");
        let (train_path, _) = write_valid_pair(&dir);
        let x = cgl_nodes(N);

        // u not square
        let f = poly_field(N_TEST, 0.0, &x, &x);
        let u = poly_field(N_TEST, 0.0, &x, &cgl_nodes(N - 1));
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
        let x = cgl_nodes(N);

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
        let path = write_npz(&dir, "bad.npz", &f, &f, &x, &cgl_nodes(N));
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
        let x = cgl_nodes(N);

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

//! The uniform loader: train/test `OperatorDataset`s on the uniform grid s = n − 1, every
//! sample transferred with T_cu and then normalized (see the module above).

use super::PoissonConfig;
use super::cheb::{PoissonField, check_nodes, read_samples};
use crate::neural_operators::chebyshev::{TRANSFER_DEGREE, Transfers, nodes};
use crate::neural_operators::data::{
    dataset::OperatorDataset,
    host_float::HostFloat,
    io::{errors::LoadError, readers::npz::NpzFileReader},
    loaders::common::{Normalizers, check_n_train, normalize_split_pair, normalize_test_split},
};
use ndarray::{ArrayD, s};
use std::path::Path;

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
/// each to the uniform grid with `transfers` ([`Transfers::to_uniform_all`]):
/// `([count, s, s], [count, s, s])`. `split` only labels the error messages.
fn read_uniform_split(
    path: &Path,
    split: &str,
    count: usize,
    transfers: &Transfers,
) -> Result<(ArrayD<f64>, ArrayD<f64>), LoadError> {
    let n = transfers.n();
    let mut reader = NpzFileReader::new(path)?;
    check_nodes(&mut reader, split, n, &nodes(n))?;
    let mut read = |field| -> Result<ArrayD<f64>, LoadError> {
        let values = read_samples(&mut reader, split, field, n, count)?;
        Ok(transfers
            .to_uniform_all(values.slice(s![..count, .., ..]))
            .into_dyn())
    };
    let f = read(PoissonField::F)?;
    let u = read(PoissonField::U)?;
    Ok((f, u))
}

#[cfg(test)]
mod tests {
    use super::super::fixtures::{
        N, N_TEST, N_TRAIN, cfg, fresh_dir, poly_field, write_npz, write_valid_pair,
    };
    use super::*;
    use crate::neural_operators::chebyshev::uniform_nodes;
    use crate::neural_operators::data::io::errors::ReaderError;
    use crate::neural_operators::data::loaders::base_dataset::DatasetConfig;
    use ndarray::{Axis, Ix3, IxDyn};

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
        let uniform = |a: ArrayD<f64>| {
            let a = a.into_dimensionality::<Ix3>().unwrap();
            t.to_uniform_all(a.view()).into_dyn()
        };
        let f_tr = uniform(poly_field(N_TRAIN, 0.0, &x, &x));
        let f_te = uniform(poly_field(N_TEST, 0.5, &x, &x));
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
}

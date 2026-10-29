//! The uniform loader: train/test `OperatorDataset`s on the uniform grid of s points
//! (default n − 1), every sample transferred with T_cu and then normalized (see the module
//! above).

use super::{PoissonConfig, PoissonField, check_nodes, read_samples};
use crate::neural_operators::chebyshev::{TRANSFER_DEGREE, Transfers, nodes};
use crate::neural_operators::data::{
    dataset::OperatorDataset,
    host_float::HostFloat,
    io::{errors::LoadError, readers::npz::NpzFileReader},
    loaders::common::{Normalizers, check_n_train, normalize_split_pair, normalize_test_split},
};
use ndarray::{Array3, ArrayD, Axis, s};
use std::path::Path;

/// Builds the Poisson dataset on the uniform grid: inputs `[N, s, s, 1]` (the normalized
/// forcing; no grid channels, the model generates them), targets `[N, s, s]`, with
/// s = `config.s()` (default n − 1).
///
/// Each file must hold `f` and `u` as `[samples, n, n]` on the CGL grid (`f[i, a, b] =
/// f(x_a, y_b)`) and `x`, `y` as the ascending CGL nodes `[n]`. The first
/// `config.base.n_test` test samples are used. The train samples are the first
/// `config.base.n_train`, or with `config.subset` the seeded nested subset of that many
/// out of its first `population` samples
/// ([`subset_indices`](super::subset_indices)), in file order; the
/// normalizers are then fitted on the subset alone (CONVENTIONS §10).
///
/// Returns `(train_dataset, test_dataset, normalizers)`. `normalizers.y` is what the
/// caller needs to decode predictions back to physical scale; `normalizers.x` is only
/// needed for encoding inputs that didn't come from this dataset.
///
/// # Errors
/// [`LoadError::Reader`] if a file can't be read or lacks `f`/`u`/`x`/`y`;
/// [`LoadError::Invalid`] if `n_train < 2` (the normalizers' sample std is undefined for
/// one sample), the subset asks for more than its `population` or its hash differs from
/// the recomputed one, `config.n <` [`Transfers::MIN_N`] or `config.s() <` [`Transfers::MIN_S`]
/// (T_uc of degree [`TRANSFER_DEGREE`] needs s − 1 ≥ d), `f` or `u` isn't `[samples, n, n]`, `x` or `y` isn't `[n]` or differs from the CGL nodes by more than
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
    let train_rows = Rows::train(config)?;

    // The transfers are built once and shared by both splits; each split's reader and
    // Chebyshev fields are freed before the next file is parsed.
    let transfers = transfers(config)?;
    let (f_train, u_train) =
        read_uniform_split(train_path.as_ref(), "train", &train_rows, &transfers)?;
    let test_rows = Rows::Prefix(config.base.n_test);
    let (f_test, u_test) = read_uniform_split(test_path.as_ref(), "test", &test_rows, &transfers)?;
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
/// [`LoadError::Invalid`] if `config.n <` [`Transfers::MIN_N`], `config.s() <`
/// [`Transfers::MIN_S`], `f` or `u` isn't
/// `[samples, n, n]`, `x` or `y` isn't `[n]` or differs from the CGL nodes by more than
/// 1e-14, the file has fewer than `n_test` samples, or `normalizers.x` was not fitted on
/// `[s, s]`.
pub fn load_poisson_uniform_test<T: HostFloat>(
    test_path: impl AsRef<Path>,
    config: &PoissonConfig,
    normalizers: &Normalizers,
) -> Result<OperatorDataset<T>, LoadError> {
    let transfers = transfers(config)?;
    let rows = Rows::Prefix(config.base.n_test);
    let (f_test, u_test) = read_uniform_split(test_path.as_ref(), "test", &rows, &transfers)?;
    normalize_test_split(f_test, u_test, normalizers)
}

/// The transfers for `config`'s n and s ([`Transfers::with_s`]), or [`LoadError::Invalid`]
/// where that would panic: n below [`Transfers::MIN_N`] or s below [`Transfers::MIN_S`].
fn transfers(config: &PoissonConfig) -> Result<Transfers, LoadError> {
    let (n, s) = (config.n, config.s());
    if n < Transfers::MIN_N {
        return Err(LoadError::Invalid(format!(
            "n must be >= {} (the smallest supported CGL grid), got {n}",
            Transfers::MIN_N
        )));
    }
    if s < Transfers::MIN_S {
        return Err(LoadError::Invalid(format!(
            "s must be >= {} (T_uc of degree {TRANSFER_DEGREE} needs s - 1 >= d), got {s}",
            Transfers::MIN_S
        )));
    }
    Ok(Transfers::with_s(n, s))
}

/// The samples of a split to load.
enum Rows {
    /// The first `n`.
    Prefix(usize),
    /// These sorted rows of a split of at least `population` samples.
    Subset { population: usize, rows: Vec<usize> },
}

impl Rows {
    /// The train rows of `config`: its subset's, recomputed and checked against the hash
    /// before any file is read ([`SubsetConfig::rows`](super::SubsetConfig::rows)), else
    /// the first `base.n_train`.
    fn train(config: &PoissonConfig) -> Result<Self, LoadError> {
        let n = config.base.n_train;
        Ok(match &config.subset {
            Some(subset) => Self::Subset {
                population: subset.population,
                rows: subset.rows(n).map_err(LoadError::Invalid)?,
            },
            None => Self::Prefix(n),
        })
    }

    /// The samples the split must hold.
    fn required(&self) -> usize {
        match self {
            Self::Prefix(n) => *n,
            Self::Subset { population, .. } => *population,
        }
    }

    /// The selected samples of `values` (`[samples, n, n]`), in file order.
    fn take(&self, values: Array3<f64>) -> Array3<f64> {
        match self {
            Self::Prefix(n) => values.slice_move(s![..*n, .., ..]),
            Self::Subset { rows, .. } => values.select(Axis(0), rows),
        }
    }
}

/// Reads one split's `f` and `u`, validated (at least [`Rows::required`] samples), and
/// transfers the `rows` of each to the uniform grid with `transfers`
/// ([`Transfers::to_uniform_all`]): `([k, s, s], [k, s, s])` for k samples. `split` only
/// labels the error messages.
fn read_uniform_split(
    path: &Path,
    split: &str,
    rows: &Rows,
    transfers: &Transfers,
) -> Result<(ArrayD<f64>, ArrayD<f64>), LoadError> {
    let n = transfers.n();
    let mut reader = NpzFileReader::new(path)?;
    check_nodes(&mut reader, split, n, &nodes(n))?;
    let mut read = |field| -> Result<ArrayD<f64>, LoadError> {
        let values = read_samples(&mut reader, split, field, n, rows.required())?;
        Ok(transfers
            .to_uniform_all(rows.take(values).view())
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

    // --- phase 5 T3: one dataset loads at any learning grid s ---

    #[test]
    fn loads_at_any_s_as_the_restriction() {
        let dir = fresh_dir("any_s");
        let (train_path, test_path) = write_valid_pair(&dir);
        let x = nodes(N);
        let u_train = poly_field(N_TRAIN, 0.0, &x, &x) * -2.0;
        let u_test = poly_field(N_TEST, 0.5, &x, &x) * -2.0;
        for s in [5, 12] {
            let config = cfg(N_TRAIN, N_TEST).with_s(Some(s));
            let (train, test, _) =
                load_poisson_uniform::<f64>(&train_path, &test_path, &config).unwrap();
            assert_eq!(train.inputs().shape(), [N_TRAIN, s, s, 1], "s={s}");
            assert_eq!(train.targets().shape(), [N_TRAIN, s, s], "s={s}");
            assert_eq!(test.inputs().shape(), [N_TEST, s, s, 1], "s={s}");
            assert_eq!(test.targets().shape(), [N_TEST, s, s], "s={s}");

            // the same fields, restricted to this s by T2's transfers
            let t = Transfers::with_s(N, s);
            let restrict = |a: &ArrayD<f64>| {
                let a = a.view().into_dimensionality::<Ix3>().unwrap();
                t.to_uniform_all(a).into_dyn()
            };
            assert_eq!(*train.targets(), restrict(&u_train), "s={s}");
            assert_eq!(*test.targets(), restrict(&u_test), "s={s}");
        }
    }

    #[test]
    fn s_below_three_is_invalid_not_a_panic() {
        for s in [0, 1, 2] {
            expect_invalid(
                load_poisson_uniform(
                    "/nope/train.npz",
                    "/nope/test.npz",
                    &cfg(2, 1).with_s(Some(s)),
                ),
                "s must be >= 3",
            );
        }
    }

    // --- phase 7 T1: nested seeded subsets, normalizers fitted on the subset ---

    #[test]
    fn subset_loads_the_selected_samples_and_fits_on_them() {
        use crate::neural_operators::data::loaders::poisson::{SubsetConfig, subset_indices};

        let dir = fresh_dir("subset");
        let (_, test_path) = write_valid_pair(&dir);
        let x = nodes(N);
        let (population, n_sub, seed) = (8, 3, 0);
        let f = poly_field(population, 0.0, &x, &x);
        let train_path = write_npz(&dir, "train8.npz", &f, &(&f * -2.0), &x, &x);

        let rows = subset_indices(population, n_sub, seed);
        assert_ne!(rows, [0, 1, 2], "the subset is not the prefix");
        let config =
            cfg(n_sub, N_TEST).with_subset(Some(SubsetConfig::draw(seed, population, n_sub)));
        let (train, test, norms) =
            load_poisson_uniform::<f64>(&train_path, &test_path, &config).unwrap();

        // the same samples written as a file of their own, loaded as a plain prefix
        let f_sel = f.select(Axis(0), &rows);
        let sel_path = write_npz(&dir, "selected.npz", &f_sel, &(&f_sel * -2.0), &x, &x);
        let (want_train, want_test, want_norms) =
            load_poisson_uniform::<f64>(&sel_path, &test_path, &cfg(n_sub, N_TEST)).unwrap();
        assert_eq!(train.inputs(), want_train.inputs());
        assert_eq!(train.targets(), want_train.targets());
        assert_eq!(test.inputs(), want_test.inputs());
        assert_eq!(norms.x.mean(), want_norms.x.mean());
        assert_eq!(norms.y.std(), want_norms.y.std());

        // not the normalizers of the whole population
        let (_, _, all) =
            load_poisson_uniform::<f64>(&train_path, &test_path, &cfg(population, N_TEST)).unwrap();
        assert_ne!(norms.x.mean(), all.x.mean());
    }

    #[test]
    fn subset_with_wrong_hash_or_population_is_invalid() {
        use crate::neural_operators::data::loaders::poisson::SubsetConfig;

        // nonexistent paths: both checks run before any file I/O
        let wrong = SubsetConfig::new(0, 8, "0000000000000000".to_string());
        expect_invalid(
            load_poisson_uniform(
                "/nope/train.npz",
                "/nope/test.npz",
                &cfg(3, 1).with_subset(Some(wrong)),
            ),
            "differs from the configured 0000000000000000",
        );
        let small = SubsetConfig::new(0, 2, String::new());
        expect_invalid(
            load_poisson_uniform(
                "/nope/train.npz",
                "/nope/test.npz",
                &cfg(3, 1).with_subset(Some(small)),
            ),
            "needs a population of at least 3, got 2",
        );
    }

    #[test]
    fn subset_population_beyond_the_file_is_invalid() {
        use crate::neural_operators::data::loaders::poisson::SubsetConfig;

        let dir = fresh_dir("subset_population");
        let (train_path, test_path) = write_valid_pair(&dir);
        let config = cfg(2, N_TEST).with_subset(Some(SubsetConfig::draw(0, N_TRAIN + 1, 2)));
        expect_invalid(
            load_poisson_uniform(&train_path, &test_path, &config),
            "requested 4 train samples but 'f' has 3",
        );
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

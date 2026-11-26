//! The 3D uniform loader (3D CL3.3): the 3D splits of `generate_poisson3d` on the uniform
//! grid of s points per axis (default n − 1), every sample transferred with T_cu along x,
//! then y, then z ([`Transfers::to_uniform_all_nd`]) and then normalized, as the 2D loader
//! does. The validation and the train rows are the 2D loader's, with a third node field
//! `z` and fields `[samples, n, n, n]` stored 'ijk' (3D proposal §3.2).

use super::AXES_3D;
use super::PoissonConfig;
use super::uniform::{Rows, read_uniform_split, transfers};
use crate::neural_operators::data::{
    dataset::OperatorDataset,
    host_float::HostFloat,
    io::errors::LoadError,
    loaders::common::{Normalizers, check_n_train, normalize_split_pair, normalize_test_split},
};
use std::path::Path;

/// [`load_poisson_uniform`](super::load_poisson_uniform) for the 3D splits: inputs
/// `[N, s, s, s, 1]` (the normalized forcing; the model appends the grid channels),
/// targets `[N, s, s, s]` in physical units (CONVENTIONS §10), s = `config.s()`.
///
/// Each file must hold `f` and `u` as `[samples, n, n, n]` on the CGL grid, stored 'ijk'
/// (`f[i, a, b, c] = f(x_a, y_b, z_c)`), and `x`, `y`, `z` as the ascending CGL nodes
/// `[n]`. The rows, the normalizers and the precision are those of the 2D loader.
///
/// # Errors
/// As [`load_poisson_uniform`](super::load_poisson_uniform), with `z` among the node fields
/// and `f`, `u` required to be `[samples, n, n, n]`.
pub fn load_poisson3d_uniform<T: HostFloat>(
    train_path: impl AsRef<Path>,
    test_path: impl AsRef<Path>,
    config: &PoissonConfig,
) -> Result<(OperatorDataset<T>, OperatorDataset<T>, Normalizers), LoadError> {
    check_n_train(config.base.n_train)?;
    let train_rows = Rows::train(config)?;
    let transfers = transfers(config)?;
    let (f_train, u_train) = read_uniform_split(
        train_path.as_ref(),
        "train",
        &train_rows,
        &transfers,
        &AXES_3D,
    )?;
    let test_rows = Rows::Prefix(config.base.n_test);
    let (f_test, u_test) =
        read_uniform_split(test_path.as_ref(), "test", &test_rows, &transfers, &AXES_3D)?;
    Ok(normalize_split_pair(f_train, u_train, f_test, u_test))
}

/// [`load_poisson_uniform_test`](super::load_poisson_uniform_test) for the 3D splits: the
/// test split alone, encoded with a run's `normalizers`: inputs `[n_test, s, s, s, 1]`,
/// targets `[n_test, s, s, s]`. With the normalizers [`load_poisson3d_uniform`] fits, it
/// equals that function's test split bit for bit.
///
/// # Errors
/// As [`load_poisson3d_uniform`], or [`LoadError::Invalid`] if `normalizers.x` was not
/// fitted on `[s, s, s]`.
pub fn load_poisson3d_uniform_test<T: HostFloat>(
    test_path: impl AsRef<Path>,
    config: &PoissonConfig,
    normalizers: &Normalizers,
) -> Result<OperatorDataset<T>, LoadError> {
    let transfers = transfers(config)?;
    let rows = Rows::Prefix(config.base.n_test);
    let (f_test, u_test) =
        read_uniform_split(test_path.as_ref(), "test", &rows, &transfers, &AXES_3D)?;
    normalize_test_split(f_test, u_test, normalizers)
}

#[cfg(test)]
mod tests {
    use super::super::fixtures::{
        N, N_TEST, N_TRAIN, cfg, fresh_dir, poly_field3, write_npz3, write_valid_pair3,
    };
    use super::super::{PoissonField, SubsetConfig, load_poisson3d_cheb_field};
    use super::*;
    use crate::neural_operators::chebyshev::{Transfers, nodes, uniform_nodes};
    use crate::neural_operators::data::io::errors::ReaderError;
    use ndarray::{ArrayD, Axis, IxDyn};

    fn rel_err(a: &ArrayD<f64>, b: &ArrayD<f64>) -> f64 {
        (a - b).mapv(|v| v * v).sum().sqrt() / b.mapv(|v| v * v).sum().sqrt()
    }

    fn expect_invalid<D>(result: Result<D, LoadError>, needle: &str) {
        let err = result.err().expect("loading should fail");
        assert!(matches!(err, LoadError::Invalid(_)), "{err:?}");
        assert!(err.to_string().contains(needle), "{err}");
    }

    #[test]
    fn shapes_and_closed_form_at_any_s() {
        let dir = fresh_dir("3d_closed_form");
        let (train_path, test_path) = write_valid_pair3(&dir);
        for s in [None, Some(5)] {
            let config = cfg(N_TRAIN, N_TEST).with_s(s);
            let s = config.s();
            let (train, test, _) =
                load_poisson3d_uniform::<f64>(&train_path, &test_path, &config).unwrap();
            assert_eq!(train.inputs().shape(), [N_TRAIN, s, s, s, 1]);
            assert_eq!(train.targets().shape(), [N_TRAIN, s, s, s]);
            assert_eq!(test.inputs().shape(), [N_TEST, s, s, s, 1]);
            assert_eq!(test.targets().shape(), [N_TEST, s, s, s]);
            // Targets in physical units: u = −2 f on the uniform nodes, T_cu exact.
            let xu = uniform_nodes(s);
            let train_u = poly_field3(N_TRAIN, 0.0, &xu) * -2.0;
            let test_u = poly_field3(N_TEST, 0.5, &xu) * -2.0;
            assert!(rel_err(train.targets(), &train_u) < 1e-12, "train, s = {s}");
            assert!(rel_err(test.targets(), &test_u) < 1e-12, "test, s = {s}");
        }
    }

    #[test]
    fn transfer_and_normalization_bit_for_bit() {
        let dir = fresh_dir("3d_normalization");
        let (train_path, test_path) = write_valid_pair3(&dir);
        let config = cfg(N_TRAIN, N_TEST);
        let s = config.s();
        let (train, test, norms) =
            load_poisson3d_uniform::<f64>(&train_path, &test_path, &config).unwrap();

        // The transfer computed independently of the loader's file path.
        let t = Transfers::new(N);
        let x = nodes(N);
        let f_tr = t.to_uniform_all_nd(poly_field3(N_TRAIN, 0.0, &x).view());
        let f_te = t.to_uniform_all_nd(poly_field3(N_TEST, 0.5, &x).view());
        let (u_tr, u_te) = (&f_tr * -2.0, &f_te * -2.0);
        let as_input = |a: ArrayD<f64>| {
            let n = a.shape()[0];
            a.into_shape_with_order(IxDyn(&[n, s, s, s, 1])).unwrap()
        };
        // x encoded with the train normalizer, both y splits raw (CONVENTIONS §10).
        assert_eq!(*train.inputs(), as_input(norms.x.encode(f_tr)));
        assert_eq!(*test.inputs(), as_input(norms.x.encode(f_te)));
        assert_eq!(*train.targets(), u_tr);
        assert_eq!(*test.targets(), u_te);

        let alone = load_poisson3d_uniform_test::<f64>(&test_path, &config, &norms).unwrap();
        assert_eq!(alone.inputs(), test.inputs());
        assert_eq!(alone.targets(), test.targets());
    }

    #[test]
    fn subset_takes_the_selected_rows() {
        let dir = fresh_dir("3d_subset");
        let x = nodes(N);
        let axes = [("x", &x), ("y", &x), ("z", &x)];
        let train_path = write_npz3(&dir, "train.npz", &poly_field3(5, 0.0, &x), &axes);
        let test_path = write_npz3(&dir, "test.npz", &poly_field3(N_TEST, 0.5, &x), &axes);
        let whole = cfg(5, N_TEST);
        let (all, _, _) = load_poisson3d_uniform::<f64>(&train_path, &test_path, &whole).unwrap();
        let subset = SubsetConfig::draw(7, 5, 3);
        let rows = subset.rows(3).unwrap();
        let config = cfg(3, N_TEST).with_subset(Some(subset));
        let (some, _, _) = load_poisson3d_uniform::<f64>(&train_path, &test_path, &config).unwrap();
        assert_eq!(*some.targets(), all.targets().select(Axis(0), &rows));
    }

    #[test]
    fn invalid_files_are_rejected() {
        let dir = fresh_dir("3d_invalid");
        let (train_path, _) = write_valid_pair3(&dir);
        let x = nodes(N);
        let f = poly_field3(N_TEST, 0.5, &x);
        let config = cfg(N_TRAIN, N_TEST);

        // a 2D-style file: no z
        let test_path = write_npz3(&dir, "no_z.npz", &f, &[("x", &x), ("y", &x)]);
        let err = load_poisson3d_uniform::<f64>(&train_path, &test_path, &config).err();
        assert!(
            matches!(err, Some(LoadError::Reader(ReaderError::FieldNotFound(_)))),
            "{err:?}"
        );

        // z not the CGL nodes
        let bad = &x * 0.5;
        let test_path = write_npz3(&dir, "bad_z.npz", &f, &[("x", &x), ("y", &x), ("z", &bad)]);
        expect_invalid(
            load_poisson3d_uniform::<f64>(&train_path, &test_path, &config),
            "test field 'z' must hold the 9 ascending CGL nodes",
        );

        // 2D fields with 3D nodes
        let f2 = f.index_axis(Axis(3), 0).to_owned();
        let axes = [("x", &x), ("y", &x), ("z", &x)];
        let test_path = write_npz3(&dir, "f2.npz", &f2, &axes);
        expect_invalid(
            load_poisson3d_uniform::<f64>(&train_path, &test_path, &config),
            "test field 'f' must be [samples, 9, 9, 9], got shape [2, 9, 9]",
        );

        // too few samples
        expect_invalid(
            load_poisson3d_uniform::<f64>(&train_path, &train_path, &cfg(N_TRAIN, N_TRAIN + 1)),
            "requested 4 test samples but 'f' has 3",
        );
    }

    /// Error 2 of the n = 65 test splits, 2D and 3D, at s = 32 and 64 (3D CL3.3, proposal
    /// §9): the loader's targets (R u) as the prediction, so error 3 is exactly 0 and error 4
    /// equals error 2; at s = 64 the sidecar's `error2` recomputed. Prints the table of
    /// `docs/laplace3d/results/cl3-3.md`. Run with
    /// `cargo test --release --lib <full path> -- --exact --ignored --nocapture`.
    #[test]
    #[ignore = "needs datasets/poisson from generate_poisson and generate_poisson3d"]
    fn error2_n65_against_2d() {
        use crate::neural_operators::data::catalogue::{Poisson3dSet, PoissonSet};
        use crate::neural_operators::data::loaders::base_dataset::DatasetConfig;
        use crate::neural_operators::data::loaders::poisson::{
            SidecarRecord, load_poisson_cheb_field, load_poisson_uniform,
        };
        use crate::neural_operators::metrics::{poisson::poisson_errors_nd, stats::ErrorStats};
        use burn::config::Config;

        let root = Path::new("datasets/poisson");
        let n_test = 200;
        // The test split stands in for the train split too: only targets are used.
        let base = DatasetConfig {
            n_train: n_test,
            n_test,
        };
        let config = |s| PoissonConfig::new(base.clone(), 65).with_s(Some(s));
        let set2 = PoissonSet { n: 65, k: 32 };
        let set3 = Poisson3dSet { n: 65, k: 32 };
        println!("| dim | s | error 2 mean | error 2 max | load (s) |");
        for (dim, npz, json) in [
            (2, set2.npz("test"), set2.json("test")),
            (3, set3.npz("test"), set3.json("test")),
        ] {
            let path = root.join(&npz);
            let sidecar = SidecarRecord::load(root.join(&json)).unwrap();
            assert_eq!(sidecar.error2.s, 64, "{dim}D sidecar's s");
            for s in [32, 64] {
                let start = std::time::Instant::now();
                let (targets, u) = if dim == 2 {
                    let (_, test, _) =
                        load_poisson_uniform::<f64>(&path, &path, &config(s)).unwrap();
                    let u = load_poisson_cheb_field(&path, PoissonField::U, 65, n_test).unwrap();
                    (test.targets().clone(), u.into_dyn())
                } else {
                    let (_, test, _) =
                        load_poisson3d_uniform::<f64>(&path, &path, &config(s)).unwrap();
                    let u = load_poisson3d_cheb_field(&path, PoissonField::U, 65, n_test).unwrap();
                    (test.targets().clone(), u)
                };
                let load_s = start.elapsed().as_secs_f64();
                let errors = poisson_errors_nd(targets.view(), u.view(), &Transfers::with_s(65, s));
                assert!(
                    errors.model.iter().all(|&e| e == 0.0),
                    "error 3, {dim}D s = {s}"
                );
                assert_eq!(errors.total, errors.transfer, "error 4, {dim}D s = {s}");
                let e2 = ErrorStats::of(&errors.transfer);
                println!(
                    "| {dim}D | {s} | {:.3e} | {:.3e} | {load_s:.1} |",
                    e2.mean, e2.max
                );
                if s == 64 {
                    let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs();
                    assert!(close(e2.mean, sidecar.error2.mean), "{dim}D sidecar mean");
                    assert!(close(e2.max, sidecar.error2.max), "{dim}D sidecar max");
                }
            }
        }
    }
}

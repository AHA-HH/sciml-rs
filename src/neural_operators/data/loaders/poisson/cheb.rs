//! The Chebyshev reader: one split's raw fields on the CGL grid (n × n, or n × n × n for
//! the 3D splits), validated and untransferred (design §7, CONVENTIONS §12), with the
//! validation the uniform loader shares (`check_nodes`, `read_samples` in the parent
//! module).

use super::{AXES_2D, AXES_3D, PoissonField, check_nodes, read_samples};
use crate::neural_operators::chebyshev::nodes;
use crate::neural_operators::data::io::{errors::LoadError, readers::npz::NpzFileReader};
use ndarray::{Array3, ArrayD, Axis, Ix3, Slice};
use std::path::Path;

/// Reads the first `count` samples of the raw field `field` of one split on the n × n CGL
/// grid, untransferred. Returns `[count, n, n]` with `field[i, a, b] = field_i(x_a, y_b)`
/// (CONVENTIONS §12). With [`PoissonField::U`] it is the reference of the errors on the
/// Chebyshev grid (design §7).
///
/// The file is validated as
/// [`load_poisson_uniform`](super::load_poisson_uniform) validates it: `x`, `y` must be
/// the ascending CGL nodes `[n]` and `field` must be `[samples, n, n]`. The other field
/// and the sidecar are not read.
///
/// # Errors
/// [`LoadError::Reader`] if the file can't be read or lacks `field`/`x`/`y`;
/// [`LoadError::Invalid`] if `n < 3`, `field` isn't `[samples, n, n]`, `x` or `y` isn't
/// `[n]` or differs from the CGL nodes by more than 1e-14, or the file has fewer than
/// `count` samples. Messages name the file in place of the split.
pub fn load_poisson_cheb_field(
    path: impl AsRef<Path>,
    field: PoissonField,
    n: usize,
    count: usize,
) -> Result<Array3<f64>, LoadError> {
    Ok(read_cheb_field(path.as_ref(), field, n, count, &AXES_2D)?
        .into_dimensionality::<Ix3>()
        .expect("checked rank"))
}

/// [`load_poisson_cheb_field`] for the 3D splits (3D CL3.3): the first `count` samples of
/// `field`, `[count, n, n, n]` on the CGL grid, stored 'ijk', untransferred: the reference
/// of the 3D errors (`poisson_errors_nd`).
///
/// # Errors
/// As [`load_poisson_cheb_field`], with `z` among the node fields and `field` required to
/// be `[samples, n, n, n]`.
pub fn load_poisson3d_cheb_field(
    path: impl AsRef<Path>,
    field: PoissonField,
    n: usize,
    count: usize,
) -> Result<ArrayD<f64>, LoadError> {
    read_cheb_field(path.as_ref(), field, n, count, &AXES_3D)
}

/// The reader behind both: `field` validated with the node fields `axes`, its first
/// `count` samples, untransferred. Messages name the file in place of the split.
fn read_cheb_field(
    path: &Path,
    field: PoissonField,
    n: usize,
    count: usize,
    axes: &[&str],
) -> Result<ArrayD<f64>, LoadError> {
    if n < 3 {
        return Err(LoadError::Invalid(format!(
            "n must be >= 3 (the uniform grid s = n - 1 needs both endpoints), got {n}"
        )));
    }
    let label = path.file_name().map_or_else(
        || path.display().to_string(),
        |f| f.to_string_lossy().into(),
    );
    let mut reader = NpzFileReader::new(path)?;
    check_nodes(&mut reader, &label, n, &nodes(n), axes)?;
    let mut values = read_samples(&mut reader, &label, field, n, axes.len(), count)?;
    values.slice_axis_inplace(Axis(0), Slice::from(..count));
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::super::fixtures::{
        N, N_TEST, fresh_dir, poly_field, poly_field3, write_npz, write_valid_pair,
        write_valid_pair3,
    };
    use super::*;
    use crate::neural_operators::chebyshev::uniform_nodes;
    use crate::neural_operators::data::io::errors::ReaderError;

    #[test]
    fn cheb_loader_reads_raw_u() {
        let dir = fresh_dir("cheb_reads");
        let (train_path, test_path) = write_valid_pair(&dir);
        let x = nodes(N);

        let u = load_poisson_cheb_field(&test_path, PoissonField::U, N, N_TEST).unwrap();
        assert_eq!(u.shape(), [N_TEST, N, N]);
        assert_eq!(u.into_dyn(), poly_field(N_TEST, 0.5, &x, &x) * -2.0);

        // fewer than stored: the first `count` samples
        let u = load_poisson_cheb_field(&train_path, PoissonField::U, N, 2).unwrap();
        assert_eq!(u.shape(), [2, N, N]);
        assert_eq!(u.into_dyn(), poly_field(2, 0.0, &x, &x) * -2.0);
    }

    #[test]
    fn cheb_loader_missing_file_is_a_reader_error() {
        let err = load_poisson_cheb_field("/nope/test.npz", PoissonField::U, N, 1)
            .expect_err("loading should fail");
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
            load_poisson_cheb_field(&path, PoissonField::U, N, N_TEST),
            "bad.npz field 'x' must hold the 9 ascending CGL nodes",
        );
        expect_cheb_invalid(
            load_poisson_cheb_field(&path, PoissonField::U, 2, 1),
            "n must be >= 3",
        );
    }

    #[test]
    fn cheb_loader_too_few_samples_is_invalid() {
        let dir = fresh_dir("cheb_too_few");
        let (_, test_path) = write_valid_pair(&dir);
        expect_cheb_invalid(
            load_poisson_cheb_field(&test_path, PoissonField::U, N, N_TEST + 1),
            "requested 3 test.npz samples but 'u' has 2",
        );
    }

    #[test]
    fn cheb_field_loader_reads_f() {
        let dir = fresh_dir("cheb_field_f");
        let (train_path, test_path) = write_valid_pair(&dir);
        let x = nodes(N);

        let f = load_poisson_cheb_field(&test_path, PoissonField::F, N, N_TEST).unwrap();
        assert_eq!(f.shape(), [N_TEST, N, N]);
        assert_eq!(f.into_dyn(), poly_field(N_TEST, 0.5, &x, &x));

        // fewer than stored: the first `count` samples
        let f = load_poisson_cheb_field(&train_path, PoissonField::F, N, 2).unwrap();
        assert_eq!(f.shape(), [2, N, N]);
        assert_eq!(f.into_dyn(), poly_field(2, 0.0, &x, &x));
    }

    #[test]
    fn cheb_loader_3d_reads_raw_u() {
        let dir = fresh_dir("3d_cheb");
        let (_, test_path) = write_valid_pair3(&dir);
        let u = load_poisson3d_cheb_field(&test_path, PoissonField::U, N, 1).unwrap();
        let x = nodes(N);
        let written = poly_field3(N_TEST, 0.5, &x) * -2.0;
        assert_eq!(u, written.slice_axis(Axis(0), (..1).into()));
        let err = load_poisson3d_cheb_field(&test_path, PoissonField::U, N, N_TEST + 1)
            .expect_err("too few samples");
        assert!(matches!(err, LoadError::Invalid(_)), "{err:?}");
        assert!(
            err.to_string()
                .contains("requested 3 test.npz samples but 'u' has 2"),
            "{err}"
        );
    }
}

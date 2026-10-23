//! The Chebyshev reader: one split's raw fields on the n × n CGL grid, validated and
//! untransferred (design §7, CONVENTIONS §12). [`load_poisson_cheb_field`] is public; the
//! uniform loader reads through the same validation (`check_nodes`, `read_samples`).

use crate::neural_operators::chebyshev::nodes;
use crate::neural_operators::data::io::{errors::LoadError, readers::npz::NpzFileReader};
use ndarray::{Array1, Array3, Ix3, s};
use std::path::Path;

/// Tolerance on the stored `x`, `y` nodes against the closed-form CGL nodes.
const NODE_TOL: f64 = 1e-14;

/// A field of a Poisson split file: the forcing `f` or the solution `u`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoissonField {
    /// The forcing f.
    F,
    /// The solution u.
    U,
}

impl PoissonField {
    /// The field's `.npz` key (datasets/README.md): `"f"` or `"u"`.
    pub fn key(self) -> &'static str {
        match self {
            Self::F => "f",
            Self::U => "u",
        }
    }
}

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
pub(super) fn check_nodes(
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

/// Reads `field`, checks that it is `[samples, n, n]` with at least `count` samples, and
/// returns it whole. `split` only labels the error messages.
pub(super) fn read_samples(
    reader: &mut NpzFileReader,
    split: &str,
    field: PoissonField,
    n: usize,
    count: usize,
) -> Result<Array3<f64>, LoadError> {
    let name = field.key();
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

#[cfg(test)]
mod tests {
    use super::super::fixtures::{N, N_TEST, fresh_dir, poly_field, write_npz, write_valid_pair};
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
}

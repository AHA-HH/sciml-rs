//! .mat file reader (legacy and modern MATLAB formats).

use crate::neural_operators::data::io::{
    errors::{ReaderError, ReaderResult},
    traits::FieldReader,
};
use matfile;
use ndarray::{ArrayD, ArrayView, IxDyn, ShapeBuilder};
use std::path::Path;

/// Reads named fields from a `.mat` file.
///
/// Unlike `NpyFileReader`/`NpzFileReader`, the file is opened and parsed
/// eagerly in `new()`; failures are returned as `Err` there.
pub struct MatFileReader {
    mat: matfile::MatFile,
}

impl MatFileReader {
    /// Opens and parses `path`.
    pub fn new(path: &Path) -> ReaderResult<Self> {
        if !path.exists() {
            return Err(ReaderError::FileNotFound(path.display().to_string()));
        }
        let file = std::fs::File::open(path)
            .map_err(|e| ReaderError::ParseError(format!("{}: {e}", path.display())))?;
        let mat = matfile::MatFile::parse(file)
            .map_err(|e| ReaderError::ParseError(format!("{}: {e}", path.display())))?;
        Ok(Self { mat })
    }
}

/// Builds a row-major (standard layout) array from MATLAB's column-major data.
///
/// `size` is MATLAB's dimension list. The data is viewed in place as a
/// column-major array of that shape (no copy) and written into a C-order
/// array with `zip_mut_with`, which picks a cache-friendly traversal: one
/// allocation per field. (Iterating the view in logical order instead is ~7x
/// slower on a 1024x421x421 field.) C order matters because downstream
/// `into_shape_with_order` (row-major) reshapes fail with `IncompatibleLayout`
/// on a Fortran-order array (REVIEW.md 2.4).
fn to_array<T: Copy + Into<f64>>(size: &[usize], data: &[T]) -> ReaderResult<ArrayD<f64>> {
    let column_major = ArrayView::from_shape(IxDyn(size).f(), data)
        .map_err(|e| ReaderError::ParseError(e.to_string()))?;
    let mut out = ArrayD::<f64>::zeros(IxDyn(size));
    out.zip_mut_with(&column_major, |o, &x| *o = x.into());
    Ok(out)
}

impl FieldReader for MatFileReader {
    /// Reads field `name` as a row-major `f64` array with MATLAB's shape.
    ///
    /// Supports real `double`, `single` and 8/16/32-bit integer fields.
    /// Complex fields and 64-bit integers (not losslessly convertible to
    /// `f64`) are rejected with `ParseError`.
    fn read_field(&self, name: &str) -> ReaderResult<ArrayD<f64>> {
        let array = self
            .mat
            .find_by_name(name)
            .ok_or_else(|| ReaderError::FieldNotFound(format!("field '{}' not found", name)))?;
        numeric_to_array(name, array.size(), array.data())
    }
}

/// Converts one field's numeric payload; split from `read_field` so the dtype
/// dispatch can be tested without a `.mat` file.
fn numeric_to_array(
    name: &str,
    size: &[usize],
    data: &matfile::NumericData,
) -> ReaderResult<ArrayD<f64>> {
    use matfile::NumericData as D;

    match data {
        D::Double { imag: Some(_), .. }
        | D::Single { imag: Some(_), .. }
        | D::Int8 { imag: Some(_), .. }
        | D::UInt8 { imag: Some(_), .. }
        | D::Int16 { imag: Some(_), .. }
        | D::UInt16 { imag: Some(_), .. }
        | D::Int32 { imag: Some(_), .. }
        | D::UInt32 { imag: Some(_), .. } => Err(ReaderError::ParseError(format!(
            "field '{name}' is complex; only real fields are supported"
        ))),
        D::Double { real, .. } => to_array(size, real),
        D::Single { real, .. } => to_array(size, real),
        D::Int8 { real, .. } => to_array(size, real),
        D::UInt8 { real, .. } => to_array(size, real),
        D::Int16 { real, .. } => to_array(size, real),
        D::UInt16 { real, .. } => to_array(size, real),
        D::Int32 { real, .. } => to_array(size, real),
        D::UInt32 { real, .. } => to_array(size, real),
        // Int64/UInt64 don't convert to f64 losslessly.
        _ => Err(ReaderError::ParseError(format!(
            "unsupported dtype in field '{}'",
            name
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_missing_file() {
        match MatFileReader::new(Path::new("does_not_exist.mat")) {
            Err(ReaderError::FileNotFound(msg)) => assert!(
                msg.contains("does_not_exist.mat"),
                "message should name the path, got: {msg}"
            ),
            Err(e) => panic!("expected FileNotFound, got {e:?}"),
            Ok(_) => panic!("expected an error"),
        }
    }

    // --- REVIEW.md 2.4 / 5.5 ---

    use matfile::NumericData as D;
    use ndarray::IxDyn;

    #[test]
    fn to_array_follows_matlab_column_major_order() {
        // MATLAB [2, 3] matrix [[1, 3, 5], [2, 4, 6]] is stored column by column.
        let a = to_array(&[2, 3], &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(
            a.into_dimensionality::<ndarray::Ix2>().unwrap(),
            ndarray::array![[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
        );
    }

    #[test]
    fn to_array_3d_indexing_and_standard_layout() {
        // Element (i, j, k) of a MATLAB [2, 3, 4] array sits at i + 2j + 6k.
        let data: Vec<f64> = (0..24).map(f64::from).collect();
        let a = to_array(&[2, 3, 4], &data).unwrap();
        assert_eq!(a.shape(), &[2, 3, 4]);
        assert!(a.is_standard_layout());
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(a[[i, j, k]], data[i + 2 * j + 6 * k], "({i},{j},{k})");
                }
            }
        }
    }

    #[test]
    fn row_major_reshape_works_on_reader_output_but_not_on_fortran_order() {
        // The loaders' `into_shape_with_order` (row-major) reshape: it used to
        // fail on the reader's Fortran-order output whenever no slice or
        // subsample happened to copy it into C order first (REVIEW.md 2.4).
        let data: Vec<f64> = (0..24).map(f64::from).collect();
        let old = ArrayD::from_shape_vec(IxDyn(&[4, 3, 2]), data.clone())
            .unwrap()
            .reversed_axes();
        assert!(old.into_shape_with_order(IxDyn(&[2, 3, 4, 1])).is_err());

        let new = to_array(&[2, 3, 4], &data).unwrap();
        let expected: Vec<f64> = new.iter().copied().collect();
        let reshaped = new.into_shape_with_order(IxDyn(&[2, 3, 4, 1])).unwrap();
        assert_eq!(reshaped.iter().copied().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn integer_and_single_fields_convert_losslessly() {
        let size = [2, 2];
        let cases = [
            D::Single {
                real: vec![0.5f32, -1.5, 2.0, 3.0],
                imag: None,
            },
            D::Int16 {
                real: vec![1i16, -2, 3, i16::MIN],
                imag: None,
            },
            D::UInt16 {
                real: vec![1u16, 2, 3, u16::MAX],
                imag: None,
            },
            D::UInt32 {
                real: vec![1u32, 2, 3, u32::MAX],
                imag: None,
            },
        ];
        let expected = [
            [0.5, -1.5, 2.0, 3.0],
            [1.0, -2.0, 3.0, i16::MIN as f64],
            [1.0, 2.0, 3.0, u16::MAX as f64],
            [1.0, 2.0, 3.0, u32::MAX as f64],
        ];
        for (data, want) in cases.iter().zip(expected) {
            let a = numeric_to_array("f", &size, data).unwrap();
            // column-major [a, b, c, d] -> [[a, c], [b, d]]
            assert_eq!(a[[0, 0]], want[0]);
            assert_eq!(a[[1, 0]], want[1]);
            assert_eq!(a[[0, 1]], want[2]);
            assert_eq!(a[[1, 1]], want[3]);
        }
    }

    #[test]
    fn complex_field_is_rejected_not_truncated() {
        let data = D::Double {
            real: vec![1.0, 2.0],
            imag: Some(vec![0.5, -0.5]),
        };
        match numeric_to_array("z", &[1, 2], &data) {
            Err(ReaderError::ParseError(msg)) => assert!(msg.contains("'z' is complex"), "{msg}"),
            other => panic!("expected ParseError, got {other:?}"),
        }
    }

    #[test]
    fn int64_field_is_rejected() {
        let data = D::Int64 {
            real: vec![1, 2],
            imag: None,
        };
        assert!(matches!(
            numeric_to_array("big", &[1, 2], &data),
            Err(ReaderError::ParseError(_))
        ));
    }
}

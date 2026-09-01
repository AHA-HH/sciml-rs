//! .mat file reader (legacy and modern MATLAB formats).

use crate::neural_operators::data::io::{
    errors::{ReaderError, ReaderResult},
    traits::FieldReader,
};
use matfile;
use ndarray::{Array, ArrayD, IxDyn};
use std::path::Path;

/// Reads named fields from a `.mat` file.
///
/// Note: unlike `NpyFileReader`/`NpzFileReader`, this reader opens and parses
/// eagerly in `new()` and panics on failure rather than returning `Err` — not
/// yet made consistent with the other `FieldReader` implementors.
pub struct MatFileReader {
    mat: matfile::MatFile,
}

impl MatFileReader {
    /// Opens and parses `path`. Panics if the file can't be opened or parsed.
    pub fn new(path: &Path) -> Self {
        let file = std::fs::File::open(path).expect("failed to open .mat file");
        let mat = matfile::MatFile::parse(file).expect("failed to parse .mat file");
        Self { mat }
    }
}

impl FieldReader for MatFileReader {
    /// Reads field `name`. MATLAB stores arrays column-major (Fortran order);
    /// `ndarray::Array::from_shape_vec` assumes row-major, so dims are reversed
    /// before construction and `.reversed_axes()` restores the correct shape —
    /// applied identically across every supported dtype below.
    fn read_field(&self, name: &str) -> ReaderResult<ArrayD<f64>> {
        let array = self
            .mat
            .find_by_name(name)
            .ok_or_else(|| ReaderError::FieldNotFound(format!("field '{}' not found", name)))?;

        let dims: Vec<usize> = array.size().iter().rev().cloned().collect();
        let shape_reversed = IxDyn(&dims);

        match array.data() {
            matfile::NumericData::Double { real, .. } => {
                Array::from_shape_vec(shape_reversed, real.to_vec())
                    .map_err(|e| ReaderError::ParseError(e.to_string()))
                    .map(|a| a.reversed_axes())
            }
            matfile::NumericData::Single { real, .. } => {
                Array::from_shape_vec(shape_reversed, real.iter().map(|&x| x as f64).collect())
                    .map_err(|e| ReaderError::ParseError(e.to_string()))
                    .map(|a| a.reversed_axes())
            }
            matfile::NumericData::Int8 { real, .. } => {
                Array::from_shape_vec(shape_reversed, real.iter().map(|&x| x as f64).collect())
                    .map_err(|e| ReaderError::ParseError(e.to_string()))
                    .map(|a| a.reversed_axes())
            }
            matfile::NumericData::Int32 { real, .. } => {
                Array::from_shape_vec(shape_reversed, real.iter().map(|&x| x as f64).collect())
                    .map_err(|e| ReaderError::ParseError(e.to_string()))
                    .map(|a| a.reversed_axes())
            }
            matfile::NumericData::UInt8 { real, .. } => {
                Array::from_shape_vec(shape_reversed, real.iter().map(|&x| x as f64).collect())
                    .map_err(|e| ReaderError::ParseError(e.to_string()))
                    .map(|a| a.reversed_axes())
            }
            // Int16/UInt16/UInt32/Complex/Char not handled — extend here if needed
            _ => Err(ReaderError::ParseError(format!(
                "unsupported dtype in field '{}'",
                name
            ))),
        }
    }
}

//! Common interface implemented by each format-specific file reader
//! (`NpyFileReader`, `NpzFileReader`, `MatFileReader`), so the data
//! pipeline can load a named field without depending on source format.

use crate::neural_operators::data::io::errors::ReaderResult;
use ndarray::ArrayD;

/// Reads a named field out of a data file, regardless of underlying format.
pub trait FieldReader {
    /// Returns the array stored under `name`, or an error if the file
    /// couldn't be read or the field doesn't exist.
    fn read_field(&self, name: &str) -> ReaderResult<ArrayD<f64>>;
}
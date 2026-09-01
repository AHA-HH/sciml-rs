//! .npy file reader - single array per file, so `read_field`'s `name` is unused.

use crate::neural_operators::data::io::{
    errors::{ReaderError, ReaderResult},
    traits::FieldReader,
};
use ndarray::ArrayD;
use ndarray_npy::read_npy;
use std::path::{Path, PathBuf};

/// Reads a `.npy` file's single array. Path existence and parsing are
/// checked lazily in `read_field`, not at construction.
pub struct NpyFileReader {
    path: PathBuf,
}

impl NpyFileReader {
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
        }
    }
}

impl FieldReader for NpyFileReader {
    /// `_name` is ignored - a `.npy` file holds exactly one array.
    /// Tries f64 first; on failure, falls back to f32 and casts up.
    fn read_field(&self, _name: &str) -> ReaderResult<ArrayD<f64>> {
        if !self.path.exists() {
            return Err(ReaderError::FileNotFound(format!(
                "{}",
                self.path.display()
            )));
        }

        let array: ArrayD<f64> = match read_npy(&self.path) {
            Ok(array) => array,
            Err(_) => {
                let array_f32: ArrayD<f32> = read_npy(&self.path).map_err(|e| {
                    ReaderError::ParseError(format!("{}: {}", self.path.display(), e))
                })?;
                array_f32.mapv(|x| x as f64)
            }
        };

        Ok(array)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_missing_file() {
        let reader = NpyFileReader::new(Path::new("does_not_exist.npy"));
        match reader.read_field("unused") {
            Err(ReaderError::FileNotFound(msg)) => assert!(
                msg.contains("does_not_exist.npy"),
                "message should name the path, got: {msg}"
            ),
            other => panic!("expected FileNotFound, got {other:?}"),
        }
    }
}

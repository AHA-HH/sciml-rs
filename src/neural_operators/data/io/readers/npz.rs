//! .npz file reader - multiple named arrays per file, `name` selects which one.

use crate::neural_operators::data::io::{
    errors::{ReaderError, ReaderResult},
    traits::FieldReader,
};
use ndarray::ArrayD;
use ndarray_npy::NpzReader;
use std::fs::File;
use std::path::{Path, PathBuf};

/// Reads a named array out of a `.npz` archive. Path existence and parsing
/// are checked lazily in `read_field`, not at construction.
pub struct NpzFileReader {
    path: PathBuf,
}

impl NpzFileReader {
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
        }
    }
}

impl FieldReader for NpzFileReader {
    /// Tries f64 first; on failure, falls back to f32 and casts up. A
    /// second failure means the field genuinely isn't present.
    fn read_field(&self, name: &str) -> ReaderResult<ArrayD<f64>> {
        if !self.path.exists() {
            return Err(ReaderError::FileNotFound(format!(
                "{}",
                self.path.display()
            )));
        }

        let mut npz = NpzReader::new(
            File::open(&self.path)
                .map_err(|e| ReaderError::ParseError(format!("{}: {}", self.path.display(), e)))?,
        )
        .map_err(|e| ReaderError::ParseError(format!("{}: {}", self.path.display(), e)))?;

        let array: ArrayD<f64> = match npz.by_name(name) {
            Ok(array) => array,
            Err(_) => {
                let array_f32: ArrayD<f32> = npz.by_name(name).map_err(|_| {
                    ReaderError::FieldNotFound(format!(
                        "field '{}' not found in {}",
                        name,
                        self.path.display()
                    ))
                })?;
                array_f32.mapv(|x| x as f64)
            }
        };

        Ok(array)
    }
}

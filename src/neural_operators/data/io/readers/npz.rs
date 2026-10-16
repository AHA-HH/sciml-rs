//! .npz file reader - multiple named arrays per file, `name` selects which one.

use crate::neural_operators::data::io::errors::{ReaderError, ReaderResult};
use ndarray::{ArrayD, IxDyn, OwnedRepr};
use ndarray_npy::{NpzReader, ReadNpyError, ReadNpzError};
use std::fs::File;
use std::path::{Path, PathBuf};

/// Reads named arrays out of a `.npz` archive.
///
/// The archive is opened and its index parsed once, eagerly in `new()`;
/// every [`read_field`](Self::read_field) reuses it.
pub struct NpzFileReader {
    path: PathBuf,
    npz: NpzReader<File>,
    /// Array names in the archive, `.npy` suffix stripped.
    names: Vec<String>,
}

impl NpzFileReader {
    /// Opens `path` and reads the archive's index.
    ///
    /// # Errors
    /// [`ReaderError::FileNotFound`] if `path` doesn't exist;
    /// [`ReaderError::ParseError`] if it can't be opened or isn't a zip archive.
    pub fn new(path: &Path) -> ReaderResult<Self> {
        if !path.exists() {
            return Err(ReaderError::FileNotFound(path.display().to_string()));
        }
        let parse =
            |e: &dyn std::fmt::Display| ReaderError::ParseError(format!("{}: {e}", path.display()));
        let file = File::open(path).map_err(|e| parse(&e))?;
        let mut npz = NpzReader::new(file).map_err(|e| parse(&e))?;
        let names = npz.names().map_err(|e| parse(&e))?;
        Ok(Self {
            path: path.to_path_buf(),
            npz,
            names,
        })
    }

    /// Reads the array `name` as `f64`, of whatever shape it was stored with.
    ///
    /// `f64` arrays are read as is; `f32` arrays are cast up.
    ///
    /// # Errors
    /// [`ReaderError::FieldNotFound`] if the archive has no array `name`;
    /// [`ReaderError::ParseError`] if it has one of another dtype, or one
    /// that can't be parsed.
    pub fn read_field(&mut self, name: &str) -> ReaderResult<ArrayD<f64>> {
        if !self.names.iter().any(|n| n == name) {
            return Err(ReaderError::FieldNotFound(format!(
                "field '{name}' not found in {}",
                self.path.display()
            )));
        }
        let parse = |e: ReadNpzError| {
            ReaderError::ParseError(format!("field '{name}' in {}: {e}", self.path.display()))
        };
        match self.npz.by_name::<_, IxDyn>(name) {
            Ok(array) => Ok(array),
            Err(ReadNpzError::Npy(ReadNpyError::WrongDescriptor(_))) => self
                .npz
                .by_name::<OwnedRepr<f32>, IxDyn>(name)
                .map(|a| a.mapv(f64::from))
                .map_err(parse),
            Err(e) => Err(parse(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array3};
    use ndarray_npy::NpzWriter;

    #[test]
    fn reports_missing_file() {
        match NpzFileReader::new(Path::new("does_not_exist.npz")) {
            Err(ReaderError::FileNotFound(msg)) => assert!(
                msg.contains("does_not_exist.npz"),
                "message should name the path, got: {msg}"
            ),
            Err(e) => panic!("expected FileNotFound, got {e:?}"),
            Ok(_) => panic!("expected an error"),
        }
    }

    /// A fresh temp path for `test`, unique per process.
    fn temp_npz(test: &str) -> PathBuf {
        std::env::temp_dir().join(format!("sciml_rs_npz_{test}_{}.npz", std::process::id()))
    }

    #[test]
    fn npz_wrong_dtype_is_parse_error() {
        let path = temp_npz("wrong_dtype");
        let mut w = NpzWriter::new(File::create(&path).unwrap());
        w.add_array("a", &Array1::from(vec![1i32, 2, 3])).unwrap();
        w.finish().unwrap();

        let mut reader = NpzFileReader::new(&path).unwrap();
        let wrong = reader.read_field("a");
        let missing = reader.read_field("b");
        let _ = std::fs::remove_file(&path);

        match wrong {
            Err(ReaderError::ParseError(msg)) => assert!(msg.contains("'a'"), "{msg}"),
            other => panic!("expected ParseError for an i32 field, got {other:?}"),
        }
        assert!(
            matches!(missing, Err(ReaderError::FieldNotFound(_))),
            "expected FieldNotFound for a missing field, got {missing:?}"
        );
    }

    #[test]
    fn npz_reads_several_fields_from_one_open() {
        let path = temp_npz("several_fields");
        let x = Array1::from(vec![-1.0f64, 0.0, 1.0]);
        let y = Array1::from(vec![0.25f32, -0.5, 2.0]);
        let u = Array3::from_shape_fn((2, 3, 3), |(i, a, b)| (i * 9 + a * 3 + b) as f64 / 7.0);
        let mut w = NpzWriter::new(File::create(&path).unwrap());
        w.add_array("x", &x).unwrap();
        w.add_array("y", &y).unwrap();
        w.add_array("u", &u).unwrap();
        w.finish().unwrap();

        let mut one_open = NpzFileReader::new(&path).unwrap();
        let read: Vec<ArrayD<f64>> = ["x", "u", "y", "x"]
            .iter()
            .map(|name| one_open.read_field(name).unwrap())
            .collect();
        let per_field: Vec<ArrayD<f64>> = ["x", "u", "y", "x"]
            .iter()
            .map(|name| NpzFileReader::new(&path).unwrap().read_field(name).unwrap())
            .collect();
        let _ = std::fs::remove_file(&path);

        assert_eq!(read, per_field);
        assert_eq!(read[0], x.into_dyn());
        assert_eq!(read[1], u.into_dyn());
        assert_eq!(read[2], y.mapv(f64::from).into_dyn());
        assert_eq!(read[1].shape(), &[2, 3, 3]);
    }
}

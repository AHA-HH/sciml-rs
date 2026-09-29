//! Error types for the `.npy`/`.npz`/`.mat` file readers and the dataset loaders.
//!
//! Concrete readers (see FieldReader implementors) return `ReaderResult<T>`,
//! surfacing missing files, malformed data, or missing named fields
//! uniformly regardless of source format.

/// Errors that can occur while reading a data file into an array.
#[derive(Debug)]
pub enum ReaderError {
    /// The requested file path does not exist or could not be opened.
    FileNotFound(String),
    /// The file was found but its contents could not be parsed as expected.
    ParseError(String),
    /// The file was parsed, but the requested named field is not present.
    FieldNotFound(String),
}

impl std::fmt::Display for ReaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReaderError::FileNotFound(p) => write!(f, "File not found: {}", p),
            ReaderError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            ReaderError::FieldNotFound(n) => write!(f, "Field not found: {}", n),
        }
    }
}

impl std::error::Error for ReaderError {}

/// Result alias used by all file readers in this module.
pub type ReaderResult<T> = Result<T, ReaderError>;

/// Errors returned by the dataset loaders (`load_burgers_uniform`,
/// `load_darcy_uniform`).
#[derive(Debug)]
pub enum LoadError {
    /// A source file could not be opened, parsed, or lacks a required field.
    Reader(ReaderError),
    /// The file was read, but its contents don't fit the requested config
    /// (e.g. too few samples, wrong rank, or a resolution mismatch).
    Invalid(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Reader(e) => write!(f, "{e}"),
            LoadError::Invalid(msg) => write!(f, "Invalid dataset: {msg}"),
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoadError::Reader(e) => Some(e),
            LoadError::Invalid(_) => None,
        }
    }
}

impl From<ReaderError> for LoadError {
    fn from(e: ReaderError) -> Self {
        LoadError::Reader(e)
    }
}

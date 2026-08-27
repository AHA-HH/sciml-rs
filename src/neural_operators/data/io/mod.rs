//! Dataset I/O
//!
//! Format-agnostic field readers (`.npy`, `.npz`, `.mat`) behind a shared
//! trait, plus their error types.

pub mod errors;
pub mod readers;
pub mod traits;

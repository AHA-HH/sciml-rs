//! Utilities
//!
//! Shared numerical helpers, including inverse complex FFT routines built on
//! Burn's signal ops, facts about the build (the backends compiled in, the git
//! commit), and a little-endian reader for the crate's binary formats.

pub mod backend;
pub(crate) mod bytes;
pub mod fft;

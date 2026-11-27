//! Library CM1.5a in f64: 2.0.0's `FNO` at its defaults (`fno1d_f64`, `fno2d_f64`) against
//! the crate's `FNO`, within D3's f64 bound (relative max error ≤ 1e-12). Both are
//! f32-accurate (Flex's f64 `rfft` uses f32 twiddles, and 2.0.0 builds its grid in f32), so
//! they are `#[ignore]`d with the error measured (Flex, M2 Pro), D3 as amended 2026-11-25.
//! Its own test binary: it configures Flex's process-wide float dtype to f64.

#[cfg(feature = "flex")]
mod flex_f64;
#[cfg(feature = "flex")]
#[path = "parity_common/fno.rs"]
mod fno;
#[cfg(feature = "flex")]
#[path = "parity_common/fno_map.rs"]
mod fno_map;
mod parity_common;

#[cfg(feature = "flex")]
use flex_f64::flex_f64;
#[cfg(feature = "flex")]
use fno::*;
#[cfg(feature = "flex")]
use parity_common::Precision::F64;

#[cfg(feature = "flex")]
#[test]
#[ignore = "f32-accurate (Flex's f32 twiddles, 2.0.0's f32 grid): 1.211e-8 on the M2 Pro"]
fn fno1d_f64_matches_2_0_0() {
    check::<3>(&FNO1D, F64, &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
#[ignore = "f32-accurate (Flex's f32 twiddles, 2.0.0's f32 grid): 3.786e-9 on the M2 Pro"]
fn fno2d_f64_matches_2_0_0() {
    check::<4>(&FNO2D, F64, &flex_f64());
}

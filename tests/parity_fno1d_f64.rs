//! Library CM0.5 in f64: CM0.2's `fno1d_f64` fixture through the parity harness. Its own
//! test binary because it configures Flex's process-wide float dtype to f64.

#[cfg(feature = "flex")]
mod flex_f64;
#[cfg(feature = "flex")]
#[path = "parity_common/fno1d.rs"]
mod fno1d;
mod parity_common;

#[cfg(feature = "flex")]
use flex_f64::flex_f64;
#[cfg(feature = "flex")]
use parity_common::assert_parity;

/// The f64 fixture within D3's f64 bound. Flex's f64 `rfft` multiplies by f32 twiddles
/// (`tests/gauss_newton_f64.rs`), so this cannot reach 1e-12 until that is fixed upstream.
#[cfg(feature = "flex")]
#[test]
#[ignore = "Burn bug: flex's f64 rfft uses f32 twiddles (tests/gauss_newton_f64.rs)"]
fn fno1d_f64_matches_2_0_0() {
    let (out, reference, d3) = fno1d::run("fno1d_f64", &flex_f64());
    let err = assert_parity(out, &reference, d3, "fno1d_f64 output");
    println!("fno1d_f64: relative max error {err:.3e}");
}

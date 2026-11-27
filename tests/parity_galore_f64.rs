//! Library CM3.6a in f64: `parity_galore_common` on the f64 fixture (D3: 1e-12). Its own test
//! binary because it configures Flex's process-wide float dtype to f64. No FFT is involved,
//! so the tests run.

#[cfg(feature = "flex")]
mod flex_f64;
// Each test binary uses part of the shared harness.
#[cfg(feature = "flex")]
#[allow(dead_code)]
mod parity_common;
#[cfg(feature = "flex")]
#[allow(dead_code)]
mod parity_galore_common;

#[cfg(feature = "flex")]
use flex_f64::flex_f64;
#[cfg(feature = "flex")]
use parity_galore_common::run;

#[cfg(feature = "flex")]
#[test]
fn matrix_f64_matches_2_0_0() {
    run::<2>("galore_f64", "matrix", &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
fn tensor4_f64_matches_2_0_0() {
    run::<4>("galore_f64", "tensor4", &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
fn fraction_f64_matches_2_0_0() {
    run::<3>("galore_f64", "fraction", &flex_f64());
}

//! Library CM1.4 in f64: 2.0.0's embeddings and the `FNO` lifting and projection, f64
//! fixtures within D3's f64 bound (relative max error ≤ 1e-12). No FFT runs, but 2.0.0
//! builds its grid and frequencies in f32 (torch's default dtype) whatever the input's, so
//! every fixture with one is f32-accurate: those tests are `#[ignore]`d with the error
//! measured (Flex, M2 Pro). The FNO without a grid passes. Its own test binary: it
//! configures Flex's process-wide float dtype to f64.

#[cfg(feature = "flex")]
#[path = "parity_common/embeddings.rs"]
mod embeddings;
#[cfg(feature = "flex")]
mod flex_f64;
#[cfg(feature = "flex")]
#[path = "parity_common/fno_map.rs"]
mod fno_map;
mod parity_common;

#[cfg(feature = "flex")]
use embeddings::*;
#[cfg(feature = "flex")]
use flex_f64::flex_f64;
#[cfg(feature = "flex")]
use parity_common::Precision::F64;

#[cfg(feature = "flex")]
#[test]
#[ignore = "2.0.0 builds its grid in f32: 1.213e-8 on the M2 Pro"]
fn fno_lift_project1d_f64_matches_2_0_0() {
    fno_lift_project1d(F64, &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
#[ignore = "2.0.0 builds its grid in f32: 5.130e-9 on the M2 Pro"]
fn fno_lift_project2d_linear_f64_matches_2_0_0() {
    fno_lift_project2d_linear(F64, &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
fn fno_lift_project3d_no_grid_f64_matches_2_0_0() {
    fno_lift_project3d_no_grid(F64, &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
#[ignore = "2.0.0 builds its grid in f32: 4.892e-9 on the M2 Pro"]
fn fno_lift_project2d_boundaries_f64_matches_2_0_0() {
    fno_lift_project2d_boundaries(F64, &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
#[ignore = "2.0.0 builds its grid in f32: 2.386e-8 on the M2 Pro"]
fn grid_embedding2d_f64_matches_2_0_0() {
    grid_embedding2d(F64, &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
#[ignore = "2.0.0 builds its frequencies in f32: 5.018e-10 on the M2 Pro"]
fn sinusoidal_transformer_f64_matches_2_0_0() {
    sinusoidal_transformer(F64, &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
#[ignore = "2.0.0 builds its frequencies (2^k π) in f32: 1.561e-6 on the M2 Pro"]
fn sinusoidal_nerf_f64_matches_2_0_0() {
    sinusoidal_nerf(F64, &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
#[ignore = "2.0.0 builds its inverse frequencies in f32: 1.810e-7 on the M2 Pro"]
fn rotary2d_f64_matches_2_0_0() {
    rotary2d(F64, &flex_f64());
}

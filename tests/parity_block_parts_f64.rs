//! Library CM1.3a in f64: the block parts' f64 fixtures within D3's f64 bound (relative max
//! error ≤ 1e-12). No part runs an FFT, so Flex's f32 `rfft` twiddles do not enter. Its own
//! test binary because it configures Flex's process-wide float dtype to f64.

#[cfg(feature = "flex")]
#[path = "parity_common/block_parts.rs"]
mod block_parts;
#[cfg(feature = "flex")]
mod flex_f64;
mod parity_common;

#[cfg(feature = "flex")]
use block_parts::*;
#[cfg(feature = "flex")]
use flex_f64::flex_f64;
#[cfg(feature = "flex")]
use parity_common::Precision::F64;

#[cfg(feature = "flex")]
#[test]
fn channel_mlp1d_f64_matches_2_0_0() {
    channel_mlp1d(F64, &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
fn channel_mlp3d_f64_matches_2_0_0() {
    channel_mlp3d(F64, &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
fn linear_skip3d_f64_matches_2_0_0() {
    linear_skip3d(F64, &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
fn soft_gating2d_f64_matches_2_0_0() {
    soft_gating2d(F64, &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
fn instance_norm1d_f64_matches_2_0_0() {
    instance_norm1d(F64, &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
fn group_norm3d_f64_matches_2_0_0() {
    group_norm3d(F64, &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
fn ada_in2d_f64_matches_2_0_0() {
    ada_in2d(F64, &flex_f64());
}

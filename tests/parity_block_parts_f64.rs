//! Library CM1.3a in f64: the block parts' f64 fixtures within D3's f64 bound (relative max
//! error ≤ 1e-12). No part runs an FFT, so Flex's f32 `rfft` twiddles do not enter. Its own
//! test binary because it configures Flex's process-wide float dtype to f64.

#[cfg(feature = "flex")]
#[path = "parity_common/block_parts.rs"]
mod block_parts;
mod parity_common;

#[cfg(feature = "flex")]
use std::sync::OnceLock;

#[cfg(feature = "flex")]
use block_parts::{Part, check};
#[cfg(feature = "flex")]
use burn::{prelude::*, tensor::FloatDType};
#[cfg(feature = "flex")]
use parity_common::Precision::F64;

/// Flex configured for f64, once per process.
#[cfg(feature = "flex")]
fn flex_f64() -> Device {
    static DEVICE: OnceLock<Device> = OnceLock::new();
    DEVICE
        .get_or_init(|| {
            let mut device = Device::flex();
            device
                .configure(FloatDType::F64)
                .expect("flex accepts f64 as its default float dtype");
            device
        })
        .clone()
}

#[cfg(feature = "flex")]
fn f64(part: Part) {
    check(part, F64, &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
fn channel_mlp1d_f64_matches_2_0_0() {
    f64(Part::ChannelMlp1d);
}

#[cfg(feature = "flex")]
#[test]
fn channel_mlp3d_f64_matches_2_0_0() {
    f64(Part::ChannelMlp3d);
}

#[cfg(feature = "flex")]
#[test]
fn linear_skip3d_f64_matches_2_0_0() {
    f64(Part::LinearSkip3d);
}

#[cfg(feature = "flex")]
#[test]
fn soft_gating2d_f64_matches_2_0_0() {
    f64(Part::SoftGating2d);
}

#[cfg(feature = "flex")]
#[test]
fn instance_norm1d_f64_matches_2_0_0() {
    f64(Part::InstanceNorm1d);
}

#[cfg(feature = "flex")]
#[test]
fn group_norm3d_f64_matches_2_0_0() {
    f64(Part::GroupNorm3d);
}

#[cfg(feature = "flex")]
#[test]
fn ada_in2d_f64_matches_2_0_0() {
    f64(Part::AdaIn2d);
}

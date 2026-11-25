//! Library CM1.3a: the block parts of 2.0.0's `FNOBlocks` against their fixtures, in f32
//! (D3: relative max error ≤ 1e-5). The f64 fixtures are `tests/parity_block_parts_f64.rs`.

#[path = "parity_common/block_parts.rs"]
mod block_parts;
mod parity_common;

use block_parts::*;
use burn::prelude::*;
use parity_common::Precision::F32;

#[test]
fn channel_mlp1d_f32_matches_2_0_0() {
    channel_mlp1d(F32, &Device::default());
}

#[test]
fn channel_mlp3d_f32_matches_2_0_0() {
    channel_mlp3d(F32, &Device::default());
}

#[test]
fn linear_skip3d_f32_matches_2_0_0() {
    linear_skip3d(F32, &Device::default());
}

#[test]
fn soft_gating2d_f32_matches_2_0_0() {
    soft_gating2d(F32, &Device::default());
}

#[test]
fn instance_norm1d_f32_matches_2_0_0() {
    instance_norm1d(F32, &Device::default());
}

#[test]
fn group_norm3d_f32_matches_2_0_0() {
    group_norm3d(F32, &Device::default());
}

#[test]
fn ada_in2d_f32_matches_2_0_0() {
    ada_in2d(F32, &Device::default());
}

//! Library CM1.3a: the block parts of 2.0.0's `FNOBlocks` against their fixtures, in f32
//! (D3: relative max error ≤ 1e-5). The f64 fixtures are `tests/parity_block_parts_f64.rs`.

#[path = "parity_common/block_parts.rs"]
mod block_parts;
mod parity_common;

use block_parts::{Part, check};
use burn::prelude::*;
use parity_common::Precision::F32;

fn f32(part: Part) {
    check(part, F32, &Device::default());
}

#[test]
fn channel_mlp1d_f32_matches_2_0_0() {
    f32(Part::ChannelMlp1d);
}

#[test]
fn channel_mlp3d_f32_matches_2_0_0() {
    f32(Part::ChannelMlp3d);
}

#[test]
fn linear_skip3d_f32_matches_2_0_0() {
    f32(Part::LinearSkip3d);
}

#[test]
fn soft_gating2d_f32_matches_2_0_0() {
    f32(Part::SoftGating2d);
}

#[test]
fn instance_norm1d_f32_matches_2_0_0() {
    f32(Part::InstanceNorm1d);
}

#[test]
fn group_norm3d_f32_matches_2_0_0() {
    f32(Part::GroupNorm3d);
}

#[test]
fn ada_in2d_f32_matches_2_0_0() {
    f32(Part::AdaIn2d);
}

//! Library CM1.3b: 2.0.0's `FNOBlocks` (legacy convolution) against the FNO's Fourier layers
//! built from the block options, post- and pre-activation, for each norm, in f32 (D3 as
//! amended 2026-11-25). No f64 fixtures: the legacy layer computes in `cfloat`, so an f64
//! reference would be f32-accurate (`tools/parity/generate.py`, `F32_ONLY`).

#[path = "parity_common/fno_blocks.rs"]
mod fno_blocks;
mod parity_common;

use burn::prelude::*;
use fno_blocks::*;

#[test]
fn fno_blocks_post1d_matches_2_0_0() {
    check::<3>(&POST_1D, &Device::default());
}

#[test]
fn fno_blocks_pre1d_matches_2_0_0() {
    check::<3>(&PRE_1D, &Device::default());
}

#[test]
fn fno_blocks_instance_post2d_matches_2_0_0() {
    check::<4>(&INSTANCE_POST_2D, &Device::default());
}

#[test]
fn fno_blocks_instance_pre2d_matches_2_0_0() {
    check::<4>(&INSTANCE_PRE_2D, &Device::default());
}

#[test]
fn fno_blocks_group_post3d_matches_2_0_0() {
    check::<5>(&GROUP_POST_3D, &Device::default());
}

#[test]
fn fno_blocks_group_pre3d_matches_2_0_0() {
    check::<5>(&GROUP_PRE_3D, &Device::default());
}

#[test]
fn fno_blocks_ada_in_post2d_matches_2_0_0() {
    check::<4>(&ADA_IN_POST_2D, &Device::default());
}

#[test]
fn fno_blocks_ada_in_pre2d_matches_2_0_0() {
    check::<4>(&ADA_IN_PRE_2D, &Device::default());
}

/// 2.0.0's repeated activation without a channel MLP or norm.
#[test]
fn fno_blocks_plain1d_matches_2_0_0() {
    check::<3>(&PLAIN_1D, &Device::default());
}

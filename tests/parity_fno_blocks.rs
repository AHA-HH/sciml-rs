//! Library CM1.3b: 2.0.0's `FNOBlocks` (legacy convolution) against the FNO's Fourier layers
//! built from the block options, post- and pre-activation, for each norm, in f32 (D3 as
//! amended 2026-11-25). No f64 fixtures: the legacy layer computes in `cfloat`, so an f64
//! reference would be f32-accurate (`tools/parity/generate.py`, `F32_ONLY`).

#[path = "parity_common/fno_blocks.rs"]
mod fno_blocks;
mod parity_common;

use burn::prelude::*;
use fno_blocks::*;
use ndarray::{Axis, Slice};
use parity_common::{Fixture, assert_parity};
use sciml_rs::neural_operators::models::fno::Activation;

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

/// 2.0.0's repeated activation (tanh) without a channel MLP or norm.
#[test]
fn fno_blocks_plain1d_matches_2_0_0() {
    check::<3>(&PLAIN_1D, &Device::default());
}

/// Every `FNOConfig::activation` against torch's function (`activations1d`: each applied to
/// one seeded input, stacked along the channels in the enum's order).
#[test]
fn activations_match_torch() {
    use Activation::*;
    let mut fixture = Fixture::open("activations1d_f32");
    let x = fixture.input::<3>(&Device::default());
    let expected = fixture.output();
    let all = [
        Relu, Gelu, Tanh, Sigmoid, Silu, LeakyRelu, Selu, Mish, HardSwish, Softsign,
    ];
    assert_eq!(expected.shape()[1], all.len(), "one channel per activation");
    for (k, activation) in all.iter().enumerate() {
        let channel = expected
            .slice_axis(Axis(1), Slice::from(k..k + 1))
            .to_owned();
        let what = format!("activations1d_f32, {activation:?}");
        let err = assert_parity(
            activation.apply(x.clone()),
            &channel,
            fixture.precision.d3(),
            &what,
        );
        println!("{what}: relative max error {err:.3e}");
    }
}

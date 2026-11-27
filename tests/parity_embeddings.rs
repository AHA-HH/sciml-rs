//! Library CM1.4: 2.0.0's embeddings and the `FNO` lifting and projection, f32 fixtures
//! within D3 (relative max error ≤ 1e-5).

#[path = "parity_common/embeddings.rs"]
mod embeddings;
#[path = "parity_common/fno_map.rs"]
mod fno_map;
mod parity_common;

use burn::prelude::*;
use embeddings::*;
use parity_common::Precision::F32;

#[test]
fn fno_lift_project1d_matches_2_0_0() {
    fno_lift_project1d(F32, &Device::default());
}

#[test]
fn fno_lift_project2d_linear_matches_2_0_0() {
    fno_lift_project2d_linear(F32, &Device::default());
}

#[test]
fn fno_lift_project3d_no_grid_matches_2_0_0() {
    fno_lift_project3d_no_grid(F32, &Device::default());
}

#[test]
fn fno_lift_project2d_boundaries_matches_2_0_0() {
    fno_lift_project2d_boundaries(F32, &Device::default());
}

#[test]
fn grid_embedding2d_matches_2_0_0() {
    grid_embedding2d(F32, &Device::default());
}

#[test]
fn sinusoidal_transformer_matches_2_0_0() {
    sinusoidal_transformer(F32, &Device::default());
}

#[test]
fn sinusoidal_nerf_matches_2_0_0() {
    sinusoidal_nerf(F32, &Device::default());
}

#[test]
fn rotary2d_matches_2_0_0() {
    rotary2d(F32, &Device::default());
}

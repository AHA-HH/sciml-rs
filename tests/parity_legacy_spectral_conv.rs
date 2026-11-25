//! Library CM1.1: 2.0.0's legacy corner-layout `SpectralConv` against the crate's
//! `SpectralConv<R>`, 1D to 3D, in f32 (D3 as amended 2026-11-25). The f64 fixtures are
//! `tests/parity_legacy_spectral_conv_f64.rs`.

#[macro_use] // `corner!`
#[path = "parity_common/legacy_conv.rs"]
mod legacy_conv;
mod parity_common;

use burn::prelude::*;
use legacy_conv::*;
use parity_common::{Take, assert_parity, rel_max_error};

#[test]
fn legacy_spectral_conv1d_f32_matches_2_0_0() {
    let (out, reference, d3) = run::<3>(&CASE_1D, "f32", &Device::default());
    let err = assert_parity(out, &reference, d3, "legacy_spectral_conv1d_f32");
    println!("legacy_spectral_conv1d_f32: relative max error {err:.3e}");
}

#[test]
fn legacy_spectral_conv2d_f32_matches_2_0_0() {
    let (out, reference, d3) = run::<4>(&CASE_2D, "f32", &Device::default());
    let err = assert_parity(out, &reference, d3, "legacy_spectral_conv2d_f32");
    println!("legacy_spectral_conv2d_f32: relative max error {err:.3e}");
}

#[test]
fn legacy_spectral_conv3d_f32_matches_2_0_0() {
    let (out, reference, d3) = run::<5>(&CASE_3D, "f32", &Device::default());
    let err = assert_parity(out, &reference, d3, "legacy_spectral_conv3d_f32");
    println!("legacy_spectral_conv3d_f32: relative max error {err:.3e}");
}

/// The 3D map's corner swap is needed: 2.0.0's corners loaded in their own order fail.
#[test]
fn legacy_3d_corner_order_matters() {
    let unpermuted = [
        corner!(0, 0).as_slice(),
        &corner!(1, 1),
        &corner!(2, 2),
        &corner!(3, 3),
        &[BIAS],
    ]
    .concat();
    let (out, reference, d3) = run_with::<5>(&CASE_3D, "f32", &unpermuted, &Device::default());
    let err = rel_max_error(out, &reference);
    assert!(err > d3, "unpermuted corners went unnoticed: {err:.3e}");
}

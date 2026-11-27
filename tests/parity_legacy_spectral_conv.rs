//! Library CM1.1: 2.0.0's legacy corner-layout `SpectralConv` against the crate's
//! `SpectralConv<R>`, 1D to 3D, in f32 (D3 as amended 2026-11-25). The f64 fixtures are
//! `tests/parity_legacy_spectral_conv_f64.rs`.

#[path = "parity_common/biased_conv.rs"]
mod biased_conv;
#[path = "parity_common/legacy_conv.rs"]
mod legacy_conv;
mod parity_common;

use burn::prelude::*;
use legacy_conv::*;
use parity_common::{Precision::F32, rel_max_error};

#[test]
fn legacy_spectral_conv1d_f32_matches_2_0_0() {
    check::<3>(&CASE_1D, F32, &Device::default());
}

#[test]
fn legacy_spectral_conv2d_f32_matches_2_0_0() {
    check::<4>(&CASE_2D, F32, &Device::default());
}

#[test]
fn legacy_spectral_conv3d_f32_matches_2_0_0() {
    check::<5>(&CASE_3D, F32, &Device::default());
}

/// The 3D map's corner swap is needed: 2.0.0's corners loaded in their own order fail.
#[test]
fn legacy_3d_corner_order_matters() {
    let mut map = map_3d();
    // Entries 2–3 take 2.0.0's `weight.1` to corner 2 and 4–5 `weight.2` to corner 1: undo it.
    for k in 2..4 {
        let to = map[k].1;
        map[k].1 = map[k + 2].1;
        map[k + 2].1 = to;
    }
    let (out, mut fixture) = output::<5>(&CASE_3D, F32, &map, &Device::default());
    let err = rel_max_error(out, &fixture.output());
    println!("legacy_spectral_conv3d_f32, corners unpermuted: relative max error {err:.3e}");
    assert!(
        err > F32.d3(),
        "unpermuted corners went unnoticed: {err:.3e}"
    );
}

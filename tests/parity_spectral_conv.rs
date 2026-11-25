//! Library CM1.2: 2.0.0's centred `SpectralConv` against the crate's corner-layout
//! `SpectralConv<R>`, the weights converted at load (D5), in f32 (D3 as amended
//! 2026-11-25): dense in 2D and 3D, `fft_norm` forward and backward, odd mode counts,
//! separable weights, and modes below `max_n_modes`. The f64 fixtures are
//! `tests/parity_spectral_conv_f64.rs`.

#[path = "parity_common/centred_conv.rs"]
mod centred_conv;
mod parity_common;

use burn::prelude::*;
use centred_conv::*;
use ndarray::{ArrayD, Axis, IxDyn, Slice};
use parity_common::{Precision::F32, rel_max_error};

#[test]
fn spectral_conv2d_f32_matches_2_0_0() {
    check::<4>(&CASE_2D, F32, &Device::default());
}

#[test]
fn spectral_conv2d_backward_f32_matches_2_0_0() {
    check::<4>(&CASE_2D_BACKWARD, F32, &Device::default());
}

#[test]
fn spectral_conv2d_odd_f32_matches_2_0_0() {
    check::<4>(&CASE_2D_ODD, F32, &Device::default());
}

#[test]
fn spectral_conv3d_f32_matches_2_0_0() {
    check::<5>(&CASE_3D, F32, &Device::default());
}

#[test]
fn spectral_conv2d_separable_f32_matches_2_0_0() {
    check::<4>(&CASE_2D_SEPARABLE, F32, &Device::default());
}

#[test]
fn spectral_conv2d_max_modes_f32_matches_2_0_0() {
    check::<4>(&CASE_2D_MAX_MODES, F32, &Device::default());
}

/// The conversion is needed: the centred weight cut into halves as if it were corners
/// (first half low, second high) fails D3.
#[test]
fn centred_layout_needs_conversion() {
    let halves = |a: ArrayD<f64>| {
        let k = a.shape()[2];
        [0..k / 2, k / 2..k]
            .map(|r| a.slice_axis(Axis(2), Slice::from(r)).to_owned())
            .to_vec()
    };
    let (out, reference) = output::<4>(&CASE_2D, F32, halves, &Device::default());
    let err = rel_max_error(out, &reference);
    println!("spectral_conv2d_f32, centred halves unconverted: relative max error {err:.3e}");
    assert!(
        err > F32.d3(),
        "unconverted weights went unnoticed: {err:.3e}"
    );
}

// Here, not in `centred_conv.rs`: the f64 binary would not use it (dead code).
/// [`CASE_2D`] on a 23-point grid, which corners cannot represent: on an odd-extent axis
/// 2.0.0 applies `fftshift` (not `ifftshift`) before the inverse, moving its output
/// spectrum by one bin.
pub const CASE_2D_ODD_EXTENT: Case = Case {
    stem: "spectral_conv2d_odd_extent",
    ..CASE_2D
};

/// On an odd extent 2.0.0's output is not a spectral convolution in corners: it applies
/// `fftshift` twice, which for odd n is a shift by one bin, not its inverse. The converted
/// weights fail D3 there (a numpy evaluation with `ifftshift` agrees with the crate, one
/// with `fftshift` with 2.0.0; see the plan). Recorded, so the limit is a test, not prose.
#[test]
fn odd_extent_is_not_representable() {
    let (out, reference) = output::<4>(
        &CASE_2D_ODD_EXTENT,
        F32,
        |a| centred_to_corners(a, 2),
        &Device::default(),
    );
    let err = rel_max_error(out, &reference);
    println!("spectral_conv2d_odd_extent_f32: relative max error {err:.3e}");
    assert!(
        err > F32.d3(),
        "2.0.0's odd-extent shift now matches: {err:.3e}"
    );
}

/// `(1, 1, shape.., 2)` holding 1, 2, 3, … in its real part and 0 in its imaginary part,
/// so a converted entry names the index it came from (0 is the zero row).
fn numbered(shape: &[usize]) -> ArrayD<f64> {
    let mut full = vec![1, 1];
    full.extend_from_slice(shape);
    full.push(2);
    let mut a = ArrayD::zeros(IxDyn(&full));
    let n: usize = shape.iter().product();
    for (i, v) in a.iter_mut().step_by(2).take(n).enumerate() {
        *v = (i + 1) as f64;
    }
    a
}

/// The real parts of `a`, row-major.
fn re(a: &ArrayD<f64>) -> Vec<f64> {
    a.index_axis(Axis(a.ndim() - 1), 0)
        .iter()
        .copied()
        .collect()
}

#[test]
fn centred_to_corners_even_count() {
    // K = 4 on axis 0 (frequencies −2, −1, 0, 1), last axis 1.
    let c = centred_to_corners(numbered(&[4, 1]), 2);
    assert_eq!(c.len(), 2);
    assert_eq!(re(&c[0]), [3.0, 4.0]); // frequencies 0, 1
    assert_eq!(re(&c[1]), [1.0, 2.0]); // frequencies −2, −1
}

#[test]
fn centred_to_corners_odd_count_pads_the_high_corner() {
    // K = 5 on axis 0 (frequencies −2..2): 3 per corner, frequency −3 zero.
    let c = centred_to_corners(numbered(&[5, 1]), 2);
    assert_eq!(re(&c[0]), [3.0, 4.0, 5.0]);
    assert_eq!(re(&c[1]), [0.0, 1.0, 2.0]);
    assert!(
        c[1].iter().skip(1).step_by(2).all(|&v| v == 0.0),
        "imaginary parts"
    );
}

#[test]
fn centred_to_corners_3d_corner_order() {
    // K = (2, 3) on the shifted axes, last axis 1. Bit j of the corner is axis j high.
    let c = centred_to_corners(numbered(&[2, 3, 1]), 3);
    assert_eq!(c.len(), 4);
    // Entry (i, j) of the input is 3i + j + 1.
    assert_eq!(re(&c[0]), [5.0, 6.0]); // axis 0 low (i = 1), axis 1 low (j = 1, 2)
    assert_eq!(re(&c[1]), [2.0, 3.0]); // axis 0 high (i = 0)
    assert_eq!(re(&c[2]), [0.0, 4.0]); // axis 1 high: zero row, then j = 0
    assert_eq!(re(&c[3]), [0.0, 1.0]);
    assert_eq!(c[2].shape(), [1, 1, 1, 2, 1, 2]);
}

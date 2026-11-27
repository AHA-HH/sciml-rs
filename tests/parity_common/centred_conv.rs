//! 2.0.0's centred `SpectralConv` (`layers/spectral_convolution.py`) as the crate's
//! `SpectralConv<R>` plus a bias (`biased_conv.rs`), for library CM1.2's
//! `spectral_conv*_{f32,f64}` fixtures. Test code only.
//!
//! 2.0.0 stores one weight `(in, out, *max_n_modes)` (`(channels, *max_n_modes)` if
//! separable) in its centred layout, a complex tensor saved with a trailing (re, im) axis,
//! and a bias `(out, 1, …)` added after the inverse transform. At load the weight becomes
//! the crate's corners by `parity_common::centred_to_corners` (D5) and the bias gains a
//! leading axis.
//! Its default `fft_norm="forward"` needs no conversion: the layer is one forward and one
//! inverse transform at the same size, so the scale factors multiply to 1/N under either
//! normalisation.

use burn::prelude::*;
use ndarray::{ArrayD, Axis};
use sciml_rs::neural_operators::layers::spectral_convolution::{SpectralConv, SpectralInit};

use crate::{
    biased_conv::{BiasedConv, run},
    parity_common::{Conversion, Fixture, MapEntry, Precision, Take, centred_to_corners, check_d3},
};

/// A fixture's configuration, in the crate's terms.
pub struct Case {
    pub stem: &'static str,
    pub in_channels: usize,
    pub out_channels: usize,
    pub separable: bool,
    /// Per corner, from 2.0.0's stored weight: `K − K/2` on a non-last axis, `K` on the last.
    pub stored: &'static [usize],
    /// The modes used: 2.0.0's `n_modes` the same way (last axis `n // 2 + 1`).
    pub active: &'static [usize],
}

impl Case {
    /// Arbitrary weights of the stored shape, to be replaced by [`load_into`].
    fn layer<const R: usize>(&self, device: &Device) -> BiasedConv<R> {
        let conv = if self.separable {
            SpectralConv::new_separable(device, self.in_channels, self.stored, SpectralInit::Normal)
        } else {
            SpectralConv::new(device, self.in_channels, self.out_channels, self.stored)
        };
        BiasedConv::new(conv.with_modes(self.active), self.out_channels, device)
    }
}

/// As `tools/parity/generate.py`'s `CONFIGS`.
pub const CASE_2D: Case = Case {
    stem: "spectral_conv2d",
    in_channels: 3,
    out_channels: 4,
    separable: false,
    stored: &[4, 4],
    active: &[4, 4],
};
/// `fft_norm="backward"`, otherwise [`CASE_2D`]: the same converter.
pub const CASE_2D_BACKWARD: Case = Case {
    stem: "spectral_conv2d_backward",
    ..CASE_2D
};
/// `n_modes=(7, 6)`: the odd count's zero row.
pub const CASE_2D_ODD: Case = Case {
    stem: "spectral_conv2d_odd",
    ..CASE_2D
};
/// `n_modes=(6, 5, 4)`: an even and an odd count, 3D corner order.
pub const CASE_3D: Case = Case {
    stem: "spectral_conv3d",
    in_channels: 2,
    out_channels: 3,
    separable: false,
    stored: &[3, 3, 3],
    active: &[3, 3, 3],
};
/// `separable=True`, 4 channels.
pub const CASE_2D_SEPARABLE: Case = Case {
    stem: "spectral_conv2d_separable",
    in_channels: 4,
    out_channels: 4,
    separable: true,
    stored: &[4, 4],
    active: &[4, 4],
};
/// `n_modes=(6, 6)`, `max_n_modes=(10, 8)`: stored (10, 8), used (6, 4).
pub const CASE_2D_MAX_MODES: Case = Case {
    stem: "spectral_conv2d_max_modes",
    in_channels: 3,
    out_channels: 4,
    separable: false,
    stored: &[5, 8],
    active: &[3, 4],
};

/// One corner: the converted `weight.tensor.corner<c>` to the crate's corner `c`.
macro_rules! corner {
    ($c:literal) => {
        [
            (
                concat!("weight.tensor.corner", $c),
                concat!("conv.weights_re.", $c),
                Take::Re,
            ),
            (
                concat!("weight.tensor.corner", $c),
                concat!("conv.weights_im.", $c),
                Take::Im,
            ),
        ]
    };
}

/// The map for `d` spatial axes: corner to corner in the same order, and the bias.
pub fn map(d: usize) -> Vec<MapEntry> {
    let corners: &[[MapEntry; 2]] = &[corner!(0), corner!(1), corner!(2), corner!(3)];
    let mut map: Vec<MapEntry> = corners[..1 << (d - 1)].concat();
    map.push(("bias", "bias", Take::Whole));
    map
}

/// The conversions of `case`'s opened fixture for its load: the bias to `[1, out, 1, …]`,
/// and the weight, given by `corners` from its 2.0.0 array (separable: with a leading axis
/// of 1 added), to `weight.tensor.corner<c>`.
pub fn convert(
    fixture: &mut Fixture,
    case: &Case,
    corners: impl FnOnce(ArrayD<f64>) -> Vec<ArrayD<f64>>,
) -> Vec<Conversion> {
    let bias = fixture.array("param.bias").insert_axis(Axis(0));
    let mut weight = fixture.array("param.weight.tensor");
    if case.separable {
        weight = weight.insert_axis(Axis(0));
    }
    let parts = corners(weight)
        .into_iter()
        .enumerate()
        .map(|(c, a)| (format!("weight.tensor.corner{c}"), a))
        .collect();
    vec![
        Conversion::whole("bias", bias),
        Conversion {
            from: "weight.tensor".into(),
            parts,
        },
    ]
}

/// `case`'s fixture in `precision`, its weight turned into corners by `corners`, run
/// through the layer: its output on the fixture's input, and the open fixture.
pub fn output<const R: usize>(
    case: &Case,
    precision: Precision,
    corners: impl FnOnce(ArrayD<f64>) -> Vec<ArrayD<f64>>,
    device: &Device,
) -> (Tensor<R>, Fixture) {
    let mut fixture = Fixture::open_as(case.stem, precision);
    let converted = convert(&mut fixture, case, corners);
    run(fixture, case.layer(device), &map(R - 2), &converted, device)
}

/// Asserts `case` in `precision`, through `centred_to_corners`, is within D3; prints and
/// returns the error.
///
/// # Panics
/// If the error exceeds D3's bound for `precision`.
pub fn check<const R: usize>(case: &Case, precision: Precision, device: &Device) -> f64 {
    let (out, mut fixture) = output::<R>(case, precision, |a| centred_to_corners(a, R - 2), device);
    check_d3(out, &mut fixture)
}

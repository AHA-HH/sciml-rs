//! 2.0.0's legacy `SpectralConv` (`layers/legacy_spectral_convolution.py`) as the crate's
//! `SpectralConv<R>` plus a bias (`biased_conv.rs`), for library CM1.1's `legacy_spectral_conv{1,2,3}d_{f32,f64}`
//! fixtures, and the hand-written maps to it. Test code only.
//!
//! The legacy layer keeps corner weights `(in, out, *[m // 2 for m in n_modes])`, stored by
//! tltorch as real tensors with a trailing (re, im) axis, uses `fft_norm="backward"` as
//! CONVENTIONS §4 does, and adds a bias `(n_layers, out, 1, …)` (`n_layers` = 1 here) after the inverse transform. Its
//! corners run in `itertools.product` order, axis 0 most significant; the crate's corner
//! index has axis `j` in bit `j` (CONVENTIONS §5). The two agree in 1D and 2D; in 3D 2.0.0's
//! `weight.1` (low axis 0, high axis 1) is corner 2 here and `weight.2` corner 1.

use burn::prelude::*;
use ndarray::ArrayD;
use sciml_rs::neural_operators::layers::spectral_convolution::SpectralConv;

use crate::{
    biased_conv::BiasedConv,
    parity_common::{Fixture, MapEntry, Precision, Take, assert_parity, load_into},
};

/// One corner: 2.0.0's `weight.<i>.tensor` to the crate's corner `c`.
macro_rules! corner {
    ($i:literal, $c:literal) => {
        [
            (
                concat!("weight.", $i, ".tensor"),
                concat!("conv.weights_re.", $c),
                crate::parity_common::Take::Re,
            ),
            (
                concat!("weight.", $i, ".tensor"),
                concat!("conv.weights_im.", $c),
                crate::parity_common::Take::Im,
            ),
        ]
    };
}

/// The bias, as stored.
pub const BIAS: MapEntry = ("bias", "bias", Take::Whole);

/// 1D: one corner.
pub fn map_1d() -> Vec<MapEntry> {
    [corner!(0, 0).as_slice(), &[BIAS]].concat()
}

/// 2D: corners in the same order.
pub fn map_2d() -> Vec<MapEntry> {
    [corner!(0, 0).as_slice(), &corner!(1, 1), &[BIAS]].concat()
}

/// 3D: 2.0.0's corners 1 and 2 swap.
pub fn map_3d() -> Vec<MapEntry> {
    [
        corner!(0, 0).as_slice(),
        &corner!(1, 2),
        &corner!(2, 1),
        &corner!(3, 3),
        &[BIAS],
    ]
    .concat()
}

/// A fixture's configuration: name stem, channels in and out, the crate's modes, the map.
pub struct Case {
    pub stem: &'static str,
    pub in_channels: usize,
    pub out_channels: usize,
    pub modes: &'static [usize],
    pub map: fn() -> Vec<MapEntry>,
}

/// As `tools/parity/generate.py`'s `CONFIGS`, `n_modes` halved.
pub const CASE_1D: Case = Case {
    stem: "legacy_spectral_conv1d",
    in_channels: 3,
    out_channels: 4,
    modes: &[6],
    map: map_1d,
};
pub const CASE_2D: Case = Case {
    stem: "legacy_spectral_conv2d",
    in_channels: 3,
    out_channels: 4,
    modes: &[4, 3],
    map: map_2d,
};
pub const CASE_3D: Case = Case {
    stem: "legacy_spectral_conv3d",
    in_channels: 2,
    out_channels: 3,
    modes: &[3, 2, 2],
    map: map_3d,
};

/// `case`'s fixture in `precision` loaded through `map`: the layer's output on the fixture's
/// input, and the reference output.
pub fn output<const R: usize>(
    case: &Case,
    precision: Precision,
    map: &[MapEntry],
    device: &Device,
) -> (Tensor<R>, ArrayD<f64>) {
    let mut fixture = Fixture::open_as(case.stem, precision);
    let conv = SpectralConv::new(device, case.in_channels, case.out_channels, case.modes);
    let mut layer = BiasedConv::<R>::new(conv, case.out_channels, device);
    load_into(&mut layer, &mut fixture, map, &[]);
    let out = layer.forward(fixture.input(device));
    (out, fixture.output())
}

/// Asserts `case` in `precision`, through its own map, is within D3; prints and returns the
/// error.
///
/// # Panics
/// If the error exceeds D3's bound for `precision`.
pub fn check<const R: usize>(case: &Case, precision: Precision, device: &Device) -> f64 {
    let (out, reference) = output::<R>(case, precision, &(case.map)(), device);
    let what = format!("{}_{}", case.stem, precision.suffix());
    let err = assert_parity(out, &reference, precision.d3(), &what);
    println!("{what}: relative max error {err:.3e}");
    err
}

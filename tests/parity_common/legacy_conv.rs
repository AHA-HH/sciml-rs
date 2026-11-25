//! 2.0.0's legacy `SpectralConv` (`layers/legacy_spectral_convolution.py`) as the crate's
//! `SpectralConv<R>` plus a bias, for library CM1.1's `legacy_spectral_conv{1,2,3}d_{f32,f64}`
//! fixtures, and the hand-written maps to it. Test code only.
//!
//! The legacy layer keeps corner weights `(in, out, *[m // 2 for m in n_modes])`, stored by
//! tltorch as real tensors with a trailing (re, im) axis, uses `fft_norm="backward"` as
//! CONVENTIONS §4 does, and adds a bias `(1, out, 1, …)` after the inverse transform. Its
//! corners run in `itertools.product` order, axis 0 most significant; the crate's corner
//! index has axis `j` in bit `j` (CONVENTIONS §5). The two agree in 1D and 2D; in 3D 2.0.0's
//! `weight.1` (low axis 0, high axis 1) is corner 2 here and `weight.2` corner 1.

use burn::{module::Param, prelude::*};
use ndarray::ArrayD;
use sciml_rs::neural_operators::layers::spectral_convolution::SpectralConv;

use crate::parity_common::{Fixture, MapEntry, Take, load_into};

/// 2.0.0's legacy layer: the spectral convolution, then the bias.
#[derive(Module, Debug)]
pub struct LegacyConv<const R: usize> {
    pub conv: SpectralConv<R>,
    /// `[1, out_channels, 1, …]`.
    pub bias: Param<Tensor<R>>,
}

impl<const R: usize> LegacyConv<R> {
    /// Arbitrary weights, to be replaced by [`load_into`]. `modes` per axis, as the crate
    /// counts them (2.0.0's `n_modes // 2`).
    pub fn new(device: &Device, in_channels: usize, out_channels: usize, modes: &[usize]) -> Self {
        let mut bias = [1; R];
        bias[1] = out_channels;
        Self {
            conv: SpectralConv::new(device, in_channels, out_channels, modes),
            bias: Param::from_tensor(Tensor::zeros(bias, device)),
        }
    }

    /// `x`: `[batch, in_channels, spatial..]` → `[batch, out_channels, spatial..]`.
    pub fn forward(&self, x: Tensor<R>) -> Tensor<R> {
        self.conv.forward(x) + self.bias.val()
    }
}

/// One corner: 2.0.0's `weight.<i>.tensor` to the crate's corner `c`.
macro_rules! corner {
    ($i:literal, $c:literal) => {
        [
            (
                concat!("weight.", $i, ".tensor"),
                concat!("conv.weights_re.", $c),
                Take::Re,
            ),
            (
                concat!("weight.", $i, ".tensor"),
                concat!("conv.weights_im.", $c),
                Take::Im,
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

/// `case`'s fixture of precision `dtype` (`"f32"`, `"f64"`) loaded through `map`, its
/// output on the fixture's input, the reference output and D3's bound.
pub fn run_with<const R: usize>(
    case: &Case,
    dtype: &str,
    map: &[MapEntry],
    device: &Device,
) -> (Tensor<R>, ArrayD<f64>, f64) {
    let mut fixture = Fixture::open(&format!("{}_{dtype}", case.stem));
    let mut layer = LegacyConv::<R>::new(device, case.in_channels, case.out_channels, case.modes);
    load_into(&mut layer, &mut fixture, map);
    let out = layer.forward(fixture.input(device));
    (out, fixture.output(), fixture.precision.d3())
}

/// [`run_with`] through the case's own map.
pub fn run<const R: usize>(
    case: &Case,
    dtype: &str,
    device: &Device,
) -> (Tensor<R>, ArrayD<f64>, f64) {
    run_with(case, dtype, &(case.map)(), device)
}

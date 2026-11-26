//! 2.0.0's centred `SpectralConv` (`layers/spectral_convolution.py`) as the crate's
//! `SpectralConv<R>` plus a bias, for library CM1.2's `spectral_conv*_{f32,f64}` fixtures.
//! Test code only.
//!
//! 2.0.0 stores one weight `(in, out, *max_n_modes)` (`(channels, *max_n_modes)` if
//! separable) in its centred layout, a complex tensor saved with a trailing (re, im) axis,
//! and a bias `(out, 1, …)` added after the inverse transform. At load the weight becomes
//! the crate's corners by [`centred_to_corners`] (D5) and the bias gains a leading axis.
//! Its default `fft_norm="forward"` needs no conversion: the layer is one forward and one
//! inverse transform at the same size, so the scale factors multiply to 1/N under either
//! normalisation.

use burn::{module::Param, prelude::*};
use ndarray::{ArrayD, Axis, Slice, concatenate};
use sciml_rs::neural_operators::layers::spectral_convolution::{SpectralConv, SpectralInit};

use crate::parity_common::{Fixture, MapEntry, Precision, Take, assert_parity, load_into};

/// 2.0.0's centred spectral weight `a`, `(lead.., K_0, .., K_{d−1}, 2)` with a trailing
/// (re, im) axis, as the crate's `2^(d−1)` mode corners, in corner order (bit `j` of the
/// index is axis `j`, CONVENTIONS §5), each `(lead.., m_0, .., m_{d−2}, K_{d−1}, 2)`.
///
/// On a non-last axis 2.0.0 keeps `K` modes around frequency 0 after `fftshift`: index `i`
/// is frequency `i − K/2` (integer division), so `K/2` negative and `K − K/2` non-negative
/// frequencies. Each corner gets `m = K − K/2`: the low corner indices `K/2..K` (frequencies
/// `0..m`), the high corner indices `K/2 − m..K/2` (frequencies `−m..−1`); for odd `K` that
/// starts one before 0, and the row is zero, a frequency 2.0.0 does not keep. The last axis
/// (the real FFT's, frequencies `0..K`) is taken whole.
///
/// The corners reproduce 2.0.0 only where the grid's extent is even on every non-last
/// axis: on an odd extent 2.0.0 applies `fftshift` (not `ifftshift`) again before the
/// inverse, which moves its output spectrum by one bin.
///
/// # Panics
/// If `a` has fewer than `d + 1` axes or no trailing axis of 2.
pub fn centred_to_corners(a: ArrayD<f64>, d: usize) -> Vec<ArrayD<f64>> {
    assert!(
        a.ndim() > d,
        "centred_to_corners: {} axes for d = {d}",
        a.ndim()
    );
    assert_eq!(a.shape()[a.ndim() - 1], 2, "no trailing (re, im) axis");
    let lead = a.ndim() - 1 - d;
    (0..1usize << (d - 1))
        .map(|corner| {
            let mut c = a.clone();
            for j in 0..d - 1 {
                let axis = Axis(lead + j);
                let k = c.shape()[lead + j];
                let half = k / 2;
                c = if corner >> j & 1 == 0 {
                    c.slice_axis(axis, Slice::from(half..k)).to_owned()
                } else if k.is_multiple_of(2) {
                    c.slice_axis(axis, Slice::from(0..half)).to_owned()
                } else {
                    let mut zero_shape = c.shape().to_vec();
                    zero_shape[lead + j] = 1;
                    let zero = ArrayD::zeros(zero_shape);
                    let neg = c.slice_axis(axis, Slice::from(0..half));
                    concatenate(axis, &[zero.view(), neg]).unwrap()
                };
            }
            c
        })
        .collect()
}

impl Fixture {
    /// Replaces the parameter `name` by `parts` (each a new parameter name and its array),
    /// for this load only, so that [`load_into`]'s checks see the converted parameters.
    ///
    /// # Panics
    /// If the fixture has no parameter `name`.
    pub fn replace(&mut self, name: &str, parts: Vec<(String, ArrayD<f64>)>) {
        let i = self
            .params
            .iter()
            .position(|p| p == name)
            .unwrap_or_else(|| panic!("fixture: no parameter `{name}`"));
        self.params.remove(i);
        self.replaced.remove(&format!("param.{name}"));
        for (part, a) in parts {
            self.replaced.insert(format!("param.{part}"), a);
            self.params.push(part);
        }
    }
}

/// 2.0.0's centred layer: the spectral convolution, then the bias.
#[derive(Module, Debug)]
pub struct CentredConv<const R: usize> {
    pub conv: SpectralConv<R>,
    /// `[1, out_channels, 1, …]`.
    pub bias: Param<Tensor<R>>,
}

impl<const R: usize> CentredConv<R> {
    /// `x`: `[batch, in_channels, spatial..]` → `[batch, out_channels, spatial..]`.
    pub fn forward(&self, x: Tensor<R>) -> Tensor<R> {
        self.conv.forward(x) + self.bias.val()
    }
}

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
    fn layer<const R: usize>(&self, device: &Device) -> CentredConv<R> {
        let conv = if self.separable {
            SpectralConv::new_separable(device, self.in_channels, self.stored, SpectralInit::Normal)
        } else {
            SpectralConv::new(device, self.in_channels, self.out_channels, self.stored)
        };
        let mut bias = [1; R];
        bias[1] = self.out_channels;
        CentredConv {
            conv: conv.with_modes(self.active),
            bias: Param::from_tensor(Tensor::zeros(bias, device)),
        }
    }

    /// The fixture's name in `precision`.
    pub fn name(&self, precision: Precision) -> String {
        let dtype = match precision {
            Precision::F32 => "f32",
            Precision::F64 => "f64",
        };
        format!("{}_{dtype}", self.stem)
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

/// Converts the opened fixture of `case` at load: the bias to `[1, out, 1, …]`, and the
/// weight, given by `corners` from its 2.0.0 array (separable: with a leading axis of 1
/// added), to `weight.tensor.corner<c>`.
pub fn convert(
    fixture: &mut Fixture,
    case: &Case,
    corners: impl FnOnce(ArrayD<f64>) -> Vec<ArrayD<f64>>,
) {
    let bias = fixture.array("param.bias").insert_axis(Axis(0));
    fixture.replace("bias", vec![("bias".into(), bias)]);
    let mut weight = fixture.array("param.weight.tensor");
    if case.separable {
        weight = weight.insert_axis(Axis(0));
    }
    let parts = corners(weight)
        .into_iter()
        .enumerate()
        .map(|(c, a)| (format!("weight.tensor.corner{c}"), a))
        .collect();
    fixture.replace("weight.tensor", parts);
}

/// `case`'s fixture in `precision`, its weight turned into corners by `corners`: the
/// layer's output on the fixture's input, and the reference output.
pub fn output<const R: usize>(
    case: &Case,
    precision: Precision,
    corners: impl FnOnce(ArrayD<f64>) -> Vec<ArrayD<f64>>,
    device: &Device,
) -> (Tensor<R>, ArrayD<f64>) {
    let mut fixture = Fixture::open(&case.name(precision));
    convert(&mut fixture, case, corners);
    let mut layer = case.layer::<R>(device);
    load_into(&mut layer, &mut fixture, &map(R - 2));
    let out = layer.forward(fixture.input(device));
    (out, fixture.output())
}

/// Asserts `case` in `precision`, through [`centred_to_corners`], is within D3; prints and
/// returns the error.
///
/// # Panics
/// If the error exceeds D3's bound for `precision`.
pub fn check<const R: usize>(case: &Case, precision: Precision, device: &Device) -> f64 {
    let (out, reference) = output::<R>(case, precision, |a| centred_to_corners(a, R - 2), device);
    let what = case.name(precision);
    let err = assert_parity(out, &reference, precision.d3(), &what);
    println!("{what}: relative max error {err:.3e}");
    err
}

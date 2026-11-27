//! 2.0.0's spectral layer shape, legacy (`legacy_conv.rs`) or centred (`centred_conv.rs`):
//! the crate's `SpectralConv<R>`, then a bias added after the inverse transform. Test code
//! only (library CM1.5f, one layer for both).

use burn::{module::Param, prelude::*};
use sciml_rs::neural_operators::layers::spectral_convolution::SpectralConv;

use crate::parity_common::{Conversion, Fixture, MapEntry, load_into};

/// A spectral convolution and a bias.
#[derive(Module, Debug)]
pub struct BiasedConv<const R: usize> {
    pub conv: SpectralConv<R>,
    /// `[1, out_channels, 1, …]`.
    pub bias: Param<Tensor<R>>,
}

impl<const R: usize> BiasedConv<R> {
    /// `conv` with a zero bias of `out_channels`, to be replaced by `load_into`.
    pub fn new(conv: SpectralConv<R>, out_channels: usize, device: &Device) -> Self {
        let mut bias = [1; R];
        bias[1] = out_channels;
        Self {
            conv,
            bias: Param::from_tensor(Tensor::zeros(bias, device)),
        }
    }

    /// `x`: `[batch, in_channels, spatial..]` → `[batch, out_channels, spatial..]`.
    pub fn forward(&self, x: Tensor<R>) -> Tensor<R> {
        self.conv.forward(x) + self.bias.val()
    }
}

/// Loads `fixture` into `layer` through `map` and `converted`, and runs it on the fixture's
/// input: the output and the fixture.
///
/// # Panics
/// If the load is incomplete (`load_into`).
pub fn run<const R: usize>(
    mut fixture: Fixture,
    mut layer: BiasedConv<R>,
    map: &[MapEntry],
    converted: &[Conversion],
    device: &Device,
) -> (Tensor<R>, Fixture) {
    load_into(&mut layer, &mut fixture, map, converted);
    (layer.forward(fixture.input(device)), fixture)
}

//! Library CM1.5a: 2.0.0's `FNO`, end to end, against the crate's `FNO` built from
//! `FNOConfig`'s options: the grid embedding, the lifting, each layer's centred spectral
//! convolution (converted to corners at load, D5) with its bias, the block parts, and the
//! projection, through one map built from the configuration.

use burn::prelude::*;
use ndarray::{ArrayD, Axis};
use sciml_rs::neural_operators::models::fno::{FNO, FNOConfig};

use crate::{
    fno_map::{
        block_defaults, block_parts, leak, lift_defaults, lift_project, to_channels_first,
        to_channels_last,
    },
    parity_common::{
        Conversion, Fixture, MapEntry, Precision, Take, centred_to_corners, check_d3, load_into,
    },
};

/// A fixture's configuration, as `generate.py`'s `CONFIGS`.
pub struct Case {
    /// The fixture's name without `_f32` / `_f64`.
    pub stem: &'static str,
    /// 2.0.0's `in_channels`, `out_channels`, `hidden_channels`, `n_layers`.
    pub data: usize,
    pub out: usize,
    pub hidden: usize,
    pub n_layers: usize,
    /// 2.0.0's `n_modes` (both signs on a non-last axis; the last halved by 2.0.0).
    pub n_modes: &'static [usize],
    /// 2.0.0's `max_n_modes` (the last axis as stored, not halved), if set.
    pub max_n_modes: Option<&'static [usize]>,
    /// The options on top of 2.0.0's defaults ([`defaults`]); the map follows them.
    pub options: fn(FNOConfig) -> FNOConfig,
    /// 2.0.0's `forward(x, output_shape=…)` as a tuple (the last layer's shape), if set.
    pub output_shape: Option<&'static [usize]>,
}

/// 2.0.0's `FNO` defaults: the block's and the lifting's ([`block_defaults`],
/// [`lift_defaults`]), and the convolution's bias.
pub fn defaults(c: FNOConfig) -> FNOConfig {
    lift_defaults(block_defaults(c)).with_conv_bias(Some(true))
}

/// 2.0.0's defaults, `FNO(n_modes=(8,), in=1, out=1, hidden=8)` on (2, 1, 64): Burgers-shaped.
pub const FNO1D: Case = Case {
    stem: "fno1d",
    data: 1,
    out: 1,
    hidden: 8,
    n_layers: 4,
    n_modes: &[8],
    max_n_modes: None,
    options: |c| c,
    output_shape: None,
};

/// 2.0.0's defaults, `FNO(n_modes=(4, 4), in=1, out=1, hidden=8)` on (2, 1, 32, 32):
/// Darcy-shaped.
pub const FNO2D: Case = Case {
    stem: "fno2d",
    n_modes: &[4, 4],
    ..FNO1D
};

impl Case {
    /// The crate's configuration: 2.0.0's counts per corner (CM1.2), `K − K/2` on a
    /// non-last axis; on the last `n // 2 + 1` used, and `max_n_modes` as stored.
    pub fn config(&self) -> FNOConfig {
        let d = self.n_modes.len();
        let corners = |counts: &[usize], last: fn(usize) -> usize| -> Vec<usize> {
            let (rest, &[k]) = counts.split_at(d - 1) else {
                unreachable!("split_at(d - 1) leaves one count")
            };
            rest.iter().map(|k| k - k / 2).chain([last(k)]).collect()
        };
        let config = FNOConfig::new(corners(self.n_modes, |k| k / 2 + 1), self.data, self.out)
            .with_hidden_channels(self.hidden)
            .with_n_layers(self.n_layers)
            .with_max_modes(self.max_n_modes.map(|m| corners(m, |k| k)));
        (self.options)(defaults(config))
    }
}

/// The conversions of the opened fixture for its load: each layer's centred weight to
/// corners (`fno_blocks.convs.<l>.weight.corner<c>`; separable with a leading axis of 1),
/// and its bias `(W, 1, …)` to `[W]`.
fn convert(fixture: &mut Fixture, config: &FNOConfig, d: usize) -> Vec<Conversion> {
    let mut converted = Vec::new();
    for l in 0..config.n_layers {
        let name = format!("fno_blocks.convs.{l}.weight.tensor");
        let mut weight = fixture.array(&format!("param.{name}"));
        if config.separable == Some(true) {
            weight = weight.insert_axis(Axis(0));
        }
        let parts = centred_to_corners(weight, d)
            .into_iter()
            .enumerate()
            .map(|(c, a)| (format!("fno_blocks.convs.{l}.weight.corner{c}"), a))
            .collect();
        converted.push(Conversion { from: name, parts });
        let name = format!("fno_blocks.convs.{l}.bias");
        let bias = fixture.array(&format!("param.{name}"));
        let bias: ArrayD<f64> = bias.flatten().to_owned().into_dyn();
        converted.push(Conversion::whole(&name, bias));
    }
    converted
}

/// The whole map, from 2.0.0's `FNO` names to the crate's `FNO` paths, for `config`.
fn map(config: &FNOConfig, d: usize) -> Vec<MapEntry> {
    let mut m = lift_project(config, "");
    for l in 0..config.n_layers {
        for c in 0..1 << (d - 1) {
            let from = leak(format!("fno_blocks.convs.{l}.weight.corner{c}"));
            m.push((from, leak(format!("conv.{l}.weights_re.{c}")), Take::Re));
            m.push((from, leak(format!("conv.{l}.weights_im.{c}")), Take::Im));
        }
        // 2.0.0's convolutions always have a bias here (`defaults`).
        let from = leak(format!("fno_blocks.convs.{l}.bias"));
        m.push((from, leak(format!("conv_biases.{l}")), Take::Whole));
    }
    m.extend(block_parts(config, ""));
    m
}

/// A load of a fixture into the crate's `FNO`, not yet applied: a control corrupts its
/// `map` or `converted` before [`Load::run`].
pub struct Load<const R: usize> {
    /// 2.0.0's names to the `FNO`'s paths.
    pub map: Vec<MapEntry>,
    /// The fixture's parameters converted for this load (the centred weights and biases).
    pub converted: Vec<Conversion>,
    fno: FNO<R>,
    fixture: Fixture,
    output_shape: Option<&'static [usize]>,
    device: Device,
}

/// `case`'s fixture in `precision`, opened and converted, with its map and an `FNO` of its
/// configuration on `device`.
pub fn load<const R: usize>(case: &Case, precision: Precision, device: &Device) -> Load<R> {
    let d = R - 2;
    let config = case.config();
    let mut fixture = Fixture::open_as(case.stem, precision);
    Load {
        map: map(&config, d),
        converted: convert(&mut fixture, &config, d),
        fno: config.init::<R>(device),
        fixture,
        output_shape: case.output_shape,
        device: device.clone(),
    }
}

impl<const R: usize> Load<R> {
    /// Loads the fixture into the `FNO` and runs it on the fixture's input: the output,
    /// channels first, and the open fixture.
    ///
    /// # Panics
    /// If the load is incomplete (`load_into`).
    pub fn run(mut self) -> (Tensor<R>, Fixture) {
        load_into(&mut self.fno, &mut self.fixture, &self.map, &self.converted);
        let x = to_channels_last(self.fixture.input::<R>(&self.device));
        let y = match self.output_shape {
            Some(shape) => self.fno.forward_with_output_shape(x, shape),
            None => self.fno.forward(x),
        };
        (to_channels_first(y), self.fixture)
    }
}

/// Asserts `case`'s fixture in `precision` within D3 through the crate's `FNO`; prints and
/// returns the relative max error.
///
/// # Panics
/// If the load is incomplete or the error exceeds D3's bound for `precision`.
pub fn check<const R: usize>(case: &Case, precision: Precision, device: &Device) -> f64 {
    let (y, mut fixture) = load::<R>(case, precision, device).run();
    check_d3(y, &mut fixture)
}

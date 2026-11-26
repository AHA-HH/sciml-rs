//! Library CM1.5a: 2.0.0's `FNO`, end to end, against the crate's `FNO` built from
//! `FNOConfig`'s options: the grid embedding, the lifting, each layer's centred spectral
//! convolution (converted to corners at load, D5) with its bias, the block parts, and the
//! projection, through one map built from the configuration.

use burn::prelude::*;
use ndarray::{ArrayD, Axis};
use sciml_rs::neural_operators::{
    layers::{embedding::GridEmbeddingNd, skip::SkipKind},
    models::fno::{Activation, BlockMlpConfig, FNOConfig, PositionalEmbedding},
};

use crate::parity_common::{
    Fixture, MapEntry, Precision, Take, block_parts, centred_to_corners, check_d3, leak,
    lift_project, load_into, to_channels_first, to_channels_last,
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
}

/// 2.0.0's `FNO` defaults, as `FNOConfig`'s options: GELU, the linear FNO skip, the channel
/// MLP of expansion 0.5 with a soft-gating skip, lifting and projection ratios 2, the unit
/// grid, the convolution's bias.
pub fn defaults(c: FNOConfig) -> FNOConfig {
    let d = c.modes.len();
    c.with_activation(Some(Activation::Gelu))
        .with_fno_skip(Some(SkipKind::Linear))
        .with_channel_mlp(Some(BlockMlpConfig::new()))
        .with_lifting_channel_ratio(Some(2.0))
        .with_projection_channel_ratio(Some(2.0))
        .with_positional_embedding(Some(PositionalEmbedding::GridNd(GridEmbeddingNd::unit(d))))
        .with_conv_bias(Some(true))
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

/// Converts the opened fixture at load: each layer's centred weight to corners
/// (`fno_blocks.convs.<l>.weight.corner<c>`; separable with a leading axis of 1), and its
/// bias `(W, 1, …)` to `[W]`.
fn convert(fixture: &mut Fixture, config: &FNOConfig, d: usize) {
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
        fixture.replace(&name, parts);
        let name = format!("fno_blocks.convs.{l}.bias");
        let bias = fixture.array(&format!("param.{name}"));
        let bias: ArrayD<f64> = bias.flatten().to_owned().into_dyn();
        fixture.replace(&name, vec![(name.clone(), bias)]);
    }
}

/// The whole map, from 2.0.0's `FNO` names to the crate's `FNO` paths, for `config`.
fn map(config: &FNOConfig, d: usize) -> Vec<MapEntry> {
    let two_layer_lifting = config
        .lifting_channel_ratio
        .is_some_and(|r| (r * config.hidden_channels as f64) as usize > 0);
    let mut m = lift_project(two_layer_lifting, "");
    for l in 0..config.n_layers {
        for c in 0..1 << (d - 1) {
            let from = leak(format!("fno_blocks.convs.{l}.weight.corner{c}"));
            m.push((from, leak(format!("conv.{l}.weights_re.{c}")), Take::Re));
            m.push((from, leak(format!("conv.{l}.weights_im.{c}")), Take::Im));
        }
        if config.conv_bias == Some(true) {
            let from = leak(format!("fno_blocks.convs.{l}.bias"));
            m.push((from, leak(format!("conv_biases.{l}")), Take::Whole));
        }
    }
    m.extend(block_parts(config, ""));
    m
}

/// `case`'s fixture in `precision` through the crate's `FNO`, with `corners` applied to the
/// converted map before the load (the identity for parity, a swap for the control): the
/// output, channels first, and the open fixture.
pub fn output<const R: usize>(
    case: &Case,
    precision: Precision,
    corners: impl Fn(&mut Vec<MapEntry>),
    device: &Device,
) -> (Tensor<R>, Fixture) {
    let d = R - 2;
    let config = case.config();
    let mut fixture = Fixture::open_as(case.stem, precision);
    convert(&mut fixture, &config, d);
    let mut fno = config.init::<R>(device);
    let mut map = map(&config, d);
    corners(&mut map);
    load_into(&mut fno, &mut fixture, &map);
    let y = fno.forward(to_channels_last(fixture.input::<R>(device)));
    (to_channels_first(y), fixture)
}

/// Asserts `case`'s fixture in `precision` within D3 through the crate's `FNO`; prints and
/// returns the relative max error.
///
/// # Panics
/// If the load is incomplete or the error exceeds D3's bound for `precision`.
pub fn check<const R: usize>(case: &Case, precision: Precision, device: &Device) -> f64 {
    let (y, mut fixture) = output::<R>(case, precision, |_| {}, device);
    check_d3(y, &mut fixture)
}

//! Library CM1.4: 2.0.0's embeddings, and its `FNO` lifting and projection (fixtures of an
//! `FNO` with `n_layers=0`: embedding, lifting, projection), against the crate's
//! (`layers::embedding`, `FNOConfig`'s `lifting_channel_ratio`, `projection_channel_ratio`
//! and `positional_embedding`). Each function loads one fixture, asserts D3 for its
//! precision and returns the relative max error.

use burn::prelude::*;
use sciml_rs::neural_operators::{
    layers::embedding::{
        GridEmbeddingNd, RotaryEmbedding2d, SinusoidalEmbedding, SinusoidalKind,
        apply_2d_rotary_pos_emb,
    },
    models::fno::{Activation, FNOConfig, PositionalEmbedding},
};

use crate::parity_common::{Fixture, MapEntry, Precision, Take, assert_parity, load_into};

/// `generate.py`'s `BOUNDARIES`.
const BOUNDARIES: [(f64, f64); 2] = [(0.0, 2.0), (-1.0, 1.0)];

fn open(stem: &str, p: Precision) -> Fixture {
    Fixture::open(&format!(
        "{stem}_{}",
        if p == Precision::F32 { "f32" } else { "f64" }
    ))
}

/// Asserts D3 on `actual` against `fixture`'s output and prints the error.
fn check<const R: usize>(actual: Tensor<R>, fixture: &mut Fixture) -> f64 {
    let name = fixture.name.clone();
    let err = assert_parity(actual, &fixture.output(), fixture.precision.d3(), &name);
    println!("{name}: relative max error {err:.3e}");
    err
}

/// `[B, C, s..]` ↔ `[B, s.., C]` (the fixtures are channels-first, the FNO channels-last).
fn to_last<const R: usize>(x: Tensor<R>) -> Tensor<R> {
    x.permute(core::array::from_fn(|i| match i {
        0 => 0,
        i if i == R - 1 => 1,
        i => i + 1,
    }))
}
fn to_first<const R: usize>(x: Tensor<R>) -> Tensor<R> {
    x.permute(core::array::from_fn(|i| match i {
        0 => 0,
        1 => R - 1,
        i => i - 1,
    }))
}

/// An `FNO` fixture with `n_layers=0`: `config` (zero layers) loaded through 2.0.0's
/// `lifting.fcs.*` and `projection.fcs.*`.
fn fno<const R: usize>(stem: &str, config: FNOConfig, p: Precision, device: &Device) -> f64 {
    let mut fixture = open(stem, p);
    let mut fno = config.with_n_layers(0).init::<R>(device);
    let two_layer_lifting = fixture.params.iter().any(|n| n == "lifting.fcs.1.weight");
    let mut map: Vec<MapEntry> = vec![
        ("lifting.fcs.0.weight", "fc0.weight", Take::Transposed),
        ("lifting.fcs.0.bias", "fc0.bias", Take::Whole),
        ("projection.fcs.0.weight", "fc1.weight", Take::Transposed),
        ("projection.fcs.0.bias", "fc1.bias", Take::Whole),
        ("projection.fcs.1.weight", "fc2.weight", Take::Transposed),
        ("projection.fcs.1.bias", "fc2.bias", Take::Whole),
    ];
    if two_layer_lifting {
        map.push((
            "lifting.fcs.1.weight",
            "lifting_out.0.weight",
            Take::Transposed,
        ));
        map.push(("lifting.fcs.1.bias", "lifting_out.0.bias", Take::Whole));
    }
    load_into(&mut fno, &mut fixture, &map);
    let y = fno.forward(to_last(fixture.input::<R>(device)));
    check(to_first(y), &mut fixture)
}

/// 2.0.0's `FNO` defaults for this task's parts: GELU, ratios 2, the unit grid.
fn defaults(modes: Vec<usize>, data: usize, out: usize, hidden: usize) -> FNOConfig {
    let d = modes.len();
    FNOConfig::new(modes, data, out)
        .with_hidden_channels(hidden)
        .with_activation(Some(Activation::Gelu))
        .with_lifting_channel_ratio(Some(2.0))
        .with_projection_channel_ratio(Some(2.0))
        .with_positional_embedding(Some(PositionalEmbedding::GridNd(GridEmbeddingNd::unit(d))))
}

pub fn fno_lift_project1d(p: Precision, device: &Device) -> f64 {
    fno::<3>("fno_lift_project1d", defaults(vec![4], 2, 1, 6), p, device)
}

/// Lifting ratio 0 (one layer), projection ratio 1.5 of 5 (7 wide), tanh.
pub fn fno_lift_project2d_linear(p: Precision, device: &Device) -> f64 {
    let config = defaults(vec![2, 2], 2, 3, 5)
        .with_activation(Some(Activation::Tanh))
        .with_lifting_channel_ratio(Some(0.0))
        .with_projection_channel_ratio(Some(1.5));
    fno::<4>("fno_lift_project2d_linear", config, p, device)
}

pub fn fno_lift_project3d_no_grid(p: Precision, device: &Device) -> f64 {
    let config = defaults(vec![2, 2, 2], 3, 2, 4)
        .with_positional_embedding(Some(PositionalEmbedding::Disabled));
    fno::<5>("fno_lift_project3d_no_grid", config, p, device)
}

pub fn fno_lift_project2d_boundaries(p: Precision, device: &Device) -> f64 {
    let grid = GridEmbeddingNd::new(BOUNDARIES.to_vec());
    let config = defaults(vec![2, 2], 2, 1, 4)
        .with_positional_embedding(Some(PositionalEmbedding::GridNd(grid)));
    fno::<4>("fno_lift_project2d_boundaries", config, p, device)
}

pub fn grid_embedding2d(p: Precision, device: &Device) -> f64 {
    let mut fixture = open("grid_embedding2d", p);
    let y = GridEmbeddingNd::new(BOUNDARIES.to_vec()).forward(fixture.input::<4>(device));
    check(y, &mut fixture)
}

fn sinusoidal(stem: &str, kind: SinusoidalKind, p: Precision, device: &Device) -> f64 {
    let mut fixture = open(stem, p);
    let y = SinusoidalEmbedding::new(3, 4)
        .with_kind(kind)
        .forward(fixture.input::<3>(device));
    check(y, &mut fixture)
}

pub fn sinusoidal_transformer(p: Precision, device: &Device) -> f64 {
    let kind = SinusoidalKind::Transformer {
        max_positions: 10000.0,
    };
    sinusoidal("sinusoidal_transformer", kind, p, device)
}

pub fn sinusoidal_nerf(p: Precision, device: &Device) -> f64 {
    sinusoidal("sinusoidal_nerf", SinusoidalKind::Nerf, p, device)
}

/// `generate.py`'s `Rotary2d(8)`: channels 0 and 1 the coordinates, the rest the features.
pub fn rotary2d(p: Precision, device: &Device) -> f64 {
    let mut fixture = open("rotary2d", p);
    let x = fixture.input::<3>(device);
    let [b, n, c] = x.dims();
    let coordinate = |k| x.clone().narrow(2, k, 1).reshape([b, n]);
    let rope = RotaryEmbedding2d::new(8);
    let (fx, fy) = (rope.forward(coordinate(0)), rope.forward(coordinate(1)));
    let y = apply_2d_rotary_pos_emb(x.clone().narrow(2, 2, c - 2), fx, fy);
    check(y, &mut fixture)
}

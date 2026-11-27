//! The FNO family's shared parity pieces (library CM1.5a): 2.0.0's `FNO` defaults as
//! `FNOConfig` options, and the maps of its lifting and projection and of its block parts,
//! included with `#[path]` by `parity_fno_blocks`, `parity_embeddings*` and `parity_fno*`.
//! Each binary uses part of it.

use burn::prelude::*;
use sciml_rs::neural_operators::{
    layers::{embedding::GridEmbeddingNd, norm::NormKind, skip::SkipKind},
    models::fno::{Activation, BlockMlpConfig, FNOConfig, PositionalEmbedding},
};

use crate::parity_common::{MapEntry, Take};

/// 2.0.0's `FNOBlocks` defaults: GELU, the linear (bias-free) FNO skip, the channel MLP of
/// expansion 0.5 with a soft-gating skip.
#[allow(dead_code, reason = "each FNO-family binary uses part of this map")]
pub fn block_defaults(c: FNOConfig) -> FNOConfig {
    c.with_activation(Some(Activation::Gelu))
        .with_fno_skip(Some(SkipKind::Linear))
        .with_channel_mlp(Some(BlockMlpConfig::new()))
}

/// 2.0.0's `FNO` defaults for the lifting and projection: GELU, ratios 2, the unit grid.
#[allow(dead_code, reason = "each FNO-family binary uses part of this map")]
pub fn lift_defaults(c: FNOConfig) -> FNOConfig {
    let d = c.modes.len();
    c.with_activation(Some(Activation::Gelu))
        .with_lifting_channel_ratio(Some(2.0))
        .with_projection_channel_ratio(Some(2.0))
        .with_positional_embedding(Some(PositionalEmbedding::GridNd(GridEmbeddingNd::unit(d))))
}

/// `[B, C, s..]` → `[B, s.., C]`: a channels-first fixture tensor for the FNO.
#[allow(dead_code, reason = "each FNO-family binary uses part of this map")]
pub fn to_channels_last<const R: usize>(x: Tensor<R>) -> Tensor<R> {
    x.permute(core::array::from_fn(|i| match i {
        0 => 0,
        i if i == R - 1 => 1,
        i => i + 1,
    }))
}

/// `[B, s.., C]` → `[B, C, s..]`, the inverse of [`to_channels_last`].
#[allow(dead_code, reason = "each FNO-family binary uses part of this map")]
pub fn to_channels_first<const R: usize>(x: Tensor<R>) -> Tensor<R> {
    x.permute(core::array::from_fn(|i| match i {
        0 => 0,
        1 => R - 1,
        i => i - 1,
    }))
}

/// A map path built at run time. Test-only: leaked, so the harness's `&'static str`
/// entries can be built in a loop.
#[allow(dead_code, reason = "each FNO-family binary uses part of this map")]
pub fn leak(path: String) -> &'static str {
    path.leak()
}

/// 2.0.0's `FNO` lifting and projection (`lifting.fcs.*`, `projection.fcs.*`) to the
/// crate's `fc0`, `lifting_out.0` (when `config`'s lifting ratio gives a second layer, as
/// [`FNOConfig::init`] decides), `fc1`, `fc2`, under the Burn path prefix `to` (library
/// CM1.4).
#[allow(dead_code, reason = "each FNO-family binary uses part of this map")]
pub fn lift_project(config: &FNOConfig, to: &str) -> Vec<MapEntry> {
    let mut layers = vec![
        ("lifting.fcs.0", "fc0"),
        ("projection.fcs.0", "fc1"),
        ("projection.fcs.1", "fc2"),
    ];
    let width = |r: f64| (r * config.hidden_channels as f64) as usize;
    if config.lifting_channel_ratio.is_some_and(|r| width(r) > 0) {
        layers.push(("lifting.fcs.1", "lifting_out.0"));
    }
    layers
        .into_iter()
        .flat_map(|(from, at)| {
            [
                (
                    leak(format!("{from}.weight")),
                    leak(format!("{to}{at}.weight")),
                    Take::Transposed,
                ),
                (
                    leak(format!("{from}.bias")),
                    leak(format!("{to}{at}.bias")),
                    Take::Whole,
                ),
            ]
        })
        .collect()
}

/// The parts of 2.0.0's `FNOBlocks` (`fno_blocks.*`) other than the spectral convolutions,
/// for the block options of `options`, to the crate's `FNO` under the Burn path prefix `to`:
/// the FNO skips, channel MLPs and their skips, and the norms (library CM1.3b). Burn names
/// an enum module's parameters after the variant (`Linear`, `SoftGating`, `Group`, `AdaIn`).
#[allow(dead_code, reason = "each FNO-family binary uses part of this map")]
pub fn block_parts(options: &FNOConfig, to: &str) -> Vec<MapEntry> {
    let mut m = Vec::new();
    let skip = |m: &mut Vec<MapEntry>, from: String, to: String, kind: &Option<SkipKind>| {
        match kind {
            // 2.0.0's linear skip is a `Flattened1dConv`, its weight `conv.weight`.
            Some(SkipKind::Linear) => m.push((
                leak(format!("{from}.conv.weight")),
                leak(format!("{to}.Linear.weight")),
                Take::Whole,
            )),
            Some(SkipKind::SoftGating) => m.push((
                leak(format!("{from}.weight")),
                leak(format!("{to}.SoftGating")),
                Take::Whole,
            )),
            Some(SkipKind::Identity) | None => {}
        }
    };
    for l in 0..options.n_layers {
        skip(
            &mut m,
            format!("fno_blocks.fno_skips.{l}"),
            format!("{to}fno_skips.{l}"),
            &options.fno_skip,
        );
        if let Some(mlp) = &options.channel_mlp {
            for k in 0..2 {
                for p in ["weight", "bias"] {
                    let path = format!("channel_mlp.{l}.fcs.{k}.{p}");
                    m.push((
                        leak(format!("fno_blocks.{path}")),
                        leak(format!("{to}{path}")),
                        Take::Whole,
                    ));
                }
            }
            skip(
                &mut m,
                format!("fno_blocks.channel_mlp_skips.{l}"),
                format!("{to}channel_mlp_skips.{l}"),
                &mlp.skip,
            );
        }
    }
    for k in 0..2 * options.n_layers {
        let (from, to) = (format!("fno_blocks.norm.{k}"), format!("{to}norms.{k}"));
        match options.norm {
            Some(NormKind::Group) => {
                m.push((
                    leak(format!("{from}.weight")),
                    leak(format!("{to}.Group.gamma")),
                    Take::Whole,
                ));
                m.push((
                    leak(format!("{from}.bias")),
                    leak(format!("{to}.Group.beta")),
                    Take::Whole,
                ));
            }
            Some(NormKind::AdaIn { .. }) => {
                for (t, b) in [(0, "fc_hidden"), (2, "fc_out")] {
                    let (f, t) = (format!("{from}.mlp.{t}"), format!("{to}.AdaIn.{b}"));
                    m.push((
                        leak(format!("{f}.weight")),
                        leak(format!("{t}.weight")),
                        Take::Transposed,
                    ));
                    m.push((
                        leak(format!("{f}.bias")),
                        leak(format!("{t}.bias")),
                        Take::Whole,
                    ));
                }
            }
            Some(NormKind::Instance) | None => {}
        }
    }
    m
}

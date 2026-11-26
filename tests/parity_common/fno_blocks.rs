//! Library CM1.3b: 2.0.0's `FNOBlocks` against the FNO's Fourier layers built from the
//! block options (`FNO::fourier_layers`), on the `fno_blocks_*` fixtures of
//! `tools/parity/generate.py`. 2.0.0's block runs the legacy corner-layout convolution without
//! its bias there (the crate's `SpectralConv` has none). The fixtures hold no lifting or
//! projection, so only the FNO's block parameters are loaded and checked complete.

use burn::{module::Param, prelude::*, store::PathFilter};
use sciml_rs::neural_operators::{
    layers::{norm::NormKind, skip::SkipKind},
    models::fno::{Activation, BlockMlpConfig, FNO, FNOConfig},
};

use crate::parity_common::{Fixture, MapEntry, Take, block_parts, check_d3, leak, load_into_part};

/// The FNO and, for AdaIN, the fixture's embedding as a test-only parameter.
#[derive(Module, Debug)]
pub struct Blocks<const R: usize> {
    pub fno: FNO<R>,
    pub embedding: Option<Param<Tensor<1>>>,
}

/// A fixture's configuration, as `generate.py`'s `CONFIGS`.
pub struct Case {
    /// The fixture's name without `_f32`.
    pub stem: &'static str,
    /// Channels in and out of every layer (2.0.0's `in_channels = out_channels`).
    pub channels: usize,
    pub n_layers: usize,
    /// 2.0.0's `n_modes`; the crate keeps half per axis (CM1.1).
    pub n_modes: &'static [usize],
    /// The block options on top of `FNOConfig`'s; the map follows them.
    pub options: fn(FNOConfig) -> FNOConfig,
}

/// 2.0.0's defaults: GELU, the linear (bias-free) FNO skip, the channel MLP of expansion 0.5
/// with a soft-gating skip.
fn defaults(c: FNOConfig) -> FNOConfig {
    c.with_activation(Some(Activation::Gelu))
        .with_fno_skip(Some(SkipKind::Linear))
        .with_channel_mlp(Some(BlockMlpConfig::new()))
}

fn pre(c: FNOConfig) -> FNOConfig {
    c.with_preactivation(Some(true))
}

fn instance(c: FNOConfig) -> FNOConfig {
    defaults(c).with_norm(Some(NormKind::Instance))
}

fn ada_in(c: FNOConfig) -> FNOConfig {
    defaults(c).with_norm(Some(NormKind::AdaIn { embed_dim: 2 }))
}

pub const POST_1D: Case = Case {
    stem: "fno_blocks_post1d",
    channels: 4,
    n_layers: 2,
    n_modes: &[8],
    options: defaults,
};
pub const PRE_1D: Case = Case {
    stem: "fno_blocks_pre1d",
    options: |c| pre(defaults(c)),
    ..POST_1D
};
pub const INSTANCE_POST_2D: Case = Case {
    stem: "fno_blocks_instance_post2d",
    channels: 4,
    n_layers: 2,
    n_modes: &[6, 4],
    options: instance,
};
pub const INSTANCE_PRE_2D: Case = Case {
    stem: "fno_blocks_instance_pre2d",
    options: |c| pre(instance(c)),
    ..INSTANCE_POST_2D
};
pub const GROUP_POST_3D: Case = Case {
    stem: "fno_blocks_group_post3d",
    channels: 4,
    n_layers: 2,
    n_modes: &[4, 4, 4],
    options: |c| {
        defaults(c)
            .with_norm(Some(NormKind::Group))
            .with_fno_skip(Some(SkipKind::SoftGating))
            .with_channel_mlp(Some(
                BlockMlpConfig::new()
                    .with_expansion(1.0)
                    .with_skip(Some(SkipKind::Linear)),
            ))
    },
};
pub const GROUP_PRE_3D: Case = Case {
    stem: "fno_blocks_group_pre3d",
    options: |c| {
        pre(defaults(c))
            .with_norm(Some(NormKind::Group))
            .with_fno_skip(Some(SkipKind::Identity))
            .with_channel_mlp(Some(
                BlockMlpConfig::new().with_skip(Some(SkipKind::Identity)),
            ))
    },
    ..GROUP_POST_3D
};
pub const ADA_IN_POST_2D: Case = Case {
    stem: "fno_blocks_ada_in_post2d",
    channels: 2,
    n_layers: 1,
    n_modes: &[6, 4],
    options: ada_in,
};
pub const ADA_IN_PRE_2D: Case = Case {
    stem: "fno_blocks_ada_in_pre2d",
    options: |c| pre(ada_in(c)),
    ..ADA_IN_POST_2D
};
/// No channel MLP, no norm, tanh: 2.0.0 applies the activation twice per layer.
pub const PLAIN_1D: Case = Case {
    stem: "fno_blocks_plain1d",
    options: |c| {
        c.with_activation(Some(Activation::Tanh))
            .with_fno_skip(Some(SkipKind::SoftGating))
    },
    ..POST_1D
};

/// The crate's corner of 2.0.0's legacy weight `i`: 2.0.0 runs its `2^(D-1)` corners in
/// `itertools.product` order, axis 0 most significant; the crate has axis `j` in bit `j`
/// (CONVENTIONS §5), so the bits reverse.
fn corner(i: usize, d: usize) -> usize {
    (0..d - 1).fold(0, |c, j| c | ((i >> (d - 2 - j)) & 1) << j)
}

/// The map, for the block `options` build, from 2.0.0's `FNOBlocksRun` names to the
/// `Blocks` paths: the legacy convolution's corners, then the shared block parts.
fn map(d: usize, options: &FNOConfig) -> Vec<MapEntry> {
    let mut m = Vec::new();
    for l in 0..options.n_layers {
        for i in 0..1 << (d - 1) {
            let from = leak(format!("fno_blocks.convs.{l}.weight.{i}.tensor"));
            let c = corner(i, d);
            m.push((from, leak(format!("fno.conv.{l}.weights_re.{c}")), Take::Re));
            m.push((from, leak(format!("fno.conv.{l}.weights_im.{c}")), Take::Im));
        }
    }
    m.extend(block_parts(options, "fno."));
    if embed_dim(options).is_some() {
        m.push(("embedding", "embedding", Take::Whole));
    }
    m
}

/// AdaIN's embedding length, when the norm is AdaIN.
fn embed_dim(options: &FNOConfig) -> Option<usize> {
    match options.norm {
        Some(NormKind::AdaIn { embed_dim }) => Some(embed_dim),
        _ => None,
    }
}

/// Runs `case`'s f32 fixture through the FNO's Fourier layers and returns the relative max
/// error, asserted within D3 and printed for the record.
///
/// # Panics
/// If the load is incomplete or the error exceeds D3.
pub fn check<const R: usize>(case: &Case, device: &Device) -> f64 {
    let d = R - 2;
    let modes: Vec<usize> = case.n_modes.iter().map(|m| m / 2).collect();
    let config = (case.options)(
        FNOConfig::new(modes, 1, 1)
            .with_hidden_channels(case.channels)
            .with_n_layers(case.n_layers),
    );
    let mut blocks = Blocks::<R> {
        fno: config.init(device),
        embedding: embed_dim(&config).map(|e| Param::from_tensor(Tensor::zeros([e], device))),
    };
    let name = format!("{}_f32", case.stem);
    let mut fixture = Fixture::open(&name);
    // Every block path; the lifting `fc0` and projection `fc1`, `fc2` are not in the fixture.
    let filter = PathFilter::new()
        .with_regex(r"^(embedding|fno\.(conv|w|fno_skips|channel_mlp|channel_mlp_skips|norms)\.)");
    load_into_part(&mut blocks, &mut fixture, &map(d, &config), Some(filter));
    let embedding = blocks.embedding.as_ref().map(Param::val);
    let out = blocks
        .fno
        .fourier_layers(fixture.input::<R>(device), embedding);
    check_d3(out, &mut fixture)
}

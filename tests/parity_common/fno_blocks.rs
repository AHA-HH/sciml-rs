//! Library CM1.3b: 2.0.0's `FNOBlocks` against the FNO's Fourier layers built from the
//! block options (`FNO::fourier_layers`), on the `fno_blocks_*` fixtures of
//! `tools/parity/generate.py`. 2.0.0's block runs the legacy corner-layout convolution without
//! its bias there (the crate's `SpectralConv` has none). The fixtures hold no lifting or
//! projection, so only the FNO's block parameters are loaded and checked complete.

use burn::{module::Param, nn::activation::ActivationConfig, prelude::*, store::PathFilter};
use sciml_rs::neural_operators::{
    layers::{norm::NormKind, skip::SkipKind},
    models::fno::{BlockMlpConfig, FNO, FNOConfig},
};

use crate::parity_common::{Fixture, MapEntry, Take, assert_parity, load_into_part};

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
    /// The block options on top of `FNOConfig`'s.
    pub options: fn(FNOConfig) -> FNOConfig,
    /// The norm, for the map.
    pub norm: Option<NormKind>,
    /// AdaIN's `ada_in_features`, when the norm is AdaIN.
    pub embed_dim: Option<usize>,
}

/// 2.0.0's defaults: GELU, the linear (bias-free) FNO skip, the channel MLP of expansion 0.5
/// with a soft-gating skip.
fn defaults(c: FNOConfig) -> FNOConfig {
    c.with_activation(Some(ActivationConfig::Gelu))
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
    norm: None,
    embed_dim: None,
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
    norm: Some(NormKind::Instance),
    embed_dim: None,
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
    norm: Some(NormKind::Group),
    embed_dim: None,
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
    norm: Some(NormKind::AdaIn { embed_dim: 2 }),
    embed_dim: Some(2),
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
        c.with_activation(Some(ActivationConfig::Tanh))
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

/// `case`'s map from 2.0.0's `FNOBlocksRun` names to the `Blocks` paths. Burn names an enum
/// module's parameters after the variant (`Linear`, `SoftGating`, `Group`, `AdaIn`).
fn map(case: &Case, d: usize, options: &FNOConfig) -> Vec<MapEntry> {
    // Test-only: leaked so the harness's `&'static str` entries can be built in a loop.
    let s = |x: String| -> &'static str { x.leak() };
    let mut m = Vec::new();
    let skip = |m: &mut Vec<MapEntry>, from: String, to: String, kind: &Option<SkipKind>| {
        match kind {
            // 2.0.0's linear skip is a `Flattened1dConv`, its weight `conv.weight`.
            Some(SkipKind::Linear) => m.push((
                s(format!("{from}.conv.weight")),
                s(format!("{to}.Linear.weight")),
                Take::Whole,
            )),
            Some(SkipKind::SoftGating) => m.push((
                s(format!("{from}.weight")),
                s(format!("{to}.SoftGating")),
                Take::Whole,
            )),
            Some(SkipKind::Identity) | None => {}
        }
    };
    for l in 0..case.n_layers {
        for i in 0..1 << (d - 1) {
            let from = s(format!("fno_blocks.convs.{l}.weight.{i}.tensor"));
            let c = corner(i, d);
            m.push((from, s(format!("fno.conv.{l}.weights_re.{c}")), Take::Re));
            m.push((from, s(format!("fno.conv.{l}.weights_im.{c}")), Take::Im));
        }
        skip(
            &mut m,
            format!("fno_blocks.fno_skips.{l}"),
            format!("fno.fno_skips.{l}"),
            &options.fno_skip,
        );
        if let Some(mlp) = &options.channel_mlp {
            for k in 0..2 {
                for p in ["weight", "bias"] {
                    let path = format!("channel_mlp.{l}.fcs.{k}.{p}");
                    m.push((
                        s(format!("fno_blocks.{path}")),
                        s(format!("fno.{path}")),
                        Take::Whole,
                    ));
                }
            }
            skip(
                &mut m,
                format!("fno_blocks.channel_mlp_skips.{l}"),
                format!("fno.channel_mlp_skips.{l}"),
                &mlp.skip,
            );
        }
    }
    for k in 0..2 * case.n_layers {
        let (from, to) = (format!("fno_blocks.norm.{k}"), format!("fno.norms.{k}"));
        match case.norm {
            Some(NormKind::Group) => {
                m.push((
                    s(format!("{from}.weight")),
                    s(format!("{to}.Group.gamma")),
                    Take::Whole,
                ));
                m.push((
                    s(format!("{from}.bias")),
                    s(format!("{to}.Group.beta")),
                    Take::Whole,
                ));
            }
            Some(NormKind::AdaIn { .. }) => {
                for (t, b) in [(0, "fc_hidden"), (2, "fc_out")] {
                    let (f, t) = (format!("{from}.mlp.{t}"), format!("{to}.AdaIn.{b}"));
                    m.push((
                        s(format!("{f}.weight")),
                        s(format!("{t}.weight")),
                        Take::Transposed,
                    ));
                    m.push((s(format!("{f}.bias")), s(format!("{t}.bias")), Take::Whole));
                }
            }
            Some(NormKind::Instance) | None => {}
        }
    }
    if case.embed_dim.is_some() {
        m.push(("embedding", "embedding", Take::Whole));
    }
    m
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
        embedding: case
            .embed_dim
            .map(|e| Param::from_tensor(Tensor::zeros([e], device))),
    };
    let name = format!("{}_f32", case.stem);
    let mut fixture = Fixture::open(&name);
    // Every block path; the lifting `fc0` and projection `fc1`, `fc2` are not in the fixture.
    let filter = PathFilter::new()
        .with_regex(r"^(embedding|fno\.(conv|w|fno_skips|channel_mlp|channel_mlp_skips|norms)\.)");
    load_into_part(
        &mut blocks,
        &mut fixture,
        &map(case, d, &config),
        Some(filter),
    );
    let embedding = blocks.embedding.as_ref().map(Param::val);
    let out = blocks
        .fno
        .fourier_layers(fixture.input::<R>(device), embedding);
    let err = assert_parity(out, &fixture.output(), fixture.precision.d3(), &name);
    println!("{name}: relative max error {err:.3e}");
    err
}

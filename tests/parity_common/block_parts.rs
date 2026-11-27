//! Library CM1.3a: the block parts of 2.0.0's `FNOBlocks` against their 2.0.0 fixtures
//! (`tools/parity/generate.py`, `channel_mlp1d` … `ada_in2d`): the crate's `ChannelMlp`,
//! `Skip` and `AdaIn`, and Burn's `InstanceNorm` and `GroupNorm` as 2.0.0's block builds
//! them. Each case builds its part, loads the fixture through a hand-written map and
//! compares the output with the fixture's.

use burn::{
    module::Param,
    nn::{
        GroupNorm, GroupNormConfig, InstanceNorm, InstanceNormConfig, activation::ActivationConfig,
    },
    prelude::*,
    store::ModuleSnapshot,
};
use sciml_rs::neural_operators::layers::{
    channel_mlp::{ChannelMlp, ChannelMlpConfig},
    norm::{AdaIn, AdaInConfig},
    skip::{Skip, SkipKind},
};

use crate::parity_common::{Fixture, MapEntry, Precision, Take, check_d3, load_into};

/// 2.0.0's `AdaIN` with its embedding as a test-only parameter, so the fixture's
/// `embedding` loads through the map like a weight.
#[derive(Module, Debug)]
pub struct AdaInCase {
    pub norm: AdaIn,
    pub embedding: Param<Tensor<1>>,
}

// Each case: fixture `<case>_{f32,f64}` in `p`'s precision on `device`; returns the relative
// max error, asserted within D3. Panics if the load is incomplete or the error exceeds D3.

/// `ChannelMLP(4, hidden=2)`: GELU, two layers, out = in; 1D.
pub fn channel_mlp1d(p: Precision, device: &Device) -> f64 {
    let mlp = ChannelMlpConfig::new(4, 4, 2).init(device);
    run::<_, 3>(
        "channel_mlp1d",
        p,
        mlp,
        &FCS[..4],
        device,
        ChannelMlp::forward,
    )
}

/// `ChannelMLP(3, out=5, hidden=4, n_layers=3, non_linearity=torch.tanh)`; 3D.
pub fn channel_mlp3d(p: Precision, device: &Device) -> f64 {
    let mlp = ChannelMlpConfig::new(3, 5, 4)
        .with_n_layers(3)
        .with_activation(ActivationConfig::Tanh)
        .init(device);
    run::<_, 5>("channel_mlp3d", p, mlp, FCS, device, ChannelMlp::forward)
}

/// `skip_connection(3, 5, n_dim=3, skip_type="linear")`. Burn names an enum module's
/// parameters after the variant.
pub fn linear_skip3d(p: Precision, device: &Device) -> f64 {
    let skip = SkipKind::Linear.init::<5>(3, 5, device);
    let map = &[("conv.weight", "Linear.weight", Take::Whole)];
    run::<_, 5>("linear_skip3d", p, skip, map, device, Skip::forward)
}

/// `skip_connection(4, 4, n_dim=2, skip_type="soft-gating")`, weight redrawn.
pub fn soft_gating2d(p: Precision, device: &Device) -> f64 {
    let skip = SkipKind::SoftGating.init::<4>(4, 4, device);
    let map = &[("weight", "SoftGating", Take::Whole)];
    run::<_, 4>("soft_gating2d", p, skip, map, device, Skip::forward)
}

/// 2.0.0's `InstanceNorm()`: no affine; 1D.
pub fn instance_norm1d(p: Precision, device: &Device) -> f64 {
    let norm = InstanceNormConfig::new(4).with_affine(false).init(device);
    run::<_, 3>(
        "instance_norm1d",
        p,
        norm,
        &[],
        device,
        InstanceNorm::forward,
    )
}

/// `nn.GroupNorm(1, 4)`, as `FNOBlocks` builds it; affine redrawn; 3D.
pub fn group_norm3d(p: Precision, device: &Device) -> f64 {
    let norm = GroupNormConfig::new(1, 4).init(device);
    let map = &[
        ("weight", "gamma", Take::Whole),
        ("bias", "beta", Take::Whole),
    ];
    run::<_, 5>("group_norm3d", p, norm, map, device, GroupNorm::forward)
}

/// `AdaIN(embed_dim=3, in_channels=4)`, its embedding in the fixture; 2D.
pub fn ada_in2d(p: Precision, device: &Device) -> f64 {
    let case = AdaInCase {
        norm: AdaInConfig::new(3, 4).init(device),
        embedding: Param::from_tensor(Tensor::zeros([3], device)),
    };
    run::<_, 4>("ada_in2d", p, case, ADA_IN, device, |c, x| {
        c.norm.forward(x, c.embedding.val())
    })
}

/// Opens `<case>_{f32,f64}`, loads it into `module` through `map`, runs `forward` on its
/// input and returns the relative max error against its output, asserted within D3 and
/// printed for the record.
fn run<M: ModuleSnapshot, const R: usize>(
    case: &str,
    precision: Precision,
    mut module: M,
    map: &[MapEntry],
    device: &Device,
    forward: impl Fn(&M, Tensor<R>) -> Tensor<R>,
) -> f64 {
    let mut fixture = Fixture::open_as(case, precision);
    load_into(&mut module, &mut fixture, map);
    let out = forward(&module, fixture.input::<R>(device));
    check_d3(out, &mut fixture)
}

/// A `ChannelMLP`'s layers: 2.0.0's `fcs.k` are the crate's `fcs.k`. Two layers take the
/// first four entries.
#[rustfmt::skip]
const FCS: &[MapEntry] = &[
    ("fcs.0.weight", "fcs.0.weight", Take::Whole),
    ("fcs.0.bias", "fcs.0.bias", Take::Whole),
    ("fcs.1.weight", "fcs.1.weight", Take::Whole),
    ("fcs.1.bias", "fcs.1.bias", Take::Whole),
    ("fcs.2.weight", "fcs.2.weight", Take::Whole),
    ("fcs.2.bias", "fcs.2.bias", Take::Whole),
];

/// `AdaIN`: 2.0.0's `mlp` is `Sequential(Linear, GELU, Linear)`, so its linears are 0 and 2;
/// torch's `Linear` weights are `[out, in]`, Burn's `[in, out]`.
#[rustfmt::skip]
const ADA_IN: &[MapEntry] = &[
    ("mlp.0.weight", "norm.fc_hidden.weight", Take::Transposed),
    ("mlp.0.bias", "norm.fc_hidden.bias", Take::Whole),
    ("mlp.2.weight", "norm.fc_out.weight", Take::Transposed),
    ("mlp.2.bias", "norm.fc_out.bias", Take::Whole),
    ("embedding", "embedding", Take::Whole),
];

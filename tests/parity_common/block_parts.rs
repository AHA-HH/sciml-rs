//! Library CM1.3a: the block parts of 2.0.0's `FNOBlocks` against their 2.0.0 fixtures
//! (`tools/parity/generate.py`, `channel_mlp1d` … `ada_in2d`): the crate's `ChannelMlp`,
//! `Skip` and `AdaIn`, and Burn's `InstanceNorm` and `GroupNorm` as 2.0.0's block builds
//! them. Each case builds its part, loads the fixture through a hand-written map and
//! returns the output beside the reference.

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

use crate::parity_common::{Fixture, MapEntry, Precision, Take, assert_parity, load_into};

/// One part and its fixture, `<name>_{f32,f64}`.
#[derive(Clone, Copy, Debug)]
pub enum Part {
    /// `ChannelMLP(4, hidden=2)`: GELU, two layers, out = in; 1D.
    ChannelMlp1d,
    /// `ChannelMLP(3, out=5, hidden=4, n_layers=3, non_linearity=torch.tanh)`; 3D.
    ChannelMlp3d,
    /// `skip_connection(3, 5, n_dim=3, skip_type="linear")`.
    LinearSkip3d,
    /// `skip_connection(4, 4, n_dim=2, skip_type="soft-gating")`, weight redrawn.
    SoftGating2d,
    /// 2.0.0's `InstanceNorm()`: no affine.
    InstanceNorm1d,
    /// `nn.GroupNorm(1, 4)`, as `FNOBlocks` builds it; affine redrawn.
    GroupNorm3d,
    /// `AdaIN(embed_dim=3, in_channels=4)`, its embedding in the fixture.
    AdaIn2d,
}

impl Part {
    /// The fixture's name without the precision suffix.
    pub fn name(self) -> &'static str {
        match self {
            Part::ChannelMlp1d => "channel_mlp1d",
            Part::ChannelMlp3d => "channel_mlp3d",
            Part::LinearSkip3d => "linear_skip3d",
            Part::SoftGating2d => "soft_gating2d",
            Part::InstanceNorm1d => "instance_norm1d",
            Part::GroupNorm3d => "group_norm3d",
            Part::AdaIn2d => "ada_in2d",
        }
    }
}

/// 2.0.0's `AdaIN` with its embedding as a test-only parameter, so the fixture's
/// `embedding` loads through the map like a weight.
#[derive(Module, Debug)]
pub struct AdaInCase {
    pub norm: AdaIn,
    pub embedding: Param<Tensor<1>>,
}

/// The relative max error of `part`'s fixture in `precision` on `device`, asserted within
/// D3 and printed for the record.
///
/// # Panics
/// If the load is incomplete or the error exceeds D3.
pub fn check(part: Part, precision: Precision, device: &Device) -> f64 {
    let suffix = match precision {
        Precision::F32 => "f32",
        Precision::F64 => "f64",
    };
    let name = format!("{}_{suffix}", part.name());
    let mut fixture = Fixture::open(&name);
    let err = match part {
        Part::ChannelMlp1d => {
            let mlp = ChannelMlpConfig::new(4, 4, 2).init(device);
            run::<_, 3>(&name, mlp, &mut fixture, MLP2, device, ChannelMlp::forward)
        }
        Part::ChannelMlp3d => {
            let mlp = ChannelMlpConfig::new(3, 5, 4)
                .with_n_layers(3)
                .with_activation(ActivationConfig::Tanh)
                .init(device);
            run::<_, 5>(&name, mlp, &mut fixture, MLP3, device, ChannelMlp::forward)
        }
        Part::LinearSkip3d => {
            let skip = SkipKind::Linear.init::<5>(3, 5, device);
            let map = &[("conv.weight", "Linear.weight", Take::Whole)];
            run::<_, 5>(&name, skip, &mut fixture, map, device, Skip::forward)
        }
        Part::SoftGating2d => {
            let skip = SkipKind::SoftGating.init::<4>(4, 4, device);
            let map = &[("weight", "SoftGating", Take::Whole)];
            run::<_, 4>(&name, skip, &mut fixture, map, device, Skip::forward)
        }
        Part::InstanceNorm1d => {
            let norm = InstanceNormConfig::new(4).with_affine(false).init(device);
            run::<_, 3>(
                &name,
                norm,
                &mut fixture,
                &[],
                device,
                InstanceNorm::forward,
            )
        }
        Part::GroupNorm3d => {
            let norm = GroupNormConfig::new(1, 4).init(device);
            let map = &[
                ("weight", "gamma", Take::Whole),
                ("bias", "beta", Take::Whole),
            ];
            run::<_, 5>(&name, norm, &mut fixture, map, device, GroupNorm::forward)
        }
        Part::AdaIn2d => {
            let case = AdaInCase {
                norm: AdaInConfig::new(3, 4).init(device),
                embedding: Param::from_tensor(Tensor::zeros([3], device)),
            };
            run::<_, 4>(&name, case, &mut fixture, ADA_IN, device, |c, x| {
                c.norm.forward(x, c.embedding.val())
            })
        }
    };
    println!("{name}: relative max error {err:.3e}");
    err
}

/// Loads `fixture` into `module` through `map`, runs `forward` on its input and returns the
/// relative max error against its output, asserted within D3.
fn run<M: ModuleSnapshot, const R: usize>(
    name: &str,
    mut module: M,
    fixture: &mut Fixture,
    map: &[MapEntry],
    device: &Device,
    forward: impl Fn(&M, Tensor<R>) -> Tensor<R>,
) -> f64 {
    load_into(&mut module, fixture, map);
    let out = forward(&module, fixture.input::<R>(device));
    assert_parity(out, &fixture.output(), fixture.precision.d3(), name)
}

/// A two-layer `ChannelMLP`: 2.0.0's `fcs.k` are the crate's `fcs.k`.
#[rustfmt::skip]
const MLP2: &[MapEntry] = &[
    ("fcs.0.weight", "fcs.0.weight", Take::Whole),
    ("fcs.0.bias", "fcs.0.bias", Take::Whole),
    ("fcs.1.weight", "fcs.1.weight", Take::Whole),
    ("fcs.1.bias", "fcs.1.bias", Take::Whole),
];

/// A three-layer `ChannelMLP`.
#[rustfmt::skip]
const MLP3: &[MapEntry] = &[
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

//! Library CM1.5a: 2.0.0's `FNO`, end to end, against the crate's `FNO`, in f32 (D3 as
//! amended 2026-11-25): 2.0.0's defaults on a Burgers- and a Darcy-shaped input, and a sweep
//! of its options (`tools/parity/generate.py`, CM1.5a's `SWEEPS`, f32 only), and since CM1.5c
//! batch norm and no FNO skip. The f64
//! fixtures of the defaults are `tests/parity_fno_f64.rs`.

#[path = "parity_common/fno.rs"]
mod fno;
#[path = "parity_common/fno_map.rs"]
mod fno_map;
mod parity_common;

use burn::prelude::*;
use fno::*;
use parity_common::{D3_F32, MapEntry, Precision::F32, rel_max_error};
use sciml_rs::neural_operators::{
    layers::{norm::NormKind, skip::SkipKind},
    models::fno::{Activation, BlockMlpConfig, PositionalEmbedding},
};

/// `FNO(n_modes=(4, 4, 4), 2 → 1, hidden 4, n_layers=2)` on (2, 2, 8, 6, 5).
const FNO3D: Case = Case {
    stem: "fno3d",
    data: 2,
    out: 1,
    hidden: 4,
    n_layers: 2,
    n_modes: &[4, 4, 4],
    max_n_modes: None,
    options: |c| c,
};
/// Instance norm, pre-activation, tanh; `n_modes=(6, 4)`, hidden 6, on (2, 1, 16, 12).
const FNO2D_INSTANCE_PRE: Case = Case {
    stem: "fno2d_instance_pre",
    hidden: 6,
    n_modes: &[6, 4],
    options: |c| {
        c.with_norm(Some(NormKind::Instance))
            .with_preactivation(Some(true))
            .with_activation(Some(Activation::Tanh))
    },
    ..FNO1D
};
/// Group norm, soft-gating FNO skip, linear channel-MLP skip of expansion 1, a one-layer
/// lifting, projection ratio 1.5, no grid, SiLU; `n_modes=(10,)`, 2 → 1, on (2, 2, 40).
const FNO1D_GROUP_SKIPS: Case = Case {
    stem: "fno1d_group_skips",
    data: 2,
    hidden: 6,
    n_modes: &[10],
    options: |c| {
        c.with_norm(Some(NormKind::Group))
            .with_fno_skip(Some(SkipKind::SoftGating))
            .with_channel_mlp(Some(
                BlockMlpConfig::new()
                    .with_expansion(1.0)
                    .with_skip(Some(SkipKind::Linear)),
            ))
            .with_lifting_channel_ratio(Some(0.0))
            .with_projection_channel_ratio(Some(1.5))
            .with_positional_embedding(Some(PositionalEmbedding::Disabled))
            .with_activation(Some(Activation::Silu))
    },
    ..FNO1D
};
/// Separable weights, `n_modes=(6, 6)` of `max_n_modes=(8, 8)`, hidden 6, on (2, 1, 16, 16).
const FNO2D_SEPARABLE: Case = Case {
    stem: "fno2d_separable",
    hidden: 6,
    n_modes: &[6, 6],
    max_n_modes: Some(&[8, 8]),
    options: |c| c.with_separable(Some(true)),
    ..FNO1D
};
/// An odd centred count, `n_modes=(7, 6)` (the corners' zero row), identity FNO skip, no
/// channel-MLP skip, the tanh stabiliser; hidden 6, on (2, 1, 16, 14).
const FNO2D_ODD_STABILIZER: Case = Case {
    stem: "fno2d_odd_stabilizer",
    hidden: 6,
    n_modes: &[7, 6],
    options: |c| {
        c.with_fno_skip(Some(SkipKind::Identity))
            .with_channel_mlp(Some(BlockMlpConfig::new().with_skip(None)))
            .with_tanh_stabilizer(Some(true))
    },
    ..FNO1D
};
/// No channel MLP, pre-activation, the tanh stabiliser, ReLU; `n_modes=(8,)`, hidden 6, on
/// (2, 1, 32).
const FNO1D_PRE_NO_MLP: Case = Case {
    stem: "fno1d_pre_no_mlp",
    hidden: 6,
    options: |c| {
        c.with_channel_mlp(None)
            .with_preactivation(Some(true))
            .with_tanh_stabilizer(Some(true))
            .with_activation(Some(Activation::Relu))
    },
    ..FNO1D
};

/// 2.0.0's defaults with `norm="batch_norm"` (eval: the running statistics, redrawn);
/// `n_modes=(6, 4)`, hidden 6, on (2, 1, 16, 12). Library CM1.5c.
const FNO2D_BATCH_NORM: Case = Case {
    stem: "fno2d_batch_norm",
    hidden: 6,
    n_modes: &[6, 4],
    options: |c| c.with_norm(Some(NormKind::Batch)),
    ..FNO1D
};
/// No FNO skip at all (2.0.0's `fno_skip=None`), batch norm, pre-activation; `n_modes=(8,)`,
/// hidden 6, on (2, 1, 32). Library CM1.5c.
const FNO1D_PRE_NO_SKIP: Case = Case {
    stem: "fno1d_pre_no_skip",
    hidden: 6,
    options: |c| {
        c.with_fno_skip(Some(SkipKind::Disabled))
            .with_norm(Some(NormKind::Batch))
            .with_preactivation(Some(true))
    },
    ..FNO1D
};

#[test]
fn fno1d_matches_2_0_0() {
    check::<3>(&FNO1D, F32, &Device::default());
}

#[test]
fn fno2d_matches_2_0_0() {
    check::<4>(&FNO2D, F32, &Device::default());
}

#[test]
fn fno3d_matches_2_0_0() {
    check::<5>(&FNO3D, F32, &Device::default());
}

#[test]
fn fno2d_instance_pre_matches_2_0_0() {
    check::<4>(&FNO2D_INSTANCE_PRE, F32, &Device::default());
}

#[test]
fn fno1d_group_skips_matches_2_0_0() {
    check::<3>(&FNO1D_GROUP_SKIPS, F32, &Device::default());
}

#[test]
fn fno2d_separable_matches_2_0_0() {
    check::<4>(&FNO2D_SEPARABLE, F32, &Device::default());
}

#[test]
fn fno2d_odd_stabilizer_matches_2_0_0() {
    check::<4>(&FNO2D_ODD_STABILIZER, F32, &Device::default());
}

#[test]
fn fno1d_pre_no_mlp_matches_2_0_0() {
    check::<3>(&FNO1D_PRE_NO_MLP, F32, &Device::default());
}

#[test]
fn fno2d_batch_norm_matches_2_0_0() {
    check::<4>(&FNO2D_BATCH_NORM, F32, &Device::default());
}

#[test]
fn fno1d_pre_no_skip_matches_2_0_0() {
    check::<3>(&FNO1D_PRE_NO_SKIP, F32, &Device::default());
}

/// The batch norm's running statistics are loaded and used: the first two norms' running
/// variances loaded into each other's places break the f32 bound.
#[test]
fn swapped_batch_statistics_fail_parity() {
    let swap = swap_targets("norms.0.Batch.running_var", "norms.1.Batch.running_var");
    let (y, mut fixture) = output::<4>(&FNO2D_BATCH_NORM, F32, swap, &Device::default());
    let err = rel_max_error(y, &fixture.output());
    assert!(err > D3_F32, "swapped statistics went unnoticed: {err:.3e}");
}

/// The map is not passing vacuously: layer 0's two corners loaded into each other's places
/// (the same shape) break the f32 bound.
#[test]
fn swapped_corners_fail_parity() {
    let swap = |map: &mut Vec<MapEntry>| {
        swap_targets("conv.0.weights_re.0", "conv.0.weights_re.1")(map);
        swap_targets("conv.0.weights_im.0", "conv.0.weights_im.1")(map);
    };
    let (y, mut fixture) = output::<4>(&FNO2D, F32, swap, &Device::default());
    let err = rel_max_error(y, &fixture.output());
    assert!(err > D3_F32, "swapped corners went unnoticed: {err:.3e}");
}

/// A map edit loading the parameters bound for `a` and `b` into each other's places.
fn swap_targets(a: &'static str, b: &'static str) -> impl Fn(&mut Vec<MapEntry>) {
    move |map| {
        for (_, to, _) in map.iter_mut() {
            *to = match *to {
                t if t == a => b,
                t if t == b => a,
                other => other,
            };
        }
    }
}

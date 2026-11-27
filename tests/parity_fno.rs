//! Library CM1.5a: 2.0.0's `FNO`, end to end, against the crate's `FNO`, in f32 (D3 as
//! amended 2026-11-25): 2.0.0's defaults on a Burgers- and a Darcy-shaped input, and a sweep
//! of its options (`tools/parity/generate.py`, CM1.5a's `SWEEPS`, f32 only), and since CM1.5c
//! batch norm and no FNO skip. The f64
//! fixtures of the defaults are `tests/parity_fno_f64.rs`. Since CM1.5f also the parity
//! harness's own checks (CM0.5's, on the library `FNO`).

#[path = "parity_common/fno.rs"]
mod fno;
#[path = "parity_common/fno_map.rs"]
mod fno_map;
mod parity_common;

use burn::prelude::*;
use fno::*;
use ndarray::{ArrayD, IxDyn};
use parity_common::{D3_F32, MapEntry, Precision::F32, Take, rel_max_error, take};
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
    output_shape: None,
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

// Library CM1.5b: domain padding and resolution scaling (`generate.py`'s `SWEEPS`, f32).

/// Per layer, every axis scaled by `r`: 2.0.0's `resolution_scaling_factor` as a list of
/// scalars.
fn per_layer(factors: &[f64], d: usize) -> Vec<Vec<f64>> {
    factors.iter().map(|&r| vec![r; d]).collect()
}

/// `domain_padding=0.25`; `n_modes=(8,)`, hidden 6, on (2, 1, 40): 10 cells each side.
const FNO1D_PADDING: Case = Case {
    stem: "fno1d_padding",
    hidden: 6,
    options: |c| c.with_domain_padding(Some(vec![0.25])),
    ..FNO1D
};
/// `domain_padding=[0.2, 0.1]`; `n_modes=(6, 4)`, hidden 6, on (2, 1, 20, 18): 4 and 2.
const FNO2D_PADDING: Case = Case {
    stem: "fno2d_padding",
    hidden: 6,
    n_modes: &[6, 4],
    options: |c| c.with_domain_padding(Some(vec![0.2, 0.1])),
    ..FNO1D
};
/// `domain_padding=0.25` on FNO3D's shape (2, 2, 8, 6, 5): 2, 2 (1.5 rounded to even), 1.
const FNO3D_PADDING: Case = Case {
    stem: "fno3d_padding",
    options: |c| c.with_domain_padding(Some(vec![0.25; 3])),
    ..FNO3D
};
/// `resolution_scaling_factor=[1.5, 1, 0.5, 1]`; `n_modes=(8,)`, hidden 6, on (2, 1, 32):
/// 32 → 48 → 48 → 24 → 24, the skips resampled linearly.
const FNO1D_SCALING: Case = Case {
    stem: "fno1d_scaling",
    hidden: 6,
    options: |c| c.with_resolution_scaling_factor(Some(per_layer(&[1.5, 1.0, 0.5, 1.0], 1))),
    ..FNO1D
};
/// `resolution_scaling_factor=[[2, 1.5], [0.5, 1]]`, two layers; `n_modes=(6, 4)`, hidden 6,
/// on (2, 1, 12, 10): 24 × 15, then 12 × 15, the skips resampled bicubically.
const FNO2D_SCALING: Case = Case {
    stem: "fno2d_scaling",
    hidden: 6,
    n_layers: 2,
    n_modes: &[6, 4],
    options: |c| c.with_resolution_scaling_factor(Some(vec![vec![2.0, 1.5], vec![0.5, 1.0]])),
    ..FNO1D
};
/// `resolution_scaling_factor=[[2, 1, 1.5], [1, 0.5, 1]]` on FNO3D: 16 × 6 × 8 (7.5 rounded
/// to even), then 16 × 3 × 8, the skips resampled in Fourier space.
const FNO3D_SCALING: Case = Case {
    stem: "fno3d_scaling",
    options: |c| {
        c.with_resolution_scaling_factor(Some(vec![vec![2.0, 1.0, 1.5], vec![1.0, 0.5, 1.0]]))
    },
    ..FNO3D
};
/// `domain_padding=0.25`, `resolution_scaling_factor=2`, one layer (the only combination
/// 2.0.0 runs); hidden 6, `n_modes=(6, 4)`, on (2, 1, 12, 10): padded 18 × 14, scaled
/// 36 × 28, cropped by 6 and 4 per side to 24 × 20.
const FNO2D_PADDING_SCALING: Case = Case {
    stem: "fno2d_padding_scaling",
    hidden: 6,
    n_layers: 1,
    n_modes: &[6, 4],
    options: |c| {
        c.with_domain_padding(Some(vec![0.25; 2]))
            .with_resolution_scaling_factor(Some(per_layer(&[2.0], 2)))
    },
    ..FNO1D
};
/// `forward(x, output_shape=(20, 15))`, two layers; hidden 6, `n_modes=(6, 4)`, on
/// (2, 1, 12, 10).
const FNO2D_OUTPUT_SHAPE: Case = Case {
    stem: "fno2d_output_shape",
    hidden: 6,
    n_layers: 2,
    n_modes: &[6, 4],
    output_shape: Some(&[20, 15]),
    ..FNO1D
};

#[test]
fn fno1d_padding_matches_2_0_0() {
    check::<3>(&FNO1D_PADDING, F32, &Device::default());
}

#[test]
fn fno2d_padding_matches_2_0_0() {
    check::<4>(&FNO2D_PADDING, F32, &Device::default());
}

#[test]
fn fno3d_padding_matches_2_0_0() {
    check::<5>(&FNO3D_PADDING, F32, &Device::default());
}

#[test]
fn fno1d_scaling_matches_2_0_0() {
    check::<3>(&FNO1D_SCALING, F32, &Device::default());
}

#[test]
fn fno2d_scaling_matches_2_0_0() {
    check::<4>(&FNO2D_SCALING, F32, &Device::default());
}

#[test]
fn fno3d_scaling_matches_2_0_0() {
    check::<5>(&FNO3D_SCALING, F32, &Device::default());
}

#[test]
fn fno2d_padding_scaling_matches_2_0_0() {
    check::<4>(&FNO2D_PADDING_SCALING, F32, &Device::default());
}

#[test]
fn fno2d_output_shape_matches_2_0_0() {
    check::<4>(&FNO2D_OUTPUT_SHAPE, F32, &Device::default());
}

/// The padding is not passing vacuously: the padded fixture without its padding breaks the
/// f32 bound.
#[test]
fn padding_left_off_fails_parity() {
    let case = Case {
        options: |c| c,
        ..FNO2D_PADDING
    };
    let (y, mut fixture) = load::<4>(&case, F32, &Device::default()).run();
    let err = rel_max_error(y, &fixture.output());
    assert!(err > D3_F32, "the padding went unnoticed: {err:.3e}");
}

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
    let mut loading = load::<4>(&FNO2D_BATCH_NORM, F32, &Device::default());
    swap_targets(
        &mut loading.map,
        "norms.0.Batch.running_var",
        "norms.1.Batch.running_var",
    );
    let (y, mut fixture) = loading.run();
    let err = rel_max_error(y, &fixture.output());
    assert!(err > D3_F32, "swapped statistics went unnoticed: {err:.3e}");
}

/// The map is not passing vacuously: layer 0's two corners loaded into each other's places
/// (the same shape) break the f32 bound.
#[test]
fn swapped_corners_fail_parity() {
    let mut loading = load::<4>(&FNO2D, F32, &Device::default());
    swap_targets(
        &mut loading.map,
        "conv.0.weights_re.0",
        "conv.0.weights_re.1",
    );
    swap_targets(
        &mut loading.map,
        "conv.0.weights_im.0",
        "conv.0.weights_im.1",
    );
    let (y, mut fixture) = loading.run();
    let err = rel_max_error(y, &fixture.output());
    assert!(err > D3_F32, "swapped corners went unnoticed: {err:.3e}");
}

/// Loads the parameters bound for `a` and `b` into each other's places.
fn swap_targets(map: &mut [MapEntry], a: &'static str, b: &'static str) {
    for (_, to, _) in map.iter_mut() {
        *to = match *to {
            t if t == a => b,
            t if t == b => a,
            other => other,
        };
    }
}

// The harness's own checks (CM0.5, ported from the retired `Fno1dRef` in CM1.5f).

/// The harness can fail: one converted weight moved by 1e-3 (layer 0's bias) breaks the
/// f32 bound.
#[test]
fn perturbed_weight_fails_parity() {
    let mut loading = load::<3>(&FNO1D, F32, &Device::default());
    let (_, parts) = loading
        .converted
        .iter_mut()
        .find(|(from, _)| from == "fno_blocks.convs.0.bias")
        .expect("layer 0's bias is converted");
    parts[0].1 += 1e-3;
    let (y, mut fixture) = loading.run();
    let err = rel_max_error(y, &fixture.output());
    assert!(err > D3_F32, "a 1e-3 shift went unnoticed: {err:.3e}");
}

#[test]
#[should_panic(expected = "parameters the map leaves out")]
fn load_rejects_an_unmapped_fixture_parameter() {
    let mut loading = load::<3>(&FNO1D, F32, &Device::default());
    loading.map.remove(0);
    loading.run();
}

#[test]
#[should_panic(expected = "map paths not in the module")]
fn load_rejects_a_path_the_module_lacks() {
    let mut loading = load::<3>(&FNO1D, F32, &Device::default());
    loading.map[0].1 = "nowhere.weight";
    loading.run();
}

#[test]
#[should_panic(expected = "once each Re and Im")]
fn load_rejects_half_a_complex_parameter() {
    let mut loading = load::<3>(&FNO1D, F32, &Device::default());
    loading.map.retain(|e| e.1 != "conv.0.weights_im.0");
    loading.run();
}

/// A map entry still naming the 2.0.0 parameter a conversion replaced is refused by name,
/// not read from the file behind the conversion's back.
#[test]
#[should_panic(expected = "map names that are not fixture parameters")]
fn load_rejects_a_converted_away_parameter() {
    let mut loading = load::<3>(&FNO1D, F32, &Device::default());
    loading.map.push((
        "fno_blocks.convs.0.weight.tensor",
        "conv_biases.0",
        Take::Whole,
    ));
    loading.run();
}

#[test]
#[should_panic(expected = "no parameter `missing` to convert")]
fn load_rejects_a_conversion_of_a_missing_parameter() {
    let mut loading = load::<3>(&FNO1D, F32, &Device::default());
    loading.converted[0].0 = "missing".into();
    loading.run();
}

#[test]
fn take_splits_the_trailing_complex_axis() {
    // [[1+2i, 3+4i]] stored as shape (1, 2, 2).
    let a = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(
        take(a.clone(), Take::Re).into_raw_vec_and_offset().0,
        [1.0, 3.0]
    );
    assert_eq!(
        take(a.clone(), Take::Im).into_raw_vec_and_offset().0,
        [2.0, 4.0]
    );
    assert_eq!(take(a.clone(), Take::Whole), a);
}

#[test]
fn rel_max_error_is_relative_to_the_reference_scale() {
    let expected = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![2.0, -4.0, 0.0]).unwrap();
    let actual = Tensor::<3>::from_data([[[2.0_f32, -4.5, 0.25]]], &Device::default());
    // max |a − e| = 0.5, max |e| = 4.
    assert!((rel_max_error(actual, &expected) - 0.125).abs() < 1e-7);
}

#[test]
#[should_panic(expected = "the output has a non-finite value")]
fn rel_max_error_rejects_nan() {
    let expected = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![1.0, 2.0]).unwrap();
    let actual = Tensor::<3>::from_data([[[1.0_f32, f32::NAN]]], &Device::default());
    rel_max_error(actual, &expected);
}

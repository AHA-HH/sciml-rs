//! Library CM0.5: the parity harness (`parity_common`) on CM0.2's 1D fixture, in f32, through
//! the hand-written map of `parity_common::fno1d`; and the harness's own checks. The f64
//! fixture is `tests/parity_fno1d_f64.rs`.

mod parity_common;

use burn::prelude::*;
use ndarray::{ArrayD, IxDyn};
use parity_common::{fno1d::*, *};

/// The acceptance test: the f32 fixture within D3's f32 bound.
#[test]
fn fno1d_f32_matches_2_0_0() {
    let (out, reference, d3) = run("fno1d_f32", &Device::default());
    let err = assert_parity(out, &reference, d3, "fno1d_f32 output");
    println!("fno1d_f32: relative max error {err:.3e}");
}

/// The harness can fail: one weight moved by 1e-3 breaks the f32 bound.
#[test]
fn perturbed_weight_fails_parity() {
    let device = Device::default();
    let (mut model, mut fixture) = loaded("fno1d_f32", &device);
    let bias = model.projection[1].bias.take().unwrap();
    model.projection[1].bias = Some(bias.map(|b| b.add_scalar(1e-3)));
    let out = model.forward(fixture.input::<3>(&device));
    let err = rel_max_error(out, &fixture.output());
    assert!(err > D3_F32, "a 1e-3 shift went unnoticed: {err:.3e}");
}

#[test]
#[should_panic(expected = "parameters the map leaves out")]
fn load_rejects_an_unmapped_fixture_parameter() {
    let device = Device::default();
    load_into(
        &mut Fno1dRef::new(&device),
        &mut Fixture::open("fno1d_f32"),
        &MAP[1..],
    );
}

#[test]
#[should_panic(expected = "map paths not in the module")]
fn load_rejects_a_path_the_module_lacks() {
    let device = Device::default();
    let mut map = MAP.to_vec();
    map[0].1 = "lifting.9.weight";
    load_into(
        &mut Fno1dRef::new(&device),
        &mut Fixture::open("fno1d_f32"),
        &map,
    );
}

#[test]
#[should_panic(expected = "once each Re and Im")]
fn load_rejects_half_a_complex_parameter() {
    let device = Device::default();
    let map: Vec<_> = MAP
        .iter()
        .copied()
        .filter(|e| e.1 != "convs.0.weights_im.0")
        .collect();
    load_into(
        &mut Fno1dRef::new(&device),
        &mut Fixture::open("fno1d_f32"),
        &map,
    );
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

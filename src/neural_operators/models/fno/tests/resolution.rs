//! 2.0.0's domain padding and resolution scaling (library CM1.5b): what they build and the
//! shapes they give. Parity with 2.0.0's `FNO` is `tests/parity_fno.rs`.

use super::*;

/// 2 layers, width 4, 2D modes (3, 2).
fn config() -> FNOConfig {
    test_config(vec![3, 2], 1, 4)
}

/// Every factor 1 and every padding fraction 0 take the same operations as neither option.
#[test]
fn unit_factors_and_zero_padding_are_bitwise_identical() {
    let device = Device::default();
    let off: FNO<4> = config().init(&device);
    let on = copy_weights(
        &off,
        config()
            .with_resolution_scaling_factor(Some(vec![vec![1.0; 2]; 2]))
            .with_domain_padding(Some(vec![0.0; 2]))
            .init(&device),
    );
    let x = random_input::<4>(&[8, 6], 1, &device);
    assert_eq!(bits(on.forward(x.clone())), bits(off.forward(x)));
}

/// Layer by layer `round(s · r)`, half to even: 8 × 6 → 16 × 9 → 8 × 4 (4.5 rounds to 4).
#[test]
fn scaling_gives_the_rounded_extents() {
    let device = Device::default();
    let fno: FNO<4> = config()
        .with_resolution_scaling_factor(Some(vec![vec![2.0, 1.5], vec![0.5, 0.5]]))
        .init(&device);
    let y = fno.forward(random_input::<4>(&[8, 6], 1, &device));
    assert_eq!(y.dims(), [2, 8, 4, 1]);
}

#[test]
fn output_shape_sets_the_last_layer_s_extents() {
    let device = Device::default();
    let fno: FNO<4> = config().init(&device);
    let y = fno.forward_with_output_shape(random_input::<4>(&[8, 6], 1, &device), &[11, 7]);
    assert_eq!(y.dims(), [2, 11, 7, 1]);
}

#[test]
fn domain_padding_keeps_the_extents() {
    let device = Device::default();
    let fno: FNO<4> = config()
        .with_domain_padding(Some(vec![0.25, 0.5]))
        .init(&device);
    let y = fno.forward(random_input::<4>(&[8, 6], 1, &device));
    assert_eq!(y.dims(), [2, 8, 6, 1]);
}

#[test]
#[should_panic(expected = "exclusive")]
fn domain_padding_with_padding_panics() {
    config()
        .with_padding(Some(2))
        .with_domain_padding(Some(vec![0.1; 2]))
        .init::<4>(&Device::default());
}

#[test]
#[should_panic(expected = "domain_padding needs 2 axes")]
fn domain_padding_of_the_wrong_rank_panics() {
    config()
        .with_domain_padding(Some(vec![0.1]))
        .init::<4>(&Device::default());
}

#[test]
#[should_panic(expected = "resolution_scaling_factor needs 2 layers of 2 positive factors")]
fn scaling_of_the_wrong_shape_panics() {
    config()
        .with_resolution_scaling_factor(Some(vec![vec![2.0, 1.0]]))
        .init::<4>(&Device::default());
}

#[test]
#[should_panic(expected = "with domain padding")]
fn output_shape_with_domain_padding_panics() {
    let device = Device::default();
    let fno: FNO<4> = config()
        .with_domain_padding(Some(vec![0.1; 2]))
        .init(&device);
    fno.forward_with_output_shape(random_input::<4>(&[8, 6], 1, &device), &[8, 6]);
}

#[test]
#[should_panic(expected = "padding with resolution_scaling_factor")]
fn padding_with_scaling_panics() {
    config()
        .with_padding(Some(2))
        .with_resolution_scaling_factor(Some(vec![vec![2.0; 2]; 2]))
        .init::<4>(&Device::default());
}

#[test]
#[should_panic(expected = "at most one scaled layer")]
fn domain_padding_with_two_scaled_layers_panics() {
    config()
        .with_domain_padding(Some(vec![0.1; 2]))
        .with_resolution_scaling_factor(Some(vec![vec![2.0; 2]; 2]))
        .init::<4>(&Device::default());
}

#[test]
#[should_panic(expected = "domain padding or padding")]
fn output_shape_with_padding_panics() {
    let device = Device::default();
    let fno: FNO<4> = config().with_padding(Some(2)).init(&device);
    fno.forward_with_output_shape(random_input::<4>(&[8, 6], 1, &device), &[8, 6]);
}

/// All-zero fractions are no domain padding, as 2.0.0 builds none: an output shape is allowed.
#[test]
fn zero_domain_padding_is_none() {
    let device = Device::default();
    let fno: FNO<4> = config()
        .with_domain_padding(Some(vec![0.0; 2]))
        .init(&device);
    assert!(fno.domain_padding.is_empty());
    let y = fno.forward_with_output_shape(random_input::<4>(&[8, 6], 1, &device), &[11, 7]);
    assert_eq!(y.dims(), [2, 11, 7, 1]);
}

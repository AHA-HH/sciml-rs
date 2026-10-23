//! The forward against the channels-last reference (CONVENTIONS §7) and the shape checks.

use super::*;

fn fno_forward_shape_check<const R: usize>(
    modes: Vec<usize>,
    hidden_channels: usize,
    data_channels: usize,
    spatial: &[usize],
) {
    let device = Device::default();
    assert_eq!(spatial.len(), R - 2, "spatial dims must match modes count");

    let config = FNOConfig {
        modes,
        hidden_channels,
        data_channels,
        out_channels: 1,
        n_layers: 4,
        spectral_init: None,
        padding: None,
    };
    let model: FNO<R> = config.init::<R>(&device);

    // Channels-last data only; the model appends the grid.
    let mut x_shape = vec![1];
    x_shape.extend_from_slice(spatial);
    x_shape.push(data_channels);
    let x_shape: [usize; R] = x_shape.try_into().unwrap();

    let out = model.forward(Tensor::<R>::random(x_shape, Distribution::Default, &device));

    let mut expected = vec![1];
    expected.extend_from_slice(spatial);
    expected.push(1);
    let expected: [usize; R] = expected.try_into().unwrap();

    assert_eq!(
        out.dims(),
        expected,
        "expected [batch, spatial.., out_channels]"
    );
}

#[test]
fn forward_shape_1d() {
    fno_forward_shape_check::<3>(vec![16], 32, 1, &[64]);
}

#[test]
fn forward_shape_2d() {
    fno_forward_shape_check::<4>(vec![2, 2], 4, 1, &[4, 4]);
}

#[test]
fn forward_shape_3d() {
    // 10 stacked timesteps; the model appends 3 coordinate channels.
    fno_forward_shape_check::<5>(vec![2, 2, 2], 4, 10, &[4, 4, 4]);
}

#[test]
#[should_panic(expected = "needs 2 modes")]
fn init_rejects_rank_mismatch() {
    let device = Device::default();
    let config = FNOConfig {
        modes: vec![16], // 1 entry → R must be 3
        hidden_channels: 32,
        data_channels: 1,
        out_channels: 1,
        n_layers: 4,
        spectral_init: None,
        padding: None,
    };
    let _: FNO<4> = config.init::<4>(&device); // asking for rank 4
}

/// `init` builds any `out_channels` (phase 4s T24); only training, in `flatten_pair`,
/// needs 1.
#[test]
fn init_builds_multichannel_output() {
    let device = Device::default();
    let model: FNO<3> = FNOConfig::new(vec![2], 1, 2)
        .with_hidden_channels(4)
        .with_n_layers(1)
        .init::<3>(&device);
    let out = model.forward(Tensor::<3>::zeros([1, 8, 1], &device));
    assert_eq!(out.dims(), [1, 8, 2]);
}

// --- issue #8: selectable spectral initialisation ---

fn check_forward<const R: usize>(
    modes: Vec<usize>,
    data_channels: usize,
    hidden: usize,
    spatial: &[usize],
) {
    let device = Device::default();
    device.seed(3);
    let model: FNO<R> = test_config(modes, data_channels, hidden).init::<R>(&device);
    let x = random_input::<R>(spatial, data_channels, &device);

    let new = model.forward(x.clone());
    let old = reference_forward_padded(&model, with_old_grid(x), Pad::Off);
    assert_eq!(new.dims(), old.dims());
    let err = rel_err(flat(new), flat(old));
    println!("FNO<{R}> forward rel err {err:e}");
    assert!(
        err <= TOL,
        "FNO<{R}> forward differs from reference: {err:e}"
    );
}

#[test]
fn forward_matches_reference_1d() {
    pin_permutations();
    check_forward::<3>(vec![4], 1, 6, &[16]);
}

/// hidden = 128 makes `fc1` (128 -> 128) and every `w` square, so a
/// transposed weight would still have the right shape.
#[test]
fn forward_matches_reference_square_weights() {
    check_forward::<3>(vec![4], 1, 128, &[16]);
}

#[test]
fn forward_matches_reference_2d() {
    check_forward::<4>(vec![3, 2], 1, 6, &[8, 6]);
}

#[test]
fn forward_matches_reference_2d_multichannel() {
    check_forward::<4>(vec![3, 2], 3, 6, &[6, 8]);
}

#[test]
fn forward_matches_reference_3d() {
    check_forward::<5>(vec![2, 2, 2], 3, 6, &[4, 6, 5]);
}

/// Same loss through both forwards: the input gradient (which flows back
/// through every layer, spectral ones included) and the lift, pointwise
/// and projection parameter gradients must agree.
#[test]
fn gradients_match_reference() {
    let device = Device::default().autodiff();
    device.seed(5);
    let model: FNO<4> = FNOConfig::new(vec![3, 2], 1, 1)
        .with_hidden_channels(6)
        .with_n_layers(2)
        .init::<4>(&device);
    let x0 = Tensor::<4>::random([2, 8, 6, 1], Distribution::Normal(0.0, 1.0), &device);
    let probe = Tensor::<4>::random([2, 8, 6, 1], Distribution::Normal(0.0, 1.0), &device);

    let new = grads_of(&model, &probe, x0.clone(), &|x| model.forward(x));
    let old = grads_of(&model, &probe, with_old_grid(x0), &|x| {
        reference_forward_padded(&model, x, Pad::Off)
    });
    for ((name, g_new), (_, g_old)) in new.into_iter().zip(old) {
        let err = rel_err(g_new, g_old);
        println!("grad {name} rel err {err:e}");
        assert!(err <= TOL, "gradient of {name} differs: {err:e}");
    }
}

/// `to_channels_first` / `to_channels_last` against literal `permute`s (CONVENTIONS §1)
/// on non-square tensors, rank 4 and 5; called from `forward_matches_reference_1d`.
fn pin_permutations() {
    let device = Device::default();
    let x4 = Tensor::<4>::random([2, 3, 4, 5], Distribution::Default, &device);
    let x5 = Tensor::<5>::random([2, 3, 4, 5, 6], Distribution::Default, &device);
    assert_eq!(
        bits(to_channels_first(x4.clone())),
        bits(x4.clone().permute([0, 3, 1, 2]))
    );
    assert_eq!(
        bits(to_channels_last(x4.clone())),
        bits(x4.permute([0, 2, 3, 1]))
    );
    assert_eq!(
        bits(to_channels_first(x5.clone())),
        bits(x5.clone().permute([0, 4, 1, 2, 3]))
    );
    assert_eq!(
        bits(to_channels_last(x5.clone())),
        bits(x5.permute([0, 2, 3, 4, 1]))
    );
}

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

// --- issue #8: selectable spectral initialisation ---

fn check_forward<const R: usize>(
    modes: Vec<usize>,
    data_channels: usize,
    hidden: usize,
    spatial: &[usize],
) {
    let device = Device::default();
    device.seed(3);
    let model: FNO<R> = FNOConfig::new(modes.clone(), data_channels, 1)
        .with_hidden_channels(hidden)
        .with_n_layers(2)
        .init::<R>(&device);
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

    // The reference differentiates w.r.t. the old grid-augmented input;
    // only its data-channel slice is comparable to the new input grad.
    let grads_of = |x0: Tensor<4>, f: &dyn Fn(Tensor<4>) -> Tensor<4>| {
        let x = x0.detach().require_grad();
        let grads = (f(x.clone()) * probe.clone()).sum().backward();
        let g_in = x.grad(&grads).expect("input grad").narrow(3, 0, 1);
        assert_eq!(g_in.dims(), [2, 8, 6, 1]);
        let mut out: Vec<(String, Tensor<1>)> = vec![("input".into(), flat(g_in))];
        for (name, l) in [
            ("fc0", &model.fc0),
            ("fc1", &model.fc1),
            ("fc2", &model.fc2),
        ] {
            out.push((
                format!("{name}.weight"),
                flat(l.weight.grad(&grads).unwrap()),
            ));
            let b = l.bias.as_ref().unwrap();
            out.push((format!("{name}.bias"), flat(b.grad(&grads).unwrap())));
        }
        for (i, w) in model.w.iter().enumerate() {
            out.push((
                format!("w[{i}].weight"),
                flat(w.weight.grad(&grads).unwrap()),
            ));
            let b = w.bias.as_ref().unwrap();
            out.push((format!("w[{i}].bias"), flat(b.grad(&grads).unwrap())));
        }
        out
    };

    let new = grads_of(x0.clone(), &|x| model.forward(x));
    let old = grads_of(with_old_grid(x0), &|x| {
        reference_forward_padded(&model, x, Pad::Off)
    });
    for ((name, g_new), (_, g_old)) in new.into_iter().zip(old) {
        let err = rel_err(g_new, g_old);
        println!("grad {name} rel err {err:e}");
        assert!(err <= TOL, "gradient of {name} differs: {err:e}");
    }
}

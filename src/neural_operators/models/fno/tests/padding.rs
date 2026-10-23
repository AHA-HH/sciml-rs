//! Domain padding (CONVENTIONS §3).

use super::*;

/// `None` and `Some(0)` take the unpadded path: bit for bit the pre-#11
/// forward on the same weights.
fn check_padding_off_is_bitwise<const R: usize>(modes: Vec<usize>, spatial: &[usize]) {
    let device = Device::default();
    let cfg = test_config(modes, 1, 6);
    let off: FNO<R> = cfg.clone().with_padding(None).init::<R>(&device);
    let zero = copy_weights(&off, cfg.with_padding(Some(0)).init::<R>(&device));
    assert_eq!(zero.padding, 0);
    assert_eq!(params(&off), params(&zero));

    let x = random_input::<R>(spatial, 1, &device);
    let before = bits(pre_padding_forward(&off, with_model_grid(x.clone())));
    assert_eq!(
        bits(off.forward(x.clone())),
        before,
        "padding None, FNO<{R}>"
    );
    assert_eq!(bits(zero.forward(x)), before, "padding Some(0), FNO<{R}>");
}

#[test]
fn padding_zero_is_bitwise_identical() {
    check_padding_off_is_bitwise::<3>(vec![4], &[16]);
    check_padding_off_is_bitwise::<4>(vec![3, 2], &[8, 6]);
    check_padding_off_is_bitwise::<5>(vec![2, 2, 2], &[4, 6, 5]);
}

/// Padding adds no parameters: the same names and shapes as unpadded.
#[test]
fn padding_adds_no_parameters() {
    let device = Device::default();
    let names_shapes = |p: Option<usize>| -> Vec<(String, usize)> {
        let m: FNO<4> = test_config(vec![3, 2], 1, 6)
            .with_padding(p)
            .init::<4>(&device);
        params(&m).into_iter().map(|(n, v)| (n, v.len())).collect()
    };
    assert_eq!(names_shapes(None), names_shapes(Some(9)));
}

/// `p > 0` matches the `cat`/`narrow` reference, and measurably differs
/// from the wrong conventions (so the comparison is not vacuous). Padded
/// extents are deliberately not powers of two.
fn check_padded_forward<const R: usize>(
    modes: Vec<usize>,
    data_channels: usize,
    spatial: &[usize],
    p: usize,
) {
    let device = Device::default();
    // SymmetricUniform (±1/√I) instead of Li's U(0, 1/(I·O)) only to make
    // the spectral path, the one place padding acts, O(1) at width 6.
    let model = with_unit_lift_bias(
        test_config(modes.clone(), data_channels, 6)
            .with_spectral_init(Some(SpectralInit::SymmetricUniform))
            .with_padding(Some(p))
            .init::<R>(&device),
    );
    let x = random_input::<R>(spatial, data_channels, &device);

    let out = model.forward(x.clone());
    let x = with_old_grid(x); // the reference's input
    let mut expected = x.dims();
    expected[R - 1] = 1;
    assert_eq!(out.dims(), expected, "crop restores the input grid");

    let err = rel_err(
        flat(out.clone()),
        flat(reference_forward_padded(&model, x.clone(), Pad::End(p))),
    );
    println!("FNO<{R}> padded forward rel err {err:e}");
    assert!(err <= TOL, "FNO<{R}> padded forward differs: {err:e}");

    for wrong in [Pad::Off, Pad::BeforeFc0(p), Pad::EndCropTail(p)] {
        let err = rel_err(
            flat(out.clone()),
            flat(reference_forward_padded(&model, x.clone(), wrong)),
        );
        println!("FNO<{R}> vs {wrong:?}: rel err {err:e}");
        assert!(
            err >= 1e-3,
            "FNO<{R}> forward indistinguishable from {wrong:?}: {err:e}"
        );
    }
}

#[test]
fn padded_forward_matches_reference_1d() {
    check_padded_forward::<3>(vec![4], 1, &[16], 5); // 21
}

#[test]
fn padded_forward_matches_reference_2d() {
    check_padded_forward::<4>(vec![3, 2], 1, &[8, 6], 3); // 11 x 9
}

#[test]
fn padded_forward_matches_reference_3d() {
    check_padded_forward::<5>(vec![2, 2, 2], 3, &[4, 6, 5], 2); // 6 x 8 x 7
}

/// The pad side is immaterial once the crop is aligned with it: on the
/// padded grid every layer is circular-shift-equivariant (the truncated
/// spectral multiplier is diagonal in frequency; the 1x1 convs, biases
/// and ReLU are pointwise), so symmetric padding with a centre crop is the
/// end-padded result shifted back. Only the misaligned crop above differs.
#[test]
fn symmetric_padding_with_centre_crop_matches_end_padding() {
    let device = Device::default();
    let model = with_unit_lift_bias(
        test_config(vec![3, 2], 1, 6)
            .with_spectral_init(Some(SpectralInit::SymmetricUniform))
            .with_padding(Some(3))
            .init::<4>(&device),
    );
    let x = random_input::<4>(&[8, 6], 1, &device);
    let err = rel_err(
        flat(model.forward(x.clone())),
        flat(reference_forward_padded(
            &model,
            with_old_grid(x),
            Pad::Symmetric(3),
        )),
    );
    println!("FNO<4> vs Symmetric(3): rel err {err:e}");
    assert!(err <= TOL, "symmetric + centre crop differs: {err:e}");
}

/// `pad_spatial` puts the data at `[0..n]` and exact zeros after it on
/// every spatial axis, and `crop_spatial` undoes it bit for bit.
#[test]
fn pad_crop_helpers_exact() {
    use burn::tensor::TensorData;
    let device = Device::default();
    let (b, c, n0, n1, p) = (2, 2, 3, 4, 2);
    let vals: Vec<f32> = (0..b * c * n0 * n1).map(|i| i as f32 + 1.0).collect();
    let x = Tensor::<4>::from_data(TensorData::new(vals.clone(), [b, c, n0, n1]), &device);

    let padded = FNO::<4>::pad_spatial(x.clone(), p);
    assert_eq!(padded.dims(), [b, c, n0 + p, n1 + p]);
    let got = padded.clone().into_data().try_to_vec::<f32>().unwrap();
    let (m0, m1) = (n0 + p, n1 + p);
    for bi in 0..b {
        for ci in 0..c {
            for i in 0..m0 {
                for j in 0..m1 {
                    let want = if i < n0 && j < n1 {
                        vals[((bi * c + ci) * n0 + i) * n1 + j]
                    } else {
                        0.0
                    };
                    let at = ((bi * c + ci) * m0 + i) * m1 + j;
                    assert_eq!(got[at], want, "[{bi},{ci},{i},{j}]");
                }
            }
        }
    }

    let cropped = FNO::<4>::crop_spatial(padded, [b, c, n0, n1]);
    assert_eq!(bits(cropped), bits(x));
}

/// Autodiff through pad/crop agrees with the `cat`/`narrow` reference for
/// the input and every parameter group, the spectral weights included.
#[test]
fn padded_gradients_match_reference() {
    let device = Device::default().autodiff();
    let model = with_unit_lift_bias(
        test_config(vec![3, 2], 1, 6)
            .with_spectral_init(Some(SpectralInit::SymmetricUniform))
            .with_padding(Some(3))
            .init::<4>(&device),
    );
    let x0 = Tensor::<4>::random([2, 8, 6, 1], Distribution::Normal(0.0, 1.0), &device);
    let probe = Tensor::<4>::random([2, 8, 6, 1], Distribution::Normal(0.0, 1.0), &device);

    let new = grads_of(&model, &probe, x0.clone(), &|x| model.forward(x));
    let old = grads_of(&model, &probe, with_old_grid(x0), &|x| {
        reference_forward_padded(&model, x, Pad::End(3))
    });
    assert_eq!(new.len(), old.len());
    for ((name, g_new), (_, g_old)) in new.into_iter().zip(old) {
        // A vanishing gradient would make the comparison vacuous.
        let size = g_old.clone().abs().max().into_scalar::<f32>();
        assert!(size > 1e-4, "gradient of {name} vanishes: {size:e}");
        let err = rel_err(g_new, g_old);
        println!("padded grad {name} rel err {err:e} (max |g| {size:e})");
        assert!(err <= TOL, "padded gradient of {name} differs: {err:e}");
    }
}

/// The input gradient through pad and crop against central differences,
/// i.e. against the definition of the derivative rather than Burn's own
/// pad/slice backward (which the reference above shares in part).
///
/// `L = Σ forward(x) · probe` is O(1) in f32, so with `h = 1e-2` the
/// rounding error of a difference is ~1e-7 / 1e-2 = 1e-5 and the
/// truncation error O(h²) ~ 1e-4, both far below the 5e-2 bound; a dropped
/// or misrouted gradient gives O(1) errors. ReLU kinks crossed within ±h
/// can spoil single entries, so the median over all entries is asserted.
#[test]
fn padded_input_gradient_finite_difference() {
    use burn::tensor::TensorData;
    let device = Device::default().autodiff();
    let (n, c, p) = (8, 1, 3); // padded extent 11
    let model = with_unit_lift_bias(
        FNOConfig::new(vec![3], 1, 1)
            .with_hidden_channels(4)
            .with_n_layers(1)
            .with_spectral_init(Some(SpectralInit::SymmetricUniform))
            .with_padding(Some(p))
            .init::<3>(&device),
    );
    let x_vals: Vec<f32> = (0..n * c).map(|i| (i as f32 * 0.37).sin()).collect();
    let probe_vals: Vec<f32> = (0..n).map(|i| (i as f32 * 0.91).cos()).collect();
    let probe = Tensor::<3>::from_data(TensorData::new(probe_vals, [1, n, 1]), &device);
    let input = |v: Vec<f32>| Tensor::<3>::from_data(TensorData::new(v, [1, n, c]), &device);
    let loss = |v: Vec<f32>| -> f32 {
        (model.forward(input(v)) * probe.clone())
            .sum()
            .into_scalar::<f32>()
    };

    let x = input(x_vals.clone()).require_grad();
    let grads = (model.forward(x.clone()) * probe.clone()).sum().backward();
    let g: Vec<f32> = x
        .grad(&grads)
        .expect("input grad")
        .into_data()
        .try_to_vec()
        .unwrap();

    let h = 1e-2;
    let fd: Vec<f32> = (0..n * c)
        .map(|k| {
            let (mut plus, mut minus) = (x_vals.clone(), x_vals.clone());
            plus[k] += h;
            minus[k] -= h;
            (loss(plus) - loss(minus)) / (2.0 * h)
        })
        .collect();

    let scale = g.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(scale > 1e-3, "degenerate gradient: max |g| = {scale:e}");
    let mut errs: Vec<f32> = g
        .iter()
        .zip(&fd)
        .map(|(a, b)| (a - b).abs() / scale)
        .collect();
    println!("autodiff {g:?}\nfinite difference {fd:?}");
    errs.sort_by(f32::total_cmp);
    let median = errs[errs.len() / 2];
    println!(
        "FD rel err median {median:e}, max {:e}",
        errs[errs.len() - 1]
    );
    assert!(
        median <= 5e-2,
        "input gradient vs finite difference: {median:e}"
    );
}

#[test]
fn padding_survives_config_round_trip() {
    use burn::config::config_to_json;
    for padding in [None, Some(0), Some(9)] {
        let config = FNOConfig::new(vec![12, 12], 1, 1).with_padding(padding);
        let json = config_to_json(&config);
        let loaded = FNOConfig::load_binary(json.as_bytes()).expect("load saved config");
        assert_eq!(loaded.padding, padding, "round trip via {json}");
    }
}

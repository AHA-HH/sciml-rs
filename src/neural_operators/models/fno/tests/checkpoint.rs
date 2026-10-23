//! Config, initialisation and checkpoint round trips.

use super::*;

/// Every spectral layer gets the configured scheme. `LiUniform` draws
/// from `[0, s)`, so a negative weight in a layer means that layer used a
/// zero-mean scheme; with 2 parts × 8·8·4 = 512 symmetric draws per layer
/// (one corner in 1D), P(none negative) = 2^-512.
#[test]
fn spectral_init_reaches_every_layer() {
    let device = Device::default();
    let min_per_layer = |init: Option<SpectralInit>| -> Vec<f32> {
        let model: FNO<3> = FNOConfig::new(vec![4], 1, 1)
            .with_hidden_channels(8)
            .with_n_layers(3)
            .with_spectral_init(init)
            .init::<3>(&device);
        model
            .conv
            .iter()
            .map(|layer| {
                layer
                    .corner_weights()
                    .into_iter()
                    .flat_map(|(re, im)| [re.min().into_scalar::<f32>(), im.min().into_scalar()])
                    .fold(f32::INFINITY, f32::min)
            })
            .collect()
    };

    let default = min_per_layer(None);
    assert_eq!(default.len(), 3);
    assert!(
        default.iter().all(|&m| m >= 0.0),
        "default (Li) init must be non-negative in every layer: {default:?}"
    );

    for init in [SpectralInit::Normal, SpectralInit::SymmetricUniform] {
        let mins = min_per_layer(Some(init.clone()));
        assert!(
            mins.iter().all(|&m| m < 0.0),
            "{init:?} did not reach every layer: per-layer minima {mins:?}"
        );
    }
}

#[test]
fn spectral_init_survives_config_round_trip() {
    use burn::config::config_to_json;

    for init in [
        None,
        Some(SpectralInit::LiUniform),
        Some(SpectralInit::Normal),
        Some(SpectralInit::SymmetricUniform),
    ] {
        let config = FNOConfig::new(vec![16], 1, 1).with_spectral_init(init.clone());
        let json = config_to_json(&config);
        let loaded = FNOConfig::load_binary(json.as_bytes()).expect("load saved config");
        assert_eq!(loaded.spectral_init, init, "round trip via {json}");
    }
}

/// A `model_cfg.json` as written before `spectral_init` existed must
/// still load, and mean the Li et al. default.
#[test]
fn config_without_spectral_init_loads_as_default() {
    let old = r#"{
  "modes": [16],
  "hidden_channels": 64,
  "data_channels": 1,
  "out_channels": 1,
  "n_layers": 4
}"#;
    let loaded = FNOConfig::load_binary(old.as_bytes()).expect("load pre-#8 config");
    assert_eq!(loaded.spectral_init, None);
    assert_eq!(loaded.padding, None);
    assert_eq!(loaded.modes, vec![16]);
    assert_eq!(loaded.hidden_channels, 64);
    assert_eq!(loaded.n_layers, 4);
}

// --- merge `04ec800`: inference without autodiff ---

#[test]
fn autodiff_trained_weights_give_identical_inference_on_plain_device() {
    // Mirrors train -> save -> predict: weights from a model built on the
    // autodiff device, loaded into one on the plain device (what the
    // predict examples now use).
    use burn::tensor::TensorData;

    let ad = Device::default().autodiff();
    let plain = Device::default();
    let config = FNOConfig {
        modes: vec![4],
        hidden_channels: 8,
        data_channels: 1,
        out_channels: 1,
        n_layers: 2,
        spectral_init: None,
        padding: None,
    };

    let trained: FNO<3> = config.init::<3>(&ad);
    let loaded: FNO<3> = copy_weights(&trained, config.init::<3>(&plain));

    // [batch, s, data channels], channels-last.
    let shape = vec![2, 16, 1];
    let vals: Vec<f32> = (0..32).map(|i| (i as f32 * 0.29).sin()).collect();
    let x_ad = Tensor::<3>::from_data(TensorData::new(vals.clone(), shape.clone()), &ad);
    let x_plain = Tensor::<3>::from_data(TensorData::new(vals, shape), &plain);

    let out_ad = trained.forward(x_ad);
    let out_plain = loaded.forward(x_plain);

    // The old example path tracked a backward graph; the new one doesn't.
    assert!(out_ad.is_autodiff());
    assert!(!out_plain.is_autodiff());

    // Same backend kernels either way, so the values must match exactly.
    assert_eq!(
        out_ad.inner().into_data().try_to_vec::<f32>().unwrap(),
        out_plain.into_data().try_to_vec::<f32>().unwrap()
    );
}

/// Padding is not part of a checkpoint: weights saved from an unpadded
/// model load into a padded one and back, unchanged, and each then runs
/// its own configuration's forward.
#[test]
fn padding_does_not_change_checkpoint() {
    let device = Device::default();
    let cfg =
        test_config(vec![3, 2], 1, 6).with_spectral_init(Some(SpectralInit::SymmetricUniform));
    let unpadded = with_unit_lift_bias(cfg.clone().init::<4>(&device));
    let x = random_input::<4>(&[8, 6], 1, &device);

    // Unpadded -> padded (17 x 15).
    let padded = copy_weights(
        &unpadded,
        cfg.clone().with_padding(Some(9)).init::<4>(&device),
    );
    assert_eq!(params(&padded), params(&unpadded));
    let out = padded.forward(x.clone());
    let err = rel_err(
        flat(out.clone()),
        flat(reference_forward_padded(
            &unpadded,
            with_old_grid(x.clone()),
            Pad::End(9),
        )),
    );
    assert!(
        err <= TOL,
        "loaded padded model differs from reference: {err:e}"
    );
    let diff = rel_err(flat(out), flat(unpadded.forward(x.clone())));
    assert!(diff >= 1e-3, "padding had no effect after load: {diff:e}");

    // Padded -> unpadded.
    let back = copy_weights(&padded, cfg.with_padding(None).init::<4>(&device));
    assert_eq!(params(&back), params(&padded));
    assert_eq!(
        bits(back.forward(x.clone())),
        bits(pre_padding_forward(&padded, with_model_grid(x)))
    );
}

/// T6: the lift still takes `data_channels + D` inputs, so checkpoints
/// written before #13 keep their parameter names and shapes.
#[test]
fn lift_width_includes_grid_channels() {
    let device = Device::default();
    let m1: FNO<3> = test_config(vec![4], 1, 6).init::<3>(&device);
    let m2: FNO<4> = test_config(vec![3, 2], 1, 6).init::<4>(&device);
    let m3: FNO<5> = test_config(vec![2, 2, 2], 10, 6).init::<5>(&device);
    assert_eq!(m1.fc0.weight.dims(), [2, 6]);
    assert_eq!(m2.fc0.weight.dims(), [3, 6]);
    assert_eq!(m3.fc0.weight.dims(), [13, 6]);
    let fc0 = |p: Vec<(String, Vec<f32>)>| {
        p.into_iter()
            .find(|(n, _)| n == "fc0.weight")
            .map(|(_, v)| v.len())
    };
    assert_eq!(fc0(params(&m2)), Some(3 * 6));
}

/// T6: weights saved by a model fed grid-augmented inputs give the same
/// predictions when loaded and fed the data alone.
#[test]
fn checkpoint_round_trip_matches_old_forward() {
    let device = Device::default();
    let cfg = test_config(vec![3, 2], 1, 6);
    let saved = with_unit_lift_bias(cfg.clone().init::<4>(&device));
    let loaded = copy_weights(&saved, cfg.init::<4>(&device));
    assert_eq!(params(&loaded), params(&saved));

    let x = random_input::<4>(&[8, 6], 1, &device);
    let new = loaded.forward(x.clone());
    let old = pre_padding_forward(&saved, with_old_grid(x));
    let bitwise = bits(new.clone()) == bits(old.clone());
    let err = rel_err(flat(new), flat(old));
    println!("checkpoint round trip: rel err {err:e}, bitwise {bitwise}");
    assert!(
        err <= TOL,
        "loaded model differs from the old forward: {err:e}"
    );
}

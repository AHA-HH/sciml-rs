//! Optimisers CO1.1: the activation option (GELU for PETScML's Burgers model, D10).

use super::*;
use burn::tensor::activation::gelu;

/// PETScML's Burgers model (§4.1.1): 16 modes, width 64, 4 Fourier layers.
fn burgers_cfg() -> FNOConfig {
    FNOConfig::new(vec![16], 1, 1)
        .with_hidden_channels(64)
        .with_n_layers(4)
}

/// PETScML state 549,569 parameters; the count is 192 (fc0) + 4 × 131,072 (spectral)
/// + 4 × 4,160 (pointwise) + 8,320 (fc1) + 129 (fc2). GELU adds none.
#[test]
fn burgers_model_has_petscml_parameter_count() {
    let device = Device::default();
    assert_eq!(burgers_cfg().init::<3>(&device).num_params(), 549_569);
    let gelu_cfg = burgers_cfg().with_activation(Some(Activation::Gelu));
    assert_eq!(gelu_cfg.init::<3>(&device).num_params(), 549_569);
}

/// `FNO::forward` with GELU wherever the default applies ReLU, written out by hand.
fn gelu_forward<const R: usize>(m: &FNO<R>, x: Tensor<R>) -> Tensor<R> {
    let grid = FNO::<R>::grid_cl(x.dims(), x.dtype(), &x.device());
    let x = Tensor::cat(vec![x, grid], R - 1);
    let mut x = FNO::<R>::linear_cf(&m.fc0, to_channels_first(x));
    let n = m.conv.len();
    for idx in 0..n {
        let x1 = m.conv[idx].forward(x.clone());
        let x2 = FNO::<R>::conv1x1_cf(&m.w[idx], x);
        x = if idx == n - 1 { x1 + x2 } else { gelu(x1 + x2) };
    }
    let x = gelu(FNO::<R>::linear_cf(&m.fc1, x));
    to_channels_last(FNO::<R>::linear_cf(&m.fc2, x))
}

#[test]
fn gelu_option_applies_gelu_at_every_activation() {
    let device = Device::default();
    let relu_model: FNO<3> = test_config(vec![4], 1, 8).init(&device);
    let gelu_model = copy_weights(
        &relu_model,
        test_config(vec![4], 1, 8)
            .with_activation(Some(Activation::Gelu))
            .init(&device),
    );
    let x = random_input::<3>(&[16], 1, &device);
    let got = gelu_model.forward(x.clone());
    assert!(
        rel_err(
            flat(got.clone()),
            flat(gelu_forward(&relu_model, x.clone()))
        ) < 1e-6
    );
    // the option changes the output, so the test sees it
    assert!(rel_err(flat(got), flat(relu_model.forward(x))) > 1e-4);
}

/// A `model_cfg.json` from before optimisers CO1.1 loads as ReLU, and `Relu` is the
/// default path bit for bit.
#[test]
fn old_config_loads_as_relu_and_relu_is_the_default() {
    let old = r#"{"modes": [16], "hidden_channels": 64, "data_channels": 1,
        "out_channels": 1, "n_layers": 4, "spectral_init": null, "padding": null}"#;
    assert!(
        FNOConfig::load_binary(old.as_bytes())
            .unwrap()
            .activation
            .is_none()
    );

    let device = Device::default();
    let default: FNO<3> = test_config(vec![4], 1, 8).init(&device);
    let relu = copy_weights(
        &default,
        test_config(vec![4], 1, 8)
            .with_activation(Some(Activation::Relu))
            .init(&device),
    );
    let x = random_input::<3>(&[16], 1, &device);
    assert_eq!(bits(default.forward(x.clone())), bits(relu.forward(x)));
}

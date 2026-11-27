//! `FNOConfig`'s optional fields (library CM1.5d, the M1-gate rule of 2026-11-27): a
//! `model_cfg.json` written before an option existed still loads, as that option's `None`.

use super::*;

/// A Phase 5 `model_cfg.json`, verbatim as `RunDir::save_configs` wrote it before library
/// CM1.3b (`runs/co0-1/poisson_fno_n129_s64_p0_seed46_*`): none of the block, lifting,
/// embedding, convolution, padding or scaling options.
const PHASE5_MODEL_CFG: &str = r#"{
  "modes": [
    12,
    12
  ],
  "data_channels": 1,
  "out_channels": 1,
  "spectral_init": null,
  "padding": 0,
  "hidden_channels": 32,
  "n_layers": 4
}"#;

/// Every option added since loads as `None`, and the model forwards bit for bit as one built
/// by `FNOConfig::new` with the same weights. A plain `bool` field (or any field without
/// `Option`) fails this load: Burn's `Config` derive gives fields no serde default.
#[test]
fn an_old_model_cfg_json_loads_with_every_newer_option_none() {
    let c = FNOConfig::load_binary(PHASE5_MODEL_CFG.as_bytes()).expect("old config loads");
    assert_eq!(c.activation, None);
    assert!(c.fno_skip.is_none() && c.channel_mlp.is_none() && c.norm.is_none());
    assert_eq!(
        (c.preactivation, c.conv_bias, c.separable, c.tanh_stabilizer),
        (None, None, None, None)
    );
    assert_eq!(
        (c.lifting_channel_ratio, c.projection_channel_ratio),
        (None, None)
    );
    assert!(c.positional_embedding.is_none() && c.max_modes.is_none());
    assert_eq!(
        (
            c.domain_padding.clone(),
            c.resolution_scaling_factor.clone()
        ),
        (None, None)
    );

    let device = Device::default();
    let new: FNO<4> = FNOConfig::new(vec![12, 12], 1, 1)
        .with_padding(Some(0))
        .init(&device);
    let old = copy_weights(&new, c.init(&device));
    let x = random_input::<4>(&[24, 24], 1, &device);
    assert_eq!(bits(old.forward(x.clone())), bits(new.forward(x)));
}

//! Generic template for FNO for n dimensions

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        Linear, LinearConfig,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::{Device, activation::relu},
};

use crate::neural_operators::layers::spectral_convolution::{SpectralConv, SpectralInit};

// FNO Architecture
#[derive(Config, Debug)]
pub struct FNOConfig {
    pub modes: Vec<usize>, // length D = R - 2
    #[config(default = 32)]
    pub hidden_channels: usize,
    pub data_channels: usize, // problem specific, 1 (coefficients), 10 (3d problem, stacked timesteps)
    pub out_channels: usize,
    #[config(default = 4)]
    pub n_layers: usize,
    /// Initialisation of every spectral layer's weights. `None` is the Li et
    /// al. default, [`SpectralInit::LiUniform`]. An `Option` so that saved
    /// configs without this field still load (as `None`).
    pub spectral_init: Option<SpectralInit>,
}

#[derive(Module, Debug)]
pub struct FNO<const R: usize> {
    fc0: Linear,
    conv: Vec<SpectralConv<R>>,
    w: Vec<Conv1d>,
    fc1: Linear,
    fc2: Linear,
}

impl FNOConfig {
    pub fn init<const R: usize>(&self, device: &Device) -> FNO<R> {
        assert_eq!(
            self.modes.len() + 2,
            R,
            "FNO<{R}> needs {} modes, config has {}",
            R - 2,
            self.modes.len()
        );
        assert_eq!(
            self.out_channels, 1,
            "only scalar-output operators are supported (see flatten_pair)"
        );

        let coord_channels = self.modes.len(); // D = R - 2, need to derive and input into layer

        FNO {
            fc0: LinearConfig::new(self.data_channels + coord_channels, self.hidden_channels)
                .init(device),

            conv: (0..self.n_layers)
                .map(|_| {
                    SpectralConv::<R>::new_with_init(
                        device,
                        self.hidden_channels,
                        self.hidden_channels,
                        &self.modes,
                        self.spectral_init.clone().unwrap_or_default(),
                    )
                })
                .collect(),

            w: (0..self.n_layers)
                .map(|_| {
                    Conv1dConfig::new(self.hidden_channels, self.hidden_channels, 1).init(device)
                })
                .collect(),

            fc1: LinearConfig::new(self.hidden_channels, 128).init(device),
            fc2: LinearConfig::new(128, self.out_channels).init(device),
        }
    }
}

impl<const R: usize> FNO<R> {
    /// `y[b, o, n] = Σ_i w[o, i] · x[b, i, n] + bias[o]` on channels-first `x`
    /// (`[B, I, spatial..]` → `[B, O, spatial..]`), as one batched matmul.
    ///
    /// The reshape to `[B, I, N]` is free on a contiguous channels-first
    /// tensor, which is what every caller in `forward` passes.
    fn pointwise(w_oi: Tensor<2>, bias: Option<Tensor<1>>, x: Tensor<R>) -> Tensor<R> {
        let mut dims = x.dims();
        let (b, i) = (dims[0], dims[1]);
        let n: usize = dims[2..].iter().product();
        let [o, w_i] = w_oi.dims();
        debug_assert_eq!(
            w_i, i,
            "pointwise weight expects {w_i} input channels, got {i}"
        );

        let y = w_oi.unsqueeze::<3>().matmul(x.reshape([b, i, n])); // [1,O,I] @ [B,I,N]
        let y = match bias {
            Some(bias) => y + bias.reshape([1, o, 1]),
            None => y,
        };
        dims[1] = o;
        y.reshape(dims)
    }

    /// A `Linear` applied along the channel axis of a channels-first tensor.
    ///
    /// Uses the layer's own parameters, so checkpoints are unchanged. `Linear`
    /// stores `[d_input, d_output]` (the default row layout used by `init`),
    /// hence the transpose to `[O, I]`.
    fn linear_cf(layer: &Linear, x: Tensor<R>) -> Tensor<R> {
        Self::pointwise(
            layer.weight.val().transpose(),
            layer.bias.as_ref().map(|b| b.val()),
            x,
        )
    }

    /// A kernel-size-1 `Conv1d` as a pointwise matmul (REVIEW.md 4.4), using
    /// its own parameters: the `[O, I, 1]` weight read as `[O, I]`.
    fn conv1x1_cf(conv: &Conv1d, x: Tensor<R>) -> Tensor<R> {
        let [o, i, k] = conv.weight.val().dims();
        debug_assert_eq!(k, 1, "pointwise path needs kernel size 1, got {k}");
        Self::pointwise(
            conv.weight.val().reshape([o, i]),
            conv.bias.as_ref().map(|b| b.val()),
            x,
        )
    }

    /// `[B, spatial.., C_in]` → `[B, spatial.., out_channels]`.
    ///
    /// Channels-first inside: the input is permuted once while it is only
    /// `C_in` wide, and the output once while it is one channel wide, so the
    /// hidden-width activations are never permuted or copied for layout.
    pub fn forward(&self, x: Tensor<R>) -> Tensor<R> {
        // [B, s.., C] -> [B, C, s..]
        let perm_in: [usize; R] = core::array::from_fn(|i| match i {
            0 => 0,
            1 => R - 1,
            i => i - 1,
        });
        let mut x = Self::linear_cf(&self.fc0, x.permute(perm_in));

        let n = self.conv.len();
        for idx in 0..n {
            let x1 = self.conv[idx].forward(x.clone());
            let x2 = Self::conv1x1_cf(&self.w[idx], x);
            x = if idx == n - 1 { x1 + x2 } else { relu(x1 + x2) };
        }

        let x = relu(Self::linear_cf(&self.fc1, x));
        let x = Self::linear_cf(&self.fc2, x);

        // [B, C, s..] -> [B, s.., C]
        let perm_out: [usize; R] = core::array::from_fn(|i| match i {
            0 => 0,
            i if i == R - 1 => 1,
            i => i + 1,
        });
        x.permute(perm_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    fn fno_forward_shape_check<const R: usize>(
        modes: Vec<usize>,
        hidden_channels: usize,
        data_channels: usize,
        spatial: &[usize],
    ) {
        let device = Device::default();
        let coord_channels = modes.len();
        assert_eq!(spatial.len(), R - 2, "spatial dims must match modes count");

        let config = FNOConfig {
            modes,
            hidden_channels,
            data_channels,
            out_channels: 1,
            n_layers: 4,
            spectral_init: None,
        };
        let model: FNO<R> = config.init::<R>(&device);

        // Channels-last on input, as fc0 expects.
        let mut x_shape = vec![1];
        x_shape.extend_from_slice(spatial);
        x_shape.push(data_channels + coord_channels);
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
        // 10 stacked timesteps + x + y + t channels.
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
        };
        let _: FNO<4> = config.init::<4>(&device); // asking for rank 4
    }

    // --- issue #8: selectable spectral initialisation ---

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
                        .flat_map(|(re, im)| {
                            [re.min().into_scalar::<f32>(), im.min().into_scalar()]
                        })
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
        assert_eq!(loaded.modes, vec![16]);
        assert_eq!(loaded.hidden_channels, 64);
        assert_eq!(loaded.n_layers, 4);
    }

    // --- REVIEW.md 2.7: inference without autodiff ---

    #[test]
    fn autodiff_trained_weights_give_identical_inference_on_plain_device() {
        // Mirrors train -> save -> predict: weights from a model built on the
        // autodiff device, loaded into one on the plain device (what the
        // predict examples now use).
        use burn::store::{BurnpackStore, ModuleSnapshot};
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
        };

        let trained: FNO<3> = config.init::<3>(&ad);
        let mut save = BurnpackStore::from_bytes(None);
        trained.save_into(&mut save).expect("save weights");
        let bytes = save.get_bytes().expect("serialise weights");

        let mut loaded: FNO<3> = config.init::<3>(&plain);
        loaded
            .load_from(&mut BurnpackStore::from_bytes(Some(bytes)))
            .expect("load weights");

        // [batch, s, data + coord channels], channels-last.
        let shape = vec![2, 16, 2];
        let vals: Vec<f32> = (0..64).map(|i| (i as f32 * 0.29).sin()).collect();
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

    // --- REVIEW.md 4.2: channels-first internals match the old forward ---

    /// The pre-4.2 `forward`, verbatim: `Linear` on channels-last, permute the
    /// hidden activation to channels-first and back.
    fn reference_forward<const R: usize>(m: &FNO<R>, x: Tensor<R>) -> Tensor<R> {
        let x = m.fc0.forward(x);
        let perm_in: [usize; R] = core::array::from_fn(|i| match i {
            0 => 0,
            1 => R - 1,
            i => i - 1,
        });
        let mut x = x.permute(perm_in);
        let n = m.conv.len();
        for idx in 0..n {
            let x1 = m.conv[idx].forward(x.clone());
            let dims = x.dims();
            let spatial: usize = dims[2..].iter().product();
            let x2 = m.w[idx]
                .forward(x.reshape([dims[0], dims[1], spatial]))
                .reshape(dims);
            x = if idx == n - 1 { x1 + x2 } else { relu(x1 + x2) };
        }
        let perm_out: [usize; R] = core::array::from_fn(|i| match i {
            0 => 0,
            i if i == R - 1 => 1,
            i => i + 1,
        });
        let x = relu(m.fc1.forward(x.permute(perm_out)));
        m.fc2.forward(x)
    }

    /// max |a - b| relative to max(1, max |b|).
    fn rel_err(a: Tensor<1>, b: Tensor<1>) -> f32 {
        let scale = b.clone().abs().max().into_scalar::<f32>().max(1.0);
        (a - b).abs().max().into_scalar::<f32>() / scale
    }

    fn flat<const R: usize>(t: Tensor<R>) -> Tensor<1> {
        let n: usize = t.dims().iter().product();
        t.reshape([n])
    }

    /// f32, and each output is a sum of at most 128 O(1) products taken in a
    /// different order than the reference, so ~1e-7 per term; 1e-5 leaves
    /// headroom without hiding a wrong weight (which gives O(1) errors).
    const TOL: f32 = 1e-5;

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
        let mut shape = vec![2];
        shape.extend_from_slice(spatial);
        shape.push(data_channels + modes.len());
        let shape: [usize; R] = shape.try_into().unwrap();
        let x = Tensor::<R>::random(shape, Distribution::Normal(0.0, 1.0), &device);

        let new = model.forward(x.clone());
        let old = reference_forward(&model, x);
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
        let x0 = Tensor::<4>::random([2, 8, 6, 3], Distribution::Normal(0.0, 1.0), &device);
        let probe = Tensor::<4>::random([2, 8, 6, 1], Distribution::Normal(0.0, 1.0), &device);

        let grads_of = |f: &dyn Fn(Tensor<4>) -> Tensor<4>| {
            let x = x0.clone().detach().require_grad();
            let grads = (f(x.clone()) * probe.clone()).sum().backward();
            let mut out: Vec<(String, Tensor<1>)> =
                vec![("input".into(), flat(x.grad(&grads).expect("input grad")))];
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

        let new = grads_of(&|x| model.forward(x));
        let old = grads_of(&|x| reference_forward(&model, x));
        for ((name, g_new), (_, g_old)) in new.into_iter().zip(old) {
            let err = rel_err(g_new, g_old);
            println!("grad {name} rel err {err:e}");
            assert!(err <= TOL, "gradient of {name} differs: {err:e}");
        }
    }
}

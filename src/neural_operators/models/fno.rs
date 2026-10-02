//! Generic template for FNO for n dimensions

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        Linear, LinearConfig,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::{DType, Device, FloatDType, Int, activation::relu, ops::PadMode},
};

use crate::neural_operators::layers::spectral_convolution::{SpectralConv, SpectralInit};

// FNO Architecture
#[derive(Config, Debug)]
pub struct FNOConfig {
    pub modes: Vec<usize>, // length D = R - 2
    #[config(default = 32)]
    pub hidden_channels: usize,
    /// Input data channels `C`, problem specific: 1 (coefficients), 10 (3D
    /// problem, stacked timesteps). Grid coordinates are not counted:
    /// [`FNO::forward`] appends the `D` coordinate channels itself, so the
    /// lifting `fc0` takes `C + D` channels.
    pub data_channels: usize,
    pub out_channels: usize,
    #[config(default = 4)]
    pub n_layers: usize,
    /// Initialisation of every spectral layer's weights. `None` is the Li et
    /// al. default, [`SpectralInit::LiUniform`]. An `Option` so that saved
    /// configs without this field still load (as `None`).
    pub spectral_init: Option<SpectralInit>,
    /// Domain padding for non-periodic problems: `p` zero cells appended at
    /// the **end** of every spatial axis after the lifting `fc0`, and cropped
    /// off again after the last spectral layer, before the projection `fc1`.
    /// This is the one-sided `F.pad(x, [0, p, 0, p])` of Li et al.'s
    /// `fourier_2d.py` (which uses 9 for Darcy), so the FFT's implied
    /// periodicity wraps through a zero buffer instead of joining opposite
    /// boundaries directly.
    ///
    /// `None` (the default) and `Some(0)` both mean no padding. An `Option`
    /// so that saved configs without this field still load (as `None`).
    ///
    /// - Every spatial axis is padded by the same `p`, also in 3D (Li et al.'s
    ///   `fourier_3d.py` pads only the last, time, axis).
    /// - The padded extent `n + p` is used as is, not rounded up to a power
    ///   of two; any grid size is supported.
    /// - The spectral layers run on `(n + p)^D` points, and `modes` are
    ///   checked against the padded extent `n + p`, not `n`.
    pub padding: Option<usize>,
}

#[derive(Module, Debug)]
pub struct FNO<const R: usize> {
    fc0: Linear,
    conv: Vec<SpectralConv<R>>,
    w: Vec<Conv1d>,
    fc1: Linear,
    fc2: Linear,
    /// Zero cells appended to each spatial axis; 0 disables padding.
    /// Not a parameter, so it is not part of a saved checkpoint.
    padding: usize,
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
            padding: self.padding.unwrap_or(0),
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

    /// Appends `p` zeros at the end of every spatial axis of a channels-first
    /// `[B, C, spatial..]` tensor.
    fn pad_spatial(x: Tensor<R>, p: usize) -> Tensor<R> {
        let pairs: [(usize, usize); R] =
            core::array::from_fn(|i| if i < 2 { (0, 0) } else { (0, p) });
        x.pad(pairs, PadMode::Constant(0.0))
    }

    /// Keeps the leading `dims[i]` entries of every axis: the inverse of
    /// [`Self::pad_spatial`] given the unpadded `dims`.
    fn crop_spatial(x: Tensor<R>, dims: [usize; R]) -> Tensor<R> {
        let ranges: [core::ops::Range<usize>; R] = core::array::from_fn(|i| 0..dims[i]);
        x.slice(ranges)
    }

    /// Uniform coordinate channels for a channels-last `[B, s_1..s_D, C]`
    /// tensor of shape `dims`: returns `[B, s_1..s_D, D]` in `dtype` on
    /// `device`, the same value in every batch row.
    ///
    /// Each channel is one `arange` along its axis, divided by `n - 1`
    /// (`linspace` over the closed `[0, 1]`; a single point is 0), and
    /// broadcast over the other axes, in the reverse axis order documented on
    /// [`Self::forward`]. `fc0`'s grid rows are trained against that order,
    /// so saved checkpoints depend on it.
    ///
    /// Channels-last so that `forward` concatenates it with the input before
    /// its one permute, the data flow of the old host-built grid. (A
    /// channels-first concatenation after the permute was observed to zero
    /// some grid entries on the Metal backend.)
    ///
    /// Values match the old loaders' f64 `linspace` rounded to `dtype` to
    /// within 1 ulp at 1.0 (the B2 contract); bit for bit under correctly
    /// rounded division while `n - 1` is exact in `dtype` (`n <= 2^24` for
    /// f32), as on flex.
    ///
    /// Built from scratch, so it is a constant: never tracked by autodiff.
    fn grid_cl(dims: [usize; R], dtype: DType, device: &Device) -> Tensor<R> {
        let mut full = dims;
        full[R - 1] = 1;
        let channels: Vec<Tensor<R>> = (1..R - 1)
            .rev()
            .map(|axis| {
                let n = dims[axis];
                let coords =
                    Tensor::<1, Int>::arange(0..n as i64, device).cast(FloatDType::from(dtype));
                let coords = if n > 1 {
                    coords.div_scalar((n - 1) as f64)
                } else {
                    coords // [0], as linspace(0, 1, 1); avoids 0 / 0
                };
                let mut shape = [1; R];
                shape[axis] = n;
                coords.reshape(shape).expand(full)
            })
            .collect();
        Tensor::cat(channels, R - 1)
    }

    /// `[B, spatial.., data_channels]` → `[B, spatial.., out_channels]`.
    ///
    /// The input carries data channels only: the `D` coordinate channels are
    /// generated on the input's device, in its dtype, for its own spatial
    /// shape, and appended after the data before the lifting `fc0`. A change
    /// of resolution needs no new grid from the caller. Each coordinate is a
    /// uniform grid over the closed `[0, 1]`, one channel per spatial axis in
    /// **reverse axis order**:
    ///
    /// | D | channels after the data   | convention                          |
    /// |---|---------------------------|-------------------------------------|
    /// | 1 | `x(s_1)`                  | Li et al.'s 1D script               |
    /// | 2 | `x(s_2), y(s_1)`          | Li et al.'s 2D `np.meshgrid` 'xy'   |
    /// | 3 | `x(s_3), y(s_2), z(s_1)`  | same rule; not Li's `fourier_3d.py` |
    ///
    /// where `x(s_a)` varies along spatial axis `a` only. These are the
    /// channels the Burgers and Darcy loaders used to store, so checkpoints
    /// trained on those inputs are unaffected.
    ///
    /// Channels-first inside: the input is permuted once while it is only
    /// `C_in` wide, and the output once while it is one channel wide, so the
    /// hidden-width activations are never permuted or copied for layout.
    ///
    /// With [`FNOConfig::padding`] `p > 0`, the lifted activation is
    /// zero-padded by `p` at the end of each spatial axis before the spectral
    /// layers and cropped back before `fc1`; the output shape is unchanged.
    ///
    /// # Panics
    /// If the input's last axis is not [`FNOConfig::data_channels`] wide, e.g.
    /// an input that still carries grid channels.
    pub fn forward(&self, x: Tensor<R>) -> Tensor<R> {
        let dims = x.dims();
        let data_channels = self.fc0.weight.dims()[0] - (R - 2);
        assert_eq!(
            dims[R - 1],
            data_channels,
            "FNO<{R}> expects data_channels = {data_channels} input channels, got {}; \
             the model generates the grid coordinates, so inputs must not include them",
            dims[R - 1]
        );
        // [B, s.., C] ++ [B, s.., D] -> [B, s.., C + D]
        let grid = Self::grid_cl(dims, x.dtype(), &x.device());
        let x = Tensor::cat(vec![x, grid], R - 1);

        // [B, s.., C + D] -> [B, C + D, s..]
        let perm_in: [usize; R] = core::array::from_fn(|i| match i {
            0 => 0,
            1 => R - 1,
            i => i - 1,
        });
        let mut x = Self::linear_cf(&self.fc0, x.permute(perm_in));

        let unpadded = x.dims();
        if self.padding > 0 {
            x = Self::pad_spatial(x, self.padding);
        }

        let n = self.conv.len();
        for idx in 0..n {
            let x1 = self.conv[idx].forward(x.clone());
            let x2 = Self::conv1x1_cf(&self.w[idx], x);
            x = if idx == n - 1 { x1 + x2 } else { relu(x1 + x2) };
        }

        if self.padding > 0 {
            x = Self::crop_spatial(x, unpadded);
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
        assert_eq!(loaded.padding, None);
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
            padding: None,
        };

        let trained: FNO<3> = config.init::<3>(&ad);
        let mut save = BurnpackStore::from_bytes(None);
        trained.save_into(&mut save).expect("save weights");
        let bytes = save.get_bytes().expect("serialise weights");

        let mut loaded: FNO<3> = config.init::<3>(&plain);
        loaded
            .load_from(&mut BurnpackStore::from_bytes(Some(bytes)))
            .expect("load weights");

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
        let x = random_input::<R>(spatial, data_channels, &device);

        let new = model.forward(x.clone());
        let old = reference_forward(&model, with_old_grid(x));
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
        let old = grads_of(with_old_grid(x0), &|x| reference_forward(&model, x));
        for ((name, g_new), (_, g_old)) in new.into_iter().zip(old) {
            let err = rel_err(g_new, g_old);
            println!("grad {name} rel err {err:e}");
            assert!(err <= TOL, "gradient of {name} differs: {err:e}");
        }
    }

    // --- issue #11: domain padding ---

    /// Where and how a reference forward pads the domain.
    #[derive(Clone, Copy, Debug)]
    enum Pad {
        /// No padding: the pre-#11 forward.
        Off,
        /// `p` zeros at the end of each spatial axis after `fc0`, keep the
        /// leading `n` (Li et al. `fourier_2d.py`; what `FNO::forward` does).
        End(usize),
        /// As `End`, but keep the trailing `n` after the layers: a crop
        /// misaligned with the pad.
        EndCropTail(usize),
        /// Zeros appended to the raw channels-last input before `fc0`, so the
        /// padded cells hold `fc0`'s bias rather than 0.
        BeforeFc0(usize),
        /// `⌊p/2⌋` zeros before and `⌈p/2⌉` after each axis, centre crop.
        Symmetric(usize),
    }

    /// `x` with `before` zeros prepended and `after` appended on `dim`, built
    /// with `cat` so it is independent of `Tensor::pad`.
    fn zero_extend<const R: usize>(
        x: Tensor<R>,
        dim: usize,
        before: usize,
        after: usize,
    ) -> Tensor<R> {
        let device = x.device();
        let block = |len: usize| {
            let mut dims = x.dims();
            dims[dim] = len;
            Tensor::<R>::zeros(dims, &device)
        };
        let mut parts = Vec::new();
        if before > 0 {
            parts.push(block(before));
        }
        parts.push(x.clone());
        if after > 0 {
            parts.push(block(after));
        }
        Tensor::cat(parts, dim)
    }

    /// The pre-4.2 style forward (channels-last `Linear`s, `Conv1d` layers)
    /// with domain padding done by `cat`/`narrow` according to `pad`.
    fn reference_forward_padded<const R: usize>(m: &FNO<R>, x: Tensor<R>, pad: Pad) -> Tensor<R> {
        let spatial_axes = 1..R - 1; // channels-last: [B, s.., C]
        let n_in: Vec<usize> = x.dims()[spatial_axes.clone()].to_vec();

        let mut x = x;
        if let Pad::BeforeFc0(p) = pad {
            for d in spatial_axes.clone() {
                x = zero_extend(x, d, 0, p);
            }
        }
        let x = m.fc0.forward(x);
        let perm_in: [usize; R] = core::array::from_fn(|i| match i {
            0 => 0,
            1 => R - 1,
            i => i - 1,
        });
        let mut x = x.permute(perm_in); // [B, H, s..]
        for d in 2..R {
            x = match pad {
                Pad::End(p) | Pad::EndCropTail(p) => zero_extend(x, d, 0, p),
                Pad::Symmetric(p) => zero_extend(x, d, p / 2, p - p / 2),
                Pad::Off | Pad::BeforeFc0(_) => x,
            };
        }

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

        for d in 2..R {
            let n_d = n_in[d - 2];
            x = match pad {
                Pad::Off => x,
                Pad::End(_) | Pad::BeforeFc0(_) => x.narrow(d, 0, n_d),
                Pad::EndCropTail(p) => x.narrow(d, p, n_d),
                Pad::Symmetric(p) => x.narrow(d, p / 2, n_d),
            };
        }

        let perm_out: [usize; R] = core::array::from_fn(|i| match i {
            0 => 0,
            i if i == R - 1 => 1,
            i => i + 1,
        });
        let x = relu(m.fc1.forward(x.permute(perm_out)));
        m.fc2.forward(x)
    }

    /// `FNO::forward` exactly as it was before padding existed.
    fn pre_padding_forward<const R: usize>(m: &FNO<R>, x: Tensor<R>) -> Tensor<R> {
        let perm_in: [usize; R] = core::array::from_fn(|i| match i {
            0 => 0,
            1 => R - 1,
            i => i - 1,
        });
        let mut x = FNO::<R>::linear_cf(&m.fc0, x.permute(perm_in));
        let n = m.conv.len();
        for idx in 0..n {
            let x1 = m.conv[idx].forward(x.clone());
            let x2 = FNO::<R>::conv1x1_cf(&m.w[idx], x);
            x = if idx == n - 1 { x1 + x2 } else { relu(x1 + x2) };
        }
        let x = relu(FNO::<R>::linear_cf(&m.fc1, x));
        let x = FNO::<R>::linear_cf(&m.fc2, x);
        let perm_out: [usize; R] = core::array::from_fn(|i| match i {
            0 => 0,
            i if i == R - 1 => 1,
            i => i + 1,
        });
        x.permute(perm_out)
    }

    /// Every parameter, by name, as raw f32 values.
    fn params<const R: usize>(m: &FNO<R>) -> Vec<(String, Vec<f32>)> {
        use burn::store::{ModuleSnapshot, bridge::to_data};
        let mut out: Vec<_> = m
            .collect(None, None, false)
            .iter()
            .map(|t| {
                (
                    t.name.clone(),
                    to_data(t).unwrap().try_to_vec::<f32>().unwrap(),
                )
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// `to`'s parameters replaced by `from`'s, through a burnpack checkpoint.
    fn copy_weights<const R: usize>(from: &FNO<R>, mut to: FNO<R>) -> FNO<R> {
        use burn::store::{BurnpackStore, ModuleSnapshot};
        let mut save = BurnpackStore::from_bytes(None);
        from.save_into(&mut save).expect("save weights");
        let bytes = save.get_bytes().expect("serialise weights");
        to.load_from(&mut BurnpackStore::from_bytes(Some(bytes)))
            .expect("load weights");
        to
    }

    fn bits<const R: usize>(t: Tensor<R>) -> Vec<u32> {
        t.into_data()
            .try_to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    fn padded_config(modes: Vec<usize>, data_channels: usize, hidden: usize) -> FNOConfig {
        FNOConfig::new(modes, data_channels, 1)
            .with_hidden_channels(hidden)
            .with_n_layers(2)
    }

    fn random_input<const R: usize>(
        spatial: &[usize],
        channels: usize,
        device: &Device,
    ) -> Tensor<R> {
        let mut shape = vec![2];
        shape.extend_from_slice(spatial);
        shape.push(channels);
        let shape: [usize; R] = shape.try_into().unwrap();
        Tensor::<R>::random(shape, Distribution::Normal(0.0, 1.0), device)
    }

    /// `m` with `fc0`'s bias set to 1. The lifted field then sits ~1 away
    /// from the zero padding, so a wrong convention (padding before `fc0`,
    /// which fills the pad with this bias, or no padding at all) changes the
    /// output by O(1) relative amounts instead of by the size of Linear's
    /// small random default bias, which can fall below the controls' 1e-3.
    fn with_unit_lift_bias<const R: usize>(mut m: FNO<R>) -> FNO<R> {
        use burn::module::Param;
        let bias = m.fc0.bias.take().expect("fc0 has a bias");
        let ones = bias.val().ones_like();
        m.fc0.bias = Some(Param::from_tensor(ones));
        m
    }

    /// `None` and `Some(0)` take the unpadded path: bit for bit the pre-#11
    /// forward on the same weights.
    fn check_padding_off_is_bitwise<const R: usize>(modes: Vec<usize>, spatial: &[usize]) {
        let device = Device::default();
        let cfg = padded_config(modes, 1, 6);
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
            let m: FNO<4> = padded_config(vec![3, 2], 1, 6)
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
            padded_config(modes.clone(), data_channels, 6)
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
            padded_config(vec![3, 2], 1, 6)
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
            padded_config(vec![3, 2], 1, 6)
                .with_spectral_init(Some(SpectralInit::SymmetricUniform))
                .with_padding(Some(3))
                .init::<4>(&device),
        );
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
            for (i, layer) in model.conv.iter().enumerate() {
                for (k, (re, im)) in layer.corner_weights().into_iter().enumerate() {
                    out.push((
                        format!("conv[{i}].corner[{k}].re"),
                        flat(re.grad(&grads).expect("spectral re grad")),
                    ));
                    out.push((
                        format!("conv[{i}].corner[{k}].im"),
                        flat(im.grad(&grads).expect("spectral im grad")),
                    ));
                }
            }
            out
        };

        let new = grads_of(x0.clone(), &|x| model.forward(x));
        let old = grads_of(with_old_grid(x0), &|x| {
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

    /// Padding is not part of a checkpoint: weights saved from an unpadded
    /// model load into a padded one and back, unchanged, and each then runs
    /// its own configuration's forward.
    #[test]
    fn padding_does_not_change_checkpoint() {
        let device = Device::default();
        let cfg = padded_config(vec![3, 2], 1, 6)
            .with_spectral_init(Some(SpectralInit::SymmetricUniform));
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

    // --- issue #13: grid channels generated on the device ---

    use crate::neural_operators::data::{
        dataitem::HostFloat,
        grids::{GridPlacement, append_grid, grid_from_axes, uniform_grid},
    };
    use ndarray::{Array1, ArrayD, Dimension, IxDyn};

    /// The grid the loaders used to store, built independently of the model
    /// in f64: `uniform_grid` over `[0, 1]` per axis, in reverse axis order
    /// (Darcy's `grid.reverse()`; a no-op for Burgers' single axis).
    fn old_grid_f64(spatial: &[usize]) -> Vec<ArrayD<f64>> {
        let mut grid = uniform_grid(&vec![(0.0, 1.0); spatial.len()], spatial);
        grid.reverse();
        grid
    }

    /// Channels-last `x` with `grids` appended after the data, the way the
    /// loaders did it: `append_grid` in f64, then one rounding to f32
    /// (`HostFloat::from_f64`). `x`'s own f32 values round-trip exactly.
    fn with_grid<const R: usize>(x: Tensor<R>, grids: &[ArrayD<f64>]) -> Tensor<R> {
        use burn::tensor::TensorData;
        let dims = x.dims();
        let device = x.device();
        let vals: Vec<f64> = x.into_data().iter::<f32>().map(f64::from).collect();
        let data = ArrayD::from_shape_vec(IxDyn(&dims), vals).unwrap();
        let out = append_grid(data, grids, GridPlacement::AfterData);
        let shape = out.shape().to_vec();
        let out: Vec<f32> = out
            .iter()
            .map(|&v| <f32 as HostFloat>::from_f64(v))
            .collect();
        Tensor::<R>::from_data(TensorData::new(out, shape), &device)
    }

    /// The pre-#13 model input for data-only `x`.
    fn with_old_grid<const R: usize>(x: Tensor<R>) -> Tensor<R> {
        let spatial = x.dims()[1..R - 1].to_vec();
        with_grid(x, &old_grid_f64(&spatial))
    }

    /// `x` with the model's own grid appended (channels-last), for bitwise
    /// comparisons with forwards that expect a grid-augmented input.
    fn with_model_grid<const R: usize>(x: Tensor<R>) -> Tensor<R> {
        let grid = FNO::<R>::grid_cl(x.dims(), x.dtype(), &x.device());
        Tensor::cat(vec![x, grid], R - 1)
    }

    /// T1: `grid_cl` against the old loader grid, per element and channel.
    /// Returns the max |Δ| against the old grid rounded to f32.
    ///
    /// Contract (B2): |Δ| ≤ 1 ulp at 1.0 (`f32::EPSILON`); bit for bit is
    /// expected under correctly rounded division. Endpoints are exact, a
    /// single point is 0 (not 0/0), and every batch row is the same.
    fn check_grid_values<const R: usize>(spatial: &[usize]) -> f32 {
        let device = Device::default();
        let d = R - 2;
        let mut dims = [2; R];
        dims[1..R - 1].copy_from_slice(spatial);
        dims[R - 1] = 1;
        let grid = FNO::<R>::grid_cl(dims, DType::F32, &device);
        let mut want = dims;
        want[R - 1] = d;
        assert_eq!(grid.dims(), want, "grid shape for {spatial:?}");

        let got: Vec<f32> = grid.into_data().try_to_vec::<f32>().unwrap();
        let per_row = got.len() / 2;
        assert_eq!(
            got[..per_row]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            got[per_row..]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "batch rows differ for {spatial:?}"
        );

        let old = old_grid_f64(spatial);
        let mut max_delta = 0f32;
        for (at, idx) in ndarray::indices(spatial).into_iter().enumerate() {
            for (k, old_k) in old.iter().enumerate() {
                let g = got[at * d + k];
                let want = <f32 as HostFloat>::from_f64(old_k[idx.slice()]);
                assert!(g.is_finite(), "{spatial:?} channel {k} at {idx:?}: {g}");
                max_delta = max_delta.max((g - want).abs());
                // channel k varies along spatial axis d - 1 - k only
                let axis = d - 1 - k;
                let (i, n) = (idx[axis], spatial[axis]);
                if i == 0 {
                    assert_eq!(g, 0.0, "{spatial:?} channel {k} start at {idx:?}");
                } else if i == n - 1 {
                    assert_eq!(g, 1.0, "{spatial:?} channel {k} end at {idx:?}");
                }
            }
        }
        println!("grid {spatial:?}: max |grid_new - grid_old| = {max_delta:e}");
        assert!(
            max_delta <= f32::EPSILON,
            "grid {spatial:?} deviates from the old f64 grid by {max_delta:e} > 1 ulp"
        );
        max_delta
    }

    #[test]
    fn grid_matches_old_loader_grid() {
        let mut worst = 0f32;
        for s in [1, 2, 7, 256, 8192] {
            worst = worst.max(check_grid_values::<3>(&[s]));
        }
        // Non-square: a transposed grid or swapped channels would differ.
        worst = worst.max(check_grid_values::<4>(&[5, 7]));
        worst = worst.max(check_grid_values::<4>(&[7, 5]));
        worst = worst.max(check_grid_values::<4>(&[1, 3]));
        worst = worst.max(check_grid_values::<5>(&[4, 6, 5]));
        println!("grid: max |grid_new - grid_old| over all shapes = {worst:e}");
    }

    /// Hand-pinned 2 x 3 grid: channel 0 = x along axis 2 (length 3),
    /// channel 1 = y along axis 1 (length 2), as `np.meshgrid` 'xy'.
    #[test]
    fn grid_2d_order_is_pinned() {
        let device = Device::default();
        let grid = FNO::<4>::grid_cl([1, 2, 3, 1], DType::F32, &device);
        let got = grid.into_data().try_to_vec::<f32>().unwrap();
        // [1, 2, 3, 2] channels-last: (x, y) per point, row-major points.
        #[rustfmt::skip]
        let want = [
            0.0, 0.0,   0.5, 0.0,   1.0, 0.0, // row 0: y = 0
            0.0, 1.0,   0.5, 1.0,   1.0, 1.0, // row 1: y = 1
        ];
        assert_eq!(got, want);
    }

    /// T2 controls: with the grid wrong, the same comparison as
    /// `check_forward` / `check_padded_forward` must fail by ≥ 1e-3, so a
    /// model that dropped, transposed or mis-scaled its grid is caught.
    ///
    /// The `[0, 1)` grid (`i / n`) is off from `i / (n - 1)` by at most
    /// `1 / n`, so its control needs a coarse axis to clear 1e-3 reliably;
    /// hence the small 1D grid below.
    fn check_grid_controls<const R: usize>(
        modes: Vec<usize>,
        data_channels: usize,
        spatial: &[usize],
        padding: Option<usize>,
    ) {
        let device = Device::default();
        device.seed(13);
        let model = with_unit_lift_bias(
            padded_config(modes, data_channels, 6)
                .with_spectral_init(Some(SpectralInit::SymmetricUniform))
                .with_padding(padding)
                .init::<R>(&device),
        );
        let x = random_input::<R>(spatial, data_channels, &device);
        let out = flat(model.forward(x.clone()));
        let reference = |grids: &[ArrayD<f64>]| {
            let x = with_grid(x.clone(), grids);
            flat(match padding {
                Some(p) if p > 0 => reference_forward_padded(&model, x, Pad::End(p)),
                _ => reference_forward(&model, x),
            })
        };

        let err = rel_err(out.clone(), reference(&old_grid_f64(spatial)));
        println!("FNO<{R}> {spatial:?} pad {padding:?}: rel err vs old grid {err:e}");
        assert!(
            err <= TOL,
            "FNO<{R}> differs from the old-grid forward: {err:e}"
        );

        let zeros: Vec<ArrayD<f64>> = old_grid_f64(spatial)
            .iter()
            .map(|g| ArrayD::zeros(g.raw_dim()))
            .collect();
        let half_open: Vec<ArrayD<f64>> = {
            let axes: Vec<Array1<f64>> = spatial
                .iter()
                .map(|&n| Array1::from_shape_fn(n, |i| i as f64 / n as f64))
                .collect();
            let mut grid = grid_from_axes(&axes);
            grid.reverse();
            grid
        };
        let mut wrong = vec![("no grid", zeros), ("[0, 1) grid", half_open)];
        if spatial.len() > 1 {
            wrong.push((
                "'ij' order",
                uniform_grid(&vec![(0.0, 1.0); spatial.len()], spatial),
            ));
        }
        for (name, grids) in wrong {
            let err = rel_err(out.clone(), reference(&grids));
            println!("FNO<{R}> {spatial:?} pad {padding:?} vs {name}: rel err {err:e}");
            assert!(
                err >= 1e-3,
                "FNO<{R}> forward indistinguishable from {name}: {err:e}"
            );
        }
    }

    #[test]
    fn grid_controls_1d() {
        check_grid_controls::<3>(vec![4], 1, &[6], None);
        check_grid_controls::<3>(vec![4], 1, &[6], Some(5));
    }

    #[test]
    fn grid_controls_2d() {
        check_grid_controls::<4>(vec![3, 2], 1, &[8, 6], None);
        check_grid_controls::<4>(vec![3, 2], 3, &[8, 6], Some(3));
    }

    #[test]
    fn grid_controls_3d() {
        check_grid_controls::<5>(vec![2, 2, 2], 3, &[4, 6, 5], None);
        check_grid_controls::<5>(vec![2, 2, 2], 1, &[4, 6, 5], Some(2));
    }

    /// T3: one set of weights at two resolutions; each forward matches the
    /// old path with the grid built for that resolution.
    fn check_resolutions<const R: usize>(modes: Vec<usize>, resolutions: &[&[usize]]) {
        let device = Device::default();
        let model: FNO<R> = padded_config(modes, 1, 6).init::<R>(&device);
        for &spatial in resolutions {
            let x = random_input::<R>(spatial, 1, &device);
            let out = model.forward(x.clone());
            assert_eq!(&out.dims()[1..R - 1], spatial);
            let err = rel_err(flat(out), flat(reference_forward(&model, with_old_grid(x))));
            println!("FNO<{R}> at {spatial:?}: rel err {err:e}");
            assert!(err <= TOL, "FNO<{R}> at {spatial:?} differs: {err:e}");
        }
    }

    #[test]
    fn resolution_change_builds_that_grid() {
        check_resolutions::<3>(vec![4], &[&[16], &[48]]);
        check_resolutions::<4>(vec![3, 2], &[&[8, 6], &[16, 12]]);
    }

    /// Regression (review R1): on Metal the expanded grid came out different
    /// per batch row after earlier device work, silently corrupting some
    /// samples' coordinates. Two identical samples must give bit-identical
    /// outputs, match the old host-grid forward, and leave `grid_cl`'s rows
    /// equal to each other and to the f64 grid, all after real model work.
    /// The input is itself a `cat` of one sample, the case in which the old
    /// channels-first concatenation lost grid entries.
    fn check_batch_rows_identical<const R: usize>(modes: Vec<usize>, spatial: &[usize]) {
        let device = Device::default();
        device.seed(13);
        let model = with_unit_lift_bias(
            padded_config(modes, 1, 6)
                .with_spectral_init(Some(SpectralInit::SymmetricUniform))
                .init::<R>(&device),
        );
        let mut one = vec![1];
        one.extend_from_slice(spatial);
        one.push(1);
        let one: [usize; R] = one.try_into().unwrap();
        let sample = Tensor::<R>::random(one, Distribution::Normal(0.0, 1.0), &device);
        let x = Tensor::cat(vec![sample.clone(), sample], 0);

        let out = model.forward(x.clone());
        let rows = bits(out.clone());
        let half = rows.len() / 2;
        assert_eq!(
            rows[..half],
            rows[half..],
            "FNO<{R}> {spatial:?}: batch rows differ"
        );
        let err = rel_err(flat(out), flat(reference_forward(&model, with_old_grid(x))));
        println!("FNO<{R}> {spatial:?} batch-row check: rel err vs old grid {err:e}");
        assert!(
            err <= TOL,
            "FNO<{R}> {spatial:?} differs from the old-grid forward: {err:e}"
        );

        let _ = check_grid_values::<R>(spatial);
    }

    #[test]
    fn batch_rows_see_identical_grids() {
        check_batch_rows_identical::<3>(vec![4], &[6]);
        check_batch_rows_identical::<3>(vec![4], &[48]);
        check_batch_rows_identical::<4>(vec![3, 2], &[8, 6]);
        check_batch_rows_identical::<5>(vec![2, 2, 2], &[4, 6, 5]);
    }

    /// T5: an old-style input that still carries the grid is rejected.
    #[test]
    #[should_panic(expected = "data_channels = 1 input channels, got 3")]
    fn forward_rejects_grid_augmented_input() {
        let device = Device::default();
        let model: FNO<4> = padded_config(vec![3, 2], 1, 6).init::<4>(&device);
        let _ = model.forward(random_input::<4>(&[8, 6], 3, &device));
    }

    /// T6: the lift still takes `data_channels + D` inputs, so checkpoints
    /// written before #13 keep their parameter names and shapes.
    #[test]
    fn lift_width_includes_grid_channels() {
        let device = Device::default();
        let m1: FNO<3> = padded_config(vec![4], 1, 6).init::<3>(&device);
        let m2: FNO<4> = padded_config(vec![3, 2], 1, 6).init::<4>(&device);
        let m3: FNO<5> = padded_config(vec![2, 2, 2], 10, 6).init::<5>(&device);
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
        let cfg = padded_config(vec![3, 2], 1, 6);
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
}

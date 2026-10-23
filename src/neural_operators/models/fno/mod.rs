//! The Fourier Neural Operator, generic over tensor rank `R = D + 2`: [`FNOConfig`] and
//! [`FNO`].

// Burn's `#[derive(Config)]` writes `field: field` in the `new` it generates, in an
// impl block outside the struct's attributes (burn-derive `config/analyzer_struct.rs`).
#![allow(clippy::redundant_field_names)]

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

/// Hyperparameters of an [`FNO`], saved with every run
/// ([`RunDir::save_configs`](crate::neural_operators::runs::RunDir::save_configs)).
///
/// The spatial dimension `D` is `modes.len()`; [`FNOConfig::init`] builds an `FNO<R>`
/// with `R = D + 2`.
#[derive(Config, Debug)]
pub struct FNOConfig {
    /// Fourier modes kept per spatial axis, one entry per axis (length `D = R - 2`;
    /// CONVENTIONS §5).
    pub modes: Vec<usize>,
    /// Width of the hidden activations between the lifting `fc0` and the projection `fc1`.
    #[config(default = 32)]
    pub hidden_channels: usize,
    /// Input data channels `C`, problem specific: 1 (coefficients), 10 (3D
    /// problem, stacked timesteps). Grid coordinates are not counted:
    /// [`FNO::forward`] appends the `D` coordinate channels itself, so the
    /// lifting `fc0` takes `C + D` channels.
    pub data_channels: usize,
    /// Output channels of the projection `fc2`. Any width builds; the trainer's
    /// `flatten_pair` accepts only 1.
    pub out_channels: usize,
    /// Number of Fourier layers (a spectral convolution plus a pointwise convolution each).
    #[config(default = 4)]
    pub n_layers: usize,
    /// Initialisation of every spectral layer's weights. `None` is the Li et
    /// al. default, [`SpectralInit::LiUniform`]. An `Option` so that saved
    /// configs without this field still load (as `None`).
    pub spectral_init: Option<SpectralInit>,
    /// Domain padding `p` for non-periodic problems (CONVENTIONS §3). `None` (the
    /// default) and `Some(0)` both mean no padding. An `Option` so that saved configs
    /// without this field still load (as `None`).
    pub padding: Option<usize>,
}

/// A Fourier Neural Operator on rank-`R` channels-last tensors `[B, s_1..s_D, C]`
/// (CONVENTIONS §1): lifting `fc0`, [`FNOConfig::n_layers`] Fourier layers, projection
/// `fc1`, `fc2`. Built by [`FNOConfig::init`].
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
    /// Builds an `FNO<R>` on `device`, with freshly initialised weights.
    ///
    /// # Panics
    /// If `modes.len() != R - 2`. Modes that do not fit the grid panic later, in
    /// [`FNO::forward`].
    pub fn init<const R: usize>(&self, device: &Device) -> FNO<R> {
        assert_eq!(
            self.modes.len() + 2,
            R,
            "FNO<{R}> needs {} modes, config has {}",
            R - 2,
            self.modes.len()
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

    /// A kernel-size-1 `Conv1d` as a pointwise matmul (commit `52b8c5f`), using
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

    /// The coordinate channels of CONVENTIONS §2 for a channels-last `[B, s_1..s_D, C]`
    /// tensor of shape `dims`: returns `[B, s_1..s_D, D]` in `dtype` on `device`, the
    /// same value in every batch row. Each channel is one `arange` along its axis,
    /// divided by `n - 1` and broadcast over the other axes.
    ///
    /// Channels-last so that `forward` concatenates it with the input before its one
    /// permute: a channels-first concatenation after the permute was observed to zero
    /// some grid entries on the Metal backend.
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
        let mut channels: Vec<Tensor<R>> = (1..R - 1)
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
        // One channel (1D) is returned as is: a single-input `cat` of the
        // expanded channel, followed by `forward`'s `cat`, gives wrong values
        // on the fused GPU backends once the chain has run at another shape
        // (`upstream/burn/fused-cat-reused-at-new-shape/`).
        if channels.len() == 1 {
            return channels.pop().expect("one channel");
        }
        Tensor::cat(channels, R - 1)
    }

    /// `[B, spatial.., data_channels]` → `[B, spatial.., out_channels]`.
    ///
    /// The input carries data channels only: the model appends the `D` coordinate
    /// channels of CONVENTIONS §2 for the input's own resolution before the lifting
    /// `fc0`, so a change of resolution needs no new grid from the caller.
    ///
    /// Channels-first inside: the input is permuted once while it is only
    /// `C_in` wide, and the output once while it is one channel wide, so the
    /// hidden-width activations are never permuted or copied for layout.
    ///
    /// With [`FNOConfig::padding`] `p > 0`, the lifted activation is padded and cropped
    /// as in CONVENTIONS §3; the output shape is unchanged.
    ///
    /// # Panics
    /// If the input's last axis is not [`FNOConfig::data_channels`] wide, e.g.
    /// an input that still carries grid channels, or if [`FNOConfig::modes`] do not fit
    /// the (padded) grid ([`SpectralConv::forward`]).
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

        let mut x = Self::linear_cf(&self.fc0, to_channels_first(x));

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

        to_channels_last(x)
    }
}

/// `[B, s.., C] -> [B, C, s..]` (CONVENTIONS §1).
pub(crate) fn to_channels_first<const R: usize>(x: Tensor<R>) -> Tensor<R> {
    x.permute(core::array::from_fn::<usize, R, _>(|i| match i {
        0 => 0,
        1 => R - 1,
        i => i - 1,
    }))
}

/// `[B, C, s..] -> [B, s.., C]` (CONVENTIONS §1), the inverse of [`to_channels_first`].
pub(crate) fn to_channels_last<const R: usize>(x: Tensor<R>) -> Tensor<R> {
    x.permute(core::array::from_fn::<usize, R, _>(|i| match i {
        0 => 0,
        i if i == R - 1 => 1,
        i => i + 1,
    }))
}

#[cfg(test)]
mod tests;

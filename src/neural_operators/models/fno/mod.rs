//! The Fourier Neural Operator, generic over tensor rank `R = D + 2`: [`FNOConfig`] and
//! [`FNO`].

// Burn's `#[derive(Config)]` writes `field: field` in the `new` it generates, in an
// impl block outside the struct's attributes (burn-derive `config/analyzer_struct.rs`).
#![allow(clippy::redundant_field_names)]

use burn::{
    Tensor,
    config::Config,
    module::{Module, Param},
    nn::{
        Initializer, Linear, LinearConfig,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::{
        DType, Device, FloatDType, Int,
        activation::{
            gelu, hard_swish, leaky_relu, mish, relu, selu, sigmoid, silu, softsign, tanh,
        },
        ops::PadMode,
    },
};

use crate::neural_operators::layers::{
    channel_mlp::{ChannelMlp, ChannelMlpConfig},
    embedding::GridEmbeddingNd,
    norm::{Norm, NormKind},
    pointwise::{conv1x1, linear_cf},
    resample::resample,
    skip::{Skip, SkipKind},
    spectral_convolution::{SpectralConv, SpectralInit},
};

/// The channel MLP after each Fourier layer's convolution, as 2.0.0's `FNOBlocks` builds it
/// (library CM1.3b): 2.0.0's two-layer `ChannelMLP` with GELU, of the layer's width in and
/// out, and a skip from the layer's input added to its output.
#[derive(Config, Debug)]
pub struct BlockMlpConfig {
    /// Hidden width as a fraction of the layer's, rounded half to even as Python's `round`
    /// (2.0.0's `channel_mlp_expansion`).
    #[config(default = 0.5)]
    pub expansion: f64,
    /// The skip added to the MLP's output (2.0.0's `channel_mlp_skip`); `None` is none.
    #[config(default = "Some(SkipKind::SoftGating)")]
    pub skip: Option<SkipKind>,
}

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
    /// lifting `fc0` takes `C + D` channels (`C` with [`PositionalEmbedding::Disabled`]).
    pub data_channels: usize,
    /// Output channels of the projection `fc2`. Any width builds; the trainer's
    /// `flatten_pair` accepts only 1.
    pub out_channels: usize,
    /// Number of Fourier layers (a spectral convolution and a skip each, with the block
    /// options' parts).
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
    /// The nonlinearity after each Fourier layer but the last and after `fc1`. `None` (the
    /// default, and every config before optimisers CO1.1) is [`Activation::Relu`]. An
    /// `Option` so that saved configs without this field still load.
    pub activation: Option<Activation>,
    // The block options of 2.0.0's `FNOBlocks` (library CM1.3b; [`FNO::fourier_layers`]).
    // Each `None` is the original block, and an `Option` so that saved configs load as `None`.
    // The activation is `activation` above (2.0.0's `non_linearity`, default GELU).
    /// The skip added to each layer's convolution (2.0.0's `fno_skip`, default
    /// [`SkipKind::Linear`], without bias); `None` is the original pointwise convolution with
    /// bias.
    pub fno_skip: Option<SkipKind>,
    /// A channel MLP in each layer (2.0.0's `use_channel_mlp`, default on); `None` is none.
    pub channel_mlp: Option<BlockMlpConfig>,
    /// A norm, two per layer (2.0.0's `norm`); `None` is none.
    pub norm: Option<NormKind>,
    /// Activation and norm before the convolution (2.0.0's `preactivation`); `None` is
    /// `false`.
    pub preactivation: Option<bool>,
    // The lifting, projection and embedding of 2.0.0's `FNO` (library CM1.4). Each `None` is
    // today's model, and an `Option` so that saved configs load as `None`.
    /// The lifting as 2.0.0's two-layer channel MLP (`lifting_channel_ratio`, default 2):
    /// `fc0` `(C + D) → L`, the activation, then `L → W`, with `L = ⌊r · W⌋`; `L = 0` keeps
    /// one layer. `None` is `fc0` alone, `(C + D) → W`.
    pub lifting_channel_ratio: Option<f64>,
    /// The projection's hidden width `⌊r · W⌋` (2.0.0's `projection_channel_ratio`, default
    /// 2); `None` is 128.
    pub projection_channel_ratio: Option<f64>,
    /// The coordinate channels (2.0.0's `positional_embedding`); `None` is CONVENTIONS §2's
    /// grid.
    pub positional_embedding: Option<PositionalEmbedding>,
    // The options of 2.0.0's centred `SpectralConv` and its stabiliser (library CM1.5a).
    // Each `None` is today's model, and an `Option` so that saved configs load as `None`.
    /// The spectral convolution's bias (2.0.0's `SpectralConv` `bias`, default on): `[W]` per
    /// layer, added to the convolution's output, before the norm. `None` is `false`.
    pub conv_bias: Option<bool>,
    /// Separable spectral weights (2.0.0's `separable`): one weight per channel and mode
    /// ([`SpectralConv::new_separable`]). `None` is `false`.
    pub separable: Option<bool>,
    /// The modes stored per axis (2.0.0's `max_n_modes`), counted as [`FNOConfig::modes`],
    /// of which the leading `modes` are used ([`SpectralConv::with_modes`]). `None` is
    /// `modes`.
    pub max_modes: Option<Vec<usize>>,
    /// `tanh` on each convolution's input (2.0.0's `stabilizer="tanh"`); the skips take the
    /// input before it. `None` is `false`.
    pub tanh_stabilizer: Option<bool>,
    // 2.0.0's domain padding and resolution scaling (library CM1.5b). Each `None` is today's
    // model, and an `Option` so that saved configs load as `None`.
    /// 2.0.0's `domain_padding`, one fraction `f` per spatial axis: after the lifting,
    /// `round(f · s)` zeros (half to even, as Python) on **both** ends of each axis, cropped
    /// before the projection. Beside CONVENTIONS §3's one-sided [`FNOConfig::padding`], which
    /// must then be unset. `None` is none.
    pub domain_padding: Option<Vec<f64>>,
    /// 2.0.0's `resolution_scaling_factor`, one factor per spatial axis for each layer
    /// (`n_layers` × `D`): layer `l` maps extent `s` to `round(s · r)` (half to even), its
    /// convolution by [`SpectralConv::forward_to`], its skips by
    /// [`resample`]. With [`FNOConfig::domain_padding`], at most one layer may scale (as in
    /// 2.0.0), and `round(p · r)` cells are cropped per side of an axis padded by `p`; not with
    /// [`FNOConfig::padding`]. `None` is every factor 1.
    pub resolution_scaling_factor: Option<Vec<Vec<f64>>>,
}

/// The FNO's coordinate channels other than CONVENTIONS §2's grid
/// ([`FNOConfig::positional_embedding`], library CM1.4).
#[derive(Config, Debug)]
pub enum PositionalEmbedding {
    /// 2.0.0's `GridEmbeddingND`, its boundaries one per spatial axis: `D` channels in axis
    /// order, `j / s` on `[0, 1)` for [`GridEmbeddingNd::unit`] (2.0.0's `"grid"`).
    GridNd(GridEmbeddingNd),
    /// None (2.0.0's `positional_embedding=None`): `fc0` takes the `C` data channels only.
    Disabled,
}

/// The FNO's nonlinearity ([`FNOConfig::activation`], optimisers CO1.1).
#[derive(Config, Debug, PartialEq)]
pub enum Activation {
    /// ReLU, the default: `train_burgers`, `train_darcy` and every run before CO1.1.
    Relu,
    /// GELU in its erf form (Burn's `gelu`, PyTorch's default), as in Li et al.'s
    /// `fourier_1d.py` and PETScML's Burgers model (D10).
    Gelu,
    // Library CM1.3b: the other functions 2.0.0's `non_linearity` is commonly given, each
    // Burn's function of the same name; parity with torch in `tests/parity_fno_blocks.rs`.
    /// `torch.tanh`.
    Tanh,
    /// `torch.sigmoid`.
    Sigmoid,
    /// `F.silu` (swish): `x · sigmoid(x)`.
    Silu,
    /// `F.leaky_relu` at torch's default negative slope, 0.01.
    LeakyRelu,
    /// `F.selu`.
    Selu,
    /// `F.mish`: `x · tanh(softplus(x))`.
    Mish,
    /// `F.hardswish`.
    HardSwish,
    /// `F.softsign`: `x / (1 + |x|)`.
    Softsign,
}

impl Activation {
    /// The function applied elementwise to `x`, any shape.
    pub fn apply<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        match self {
            Activation::Relu => relu(x),
            Activation::Gelu => gelu(x),
            Activation::Tanh => tanh(x),
            Activation::Sigmoid => sigmoid(x),
            Activation::Silu => silu(x),
            Activation::LeakyRelu => leaky_relu(x, 0.01),
            Activation::Selu => selu(x),
            Activation::Mish => mish(x),
            Activation::HardSwish => hard_swish(x),
            Activation::Softsign => softsign(x),
        }
    }
}

/// A Fourier Neural Operator on rank-`R` channels-last tensors `[B, s_1..s_D, C]`
/// (CONVENTIONS §1): lifting `fc0`, [`FNOConfig::n_layers`] Fourier layers, projection
/// `fc1`, `fc2`. Built by [`FNOConfig::init`].
#[derive(Module, Debug)]
pub struct FNO<const R: usize> {
    fc0: Linear,
    conv: Vec<SpectralConv<R>>,
    /// The original FNO skip, a pointwise convolution with bias; empty when
    /// [`FNOConfig::fno_skip`] is set (then `fno_skips` holds the skips).
    w: Vec<Conv1d>,
    fc1: Linear,
    fc2: Linear,
    /// Zero cells appended to each spatial axis; 0 disables padding.
    /// Not a parameter, so it is not part of a saved checkpoint.
    padding: usize,
    /// The nonlinearity ([`FNOConfig::activation`]); not a parameter either.
    #[module(skip)]
    activation: Activation,
    // The block options (library CM1.3b), after the fields above so that their checkpoint
    // paths and parameter order are unchanged; empty with every option absent.
    /// The FNO skip of each layer when [`FNOConfig::fno_skip`] is set (`w` is then empty).
    fno_skips: Vec<Skip<R>>,
    /// The channel MLP of each layer, when [`FNOConfig::channel_mlp`] is set.
    channel_mlp: Vec<ChannelMlp>,
    /// The channel MLP's skip of each layer.
    channel_mlp_skips: Vec<Skip<R>>,
    /// Two per layer, `2i` and `2i + 1`, when [`FNOConfig::norm`] is set.
    norms: Vec<Norm>,
    preactivation: bool,
    /// Whether post-activation applies the activation a second time after the norm slot, as
    /// 2.0.0 does: only when a 2.0.0 block option (FNO skip, channel MLP, norm) is set. The
    /// original block, ReLU or GELU, applies it once.
    repeat_activation: bool,
    /// The lifting's second layer, `L → W` after `fc0`'s `(C + D) → L`, when
    /// [`FNOConfig::lifting_channel_ratio`] gives it one (library CM1.4); empty otherwise.
    lifting_out: Vec<Linear>,
    /// [`FNOConfig::positional_embedding`]; not a parameter.
    #[module(skip)]
    positional_embedding: Option<PositionalEmbedding>,
    /// Each layer's convolution bias, `[W]`, when [`FNOConfig::conv_bias`] is set (library
    /// CM1.5a); empty otherwise.
    conv_biases: Vec<Param<Tensor<1>>>,
    /// [`FNOConfig::tanh_stabilizer`].
    tanh_stabilizer: bool,
    /// [`FNOConfig::domain_padding`], empty when unset; not a parameter.
    #[module(skip)]
    domain_padding: Vec<f64>,
    /// [`FNOConfig::resolution_scaling_factor`], empty when unset; not a parameter.
    #[module(skip)]
    scaling: Vec<Vec<f64>>,
}

impl FNOConfig {
    /// Builds an `FNO<R>` on `device`, with freshly initialised weights.
    ///
    /// # Panics
    /// If `modes.len() != R - 2`, if a [`PositionalEmbedding::GridNd`] has not `R - 2`
    /// boundaries, if [`FNOConfig::projection_channel_ratio`] gives a width of 0, or if
    /// [`FNOConfig::max_modes`] has a count below `modes`' or not `R - 2` entries
    /// ([`SpectralConv::with_modes`]), if [`FNOConfig::domain_padding`] has not `R - 2`
    /// entries or is set with [`FNOConfig::padding`], or if
    /// [`FNOConfig::resolution_scaling_factor`] is not `n_layers` × `(R - 2)` positive factors,
    /// is set with [`FNOConfig::padding`], or scales more than one layer with
    /// [`FNOConfig::domain_padding`].
    /// Modes that do not fit the grid panic later, in [`FNO::forward`].
    pub fn init<const R: usize>(&self, device: &Device) -> FNO<R> {
        assert_eq!(
            self.modes.len() + 2,
            R,
            "FNO<{R}> needs {} modes, config has {}",
            R - 2,
            self.modes.len()
        );

        if let Some(PositionalEmbedding::GridNd(grid)) = &self.positional_embedding {
            assert_eq!(
                grid.boundaries.len(),
                R - 2,
                "the grid needs {} axes",
                R - 2
            );
        }
        if let Some(fractions) = &self.domain_padding {
            assert_eq!(
                fractions.len(),
                R - 2,
                "domain_padding needs {} axes",
                R - 2
            );
            assert!(
                self.padding.unwrap_or(0) == 0,
                "domain_padding and padding are exclusive"
            );
        }
        if let Some(factors) = &self.resolution_scaling_factor {
            assert!(
                factors.len() == self.n_layers
                    && factors
                        .iter()
                        .all(|f| f.len() == R - 2 && f.iter().all(|&r| r > 0.0)),
                "resolution_scaling_factor needs {} layers of {} positive factors",
                self.n_layers,
                R - 2
            );
            // CONVENTIONS §3's crop restores the unscaled extents.
            assert!(
                self.padding.unwrap_or(0) == 0,
                "padding with resolution_scaling_factor is not supported (domain_padding is)"
            );
            // 2.0.0 crops by one layer's factor (its unpad lookup fails otherwise).
            let scaled = factors
                .iter()
                .filter(|f| f.iter().any(|&r| r != 1.0))
                .count();
            assert!(
                self.domain_padding.is_none() || scaled <= 1,
                "domain_padding with resolution_scaling_factor needs at most one scaled layer"
            );
        }
        let coord_channels = coord_channels(self.positional_embedding.as_ref(), R - 2);
        let width = self.hidden_channels;
        // Python's `int(r · W)`, truncated, as 2.0.0.
        let ratio_width = |r: f64| (r * width as f64) as usize;
        let lifting = self
            .lifting_channel_ratio
            .map(ratio_width)
            .filter(|&l| l > 0);
        let projection = self.projection_channel_ratio.map_or(128, ratio_width);
        assert!(projection > 0, "the projection ratio gives a width of 0");
        fn layers<T>(n: usize, build: impl Fn() -> T) -> Vec<T> {
            (0..n).map(|_| build()).collect()
        }
        let stored = self.max_modes.as_deref().unwrap_or(&self.modes);
        let init = self.spectral_init.clone().unwrap_or_default();
        let skips = |kind: Option<&SkipKind>| match kind {
            Some(kind) => layers(self.n_layers, || kind.init(width, width, device)),
            None => Vec::new(),
        };

        FNO {
            fc0: LinearConfig::new(
                self.data_channels + coord_channels,
                lifting.unwrap_or(self.hidden_channels),
            )
            .init(device),

            conv: (0..self.n_layers)
                .map(|_| {
                    let conv = if self.separable == Some(true) {
                        SpectralConv::<R>::new_separable(device, width, stored, init.clone())
                    } else {
                        SpectralConv::<R>::new_with_init(device, width, width, stored, init.clone())
                    };
                    match self.max_modes {
                        Some(_) => conv.with_modes(&self.modes),
                        None => conv,
                    }
                })
                .collect(),

            w: match self.fno_skip {
                Some(_) => Vec::new(),
                None => layers(self.n_layers, || {
                    Conv1dConfig::new(width, width, 1).init(device)
                }),
            },

            fc1: LinearConfig::new(self.hidden_channels, projection).init(device),
            fc2: LinearConfig::new(projection, self.out_channels).init(device),
            padding: self.padding.unwrap_or(0),
            activation: self.activation.clone().unwrap_or(Activation::Relu),
            fno_skips: skips(self.fno_skip.as_ref()),
            channel_mlp: match &self.channel_mlp {
                Some(mlp) => {
                    let hidden = (width as f64 * mlp.expansion).round_ties_even() as usize;
                    layers(self.n_layers, || {
                        ChannelMlpConfig::new(width, width, hidden).init(device)
                    })
                }
                None => Vec::new(),
            },
            channel_mlp_skips: skips(self.channel_mlp.as_ref().and_then(|m| m.skip.as_ref())),
            norms: match &self.norm {
                Some(kind) => layers(2 * self.n_layers, || kind.init(width, device)),
                None => Vec::new(),
            },
            preactivation: self.preactivation.unwrap_or(false),
            repeat_activation: self.fno_skip.is_some()
                || self.channel_mlp.is_some()
                || self.norm.is_some(),
            lifting_out: lifting
                .map(|l| vec![LinearConfig::new(l, width).init(device)])
                .unwrap_or_default(),
            positional_embedding: self.positional_embedding.clone(),
            conv_biases: match self.conv_bias {
                // 2.0.0's `init_std="auto"`, √(2 / (in + out)), times a standard normal.
                Some(true) => layers(self.n_layers, || {
                    let std = (2.0 / (2 * width) as f64).sqrt();
                    Initializer::Normal { mean: 0.0, std }.init([width], device)
                }),
                _ => Vec::new(),
            },
            tanh_stabilizer: self.tanh_stabilizer == Some(true),
            domain_padding: self.domain_padding.clone().unwrap_or_default(),
            scaling: self.resolution_scaling_factor.clone().unwrap_or_default(),
        }
    }
}

impl<const R: usize> FNO<R> {
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

    /// The configured nonlinearity ([`FNOConfig::activation`]).
    fn activate(&self, x: Tensor<R>) -> Tensor<R> {
        self.activation.apply(x)
    }

    /// `[B, spatial.., data_channels]` → `[B, spatial.., out_channels]`.
    ///
    /// The input carries data channels only: the model appends the `D` coordinate
    /// channels of CONVENTIONS §2 (or of [`FNOConfig::positional_embedding`]) for the
    /// input's own resolution before the lifting `fc0`, so a change of resolution needs no
    /// new grid from the caller.
    ///
    /// Channels-first inside: the input is permuted once while it is only
    /// `C_in` wide, and the output once while it is one channel wide, so the
    /// hidden-width activations are never permuted or copied for layout.
    ///
    /// With [`FNOConfig::padding`] `p > 0`, the lifted activation is padded and cropped
    /// as in CONVENTIONS §3; the output shape is unchanged. So with
    /// [`FNOConfig::domain_padding`]; with [`FNOConfig::resolution_scaling_factor`] the
    /// output's extents are the scaled ones.
    ///
    /// # Panics
    /// If the input's last axis is not [`FNOConfig::data_channels`] wide, e.g.
    /// an input that still carries grid channels, or if [`FNOConfig::modes`] do not fit
    /// the (padded) grid ([`SpectralConv::forward`]), or if the norm is AdaIN (which needs
    /// [`FNO::forward_with_embedding`]).
    pub fn forward(&self, x: Tensor<R>) -> Tensor<R> {
        self.forward_with_embedding(x, None)
    }

    /// [`FNO::forward`] with an `embedding` (`[embed_dim]`) for the AdaIN norms
    /// ([`FNOConfig::norm`]); the other norms ignore it.
    ///
    /// # Panics
    /// As [`FNO::forward`], and for AdaIN without an embedding.
    pub fn forward_with_embedding(&self, x: Tensor<R>, embedding: Option<Tensor<1>>) -> Tensor<R> {
        self.forward_full(x, embedding, None)
    }

    /// [`FNO::forward`] with the last layer's output on the spatial extents `output_shape`
    /// (2.0.0's `forward(x, output_shape=…)` with a tuple; library CM1.5b): `[B, spatial..,
    /// data_channels]` → `[B, output_shape.., out_channels]`.
    ///
    /// # Panics
    /// As [`FNO::forward`]; if `output_shape` has not `R - 2` positive entries, or with
    /// [`FNOConfig::domain_padding`] (2.0.0 finds no crop for a shape it did not compute) or
    /// [`FNOConfig::padding`] (its crop restores the input's extents).
    pub fn forward_with_output_shape(&self, x: Tensor<R>, output_shape: &[usize]) -> Tensor<R> {
        assert!(
            self.domain_padding.is_empty() && self.padding == 0,
            "an output shape with domain padding or padding is not supported"
        );
        self.forward_full(x, None, Some(output_shape))
    }

    /// The forward of [`FNO::forward_with_embedding`] and [`FNO::forward_with_output_shape`].
    fn forward_full(
        &self,
        x: Tensor<R>,
        embedding: Option<Tensor<1>>,
        output_shape: Option<&[usize]>,
    ) -> Tensor<R> {
        let dims = x.dims();
        let data_channels =
            self.fc0.weight.dims()[0] - coord_channels(self.positional_embedding.as_ref(), R - 2);
        assert_eq!(
            dims[R - 1],
            data_channels,
            "FNO<{R}> expects data_channels = {data_channels} input channels, got {}; \
             the model generates the grid coordinates, so inputs must not include them",
            dims[R - 1]
        );
        // [B, s.., C] ++ [B, s.., D] -> [B, s.., C + D]
        let (dtype, device) = (x.dtype(), x.device());
        let x = match &self.positional_embedding {
            None => Tensor::cat(vec![x, Self::grid_cl(dims, dtype, &device)], R - 1),
            Some(PositionalEmbedding::GridNd(grid)) => {
                Tensor::cat(vec![x, grid.grid(dims, R - 1, dtype, &device)], R - 1)
            }
            Some(PositionalEmbedding::Disabled) => x,
        };

        let mut x = linear_cf(&self.fc0, to_channels_first(x));
        for layer in &self.lifting_out {
            x = linear_cf(layer, self.activate(x));
        }

        let unpadded = x.dims();
        if self.padding > 0 {
            x = Self::pad_spatial(x, self.padding);
        }
        // 2.0.0's `DomainPadding`: per axis `round(f · s)` on both ends.
        let pads: Vec<usize> = self
            .domain_padding
            .iter()
            .zip(&unpadded[2..])
            .map(|(f, &s)| (f * s as f64).round_ties_even() as usize)
            .collect();
        if !pads.is_empty() {
            let pairs: [(usize, usize); R] = core::array::from_fn(|i| {
                if i < 2 {
                    (0, 0)
                } else {
                    (pads[i - 2], pads[i - 2])
                }
            });
            x = x.pad(pairs, PadMode::Constant(0.0));
        }

        x = self.layers_to(x, embedding, output_shape);

        if self.padding > 0 {
            x = Self::crop_spatial(x, unpadded);
        }
        if !pads.is_empty() {
            let dims = x.dims();
            let ranges: [core::ops::Range<usize>; R] = core::array::from_fn(|i| {
                if i < 2 {
                    return 0..dims[i];
                }
                // At most one layer scales with domain padding (`init`): its factor.
                let scale: f64 = self.scaling.iter().map(|r| r[i - 2]).product();
                let crop = (pads[i - 2] as f64 * scale).round_ties_even() as usize;
                crop..dims[i] - crop
            });
            x = x.slice(ranges);
        }

        let x = self.activate(linear_cf(&self.fc1, x));
        let x = linear_cf(&self.fc2, x);

        to_channels_last(x)
    }

    /// The Fourier layers alone, on the lifted channels-first activation: `[B, W, s..]` →
    /// `[B, W, s..]`, `W` = [`FNOConfig::hidden_channels`]. Each layer is 2.0.0's `FNOBlocks`
    /// at that index (`fno_block.py`), post- or pre-activation by
    /// [`FNOConfig::preactivation`]; with every block option absent, the original block:
    /// `act(conv(x) + w(x))`, `act` ReLU or GELU ([`FNOConfig::activation`]), without the
    /// activation after the last layer.
    ///
    /// `embedding` (`[embed_dim]`) conditions every AdaIN norm, as 2.0.0's
    /// `set_ada_in_embeddings` with one embedding; the other norms ignore it.
    ///
    /// # Panics
    /// If the norm is AdaIN and `embedding` is `None`; as [`SpectralConv::forward`] if the
    /// modes do not fit the grid.
    pub fn fourier_layers(&self, x: Tensor<R>, embedding: Option<Tensor<1>>) -> Tensor<R> {
        self.layers_to(x, embedding, None)
    }

    /// [`FNO::fourier_layers`], each layer's output on its scaled extents
    /// ([`FNOConfig::resolution_scaling_factor`]), the last one's on `output_shape` if given.
    fn layers_to(
        &self,
        mut x: Tensor<R>,
        embedding: Option<Tensor<1>>,
        output_shape: Option<&[usize]>,
    ) -> Tensor<R> {
        let embedding = embedding.as_ref();
        let n = self.conv.len();
        for idx in 0..n {
            let dims = x.dims();
            let out: Vec<usize> = match (output_shape, self.scaling.get(idx)) {
                (Some(shape), _) if idx + 1 == n => shape.to_vec(),
                (_, Some(factors)) => dims[2..]
                    .iter()
                    .zip(factors)
                    .map(|(&s, r)| (s as f64 * r).round_ties_even() as usize)
                    .collect(),
                _ => dims[2..].to_vec(),
            };
            x = if self.preactivation {
                self.pre_block(idx, x, embedding, &out)
            } else {
                self.post_block(idx, x, embedding, &out)
            };
        }
        x
    }

    /// Layer `idx`, 2.0.0's `forward_with_postactivation`, its output on the spatial extents
    /// `out`. With every option absent: `act(conv(x) + w(x))`, the activation skipped after
    /// the last layer.
    fn post_block(
        &self,
        idx: usize,
        x: Tensor<R>,
        embedding: Option<&Tensor<1>>,
        out: &[usize],
    ) -> Tensor<R> {
        let last = idx + 1 == self.conv.len();
        let mlp_skip = self.channel_mlp_skip(idx, &x, out);
        let x_fno = self.norm(2 * idx, self.spectral(idx, x.clone(), out), embedding);
        let mut x = x_fno + resample(self.fno_skip(idx, x), out);
        if !last {
            x = self.activate(x);
        }
        x = self.norm(2 * idx + 1, self.channel_mlp(idx, x, mlp_skip), embedding);
        if !last && self.repeat_activation {
            x = self.activate(x);
        }
        x
    }

    /// Layer `idx`, 2.0.0's `forward_with_preactivation`, its output on the extents `out`.
    fn pre_block(
        &self,
        idx: usize,
        x: Tensor<R>,
        embedding: Option<&Tensor<1>>,
        out: &[usize],
    ) -> Tensor<R> {
        let last = idx + 1 == self.conv.len();
        let x = self.norm(2 * idx, self.activate(x), embedding);
        let mlp_skip = self.channel_mlp_skip(idx, &x, out);
        let mut x = self.spectral(idx, x.clone(), out) + resample(self.fno_skip(idx, x), out);
        if !last {
            x = self.activate(x);
        }
        let x = self.norm(2 * idx + 1, x, embedding);
        self.channel_mlp(idx, x, mlp_skip)
    }

    /// Layer `idx`'s spectral convolution of `x` (`[B, W, s..]`) onto the extents `out`:
    /// `tanh` first with [`FNOConfig::tanh_stabilizer`], the bias after with
    /// [`FNOConfig::conv_bias`], as in 2.0.0's `FNOBlocks`; the convolution alone with neither.
    fn spectral(&self, idx: usize, x: Tensor<R>, out: &[usize]) -> Tensor<R> {
        let x = if self.tanh_stabilizer { tanh(x) } else { x };
        let y = self.conv[idx].forward_to(x, out);
        match self.conv_biases.get(idx) {
            Some(bias) => {
                let mut shape = [1; R];
                shape[1] = bias.dims()[0];
                y + bias.val().reshape(shape)
            }
            None => y,
        }
    }

    /// Layer `idx`'s FNO skip: the configured one, or the original `w`.
    fn fno_skip(&self, idx: usize, x: Tensor<R>) -> Tensor<R> {
        match self.fno_skips.get(idx) {
            Some(skip) => skip.forward(x),
            None => conv1x1(&self.w[idx], x),
        }
    }

    /// Layer `idx`'s channel-MLP skip of `x`, resampled onto the extents `out`, if it has one.
    fn channel_mlp_skip(&self, idx: usize, x: &Tensor<R>, out: &[usize]) -> Option<Tensor<R>> {
        self.channel_mlp_skips
            .get(idx)
            .map(|skip| resample(skip.forward(x.clone()), out))
    }

    /// Layer `idx`'s channel MLP plus its skip, or `x` unchanged without an MLP.
    fn channel_mlp(&self, idx: usize, x: Tensor<R>, skip: Option<Tensor<R>>) -> Tensor<R> {
        let Some(mlp) = self.channel_mlp.get(idx) else {
            return x;
        };
        match skip {
            Some(skip) => mlp.forward(x) + skip,
            None => mlp.forward(x),
        }
    }

    /// Norm `k` (two per layer), or `x` unchanged without a norm.
    fn norm(&self, k: usize, x: Tensor<R>, embedding: Option<&Tensor<1>>) -> Tensor<R> {
        match self.norms.get(k) {
            Some(norm) => norm.forward(x, embedding),
            None => x,
        }
    }
}

/// The coordinate channels `embedding` appends in `d` spatial dimensions: `d`, or 0 when
/// [`PositionalEmbedding::Disabled`].
fn coord_channels(embedding: Option<&PositionalEmbedding>, d: usize) -> usize {
    match embedding {
        Some(PositionalEmbedding::Disabled) => 0,
        None | Some(PositionalEmbedding::GridNd(_)) => d,
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

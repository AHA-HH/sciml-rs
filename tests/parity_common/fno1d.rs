//! 2.0.0's 1D `FNO` at its defaults, laid out in Burn for CM0.2's `fno1d_{f32,f64}`
//! fixtures (`FNO(n_modes=(8,), in_channels=1, out_channels=1, hidden_channels=8)`), and
//! the hand-written map to it. Test code only: the library's 2.0.0 FNO is CM1.3–CM1.5.
//!
//! Built from Burn's `Conv1d` and `gelu` (the erf form, as torch's `F.gelu`) and the
//! crate's `SpectralConv`; the forward follows 2.0.0's `FNO.forward` and
//! `FNOBlocks.forward_with_postactivation`. 2.0.0's `fft_norm="forward"` puts the 1/n on
//! the forward transform and Burn's on the inverse; the spectral convolution is linear, so
//! the two agree without rescaling.

use burn::{
    module::Param,
    nn::conv::{Conv1d, Conv1dConfig},
    prelude::*,
    tensor::activation::gelu,
};
use ndarray::ArrayD;
use sciml_rs::neural_operators::layers::spectral_convolution::SpectralConv;

use crate::parity_common::{Fixture, MapEntry, Take, load_into};

pub const HIDDEN: usize = 8;
pub const N_LAYERS: usize = 4;
/// 2.0.0 keeps `n_modes // 2 + 1` modes of the last (real-FFT) axis: 8 → 5.
pub const MODES: usize = 5;

/// 2.0.0's 1D `FNO` at its defaults, with one data channel in and out.
#[derive(Module, Debug)]
pub struct Fno1dRef {
    /// Channel MLP, (1 data + 1 grid) → 16 → 8.
    pub lifting: Vec<Conv1d>,
    pub convs: Vec<SpectralConv<3>>,
    /// The spectral convolutions' biases, `[HIDDEN, 1]`.
    pub conv_biases: Vec<Param<Tensor<2>>>,
    /// Linear FNO skips, no bias.
    pub fno_skips: Vec<Conv1d>,
    /// Per block, a channel MLP 8 → 4 → 8.
    pub channel_mlp: Vec<Vec<Conv1d>>,
    /// Soft-gating channel-MLP skips, `[1, HIDDEN, 1]`, no bias.
    pub channel_mlp_skips: Vec<Param<Tensor<3>>>,
    /// Channel MLP, 8 → 16 → 1.
    pub projection: Vec<Conv1d>,
}

fn conv(i: usize, o: usize, bias: bool, device: &Device) -> Conv1d {
    Conv1dConfig::new(i, o, 1).with_bias(bias).init(device)
}

impl Fno1dRef {
    /// Arbitrary weights, to be replaced by [`load_into`].
    pub fn new(device: &Device) -> Self {
        let blocks = 0..N_LAYERS;
        Self {
            lifting: vec![conv(2, 16, true, device), conv(16, HIDDEN, true, device)],
            convs: blocks
                .clone()
                .map(|_| SpectralConv::new(device, HIDDEN, HIDDEN, &[MODES]))
                .collect(),
            conv_biases: blocks
                .clone()
                .map(|_| Param::from_tensor(Tensor::zeros([HIDDEN, 1], device)))
                .collect(),
            fno_skips: blocks
                .clone()
                .map(|_| conv(HIDDEN, HIDDEN, false, device))
                .collect(),
            channel_mlp: blocks
                .clone()
                .map(|_| vec![conv(HIDDEN, 4, true, device), conv(4, HIDDEN, true, device)])
                .collect(),
            channel_mlp_skips: blocks
                .map(|_| Param::from_tensor(Tensor::ones([1, HIDDEN, 1], device)))
                .collect(),
            projection: vec![conv(HIDDEN, 16, true, device), conv(16, 1, true, device)],
        }
    }

    /// `x`: `[batch, 1, n]` → `[batch, 1, n]`.
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, _, n] = x.dims();
        // GridEmbeddingND on [0, 1]: x_j = j / n, appended after the data. 2.0.0 builds it
        // in f32 and casts; here it is built in the run's dtype, the same only while j / n is
        // exact in f32 (n a power of two, as the fixture's 64).
        let grid = Tensor::<1, Int>::arange(0..n as i64, &x.device())
            .float()
            .div_scalar(n as f64)
            .cast(x.dtype())
            .reshape([1, 1, n])
            .repeat_dim(0, batch);
        let mut x = mlp(&self.lifting, Tensor::cat(vec![x, grid], 1));
        for i in 0..N_LAYERS {
            let last = i == N_LAYERS - 1;
            let skip_fno = self.fno_skips[i].forward(x.clone());
            let skip_mlp = x.clone() * self.channel_mlp_skips[i].val();
            let bias = self.conv_biases[i].val().unsqueeze_dim::<3>(0);
            x = self.convs[i].forward(x) + bias + skip_fno;
            if !last {
                x = gelu(x);
            }
            x = mlp(&self.channel_mlp[i], x) + skip_mlp;
            if !last {
                x = gelu(x);
            }
        }
        mlp(&self.projection, x)
    }
}

/// 2.0.0's `ChannelMLP`: GELU between layers, none after the last.
fn mlp(fcs: &[Conv1d], mut x: Tensor<3>) -> Tensor<3> {
    for (i, fc) in fcs.iter().enumerate() {
        x = fc.forward(x);
        if i + 1 < fcs.len() {
            x = gelu(x);
        }
    }
    x
}

/// The hand-written map: 2.0.0's `state_dict` names to [`Fno1dRef`]'s paths.
#[rustfmt::skip]
pub const MAP: &[MapEntry] = &[
    ("lifting.fcs.0.weight", "lifting.0.weight", Take::Whole),
    ("lifting.fcs.0.bias", "lifting.0.bias", Take::Whole),
    ("lifting.fcs.1.weight", "lifting.1.weight", Take::Whole),
    ("lifting.fcs.1.bias", "lifting.1.bias", Take::Whole),
    ("fno_blocks.convs.0.weight.tensor", "convs.0.weights_re.0", Take::Re),
    ("fno_blocks.convs.0.weight.tensor", "convs.0.weights_im.0", Take::Im),
    ("fno_blocks.convs.0.bias", "conv_biases.0", Take::Whole),
    ("fno_blocks.convs.1.weight.tensor", "convs.1.weights_re.0", Take::Re),
    ("fno_blocks.convs.1.weight.tensor", "convs.1.weights_im.0", Take::Im),
    ("fno_blocks.convs.1.bias", "conv_biases.1", Take::Whole),
    ("fno_blocks.convs.2.weight.tensor", "convs.2.weights_re.0", Take::Re),
    ("fno_blocks.convs.2.weight.tensor", "convs.2.weights_im.0", Take::Im),
    ("fno_blocks.convs.2.bias", "conv_biases.2", Take::Whole),
    ("fno_blocks.convs.3.weight.tensor", "convs.3.weights_re.0", Take::Re),
    ("fno_blocks.convs.3.weight.tensor", "convs.3.weights_im.0", Take::Im),
    ("fno_blocks.convs.3.bias", "conv_biases.3", Take::Whole),
    ("fno_blocks.fno_skips.0.conv.weight", "fno_skips.0.weight", Take::Whole),
    ("fno_blocks.fno_skips.1.conv.weight", "fno_skips.1.weight", Take::Whole),
    ("fno_blocks.fno_skips.2.conv.weight", "fno_skips.2.weight", Take::Whole),
    ("fno_blocks.fno_skips.3.conv.weight", "fno_skips.3.weight", Take::Whole),
    ("fno_blocks.channel_mlp.0.fcs.0.weight", "channel_mlp.0.0.weight", Take::Whole),
    ("fno_blocks.channel_mlp.0.fcs.0.bias", "channel_mlp.0.0.bias", Take::Whole),
    ("fno_blocks.channel_mlp.0.fcs.1.weight", "channel_mlp.0.1.weight", Take::Whole),
    ("fno_blocks.channel_mlp.0.fcs.1.bias", "channel_mlp.0.1.bias", Take::Whole),
    ("fno_blocks.channel_mlp.1.fcs.0.weight", "channel_mlp.1.0.weight", Take::Whole),
    ("fno_blocks.channel_mlp.1.fcs.0.bias", "channel_mlp.1.0.bias", Take::Whole),
    ("fno_blocks.channel_mlp.1.fcs.1.weight", "channel_mlp.1.1.weight", Take::Whole),
    ("fno_blocks.channel_mlp.1.fcs.1.bias", "channel_mlp.1.1.bias", Take::Whole),
    ("fno_blocks.channel_mlp.2.fcs.0.weight", "channel_mlp.2.0.weight", Take::Whole),
    ("fno_blocks.channel_mlp.2.fcs.0.bias", "channel_mlp.2.0.bias", Take::Whole),
    ("fno_blocks.channel_mlp.2.fcs.1.weight", "channel_mlp.2.1.weight", Take::Whole),
    ("fno_blocks.channel_mlp.2.fcs.1.bias", "channel_mlp.2.1.bias", Take::Whole),
    ("fno_blocks.channel_mlp.3.fcs.0.weight", "channel_mlp.3.0.weight", Take::Whole),
    ("fno_blocks.channel_mlp.3.fcs.0.bias", "channel_mlp.3.0.bias", Take::Whole),
    ("fno_blocks.channel_mlp.3.fcs.1.weight", "channel_mlp.3.1.weight", Take::Whole),
    ("fno_blocks.channel_mlp.3.fcs.1.bias", "channel_mlp.3.1.bias", Take::Whole),
    ("fno_blocks.channel_mlp_skips.0.weight", "channel_mlp_skips.0", Take::Whole),
    ("fno_blocks.channel_mlp_skips.1.weight", "channel_mlp_skips.1", Take::Whole),
    ("fno_blocks.channel_mlp_skips.2.weight", "channel_mlp_skips.2", Take::Whole),
    ("fno_blocks.channel_mlp_skips.3.weight", "channel_mlp_skips.3", Take::Whole),
    ("projection.fcs.0.weight", "projection.0.weight", Take::Whole),
    ("projection.fcs.0.bias", "projection.0.bias", Take::Whole),
    ("projection.fcs.1.weight", "projection.1.weight", Take::Whole),
    ("projection.fcs.1.bias", "projection.1.bias", Take::Whole),
];

/// The fixture `name` opened and loaded into a [`Fno1dRef`] on `device` through [`MAP`].
pub fn loaded(name: &str, device: &Device) -> (Fno1dRef, Fixture) {
    let mut fixture = Fixture::open(name);
    let mut model = Fno1dRef::new(device);
    load_into(&mut model, &mut fixture, MAP);
    (model, fixture)
}

/// The fixture `name` through [`loaded`]'s model: (output, reference, D3's bound for the
/// fixture's precision).
pub fn run(name: &str, device: &Device) -> (Tensor<3>, ArrayD<f64>, f64) {
    let (model, mut fixture) = loaded(name, device);
    let x = fixture.input::<3>(device);
    (model.forward(x), fixture.output(), fixture.precision.d3())
}

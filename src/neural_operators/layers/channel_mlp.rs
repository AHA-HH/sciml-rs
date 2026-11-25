//! The channel MLP of `neuraloperator` 2.0.0 (`layers/channel_mlp.py`, `ChannelMLP`): a stack
//! of pointwise (kernel-size-1) convolutions with an activation between them, the same at
//! every grid point, so it works at any resolution (library CM1.3a).

// Burn's `#[derive(Config)]` writes `field: field` in the `new` it generates.
#![allow(clippy::redundant_field_names)]

use super::pointwise::conv1x1;
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        activation::{Activation, ActivationConfig},
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::Device,
};

/// Hyperparameters of a [`ChannelMlp`]. 2.0.0's `ChannelMLP` without its dropout (default
/// 0, and a no-op in evaluation).
#[derive(Config, Debug)]
pub struct ChannelMlpConfig {
    /// Channels in.
    pub in_channels: usize,
    /// Channels out.
    pub out_channels: usize,
    /// Width of the layers between the first and the last.
    pub hidden_channels: usize,
    /// Number of pointwise layers, at least 1; 2.0.0's default 2.
    #[config(default = 2)]
    pub n_layers: usize,
    /// The activation between layers (none after the last); 2.0.0's default `F.gelu`, the
    /// erf form, is [`ActivationConfig::Gelu`].
    #[config(default = "ActivationConfig::Gelu")]
    pub activation: ActivationConfig,
}

/// 2.0.0's `ChannelMLP` on channels-first `[B, C, s_1..s_D]` tensors of any rank: `n_layers`
/// pointwise convolutions with bias, `in → hidden → … → hidden → out`, the activation after
/// every layer but the last. Built by [`ChannelMlpConfig::init`].
#[derive(Module, Debug)]
pub struct ChannelMlp {
    /// The layers, as 2.0.0's `fcs`: `Conv1d` of kernel size 1, weight `[out, in, 1]`.
    fcs: Vec<Conv1d>,
    activation: Activation,
}

impl ChannelMlpConfig {
    /// Builds a [`ChannelMlp`] on `device`, with Burn's default `Conv1d` initialisation.
    ///
    /// # Panics
    /// If `n_layers` is 0.
    pub fn init(&self, device: &Device) -> ChannelMlp {
        assert!(self.n_layers > 0, "a channel MLP needs at least one layer");
        let width = |i: usize| match i {
            0 => self.in_channels,
            i if i == self.n_layers => self.out_channels,
            _ => self.hidden_channels,
        };
        ChannelMlp {
            fcs: (0..self.n_layers)
                .map(|i| Conv1dConfig::new(width(i), width(i + 1), 1).init(device))
                .collect(),
            activation: self.activation.init(device),
        }
    }
}

impl ChannelMlp {
    /// `[B, in_channels, s..]` → `[B, out_channels, s..]`.
    ///
    /// # Panics
    /// If the input's channel axis (1) is not `in_channels` wide.
    pub fn forward<const R: usize>(&self, mut x: Tensor<R>) -> Tensor<R> {
        let (last, hidden) = self.fcs.split_last().expect("at least one layer");
        for fc in hidden {
            x = self.activation.forward(conv1x1(fc, x));
        }
        conv1x1(last, x)
    }
}

#[cfg(test)]
mod tests {
    use burn::{module::Param, tensor::Tolerance};

    use super::*;

    /// Two layers with ReLU between, against the sum written out by hand.
    #[test]
    fn two_layers_by_hand() {
        let device = Device::default();
        let mut mlp = ChannelMlpConfig::new(2, 1, 2)
            .with_activation(ActivationConfig::Relu)
            .init(&device);
        // Layer 0: identity weights, bias (0, -1); layer 1: weights (1, 2), bias 0.5.
        mlp.fcs[0].weight = Param::from_tensor(Tensor::<3>::from_data(
            [[[1.0], [0.0]], [[0.0], [1.0]]],
            &device,
        ));
        mlp.fcs[0].bias = Some(Param::from_tensor(Tensor::from_data([0.0, -1.0], &device)));
        mlp.fcs[1].weight = Param::from_tensor(Tensor::<3>::from_data([[[1.0], [2.0]]], &device));
        mlp.fcs[1].bias = Some(Param::from_tensor(Tensor::from_data([0.5], &device)));
        // x: one sample, 2 channels, a 1×2 grid: (3, 0.5) and (-1, 4).
        let x = Tensor::<4>::from_data([[[[3.0, -1.0]], [[0.5, 4.0]]]], &device);
        // Point 1: relu(3) + 2·relu(0.5 − 1) + 0.5 = 3.5; point 2: 0 + 2·3 + 0.5 = 6.5.
        let expected = Tensor::<4>::from_data([[[[3.5, 6.5]]]], &device);
        mlp.forward(x)
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());
    }

    #[test]
    fn widths_follow_the_config() {
        let mlp = ChannelMlpConfig::new(3, 5, 4)
            .with_n_layers(3)
            .init(&Device::default());
        let shapes: Vec<_> = mlp.fcs.iter().map(|fc| fc.weight.dims()).collect();
        assert_eq!(shapes, vec![[4, 3, 1], [4, 4, 1], [5, 4, 1]]);
        let y = mlp.forward(Tensor::<5>::zeros([2, 3, 4, 3, 2], &Device::default()));
        assert_eq!(y.dims(), [2, 5, 4, 3, 2]);
    }

    #[test]
    fn one_layer_maps_in_to_out() {
        let mlp = ChannelMlpConfig::new(3, 2, 7)
            .with_n_layers(1)
            .init(&Device::default());
        assert_eq!(mlp.fcs[0].weight.dims(), [2, 3, 1]);
    }

    #[test]
    #[should_panic(expected = "at least one layer")]
    fn zero_layers_panics() {
        ChannelMlpConfig::new(3, 2, 7)
            .with_n_layers(0)
            .init(&Device::default());
    }
}

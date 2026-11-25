//! Adaptive instance normalisation, as `neuraloperator` 2.0.0's `AdaIN`
//! (`layers/normalization_layers.py`; library CM1.3a). 2.0.0's other norms are Burn's own:
//! its `InstanceNorm()` is Burn's `InstanceNorm` with `affine = false`, and the group norm its
//! block builds, `nn.GroupNorm(1, C)`, is Burn's `GroupNorm` with one group.

// Burn's `#[derive(Config)]` writes `field: field` in the `new` it generates.
#![allow(clippy::redundant_field_names)]

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{InstanceNorm, InstanceNormConfig, Linear, LinearConfig},
    tensor::{Device, activation::gelu},
};

/// Width of the hidden layer of AdaIN's MLP, fixed in 2.0.0's `AdaIN`.
const HIDDEN: usize = 512;

/// Hyperparameters of an [`AdaIn`].
#[derive(Config, Debug)]
pub struct AdaInConfig {
    /// Length of the conditioning embedding.
    pub embed_dim: usize,
    /// Channels normalised.
    pub in_channels: usize,
    /// Added to the variance; 2.0.0's default.
    #[config(default = 1e-5)]
    pub epsilon: f64,
}

/// 2.0.0's `AdaIN` on channels-first `[B, C, s_1..s_D]` tensors of any rank: each sample and
/// channel normalised over the grid (instance norm), then scaled and shifted per channel by
/// weights an MLP, `Linear(embed_dim, 512) → GELU → Linear(512, 2C)`, computes from an
/// embedding. Built by [`AdaInConfig::init`].
#[derive(Module, Debug)]
pub struct AdaIn {
    /// `Linear(embed_dim, 512)`, weight `[embed_dim, 512]` (Burn's layout, torch's
    /// transposed).
    fc_hidden: Linear,
    /// `Linear(512, 2C)`: the scale is its first `C` outputs, the shift the last `C`.
    fc_out: Linear,
    norm: InstanceNorm,
}

impl AdaInConfig {
    /// Builds an [`AdaIn`] on `device`, with Burn's default `Linear` initialisation.
    pub fn init(&self, device: &Device) -> AdaIn {
        let linear = |i, o| LinearConfig::new(i, o).init(device);
        AdaIn {
            fc_hidden: linear(self.embed_dim, HIDDEN),
            fc_out: linear(HIDDEN, 2 * self.in_channels),
            norm: InstanceNormConfig::new(self.in_channels)
                .with_epsilon(self.epsilon)
                .with_affine(false)
                .init(device),
        }
    }
}

impl AdaIn {
    /// `x`: `[B, C, s..]`, `embedding`: `[embed_dim]` → `[B, C, s..]`. One embedding for the
    /// whole batch, as 2.0.0's `set_embedding`; here an argument rather than state.
    ///
    /// # Panics
    /// If `embedding` is not `embed_dim` long or `x` is not `C` channels wide.
    pub fn forward<const R: usize>(&self, x: Tensor<R>, embedding: Tensor<1>) -> Tensor<R> {
        let [e, _] = self.fc_hidden.weight.dims();
        assert_eq!(
            embedding.dims()[0],
            e,
            "AdaIn: embedding of length {} for embed_dim = {e}",
            embedding.dims()[0]
        );
        let c = self.norm.num_channels;
        let affine = self
            .fc_out
            .forward(gelu(self.fc_hidden.forward(embedding.unsqueeze::<2>())));
        let mut shape = [1; R];
        shape[1] = c;
        let scale = affine.clone().slice([0..1, 0..c]).reshape(shape);
        let shift = affine.slice([0..1, c..2 * c]).reshape(shape);
        self.norm.forward(x) * scale + shift
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        module::Param,
        tensor::{Distribution, Tolerance},
    };

    use super::*;

    /// With the last layer's weights zero, the scale and shift are its bias: per channel,
    /// the output is (x − mean) / √(var + ε) · a + b.
    #[test]
    fn scale_and_shift_come_from_the_mlp() {
        let device = Device::default();
        let mut ada = AdaInConfig::new(3, 2).init(&device);
        ada.fc_out.weight = Param::from_tensor(Tensor::zeros([HIDDEN, 4], &device));
        ada.fc_out.bias = Some(Param::from_tensor(Tensor::from_data(
            [2.0, -1.0, 0.5, 3.0],
            &device,
        )));
        // One sample, two channels, two points: each channel is (m − d, m + d), normalised to
        // (−1, 1) · d / √(d² + ε).
        let x = Tensor::<3>::from_data([[[1.0, 3.0], [-4.0, 0.0]]], &device);
        let y = ada.forward(x, Tensor::random([3], Distribution::Default, &device));
        let n = |d: f64| d / (d * d + 1e-5).sqrt();
        let expected = Tensor::<3>::from_data(
            [[
                [0.5 - 2.0 * n(1.0), 0.5 + 2.0 * n(1.0)],
                [3.0 + n(2.0), 3.0 - n(2.0)],
            ]],
            &device,
        );
        y.into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());
    }

    #[test]
    fn shapes_follow_the_config() {
        let device = Device::default();
        let ada = AdaInConfig::new(3, 4).init(&device);
        assert_eq!(ada.fc_hidden.weight.dims(), [3, HIDDEN]);
        assert_eq!(ada.fc_out.weight.dims(), [HIDDEN, 8]);
        let y = ada.forward(
            Tensor::<5>::ones([2, 4, 3, 2, 2], &device),
            Tensor::ones([3], &device),
        );
        assert_eq!(y.dims(), [2, 4, 3, 2, 2]);
    }

    #[test]
    #[should_panic(expected = "embedding of length 2")]
    fn wrong_embedding_length_panics() {
        let device = Device::default();
        let ada = AdaInConfig::new(3, 4).init(&device);
        ada.forward(
            Tensor::<4>::ones([1, 4, 2, 2], &device),
            Tensor::ones([2], &device),
        );
    }
}

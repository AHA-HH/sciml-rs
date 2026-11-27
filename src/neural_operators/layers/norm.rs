//! Adaptive instance normalisation, as `neuraloperator` 2.0.0's `AdaIN`
//! (`layers/normalization_layers.py`; library CM1.3a). 2.0.0's other norms are Burn's own:
//! its `InstanceNorm()` is Burn's `InstanceNorm` with `affine = false`, and the group norm its
//! block builds, `nn.GroupNorm(1, C)`, is Burn's `GroupNorm` with one group, and its
//! `BatchNorm` (torch's `BatchNorm{1,2,3}d`) is Burn's `BatchNorm` (library CM1.5c).

// Burn's `#[derive(Config)]` writes `field: field` in the `new` it generates.
#![allow(clippy::redundant_field_names)]

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig, GroupNorm, GroupNormConfig, InstanceNorm, InstanceNormConfig,
        Linear, LinearConfig,
    },
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

/// Which norm a block uses, as 2.0.0's `FNOBlocks(norm=…)`; [`NormKind::init`] builds it.
#[derive(Config, Debug)]
pub enum NormKind {
    /// `"instance_norm"`: 2.0.0's `InstanceNorm()`, without affine.
    Instance,
    /// `"group_norm"`: `nn.GroupNorm(1, C)`, one group, affine.
    Group,
    /// `"ada_in"`: [`AdaIn`] conditioned on an embedding of length `embed_dim` (2.0.0's
    /// `ada_in_features`).
    AdaIn {
        /// Length of the conditioning embedding.
        embed_dim: usize,
    },
    /// `"batch_norm"`: torch's `BatchNorm{1,2,3}d` (2.0.0's `BatchNorm`), affine, eps 1e-5,
    /// momentum 0.1, with running statistics (library CM1.5c).
    Batch,
}

/// A norm on channels-first `[B, C, s_1..s_D]` tensors of any rank, built by
/// [`NormKind::init`].
#[derive(Module, Debug)]
pub enum Norm {
    /// Burn's `InstanceNorm` without affine, eps 1e-5.
    Instance(InstanceNorm),
    /// Burn's `GroupNorm` with one group, eps 1e-5.
    Group(GroupNorm),
    /// The crate's [`AdaIn`], boxed: its MLP makes it far larger than the other variants.
    AdaIn(Box<AdaIn>),
    /// Burn's `BatchNorm`: the batch statistics on an autodiff device (training, updating
    /// the running ones), the running statistics otherwise (torch's eval mode). Burn updates
    /// the running variance with the biased batch variance, torch with the unbiased one.
    Batch(BatchNorm),
}

impl NormKind {
    /// Builds the norm of `channels` channels on `device`, as 2.0.0 initialises it.
    pub fn init(&self, channels: usize, device: &Device) -> Norm {
        match self {
            NormKind::Instance => Norm::Instance(
                InstanceNormConfig::new(channels)
                    .with_affine(false)
                    .init(device),
            ),
            NormKind::Group => Norm::Group(GroupNormConfig::new(1, channels).init(device)),
            NormKind::AdaIn { embed_dim } => Norm::AdaIn(Box::new(
                AdaInConfig::new(*embed_dim, channels).init(device),
            )),
            // Burn's defaults are torch's: γ = 1, β = 0, mean 0, variance 1.
            NormKind::Batch => Norm::Batch(BatchNormConfig::new(channels).init(device)),
        }
    }
}

impl Norm {
    /// `[B, C, s..]` → `[B, C, s..]`; `embedding` (`[embed_dim]`) conditions AdaIN and is
    /// ignored by the other norms.
    ///
    /// # Panics
    /// For AdaIN without an embedding, and as [`AdaIn::forward`].
    pub fn forward<const R: usize>(
        &self,
        x: Tensor<R>,
        embedding: Option<&Tensor<1>>,
    ) -> Tensor<R> {
        match self {
            Norm::Instance(norm) => norm.forward(x),
            Norm::Group(norm) => norm.forward(x),
            Norm::Batch(norm) => norm.forward(x),
            Norm::AdaIn(norm) => norm.forward(
                x,
                embedding.expect("an AdaIN norm needs an embedding").clone(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        module::{Param, RunningState},
        tensor::{Distribution, TensorData, Tolerance},
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
    #[should_panic(expected = "an AdaIN norm needs an embedding")]
    fn ada_in_norm_without_an_embedding_panics() {
        let device = Device::default();
        let norm = NormKind::AdaIn { embed_dim: 3 }.init(4, &device);
        norm.forward(Tensor::<3>::ones([1, 4, 2], &device), None);
    }

    #[test]
    fn norm_kinds_build_2_0_0_s_norms() {
        let device = Device::default();
        let Norm::Group(group) = NormKind::Group.init(4, &device) else {
            panic!("not a group norm")
        };
        assert_eq!((group.num_groups, group.num_channels), (1, 4));
        let Norm::Instance(instance) = NormKind::Instance.init(4, &device) else {
            panic!("not an instance norm")
        };
        assert!(instance.gamma.is_none());
    }

    /// On a device without autodiff the batch norm uses its running statistics, per
    /// channel: (x − mean) / √(var + ε) · γ + β, whatever the batch.
    #[test]
    fn batch_norm_uses_the_running_statistics_in_eval() {
        let device = Device::default();
        let Norm::Batch(mut batch) = NormKind::Batch.init(2, &device) else {
            panic!("not a batch norm")
        };
        assert_eq!(
            batch.gamma.val().into_data(),
            TensorData::from([1.0f32, 1.0])
        );
        assert_eq!(
            batch.beta.val().into_data(),
            TensorData::from([0.0f32, 0.0])
        );
        batch.gamma = Param::from_tensor(Tensor::from_data([2.0, -1.0], &device));
        batch.beta = Param::from_tensor(Tensor::from_data([0.5, 3.0], &device));
        batch.running_mean = RunningState::new(Tensor::from_data([1.0, -2.0], &device));
        batch.running_var = RunningState::new(Tensor::from_data([4.0, 0.25], &device));
        let x = Tensor::<3>::from_data([[[1.0, 3.0], [-4.0, 0.0]]], &device);
        let y = Norm::Batch(batch).forward(x, None);
        let n = |x: f64, mean: f64, var: f64| (x - mean) / (var + 1e-5).sqrt();
        let expected = Tensor::<3>::from_data(
            [[
                [0.5 + 2.0 * n(1.0, 1.0, 4.0), 0.5 + 2.0 * n(3.0, 1.0, 4.0)],
                [3.0 - n(-4.0, -2.0, 0.25), 3.0 - n(0.0, -2.0, 0.25)],
            ]],
            &device,
        );
        y.into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());
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

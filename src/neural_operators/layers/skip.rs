//! The skip connections of `neuraloperator` 2.0.0 (`layers/skip_connections.py`,
//! `skip_connection`): identity, linear and soft-gating, on channels-first tensors of any rank
//! (library CM1.3a).

use burn::{
    Tensor,
    config::Config,
    module::{Module, Param},
    nn::{
        Identity,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::Device,
};

use super::pointwise::conv1x1;

/// Which skip connection, as 2.0.0's `skip_type`; [`SkipKind::init`] builds it.
#[derive(Config, Debug)]
pub enum SkipKind {
    /// `"identity"`: the input unchanged.
    Identity,
    /// `"linear"`: a pointwise convolution without bias (2.0.0's `Flattened1dConv`).
    Linear,
    /// `"soft-gating"`: each channel scaled by a learnt weight, initially 1, without bias
    /// (2.0.0's `SoftGating`; no 2.0.0 caller sets its bias).
    SoftGating,
}

/// A skip connection on channels-first `[B, C, s_1..s_D]` tensors of rank `R`, built by
/// [`SkipKind::init`].
#[derive(Module, Debug)]
pub enum Skip<const R: usize> {
    /// The input unchanged.
    Identity(Identity),
    /// A kernel-size-1 `Conv1d` without bias, weight `[out, in, 1]`.
    Linear(Conv1d),
    /// The per-channel weight, `[1, C, 1, …, 1]` of rank `R`.
    SoftGating(Param<Tensor<R>>),
}

impl SkipKind {
    /// Builds the skip from `in_channels` to `out_channels` on `device`; the linear skip has
    /// Burn's default `Conv1d` initialisation, soft gating weights of 1.
    ///
    /// # Panics
    /// If `in_channels != out_channels` for the identity or soft gating, which keep the
    /// width (2.0.0 raises for soft gating).
    pub fn init<const R: usize>(
        &self,
        in_channels: usize,
        out_channels: usize,
        device: &Device,
    ) -> Skip<R> {
        let same_width = || {
            assert_eq!(
                in_channels, out_channels,
                "a {self:?} skip keeps the width: in_channels must equal out_channels"
            )
        };
        match self {
            SkipKind::Identity => {
                same_width();
                Skip::Identity(Identity::new())
            }
            SkipKind::Linear => Skip::Linear(
                Conv1dConfig::new(in_channels, out_channels, 1)
                    .with_bias(false)
                    .init(device),
            ),
            SkipKind::SoftGating => {
                same_width();
                let mut shape = [1; R];
                shape[1] = in_channels;
                Skip::SoftGating(Param::from_tensor(Tensor::ones(shape, device)))
            }
        }
    }
}

impl<const R: usize> Skip<R> {
    /// `[B, in_channels, s..]` → `[B, out_channels, s..]`.
    ///
    /// # Panics
    /// If the input's channel axis (1) is not `in_channels` wide (linear and soft gating).
    pub fn forward(&self, x: Tensor<R>) -> Tensor<R> {
        match self {
            Skip::Identity(identity) => identity.forward(x),
            Skip::Linear(conv) => conv1x1(conv, x),
            Skip::SoftGating(weight) => x * weight.val(),
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tolerance;

    use super::*;

    #[test]
    fn identity_returns_its_input() {
        let device = Device::default();
        let x = Tensor::<4>::random([2, 3, 4, 5], burn::tensor::Distribution::Default, &device);
        let y = SkipKind::Identity
            .init::<4>(3, 3, &device)
            .forward(x.clone());
        assert_eq!(y.into_data(), x.into_data());
    }

    /// Soft gating scales channel c by its weight, whatever the grid point.
    #[test]
    fn soft_gating_scales_each_channel() {
        let device = Device::default();
        let mut skip = SkipKind::SoftGating.init::<3>(2, 2, &device);
        if let Skip::SoftGating(w) = &mut skip {
            *w = Param::from_tensor(Tensor::from_data([[[2.0], [-0.5]]], &device));
        }
        let x = Tensor::<3>::from_data([[[1.0, 3.0], [4.0, -2.0]]], &device);
        let expected = Tensor::<3>::from_data([[[2.0, 6.0], [-2.0, 1.0]]], &device);
        skip.forward(x)
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());
    }

    #[test]
    fn linear_changes_the_width() {
        let device = Device::default();
        let skip = SkipKind::Linear.init::<5>(3, 5, &device);
        let Skip::Linear(conv) = &skip else {
            panic!("not linear")
        };
        assert!(conv.bias.is_none());
        let y = skip.forward(Tensor::<5>::zeros([2, 3, 4, 3, 2], &device));
        assert_eq!(y.dims(), [2, 5, 4, 3, 2]);
    }

    #[test]
    #[should_panic(expected = "keeps the width")]
    fn soft_gating_needs_equal_widths() {
        SkipKind::SoftGating.init::<4>(3, 4, &Device::default());
    }

    #[test]
    #[should_panic(expected = "keeps the width")]
    fn identity_needs_equal_widths() {
        SkipKind::Identity.init::<4>(3, 4, &Device::default());
    }
}

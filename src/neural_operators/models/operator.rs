//! The operator trait (library CM0.3; proposal §6.1, decision D4): what the training
//! stack needs from a model, so that the trainer, evaluation, inference, checkpoints, the
//! run directory and the Gauss–Newton residual work for any operator, not only the
//! [`FNO`].
//!
//! D4 (signed off 2026-11-24): a generic trait with associated config and input types,
//! static dispatch, no trait objects. Saving and loading need nothing here: every Burn
//! [`Module`] saves to Burnpack, and every [`Config`](burn::config::Config) to JSON.

use burn::{module::Module, tensor::Device, tensor::Tensor};

use crate::neural_operators::models::{
    fno::{FNO, FNOConfig},
    registry::OperatorConfig,
};

/// A neural operator: a Burn [`Module`] built from its [`Config`](Operator::Config) and
/// run on its [`Input`](Operator::Input).
pub trait Operator: Module {
    /// The hyperparameters, saved as the run directory's `model_cfg.json` with their
    /// [`KIND`](OperatorConfig::KIND).
    type Config: OperatorConfig;
    /// What [`forward`](Operator::forward) takes: `Tensor<R>` for an operator on a grid.
    type Input;
    /// What [`forward`](Operator::forward) returns.
    type Output;
    /// The operator's name in the registry (CM0.4): its config's
    /// [`KIND`](OperatorConfig::KIND). Never overridden: the registry and
    /// `model_cfg.json` know the operator by `KIND`.
    const NAME: &'static str = <Self::Config as OperatorConfig>::KIND;

    /// A freshly initialised operator on `device`, drawn from the device's RNG.
    fn init(config: &Self::Config, device: &Device) -> Self;

    /// The operator applied to `x`.
    fn forward(&self, x: Self::Input) -> Self::Output;
}

/// An operator on a grid, as the current training stack takes them: channels-last
/// `[B, *spatial, C]` in, `[B, *spatial, C_out]` out, both rank `R` (CONVENTIONS §1).
/// Implemented for every such [`Operator`].
pub trait GridOperator<const R: usize>: Operator<Input = Tensor<R>, Output = Tensor<R>> {}

impl<const R: usize, M: Operator<Input = Tensor<R>, Output = Tensor<R>>> GridOperator<R> for M {}

impl<const R: usize> Operator for FNO<R> {
    type Config = FNOConfig;
    type Input = Tensor<R>;
    type Output = Tensor<R>;

    /// [`FNOConfig::init`].
    ///
    /// # Panics
    /// If `config.modes.len() != R - 2`.
    fn init(config: &FNOConfig, device: &Device) -> Self {
        config.init::<R>(device)
    }

    /// [`FNO::forward`].
    ///
    /// # Panics
    /// As [`FNO::forward`].
    fn forward(&self, x: Tensor<R>) -> Tensor<R> {
        FNO::forward(self, x)
    }
}

/// A second operator for the tests of the generic stack: one pointwise `Linear` from the
/// data channels to one output channel.
#[cfg(test)]
pub(crate) mod pointwise {
    // see `models/fno/mod.rs`: Burn's `#[derive(Config)]` writes `field: field`
    #![allow(clippy::redundant_field_names)]

    use burn::{
        config::Config,
        module::Module,
        nn::{Linear, LinearConfig},
        tensor::{Device, Tensor},
    };

    use super::Operator;
    use crate::neural_operators::models::registry::OperatorConfig;

    /// [`Pointwise`]'s hyperparameters.
    #[derive(Config, Debug)]
    pub(crate) struct PointwiseConfig {
        /// Input channels.
        pub data_channels: usize,
    }

    /// `y[.., 0] = w · x[.., :] + b` at every grid point.
    #[derive(Module, Debug)]
    pub(crate) struct Pointwise<const R: usize> {
        linear: Linear,
    }

    impl OperatorConfig for PointwiseConfig {
        const KIND: &'static str = "pointwise";
    }

    impl<const R: usize> Operator for Pointwise<R> {
        type Config = PointwiseConfig;
        type Input = Tensor<R>;
        type Output = Tensor<R>;

        fn init(config: &PointwiseConfig, device: &Device) -> Self {
            Self {
                linear: LinearConfig::new(config.data_channels, 1).init(device),
            }
        }

        fn forward(&self, x: Tensor<R>) -> Tensor<R> {
            self.linear.forward(x)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use burn::tensor::TensorData;

    use super::{
        pointwise::{Pointwise, PointwiseConfig},
        *,
    };
    use crate::neural_operators::{
        data::loaders::{base_dataset::DatasetConfig, burgers::BurgersConfig},
        models::predict::predict_host,
        optim::gauss_newton::params,
        runs::RunDir,
        training::{
            config::TrainingConfig,
            fixtures::{tiny_dataset, tiny_model_cfg},
            trainer::{build_components, train_components},
        },
    };

    fn values<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    /// The trait's `forward` is the FNO's own, bit for bit. (Seeded `init` is not compared
    /// here: Flex's RNG is process-wide, so another test drawing in parallel changes the
    /// weights. `tests/fno_bit_identity.rs` shows it: its runs build through `M::init`
    /// and match `main`.)
    #[test]
    fn fno_through_the_trait_is_the_fno() {
        let device = Device::default();
        let direct = <FNO<3> as Operator>::init(&tiny_model_cfg(), &device);
        let x = Tensor::<3>::from_data(
            TensorData::new((0..16).map(|i| (i as f32 * 0.3).sin()).collect(), [2, 8, 1]),
            &device,
        );
        assert_eq!(
            values(direct.forward(x.clone())),
            values(Operator::forward(&direct, x))
        );
        assert_eq!(<FNO<3> as Operator>::NAME, "fno");
    }

    /// A model that is not the FNO trains through the generic stack, checkpoints, reloads
    /// from its run directory and predicts: the reloaded and checkpointed models are the
    /// trained one.
    #[test]
    fn a_second_operator_trains_checkpoints_and_reloads() {
        let root = std::env::temp_dir().join(format!("sciml_rs_pointwise_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let model_cfg = PointwiseConfig::new(1);
        let train_cfg = TrainingConfig::new()
            .with_epochs(2)
            .with_batch_size(2)
            .with_test_batch_size(2);
        let device = Device::default().autodiff();
        let components = build_components::<Pointwise<3>, 3, 2, _>(
            &model_cfg,
            &train_cfg,
            tiny_dataset(4),
            tiny_dataset(2),
            &device,
        );
        let run = RunDir::create(&root, "pointwise").unwrap();
        let data_cfg = BurgersConfig::new(
            DatasetConfig {
                n_train: 4,
                n_test: 2,
            },
            1,
        );
        run.save_configs(&model_cfg, &train_cfg, &data_cfg).unwrap();
        let (model, metrics) = train_components(
            components,
            None,
            &train_cfg,
            &device,
            &run,
            Some(1),
            None,
            None,
        );
        assert_eq!(metrics.len(), 2);
        assert!(metrics.iter().all(|m| m.test_l2.is_finite()));
        run.save_weights(&model).unwrap();

        let plain = Device::default();
        let loaded: Pointwise<3> = run.load_model(&plain).unwrap();
        let ck = run.checkpoint_of::<Pointwise<3>>(&plain).unwrap().unwrap();
        assert_eq!(ck.state.next_epoch, 2);
        let trained = values(params(&model).to_flat());
        assert_eq!(trained, values(params(&loaded).to_flat()));
        assert_eq!(trained, values(params(&ck.model).to_flat()));

        let inputs = tiny_dataset(3).inputs().mapv(|v| v as f32);
        let pred = predict_host::<3>(&loaded, inputs.view(), 2, &plain);
        assert_eq!(pred.shape(), &[3, 8]);
        let _ = fs::remove_dir_all(&root);
    }
}

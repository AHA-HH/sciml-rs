//! Burn `Learner` integration: [`TrainStep`] and [`InferenceStep`] for the
//! 1D FNO, so it can be trained with `SupervisedTraining` (and its TUI) as
//! an alternative to the hand-written loop in
//! [`trainer`](crate::neural_operators::training::trainer).
//!
//! Only `FNO<3>` (Burgers: batch + 1 spatial + channel) is supported. The
//! impls are per rank because `Batch<R, RM1>` would leave `RM1` unconstrained
//! in a generic `impl<const R: usize> TrainStep for FNO<R>`. Datasets that
//! need a decoder (Darcy) are not covered: the step has no postprocess hook.

use burn::{
    prelude::*,
    train::{InferenceStep, RegressionOutput, TrainOutput, TrainStep},
};

use crate::neural_operators::{
    data::batcher::Batch,
    losses::data_losses::{LpLoss, Reduction},
    models::fno::FNO,
    training::trainer::flatten_pair,
};

impl FNO<3> {
    /// Forward pass plus loss, shared by the train and inference steps.
    ///
    /// Returns the summed relative L2 over the batch (the quantity
    /// [`train_epoch`](crate::neural_operators::training::trainer::train_epoch)
    /// backpropagates) and a [`RegressionOutput`] whose `loss` is that sum
    /// divided by the batch size. The division only affects the reported
    /// metric, so the Learner shows a per-sample relative L2 comparable to
    /// `train_l2` / `test_l2` in `EpochMetrics`.
    fn learner_forward(&self, batch: Batch<3, 2>) -> (Tensor<1>, RegressionOutput) {
        let out = self.forward(batch.inputs);
        let (output, targets) = flatten_pair::<3, 2>(out, batch.targets);
        let b = targets.dims()[0];

        let l2_sum = LpLoss::new(1, 2, Reduction::Sum).rel(output.clone(), targets.clone());
        let loss = l2_sum.clone().div_scalar(b as f64);

        (
            l2_sum,
            RegressionOutput {
                loss,
                output,
                targets,
            },
        )
    }
}

impl TrainStep for FNO<3> {
    type Input = Batch<3, 2>;
    type Output = RegressionOutput;

    fn step(&self, batch: Batch<3, 2>) -> TrainOutput<RegressionOutput> {
        let (l2_sum, item) = self.learner_forward(batch);
        TrainOutput::new(self, l2_sum.backward(), item)
    }
}

impl InferenceStep for FNO<3> {
    type Input = Batch<3, 2>;
    type Output = RegressionOutput;

    fn step(&self, batch: Batch<3, 2>) -> RegressionOutput {
        self.learner_forward(batch).1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::{
        models::fno::FNOConfig,
        training::trainer::{flatten_pair, identity},
    };
    use burn::optim::{
        GradientsParams, SgdConfig, lr_scheduler::module_lr_scheduler::ModuleLearningRate,
    };

    fn tiny_model(device: &Device) -> FNO<3> {
        FNOConfig::new(vec![2], 1, 1)
            .with_hidden_channels(4)
            .with_n_layers(1)
            .init::<3>(device)
    }

    /// Batch of 3 samples on an 8-point grid; inputs carry the data channel
    /// plus the grid channel, targets are bounded away from zero.
    fn tiny_batch(device: &Device) -> Batch<3, 2> {
        let (b, s) = (3, 8);
        let inputs: Vec<f32> = (0..b * s * 2).map(|i| (i as f32 * 0.37).sin()).collect();
        let targets: Vec<f32> = (0..b * s).map(|i| 2.0 + (i as f32 * 0.7).cos()).collect();
        Batch {
            inputs: Tensor::from_data(TensorData::new(inputs, [b, s, 2]), device),
            targets: Tensor::from_data(TensorData::new(targets, [b, s]), device),
        }
    }

    fn probe(model: &FNO<3>, device: &Device) -> Vec<f32> {
        model
            .forward(tiny_batch(device).inputs)
            .into_data()
            .iter::<f32>()
            .collect()
    }

    /// The Learner's gradients must equal the hand-written loop's (summed
    /// relative L2, no rescaling). Compared through one plain SGD step, which
    /// is linear in the gradient, so a batch-mean vs batch-sum mix-up (a
    /// factor of 3 here) or a detached graph changes the updated model.
    #[test]
    fn train_step_gradients_match_hand_written_loop() {
        let device = Device::default().autodiff();
        device.seed(7);
        let model = tiny_model(&device);
        let lr = ModuleLearningRate::from(1e-1_f64);

        // Learner path.
        let learner_grads = TrainStep::step(&model, tiny_batch(&device)).grads;
        let learner_model = SgdConfig::new()
            .init()
            .step(lr.clone(), model.clone(), learner_grads);

        // Hand-written path, as in `train_epoch`.
        let batch = tiny_batch(&device);
        let out = model.forward(batch.inputs);
        let (out, target) = flatten_pair::<3, 2>(out, batch.targets);
        let (out, target) = identity(out, target);
        let l2 = LpLoss::new(1, 2, Reduction::Sum).rel(out, target);
        let manual_grads = GradientsParams::from_grads(l2.backward(), &model);
        let manual_model = SgdConfig::new()
            .init()
            .step(lr, model.clone(), manual_grads);

        let before = probe(&model, &device);
        let learner = probe(&learner_model, &device);
        let manual = probe(&manual_model, &device);

        let moved = before
            .iter()
            .zip(&manual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            moved > 1e-4,
            "SGD step too small to detect a mismatch: {moved}"
        );

        for (i, (l, m)) in learner.iter().zip(&manual).enumerate() {
            assert!((l - m).abs() <= 1e-6 * m.abs().max(1.0), "[{i}] {l} != {m}");
        }
    }

    /// The reported loss is the per-sample mean of the summed relative L2,
    /// and the inference step runs on the inner (non-autodiff) device.
    #[test]
    fn inference_step_reports_per_sample_relative_l2() {
        let device = Device::default().autodiff().inner();
        device.seed(7);
        let model = tiny_model(&device);

        let batch = tiny_batch(&device);
        let out = model.forward(batch.inputs.clone());
        let (o, t) = flatten_pair::<3, 2>(out, batch.targets.clone());
        let expected = LpLoss::new(1, 2, Reduction::Sum)
            .rel(o, t)
            .into_scalar::<f32>()
            / 3.0;

        let item = InferenceStep::step(&model, batch);
        assert!(!item.loss.is_autodiff());
        assert_eq!(item.output.dims(), [3, 8]);
        assert_eq!(item.targets.dims(), [3, 8]);

        let loss = item.loss.into_scalar::<f32>();
        assert!(
            (loss - expected).abs() <= 1e-6 * expected,
            "{loss} != {expected}"
        );
    }
}

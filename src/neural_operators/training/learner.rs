//! Burn `Learner` integration: [`TrainStep`] and [`InferenceStep`] for the
//! 1D FNO, so it can be trained with `SupervisedTraining` (its TUI, metric
//! logging and summary).
//!
//! This is a showcase, kept as the template for using Burn's `Learner`:
//! [`train_epoch`](crate::neural_operators::training::trainer::train_epoch) is
//! the canonical loop, and this is its Burgers-only (`FNO<3>`) equivalent. Both
//! take their loss from [`forward_loss`].
//!
//! `TrainStep` cannot be generic over R here: its `Input` would be
//! `Batch<R, RM1>`, and a generic `impl<const R: usize> TrainStep for FNO<R>`
//! leaves `RM1` unconstrained (stable Rust cannot write `R - 1` in a type), so
//! the impls are per rank. Datasets that need a decoder (Darcy) are not
//! covered: the step passes no postprocess hook.

use burn::{
    prelude::*,
    train::{InferenceStep, RegressionOutput, TrainOutput, TrainStep},
};

use crate::neural_operators::{
    data::batcher::Batch,
    models::fno::FNO,
    training::trainer::{forward_loss, identity},
};

impl FNO<3> {
    /// Forward pass plus loss, shared by the train and inference steps.
    ///
    /// Returns the summed relative L2 over the batch (the quantity
    /// [`train_epoch`](crate::neural_operators::training::trainer::train_epoch)
    /// backpropagates, from the same [`forward_loss`]) and a
    /// [`RegressionOutput`] whose `loss` is that sum divided by the batch size.
    /// The division only affects the reported metric, so the Learner shows a
    /// per-sample relative L2 comparable to `train_l2` / `test_l2` in
    /// `EpochMetrics`.
    fn learner_forward(&self, batch: Batch<3, 2>) -> (Tensor<1>, RegressionOutput) {
        let (l2_sum, output, targets) = forward_loss(self, batch, &identity);
        let loss = l2_sum.clone().div_scalar(targets.dims()[0] as f64);

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
        losses::data_losses::{LpLoss, Reduction},
        models::fno::FNOConfig,
        training::trainer::flatten_pair,
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
    /// (the model appends the grid), targets are bounded away from zero.
    fn tiny_batch(device: &Device) -> Batch<3, 2> {
        let (b, s) = (3, 8);
        let inputs: Vec<f32> = (0..b * s).map(|i| (i as f32 * 0.37).sin()).collect();
        let targets: Vec<f32> = (0..b * s).map(|i| 2.0 + (i as f32 * 0.7).cos()).collect();
        Batch {
            inputs: Tensor::from_data(TensorData::new(inputs, [b, s, 1]), device),
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

        // Hand-written oracle, independent of `forward_loss`.
        let batch = tiny_batch(&device);
        let out = model.forward(batch.inputs);
        let (out, target) = flatten_pair::<3, 2>(out, batch.targets);
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

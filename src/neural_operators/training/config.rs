//! Optimizer and schedule settings for a training run.

use burn::config::Config;

/// Optimizer, schedule and batching settings, saved with every run
/// ([`RunDir::save_configs`](crate::neural_operators::runs::RunDir::save_configs)).
#[derive(Config, Debug)]
pub struct TrainingConfig {
    /// Number of training epochs.
    #[config(default = 100)]
    pub epochs: usize,
    /// Training batch size; the last batch of an epoch may be partial.
    #[config(default = 20)]
    pub batch_size: usize,
    /// Evaluation batch size; it does not change the test loss, which is a per-sample mean.
    #[config(default = 20)]
    pub test_batch_size: usize,
    /// Initial (peak) learning rate of the cosine schedule.
    #[config(default = 1e-3)]
    pub learning_rate: f64,
    /// f32 to match `WeightDecayConfig::penalty`, unlike the f64 types above.
    #[config(default = 1e-4)]
    pub weight_decay: f32,
    /// Final learning rate of the cosine schedule.
    #[config(default = 1e-7)]
    pub min_lr: f64,
    /// Seeds the device RNG (weight initialisation) and the training loader's shuffle.
    #[config(default = 42)]
    pub seed: u64,
}

impl TrainingConfig {
    /// Optimisation steps a run takes on `n_train` samples: `epochs` × ⌈`n_train` /
    /// `batch_size`⌉, since an epoch's last batch may be partial. The trainer's cosine
    /// schedule spans exactly these steps.
    pub fn total_steps(&self, n_train: usize) -> usize {
        self.epochs * n_train.div_ceil(self.batch_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_steps_counts_the_partial_last_batch() {
        let cfg = TrainingConfig::new().with_epochs(500).with_batch_size(20);
        assert_eq!(cfg.total_steps(25), 1000);
        assert_eq!(cfg.total_steps(1000), 25_000);
        assert_eq!(cfg.total_steps(20), 500);
    }
}

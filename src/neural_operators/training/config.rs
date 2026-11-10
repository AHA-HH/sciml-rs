//! Optimizer and schedule settings for a training run.

use burn::config::Config;

use crate::neural_operators::optim::gauss_newton::GaussNewtonConfig;

/// Which optimiser trains the model (Phase 10 T3).
#[derive(Config, Debug)]
pub enum OptimizerConfig {
    /// Adam with weight decay and the cosine schedule, from [`TrainingConfig`]'s
    /// learning-rate and weight-decay fields: the baseline.
    Adam,
    /// Damped Gauss–Newton ([`GaussNewton`](crate::neural_operators::optim::gauss_newton::GaussNewton)),
    /// one step per batch; [`TrainingConfig`]'s learning-rate fields are unused, and the
    /// weight decay is the config's own μ.
    GaussNewton(GaussNewtonConfig),
}

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
    /// The optimiser; `None` (the default) means [`OptimizerConfig::Adam`]. An `Option`
    /// so that saved configs without this field (before Phase 10 T3) still load.
    pub optimizer: Option<OptimizerConfig>,
}

impl TrainingConfig {
    /// Optimisation steps a run takes on `n_train` samples: `epochs` × ⌈`n_train` /
    /// `batch_size`⌉, since an epoch's last batch may be partial. The trainer's cosine
    /// schedule spans exactly these steps.
    pub fn total_steps(&self, n_train: usize) -> usize {
        self.epochs * n_train.div_ceil(self.batch_size)
    }

    /// The Gauss–Newton settings if this run trains with Gauss–Newton, else `None` (Adam).
    pub fn gauss_newton(&self) -> Option<&GaussNewtonConfig> {
        match &self.optimizer {
            Some(OptimizerConfig::GaussNewton(gn)) => Some(gn),
            Some(OptimizerConfig::Adam) | None => None,
        }
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

    /// A `train_cfg.json` as runs wrote it before Phase 10 T3 loads as Adam; a
    /// Gauss–Newton config round-trips.
    #[test]
    fn old_config_loads_as_adam_and_gauss_newton_round_trips() {
        let old = r#"{"epochs": 500, "batch_size": 20, "test_batch_size": 20,
            "learning_rate": 0.001, "weight_decay": 0.0001, "min_lr": 1e-5, "seed": 42}"#;
        let cfg = TrainingConfig::load_binary(old.as_bytes()).unwrap();
        assert!(cfg.optimizer.is_none() && cfg.gauss_newton().is_none());
        assert_eq!(cfg.epochs, 500);

        let gn = GaussNewtonConfig::new().with_cg_max_iters(7);
        let cfg = TrainingConfig::new().with_optimizer(Some(OptimizerConfig::GaussNewton(gn)));
        let back: TrainingConfig =
            TrainingConfig::load_binary(burn::config::config_to_json(&cfg).as_bytes()).unwrap();
        assert_eq!(back.gauss_newton().map(|g| g.cg_max_iters), Some(7));
        let adam = TrainingConfig::new().with_optimizer(Some(OptimizerConfig::Adam));
        assert!(adam.gauss_newton().is_none());
    }
}

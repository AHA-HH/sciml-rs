//! Optimizer and schedule settings for a training run.

use burn::config::Config;

/// Training config
#[derive(Config, Debug)]
pub struct TrainingConfig {
    #[config(default = 100)]
    pub epochs: usize,
    #[config(default = 20)]
    pub batch_size: usize,
    #[config(default = 20)]
    pub test_batch_size: usize,
    #[config(default = 1e-3)]
    pub learning_rate: f64,
    /// f32 to match `WeightDecayConfig::penalty`, unlike the f64 types above.
    #[config(default = 1e-4)]
    pub weight_decay: f32,
    #[config(default = 1e-7)]
    pub min_lr: f64,
    #[config(default = 42)]
    pub seed: u64,
}

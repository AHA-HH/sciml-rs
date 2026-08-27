//! Per-epoch training metrics, collected for logging

/// Snapshot of training/eval state after one epoch.
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub train_mse: f32,
    pub train_l2: f32,
    pub test_l2: f32,
    pub current_lr: f64,
}
//! Per-epoch training metrics, collected for logging

/// Snapshot of training/eval state after one epoch.
/// ///
/// `train_mse` is averaged over batches (the MSE is already a per-batch mean);
/// `train_l2` and `test_l2` are averaged over samples (LpLoss sums per batch).
/// `current_lr` is the rate after the epoch's final step.
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub train_mse: f32,
    pub train_l2: f32,
    pub test_l2: f32,
    pub current_lr: f64,
}

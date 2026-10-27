//! `costs.json`: what a run cost and where it ran (design §13.9).

// Burn's `#[derive(Config)]` writes `field: field` in the `new` it generates, in an
// impl block outside the struct's attributes (burn-derive `config/analyzer_struct.rs`).
#![allow(clippy::redundant_field_names)]

use burn::config::Config;

/// A run's costs, written by the training example and completed by predict
/// ([`RunDir::save_costs`](super::RunDir::save_costs),
/// [`RunDir::costs`](super::RunDir::costs)). Times are wall-clock seconds.
#[derive(Config, Debug)]
pub struct RunCosts {
    /// Generation time of the train split, from its sidecar's `wall_time_s`; `None` for a
    /// dataset without sidecars.
    pub dataset_train_wall_time_s: Option<f64>,
    /// Generation time of the test split, from its sidecar's `wall_time_s`; `None` for a
    /// dataset without sidecars.
    pub dataset_test_wall_time_s: Option<f64>,
    /// Reading both splits, the transfer to the learning grid and normalisation.
    pub load_time_s: f64,
    /// Training from building the model to the end of the last epoch.
    pub training_time_s: f64,
    /// Inference on the whole test split, warm (the second of two passes); written by
    /// predict, `None` until then.
    pub inference_time_s: Option<f64>,
    /// Bytes the train split uploads to the device: inputs plus targets as stored,
    /// N × s² × (in + out channels) × 4 for f32.
    pub host_to_device_bytes_train: usize,
    /// Bytes the test split uploads to the device, as for the train split.
    pub host_to_device_bytes_test: usize,
    /// Peak GPU memory as `"<N> MiB"`, the peak `nvidia-smi` reported for the whole
    /// device while the run trained (e.g. `"12345 MiB"`), or `"unavailable"`: measured,
    /// never estimated. Nothing in the crate measures it; on cuda it is filled in from
    /// `nvidia-smi` sampled beside the run.
    pub peak_gpu_memory: String,
    /// The backend features compiled in
    /// ([`backend_features`](crate::neural_operators::utils::backend::backend_features)).
    pub backend_features: Vec<String>,
    /// The training device's debug name.
    pub device: String,
    /// The commit the run was trained at
    /// ([`git_commit`](crate::neural_operators::utils::backend::git_commit)).
    pub git_commit: String,
    /// Trainable parameters of the model (Burn's `Module::num_params`).
    pub num_params: usize,
}

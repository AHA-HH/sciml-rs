//! Export of finished run directories to a self-hosted MLflow tracking server (Phase 8
//! T1, feature `mlflow`). One run directory becomes one MLflow run; the run directory
//! stays the record, MLflow is a view of it (design §13.10).
//!
//! Two halves:
//! - [`export_requests`], pure: from a [`RunDir`](crate::neural_operators::runs::RunDir),
//!   the REST requests that create, fill and finish the MLflow run. It reads the run
//!   directory and nothing else, so it is tested without a network.
//! - [`Client`]: sends them over MLflow's REST API (`/api/2.0/mlflow/...` and the
//!   proxied `mlflow-artifacts` endpoint), threading the new run's id into the requests
//!   after `runs/create`, and finds, deletes and fails runs.
//!
//! The mapping (one MLflow run per run directory, Poisson runs):
//! - run name: the directory name; tags `sciml.run_dir`, `sciml.git_commit` and
//!   `backend` (from `costs.json`, when present), `padding`, `s`, `n`, `K` (when the data
//!   config records it), `seed`, `N` (train samples) and `group` = `p{padding}_s{s}`;
//! - params: the three configs flattened to dotted keys (`model.modes`, `train.seed`,
//!   `data.source.k`), values as JSON text (strings bare);
//! - metrics, step = epoch: `train_mse`, `train_l2`, `test_l2`, `lr` and `epoch_time_s`
//!   when recorded; at the last epoch, the mean and max of errors 2–5 from `errors.json`
//!   and the numeric fields of `costs.json`, when those files exist;
//! - artifacts: [`RunDir::artifacts`](crate::neural_operators::runs::RunDir::artifacts)
//!   (configs, `costs.json`, `errors.json`, plots), never weights or checkpoints;
//! - status `FINISHED` with `end_time`, or `FAILED` if a request fails.
//!
//! Live logging (opt-in, planning decision Q7): [`MlflowObserver`] sends the same run
//! during training (`train_poisson` with `MLFLOW_EXPERIMENT_NAME` set), built from the
//! same pieces ([`start_requests`], [`epoch_request`], [`finish_requests`]). It is
//! best-effort; the export stays the authoritative record.

mod builder;
mod client;
mod live;

pub use builder::{
    Body, MAX_KEY_LEN, MAX_METRICS, MAX_PARAMS, MAX_VALUE_LEN, Method, RUN_DIR_TAG, RUN_ID,
    Request, epoch_request, export_requests, finish_requests, start_requests,
};
pub use client::{Client, DEFAULT_TRACKING_URI, MlflowError};
pub use live::{EXPERIMENT_ENV, MlflowObserver, TRACKING_URI_ENV};

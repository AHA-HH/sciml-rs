//! Opt-in live logging during training (Phase 8 T1, planning decision Q7): the same MLflow
//! run the exporter would write, filled as training goes. Best-effort: the first failed
//! request turns it off for the run; `mlflow_export --replace` gives the clean record.

use std::error::Error;
use std::fmt::Display;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::builder::{epoch_request, finish_requests, start_requests};
use super::client::{Client, DEFAULT_TRACKING_URI};
use crate::neural_operators::runs::{EpochMetrics, RunDir};

/// The variable that switches live logging on: MLflow's own name for the experiment.
pub const EXPERIMENT_ENV: &str = "MLFLOW_EXPERIMENT_NAME";
/// The variable naming the tracking server, [`DEFAULT_TRACKING_URI`] when unset.
pub const TRACKING_URI_ENV: &str = "MLFLOW_TRACKING_URI";
/// Each live request's total time limit, so a slow or unreachable server costs training
/// at most this once.
const TIMEOUT: Duration = Duration::from_secs(5);

/// Logs one training run to MLflow live: [`start`](Self::start) opens the run,
/// [`on_epoch`](Self::on_epoch) logs each epoch at step = epoch, [`finish`](Self::finish)
/// closes it. Every request goes through the exporter's builder
/// ([`start_requests`](super::start_requests), [`epoch_request`](super::epoch_request),
/// [`finish_requests`](super::finish_requests)), so the run matches an exported one.
///
/// A failed request prints one warning to stderr and turns the observer off for the
/// rest of the run; it never panics and never fails training.
pub struct MlflowObserver {
    client: Client,
    run: RunDir,
    experiment_id: String,
    /// `None` once off.
    run_id: Option<String>,
}

impl MlflowObserver {
    /// The observer for `run`, started, if [`EXPERIMENT_ENV`] is set; `None` otherwise,
    /// and then nothing is sent. The server is [`TRACKING_URI_ENV`], or
    /// [`DEFAULT_TRACKING_URI`].
    pub fn from_env(run: &RunDir) -> Option<Self> {
        let experiment = std::env::var(EXPERIMENT_ENV).ok()?;
        let uri =
            std::env::var(TRACKING_URI_ENV).unwrap_or_else(|_| DEFAULT_TRACKING_URI.to_string());
        Some(Self::start(&uri, &experiment, run))
    }

    /// Opens `run`'s MLflow run in experiment `experiment` (created if missing) at
    /// `tracking_uri`. A run already tagged with this directory (`sciml.run_dir`, as after
    /// `--resume`) is continued; otherwise a new one is created with the exporter's tags
    /// and params. On failure the observer starts off, with a warning.
    pub fn start(tracking_uri: &str, experiment: &str, run: &RunDir) -> Self {
        let mut observer = Self {
            client: Client::with_timeout(tracking_uri, Some(TIMEOUT)),
            run: run.clone(),
            experiment_id: String::new(),
            run_id: None,
        };
        match observer.open(experiment) {
            Ok(id) => observer.run_id = Some(id),
            Err(e) => observer.disable(e),
        }
        observer
    }

    /// Whether the observer still logs: started and no request has failed.
    pub fn is_active(&self) -> bool {
        self.run_id.is_some()
    }

    /// Logs epoch `m`'s metrics, one `runs/log-batch`.
    pub fn on_epoch(&mut self, m: &EpochMetrics) {
        if let Some(id) = &self.run_id
            && let Err(e) = self.client.send_to(id, &epoch_request(m, now_ms()))
        {
            self.disable(e);
        }
    }

    /// Closes the run once the run directory is complete (`costs.json`, plots): the
    /// summary metrics, the artifacts and status `FINISHED`.
    pub fn finish(&mut self) {
        let Some(id) = self.run_id.clone() else {
            return;
        };
        let sent = finish_requests(&self.run, &self.experiment_id, now_ms()).and_then(|rs| {
            rs.iter()
                .try_for_each(|r| self.client.send_to(&id, r))
                .map_err(Into::into)
        });
        if let Err(e) = sent {
            self.disable(e);
        }
    }

    fn open(&mut self, experiment: &str) -> Result<String, Box<dyn Error>> {
        self.experiment_id = self.client.experiment_id(experiment)?;
        let name = self
            .run
            .path()
            .file_name()
            .unwrap_or_default()
            .to_string_lossy();
        if let Some(id) = self.client.find_runs(&self.experiment_id, &name)?.first() {
            return Ok(id.clone());
        }
        let requests = start_requests(&self.run, &self.experiment_id, now_ms())?;
        let (create, rest) = requests.split_first().ok_or("no requests")?;
        let id = self.client.create(create)?;
        for r in rest {
            self.client.send_to(&id, r)?;
        }
        Ok(id)
    }

    fn disable(&mut self, e: impl Display) {
        eprintln!(
            "warning: MLflow live logging is off for this run ({e}); export it afterwards \
             with `mlflow_export --replace <experiment> {}`",
            self.run.path().display()
        );
        self.run_id = None;
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    // port 1 is closed: every request fails at once with "connection refused"
    const CLOSED: &str = "http://127.0.0.1:1";

    fn epoch(epoch: usize) -> EpochMetrics {
        EpochMetrics {
            epoch,
            train_mse: 0.5,
            train_l2: 0.25,
            test_l2: 0.125,
            current_lr: 1e-3,
            epoch_time_s: None,
        }
    }

    fn run_dir(test: &str) -> RunDir {
        let dir = std::env::temp_dir().join(format!("sciml_rs_live_{test}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        RunDir::open(dir).unwrap()
    }

    #[test]
    fn unreachable_server_at_start_leaves_the_observer_off() {
        let run = run_dir("start");
        let mut observer = MlflowObserver::start(CLOSED, "scratch", &run);
        assert!(!observer.is_active());
        for e in 0..3 {
            observer.on_epoch(&epoch(e));
        }
        observer.finish();
        assert!(!observer.is_active());
        std::fs::remove_dir_all(run.path()).unwrap();
    }

    #[test]
    fn first_failed_epoch_turns_the_observer_off() {
        let run = run_dir("epoch");
        let mut observer = MlflowObserver {
            client: Client::with_timeout(CLOSED, Some(TIMEOUT)),
            run: run.clone(),
            experiment_id: "1".into(),
            run_id: Some("abc".into()),
        };
        observer.on_epoch(&epoch(0));
        assert!(!observer.is_active(), "still on after a failed request");
        // off: later epochs and the finish send nothing
        observer.on_epoch(&epoch(1));
        observer.finish();
        assert!(!observer.is_active());
        std::fs::remove_dir_all(run.path()).unwrap();
    }
}

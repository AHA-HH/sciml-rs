//! The I/O half of the exporter: sends [`Request`]s to an MLflow tracking server.

use std::fmt;

use serde_json::{Value, json};

use super::builder::{Body, Method, RUN_DIR_TAG, RUN_ID, Request};

/// A failed MLflow call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlflowError {
    /// The server answered with MLflow's error body, `{"error_code": ..., "message": ...}`.
    Api {
        /// The HTTP status.
        status: u16,
        /// MLflow's error code, e.g. `RESOURCE_DOES_NOT_EXIST`.
        error_code: String,
        /// MLflow's message.
        message: String,
    },
    /// Anything else: no connection, a non-MLflow error body, an unreadable artifact or
    /// an unexpected response.
    Other(String),
}

impl MlflowError {
    /// The error for a non-2xx response with status `status` and body `body`: [`Api`]
    /// if the body is MLflow's error JSON, else [`Other`] with the status and body.
    ///
    /// [`Api`]: MlflowError::Api
    /// [`Other`]: MlflowError::Other
    pub fn from_response(status: u16, body: &str) -> Self {
        let parsed: Option<Value> = serde_json::from_str(body).ok();
        let field = |k: &str| {
            parsed
                .as_ref()
                .and_then(|v| v.get(k))
                .and_then(Value::as_str)
                .map(String::from)
        };
        match field("error_code") {
            Some(error_code) => Self::Api {
                status,
                error_code,
                message: field("message").unwrap_or_default(),
            },
            None => Self::Other(format!("HTTP {status}: {body}")),
        }
    }
}

impl fmt::Display for MlflowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Api {
                status,
                error_code,
                message,
            } => write!(f, "MLflow {error_code} (HTTP {status}): {message}"),
            Self::Other(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for MlflowError {}

/// A blocking client of one MLflow tracking server.
pub struct Client {
    base: String,
    agent: ureq::Agent,
}

impl Client {
    /// A client of the server at `tracking_uri`, e.g. `http://127.0.0.1:5000`. Sends
    /// nothing yet.
    pub fn new(tracking_uri: &str) -> Self {
        let agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .build()
            .into();
        Self {
            base: tracking_uri.trim_end_matches('/').to_string(),
            agent,
        }
    }

    /// The id of experiment `name`, created if it does not exist.
    ///
    /// # Errors
    /// If a call fails.
    pub fn experiment_id(&self, name: &str) -> Result<String, MlflowError> {
        let url = format!("{}/api/2.0/mlflow/experiments/get-by-name", self.base);
        let found = self.agent.get(&url).query("experiment_name", name).call();
        let id = match self.read(found) {
            Ok(v) => v["experiment"]["experiment_id"].clone(),
            Err(MlflowError::Api { error_code, .. }) if error_code == "RESOURCE_DOES_NOT_EXIST" => {
                self.post("experiments/create", &json!({"name": name}))?["experiment_id"].clone()
            }
            Err(e) => return Err(e),
        };
        id.as_str()
            .map(String::from)
            .ok_or_else(|| MlflowError::Other(format!("no experiment_id for {name}")))
    }

    /// The ids of the active runs in `experiment_id` tagged `sciml.run_dir = run_dir`.
    ///
    /// # Errors
    /// If the call fails.
    pub fn find_runs(
        &self,
        experiment_id: &str,
        run_dir: &str,
    ) -> Result<Vec<String>, MlflowError> {
        let body = json!({
            "experiment_ids": [experiment_id],
            "filter": format!("tags.`{RUN_DIR_TAG}` = '{run_dir}'"),
            "max_results": 1000,
        });
        let found = self.post("runs/search", &body)?;
        Ok(found["runs"]
            .as_array()
            .map(|runs| {
                runs.iter()
                    .filter_map(|r| r["info"]["run_id"].as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default())
    }

    /// Deletes run `run_id` (MLflow's soft delete).
    ///
    /// # Errors
    /// If the call fails.
    pub fn delete_run(&self, run_id: &str) -> Result<(), MlflowError> {
        self.post("runs/delete", &json!({"run_id": run_id}))
            .map(drop)
    }

    /// Sends `requests` ([`export_requests`](super::export_requests)) in order and
    /// returns the new run's id. The first must be `runs/create`; its run id replaces
    /// [`RUN_ID`] in the paths and `run_id` fields of the rest. If a later request fails,
    /// the run is marked `FAILED` (end time `now_ms`) before the error is returned.
    ///
    /// # Errors
    /// The first failed request's error.
    pub fn export(&self, requests: &[Request], now_ms: i64) -> Result<String, MlflowError> {
        let (create, rest) = requests
            .split_first()
            .ok_or_else(|| MlflowError::Other("no requests".into()))?;
        let created = self.send(create)?;
        let run_id = created["run"]["info"]["run_id"]
            .as_str()
            .ok_or_else(|| {
                MlflowError::Other(format!("runs/create returned no run id: {created}"))
            })?
            .to_string();
        for r in rest {
            let mut r = r.clone();
            r.path = r.path.replace(RUN_ID, &run_id);
            if let Body::Json(Value::Object(fields)) = &mut r.body
                && fields.contains_key("run_id")
            {
                fields.insert("run_id".into(), run_id.clone().into());
            }
            if let Err(e) = self.send(&r) {
                let failed = json!({"run_id": run_id, "status": "FAILED", "end_time": now_ms});
                // the first error is the one to report
                let _ = self.post("runs/update", &failed);
                return Err(e);
            }
        }
        Ok(run_id)
    }

    /// The run's page in the MLflow UI.
    pub fn run_url(&self, experiment_id: &str, run_id: &str) -> String {
        format!("{}/#/experiments/{experiment_id}/runs/{run_id}", self.base)
    }

    fn post(&self, endpoint: &str, body: &Value) -> Result<Value, MlflowError> {
        self.send(&Request {
            method: Method::Post,
            path: format!("/api/2.0/mlflow/{endpoint}"),
            body: Body::Json(body.clone()),
        })
    }

    fn send(&self, r: &Request) -> Result<Value, MlflowError> {
        let url = format!("{}{}", self.base, r.path);
        let builder = match r.method {
            Method::Post => self.agent.post(&url),
            Method::Put => self.agent.put(&url),
        };
        let response = match &r.body {
            Body::Json(v) => builder
                .header("Content-Type", "application/json")
                .send(v.to_string()),
            Body::File(path) => {
                let bytes = std::fs::read(path)
                    .map_err(|e| MlflowError::Other(format!("{}: {e}", path.display())))?;
                builder
                    .header("Content-Type", "application/octet-stream")
                    .send(&bytes[..])
            }
        };
        self.read(response)
    }

    /// The JSON of a 2xx response (`{}` if empty), else its [`MlflowError`].
    fn read(
        &self,
        response: Result<ureq::http::Response<ureq::Body>, ureq::Error>,
    ) -> Result<Value, MlflowError> {
        let mut response = response.map_err(|e| MlflowError::Other(e.to_string()))?;
        let status = response.status().as_u16();
        let body = response
            .body_mut()
            .read_to_string()
            .map_err(|e| MlflowError::Other(e.to_string()))?;
        if !(200..300).contains(&status) {
            return Err(MlflowError::from_response(status, &body));
        }
        if body.trim().is_empty() {
            return Ok(json!({}));
        }
        serde_json::from_str(&body).map_err(|e| MlflowError::Other(format!("{e}: {body}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlflow_error_body_maps_to_api_error() {
        let body = r#"{"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "Could not find experiment with name 'x'"}"#;
        assert_eq!(
            MlflowError::from_response(404, body),
            MlflowError::Api {
                status: 404,
                error_code: "RESOURCE_DOES_NOT_EXIST".into(),
                message: "Could not find experiment with name 'x'".into(),
            }
        );
    }

    #[test]
    fn other_error_body_keeps_status_and_text() {
        assert_eq!(
            MlflowError::from_response(502, "Bad Gateway"),
            MlflowError::Other("HTTP 502: Bad Gateway".into())
        );
    }
}

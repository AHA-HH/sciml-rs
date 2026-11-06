//! The pure half of the exporter: a run directory to the list of MLflow requests.

use std::error::Error;
use std::path::PathBuf;

use burn::config::Config;
use serde_json::{Map, Value, json};

use crate::neural_operators::{
    data::loaders::poisson::PoissonConfig,
    metrics::poisson::{ERRORS_JSON, PoissonErrorsRecord},
    runs::{EpochMetrics, RunCosts, RunDir},
};

// Per-request limits of `runs/log-batch`, from MLflow's REST reference
// (docs/api_reference/source/rest-api.rst, "Log Batch > Request Limits"): at most 1000
// metrics, 100 params and 100 tags, 1000 entities in total, keys and param/tag values up
// to 250 characters, 1 MB per request. Each batch here holds one entity type, so the
// per-type limits also bound the total; 1000 metrics are ~100 kB of JSON.
/// Most metrics in one `runs/log-batch` request.
pub const MAX_METRICS: usize = 1000;
/// Most params in one `runs/log-batch` request.
pub const MAX_PARAMS: usize = 100;
/// Longest metric, param or tag key MLflow accepts.
pub const MAX_KEY_LEN: usize = 250;
/// Longest param or tag value MLflow accepts.
pub const MAX_VALUE_LEN: usize = 250;

/// Placeholder for the MLflow run id in the requests after `runs/create`: in their
/// `run_id` body field and in artifact paths. [`Client`](super::Client) substitutes it.
pub const RUN_ID: &str = "{run_id}";
/// The tag naming the run directory an MLflow run came from (planning decision Q5).
pub const RUN_DIR_TAG: &str = "sciml.run_dir";

const API: &str = "/api/2.0/mlflow";
const ARTIFACTS_API: &str = "/api/2.0/mlflow-artifacts/artifacts";

/// HTTP method of a [`Request`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// The `/api/2.0/mlflow/...` calls.
    Post,
    /// Artifact uploads.
    Put,
}

/// Body of a [`Request`].
#[derive(Clone, Debug, PartialEq)]
pub enum Body {
    /// A JSON body.
    Json(Value),
    /// A file sent as the raw body.
    File(PathBuf),
}

/// One REST request: method, path below the tracking URI, body.
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    /// The HTTP method.
    pub method: Method,
    /// The path, starting with `/api/2.0/`; may hold [`RUN_ID`].
    pub path: String,
    /// The body; a JSON body may hold [`RUN_ID`] as its `run_id`.
    pub body: Body,
}

/// The requests that export `run` into experiment `experiment_id`, in order: `runs/create`
/// (name and tags), `runs/log-batch` for the params, then for the metrics, one `PUT` per
/// artifact, and `runs/update` to `FINISHED`. See the [module docs](super) for the mapping.
///
/// Times are Unix milliseconds. The start is the directory's `_<unix_secs>` suffix, when
/// it has one (training creates `runs/<name>_<unix_secs>/` as it starts), else `now_ms`;
/// the end is the start plus `training_time_s` from `costs.json`, when present, else
/// `now_ms`. `now_ms` is also every metric's timestamp.
///
/// Reads `run` only: the configs as `FNOConfig`, `TrainingConfig` and `PoissonConfig`,
/// `metrics.csv`, and `costs.json` and `errors.json` when present.
///
/// # Errors
/// If a config, `metrics.csv`, `costs.json` or `errors.json` can't be read or parsed (a
/// non-Poisson run fails on its data config), the directory can't be listed, or a key or
/// value is longer than MLflow accepts.
pub fn export_requests(
    run: &RunDir,
    experiment_id: &str,
    now_ms: i64,
) -> Result<Vec<Request>, Box<dyn Error>> {
    let mut requests = start_requests(run, experiment_id, now_ms)?;
    let costs = run.costs()?;
    let epochs = run.metrics()?;
    let start_ms = start_ms(&dir_name(run)?, now_ms);
    let end_ms = costs.as_ref().map_or(now_ms, |c| {
        start_ms + (c.training_time_s * 1000.0).round() as i64
    });
    let mut metrics: Vec<Value> = epochs
        .iter()
        .flat_map(|e| epoch_metrics(e, now_ms))
        .collect();
    let last = epochs.last().map_or(0, |e| e.epoch);
    metrics.extend(summary_metrics(run, costs.as_ref(), last, now_ms)?);
    requests.extend(tail_requests(run, experiment_id, &metrics, end_ms)?);
    Ok(requests)
}

/// The requests that open `run`'s MLflow run: `runs/create` with the run's name, its start
/// time (as for [`export_requests`]) and tags, then `runs/log-batch` for the params. The
/// first half of [`export_requests`]; live logging ([`MlflowObserver`](super::MlflowObserver))
/// sends it as training starts, when the run directory holds its configs but no
/// `costs.json` yet.
///
/// # Errors
/// If a config or `costs.json` can't be read or parsed, or a key or value is longer than
/// MLflow accepts.
pub fn start_requests(
    run: &RunDir,
    experiment_id: &str,
    now_ms: i64,
) -> Result<Vec<Request>, Box<dyn Error>> {
    let name = dir_name(run)?;
    let model = run.model_config()?;
    let train = run.training_config()?;
    let data: PoissonConfig = run.data_config()?;
    let costs = run.costs()?;

    let padding = model.padding.unwrap_or(0);
    let s = data.s();
    let mut tags = vec![(RUN_DIR_TAG.to_string(), name.clone())];
    if let Some(c) = &costs {
        tags.extend(costs_tags(c));
    }
    tags.push(("padding".into(), padding.to_string()));
    tags.push(("s".into(), s.to_string()));
    tags.push(("n".into(), data.n.to_string()));
    if let Some(src) = &data.source {
        tags.push(("K".into(), src.k.to_string()));
    }
    tags.push(("seed".into(), train.seed.to_string()));
    tags.push(("N".into(), data.base.n_train.to_string()));
    tags.push(("group".into(), format!("p{padding}_s{s}")));

    let mut params = Vec::new();
    flatten("model", &to_json(&model)?, &mut params);
    flatten("train", &to_json(&train)?, &mut params);
    flatten("data", &to_json(&data)?, &mut params);
    for (k, v) in tags.iter().chain(&params) {
        if k.len() > MAX_KEY_LEN || v.len() > MAX_VALUE_LEN {
            return Err(format!("{k}: key or value longer than MLflow's limit of 250").into());
        }
    }

    let mut requests = vec![post(
        "runs/create",
        json!({
            "experiment_id": experiment_id,
            "run_name": name,
            "start_time": start_ms(&name, now_ms),
            "tags": key_values(&tags),
        }),
    )];
    for chunk in params.chunks(MAX_PARAMS) {
        requests.push(post(
            "runs/log-batch",
            json!({"run_id": RUN_ID, "params": key_values(chunk)}),
        ));
    }
    Ok(requests)
}

/// One `runs/log-batch` with epoch `m`'s metrics at step `m.epoch`, timestamp `now_ms`:
/// `train_mse`, `train_l2`, `test_l2`, `lr` and `epoch_time_s` when recorded. What live
/// logging sends after each epoch; [`export_requests`] logs the same metrics, batched.
pub fn epoch_request(m: &EpochMetrics, now_ms: i64) -> Request {
    post(
        "runs/log-batch",
        json!({"run_id": RUN_ID, "metrics": epoch_metrics(m, now_ms)}),
    )
}

/// The requests that close a live-logged run at the end of training: a `runs/log-batch`
/// of the tags from `costs.json` (commit, backend) when present, the summary metrics at
/// the last epoch of `metrics.csv` (as for [`export_requests`]), one `PUT` per artifact,
/// and `runs/update` to `FINISHED` with end time `now_ms`.
///
/// # Errors
/// If `metrics.csv`, `costs.json` or `errors.json` can't be read or parsed, or the
/// directory can't be listed.
pub fn finish_requests(
    run: &RunDir,
    experiment_id: &str,
    now_ms: i64,
) -> Result<Vec<Request>, Box<dyn Error>> {
    let costs = run.costs()?;
    let last = run.metrics()?.last().map_or(0, |e| e.epoch);
    let mut requests = Vec::new();
    if let Some(c) = &costs {
        requests.push(post(
            "runs/log-batch",
            json!({"run_id": RUN_ID, "tags": key_values(&costs_tags(c))}),
        ));
    }
    let metrics = summary_metrics(run, costs.as_ref(), last, now_ms)?;
    requests.extend(tail_requests(run, experiment_id, &metrics, now_ms)?);
    Ok(requests)
}

fn dir_name(run: &RunDir) -> Result<String, Box<dyn Error>> {
    Ok(run
        .path()
        .file_name()
        .ok_or("run directory has no name")?
        .to_string_lossy()
        .into_owned())
}

/// The directory's `_<unix_secs>` suffix in milliseconds, else `now_ms`.
fn start_ms(name: &str, now_ms: i64) -> i64 {
    name.rsplit_once('_')
        .and_then(|(_, secs)| secs.parse::<i64>().ok())
        .map_or(now_ms, |secs| secs * 1000)
}

fn costs_tags(c: &RunCosts) -> Vec<(String, String)> {
    vec![
        ("sciml.git_commit".into(), c.git_commit.clone()),
        ("backend".into(), c.backend_features.join(",")),
    ]
}

fn metric(key: &str, value: f64, step: usize, now_ms: i64) -> Value {
    json!({"key": key, "value": number(value), "timestamp": now_ms, "step": step})
}

fn epoch_metrics(e: &EpochMetrics, now_ms: i64) -> Vec<Value> {
    let mut metrics = vec![
        metric("train_mse", e.train_mse.into(), e.epoch, now_ms),
        metric("train_l2", e.train_l2.into(), e.epoch, now_ms),
        metric("test_l2", e.test_l2.into(), e.epoch, now_ms),
        metric("lr", e.current_lr, e.epoch, now_ms),
    ];
    if let Some(t) = e.epoch_time_s {
        metrics.push(metric("epoch_time_s", t, e.epoch, now_ms));
    }
    metrics
}

/// Errors 2–5 from `errors.json` and the numeric fields of `costs`, at step `last`.
fn summary_metrics(
    run: &RunDir,
    costs: Option<&RunCosts>,
    last: usize,
    now_ms: i64,
) -> Result<Vec<Value>, Box<dyn Error>> {
    let mut metrics = Vec::new();
    let errors_path = run.path().join(ERRORS_JSON);
    if errors_path.exists() {
        let r = PoissonErrorsRecord::load(&errors_path)
            .map_err(|e| format!("{}: {e}", errors_path.display()))?;
        for (key, stats) in [
            ("error2", r.error2),
            ("error3", r.error3),
            ("error4", r.error4),
            ("error5_max_ratio", r.error5_max_ratio),
            ("error5_rel_rms", r.error5_rel_rms),
        ] {
            metrics.push(metric(&format!("{key}_mean"), stats.mean, last, now_ms));
            metrics.push(metric(&format!("{key}_max"), stats.max, last, now_ms));
        }
    }
    if let Some(c) = costs
        && let Value::Object(fields) = to_json(c)?
    {
        for (key, v) in fields {
            if let Some(x) = v.as_f64() {
                metrics.push(metric(&key, x, last, now_ms));
            }
        }
    }
    Ok(metrics)
}

/// `metrics` in `runs/log-batch`es within the limits, one `PUT` per artifact, and
/// `runs/update` to `FINISHED` at `end_ms`.
fn tail_requests(
    run: &RunDir,
    experiment_id: &str,
    metrics: &[Value],
    end_ms: i64,
) -> Result<Vec<Request>, Box<dyn Error>> {
    let mut requests = Vec::new();
    for chunk in metrics.chunks(MAX_METRICS) {
        requests.push(post(
            "runs/log-batch",
            json!({"run_id": RUN_ID, "metrics": chunk}),
        ));
    }
    for file in run.artifacts()? {
        let file_name = file.file_name().unwrap_or_default().to_string_lossy();
        requests.push(Request {
            method: Method::Put,
            path: format!("{ARTIFACTS_API}/{experiment_id}/{RUN_ID}/artifacts/{file_name}"),
            body: Body::File(file.clone()),
        });
    }
    requests.push(post(
        "runs/update",
        json!({"run_id": RUN_ID, "status": "FINISHED", "end_time": end_ms}),
    ));
    Ok(requests)
}

fn post(endpoint: &str, body: Value) -> Request {
    Request {
        method: Method::Post,
        path: format!("{API}/{endpoint}"),
        body: Body::Json(body),
    }
}

fn to_json<C: Config>(config: &C) -> Result<Value, serde_json::Error> {
    serde_json::to_value(config)
}

/// Objects are flattened to `prefix.key`; any other value is a leaf: strings bare, the
/// rest (numbers, arrays, null) as JSON text.
fn flatten(prefix: &str, value: &Value, out: &mut Vec<(String, String)>) {
    match value {
        Value::Object(fields) => {
            for (k, v) in fields {
                flatten(&format!("{prefix}.{k}"), v, out);
            }
        }
        Value::String(s) => out.push((prefix.to_string(), s.clone())),
        other => out.push((prefix.to_string(), other.to_string())),
    }
}

fn key_values(pairs: &[(String, String)]) -> Vec<Value> {
    pairs
        .iter()
        .map(|(k, v)| {
            let mut m = Map::new();
            m.insert("key".into(), k.as_str().into());
            m.insert("value".into(), v.as_str().into());
            Value::Object(m)
        })
        .collect()
}

/// JSON has no NaN or infinity; MLflow's protobuf JSON parser takes them as strings.
fn number(x: f64) -> Value {
    match x {
        x if x.is_nan() => "NaN".into(),
        x if x == f64::INFINITY => "Infinity".into(),
        x if x == f64::NEG_INFINITY => "-Infinity".into(),
        x => x.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::{
        data::loaders::{
            base_dataset::DatasetConfig,
            poisson::{DatasetSource, SplitSource},
        },
        metrics::stats::ErrorStats,
        models::fno::FNOConfig,
        training::config::TrainingConfig,
    };
    use std::collections::HashSet;

    /// A fresh run directory `<tmp>/sciml_rs_tracking_<test>_<pid>/<dir>`.
    fn fresh_run(test: &str, dir: &str) -> RunDir {
        let root =
            std::env::temp_dir().join(format!("sciml_rs_tracking_{test}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join(dir)).unwrap();
        RunDir::open(root.join(dir)).unwrap()
    }

    fn model() -> FNOConfig {
        FNOConfig {
            modes: vec![12, 12],
            hidden_channels: 32,
            data_channels: 1,
            out_channels: 1,
            n_layers: 4,
            spectral_init: None,
            padding: Some(0),
        }
    }

    fn data(source: bool) -> PoissonConfig {
        let split = |name: &str| SplitSource {
            file: format!("poisson_n129_K64_{name}.npz"),
            git_commit: "abc123".into(),
        };
        PoissonConfig {
            base: DatasetConfig {
                n_train: 1000,
                n_test: 200,
            },
            n: 129,
            s: Some(64),
            source: source.then(|| DatasetSource {
                k: 64,
                d: 2,
                train: split("train"),
                test: split("test"),
            }),
            subset: None,
        }
    }

    fn write_epochs(run: &RunDir, epochs: usize) {
        for epoch in 0..epochs {
            run.append_metrics(&EpochMetrics {
                epoch,
                train_mse: 0.5,
                train_l2: 0.25,
                test_l2: 0.125,
                current_lr: 1e-3,
                epoch_time_s: Some(2.0),
            })
            .unwrap();
        }
    }

    fn write_costs(run: &RunDir) {
        run.save_costs(&RunCosts {
            dataset_train_wall_time_s: None,
            dataset_test_wall_time_s: None,
            load_time_s: 1.5,
            training_time_s: 100.0,
            inference_time_s: Some(2.0),
            host_to_device_bytes_train: 10,
            host_to_device_bytes_test: 2,
            peak_gpu_memory: "unavailable".into(),
            backend_features: vec!["flex".into()],
            device: "Cpu".into(),
            git_commit: "deadbeef".into(),
            num_params: 7,
            optimisation_steps: Some(100),
            resumes: None,
        })
        .unwrap();
    }

    fn write_errors(run: &RunDir) {
        let stats = |m: f64| ErrorStats {
            mean: m,
            max: 2.0 * m,
        };
        PoissonErrorsRecord {
            n_test: 2,
            n: 129,
            s: 64,
            device: "Cpu".into(),
            backend_features: vec!["flex".into()],
            error1: "quoted".into(),
            error2: stats(0.002),
            error2_sidecar: stats(0.002),
            error2_compared: false,
            error3: stats(0.003),
            error3_metrics_csv: 0.003,
            error4: stats(0.004),
            error4_worst_sample: 1,
            error4_per_sample: vec![0.004, 0.004],
            u_norm_per_sample: vec![1.0, 1.0],
            error4_u_norm_log_correlation: 0.0,
            error5_max_ratio: stats(0.005),
            error5_rel_rms: stats(0.006),
            boundary_to_total: 0.5,
            ablation_warranted: false,
            cross_checks_passed: true,
        }
        .save(run.path().join(ERRORS_JSON))
        .unwrap();
    }

    fn json(r: &Request) -> &Value {
        match &r.body {
            Body::Json(v) => v,
            Body::File(f) => panic!("expected a JSON body, got {}", f.display()),
        }
    }

    fn pairs(v: &Value) -> Vec<(String, String)> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|p| {
                (
                    p["key"].as_str().unwrap().to_string(),
                    p["value"].as_str().unwrap().to_string(),
                )
            })
            .collect()
    }

    fn metric_keys(requests: &[Request]) -> Vec<(String, u64)> {
        requests
            .iter()
            .filter(|r| json_has(r, "metrics"))
            .flat_map(|r| json(r)["metrics"].as_array().unwrap().clone())
            .map(|m| {
                (
                    m["key"].as_str().unwrap().to_string(),
                    m["step"].as_u64().unwrap(),
                )
            })
            .collect()
    }

    fn json_has(r: &Request, field: &str) -> bool {
        matches!(&r.body, Body::Json(v) if v.get(field).is_some())
    }

    fn artifact_names(requests: &[Request]) -> Vec<String> {
        requests
            .iter()
            .filter_map(|r| match &r.body {
                Body::File(f) => Some(f.file_name().unwrap().to_string_lossy().into_owned()),
                Body::Json(_) => None,
            })
            .collect()
    }

    #[test]
    fn full_run_gives_the_exact_request_list() {
        let run = fresh_run("full", "poisson_fno_n129_s64_p0_1793188648");
        run.save_configs(&model(), &TrainingConfig::new(), &data(true))
            .unwrap();
        write_epochs(&run, 3);
        write_costs(&run);
        write_errors(&run);
        for f in [
            "plots.png",
            "prediction_7.png",
            "model_weights.bpk",
            "checkpoint.bin",
        ] {
            std::fs::write(run.path().join(f), b"x").unwrap();
        }

        let requests = export_requests(&run, "5", 1_000).unwrap();
        let shape: Vec<(Method, &str)> = requests
            .iter()
            .map(|r| (r.method, r.path.as_str()))
            .collect();
        let put =
            |f: &str| format!("/api/2.0/mlflow-artifacts/artifacts/5/{{run_id}}/artifacts/{f}");
        let artifacts = [
            "costs.json",
            "data_cfg.json",
            "errors.json",
            "model_cfg.json",
            "plots.png",
            "prediction_7.png",
            "train_cfg.json",
        ]
        .map(put);
        let mut expected = vec![
            (Method::Post, "/api/2.0/mlflow/runs/create"),
            (Method::Post, "/api/2.0/mlflow/runs/log-batch"),
            (Method::Post, "/api/2.0/mlflow/runs/log-batch"),
        ];
        expected.extend(artifacts.iter().map(|p| (Method::Put, p.as_str())));
        expected.push((Method::Post, "/api/2.0/mlflow/runs/update"));
        assert_eq!(shape, expected);

        let create = json(&requests[0]);
        assert_eq!(create["experiment_id"], "5");
        assert_eq!(create["run_name"], "poisson_fno_n129_s64_p0_1793188648");
        // the directory's suffix; the end adds costs.json's training_time_s of 100 s
        assert_eq!(create["start_time"], 1_793_188_648_000_i64);
        let tags: Vec<(String, String)> = pairs(&create["tags"]);
        let tag = |k: &str, v: &str| (k.to_string(), v.to_string());
        assert_eq!(
            tags,
            vec![
                tag("sciml.run_dir", "poisson_fno_n129_s64_p0_1793188648"),
                tag("sciml.git_commit", "deadbeef"),
                tag("backend", "flex"),
                tag("padding", "0"),
                tag("s", "64"),
                tag("n", "129"),
                tag("K", "64"),
                tag("seed", "42"),
                tag("N", "1000"),
                tag("group", "p0_s64"),
            ]
        );

        assert_eq!(json(&requests[1])["run_id"], RUN_ID);
        assert!(json(&requests[1])["params"].is_array());

        // 3 epochs × 5 per-epoch metrics, 10 error stats, 7 numeric costs at step 2
        let metrics = metric_keys(&requests);
        assert_eq!(metrics.len(), 3 * 5 + 10 + 7);
        assert!(metrics.contains(&("test_l2".into(), 2)));
        assert!(metrics.contains(&("epoch_time_s".into(), 0)));
        assert!(metrics.contains(&("error4_mean".into(), 2)));
        assert!(metrics.contains(&("error5_rel_rms_max".into(), 2)));
        assert!(metrics.contains(&("training_time_s".into(), 2)));
        assert!(metrics.contains(&("num_params".into(), 2)));
        let m0 = &json(&requests[2])["metrics"][0];
        assert_eq!(
            *m0,
            serde_json::json!({"key": "train_mse", "value": 0.5, "timestamp": 1_000, "step": 0})
        );

        assert_eq!(
            *json(requests.last().unwrap()),
            serde_json::json!({"run_id": RUN_ID, "status": "FINISHED", "end_time": 1_793_188_748_000_i64})
        );
        std::fs::remove_dir_all(run.path().parent().unwrap()).unwrap();
    }

    #[test]
    fn old_run_has_its_curve_and_no_summary() {
        let run = fresh_run("old", "poisson_fno_p0_1791906496");
        run.save_configs(&model(), &TrainingConfig::new(), &data(false))
            .unwrap();
        std::fs::write(
            run.path().join("metrics.csv"),
            "epoch,train_mse,train_l2,test_l2,current_lr\n0,0.5,0.3,0.4,0.001\n1,0.4,0.2,0.3,0.0009\n",
        )
        .unwrap();

        let requests = export_requests(&run, "1", 0).unwrap();
        let metrics: HashSet<String> = metric_keys(&requests).into_iter().map(|(k, _)| k).collect();
        let expected: HashSet<String> = ["train_mse", "train_l2", "test_l2", "lr"]
            .map(String::from)
            .into();
        assert_eq!(metrics, expected);
        assert_eq!(metric_keys(&requests).len(), 8);
        assert_eq!(
            artifact_names(&requests),
            ["data_cfg.json", "model_cfg.json", "train_cfg.json"]
        );
        // start from the directory's suffix; no costs.json, so the end is now
        assert_eq!(json(&requests[0])["start_time"], 1_791_906_496_000_i64);
        assert_eq!(json(requests.last().unwrap())["end_time"], 0);
        let tags = pairs(&json(&requests[0])["tags"]);
        assert!(
            tags.iter()
                .all(|(k, _)| k != "K" && k != "sciml.git_commit" && k != "backend")
        );
        std::fs::remove_dir_all(run.path().parent().unwrap()).unwrap();
    }

    #[test]
    fn params_flatten_nested_keys_and_vectors() {
        let run = fresh_run("params", "poisson_run");
        run.save_configs(&model(), &TrainingConfig::new(), &data(true))
            .unwrap();
        write_epochs(&run, 1);
        let requests = export_requests(&run, "1", 7).unwrap();
        // no numeric suffix: the start is now
        assert_eq!(json(&requests[0])["start_time"], 7);
        let params = pairs(&json(&requests[1])["params"]);
        let get = |k: &str| {
            params
                .iter()
                .find(|(key, _)| key == k)
                .unwrap_or_else(|| panic!("no param {k}"))
                .1
                .clone()
        };
        assert_eq!(get("model.modes"), "[12,12]");
        assert_eq!(get("model.padding"), "0");
        assert_eq!(get("model.spectral_init"), "null");
        assert_eq!(get("train.learning_rate"), "0.001");
        assert_eq!(get("train.seed"), "42");
        assert_eq!(get("data.n"), "129");
        assert_eq!(get("data.base.n_train"), "1000");
        assert_eq!(get("data.source.train.file"), "poisson_n129_K64_train.npz");
        assert_eq!(get("data.subset"), "null");
        std::fs::remove_dir_all(run.path().parent().unwrap()).unwrap();
    }

    #[test]
    fn metrics_of_500_epochs_are_batched_within_the_limits() {
        let run = fresh_run("batching", "poisson_run");
        run.save_configs(&model(), &TrainingConfig::new(), &data(true))
            .unwrap();
        write_epochs(&run, 500);
        write_costs(&run);
        let requests = export_requests(&run, "1", 0).unwrap();
        for r in &requests {
            if let Body::Json(v) = &r.body {
                let len = |f: &str| v.get(f).and_then(Value::as_array).map_or(0, Vec::len);
                assert!(len("metrics") <= MAX_METRICS);
                assert!(len("params") <= MAX_PARAMS);
                assert!(len("metrics") + len("params") + len("tags") <= MAX_METRICS);
            }
        }
        let keys = metric_keys(&requests);
        let unique: HashSet<&(String, u64)> = keys.iter().collect();
        assert_eq!(keys.len(), unique.len(), "a (key, step) logged twice");
        assert_eq!(keys.len(), 500 * 5 + 7);
        for step in 0..500 {
            for key in ["train_mse", "train_l2", "test_l2", "lr", "epoch_time_s"] {
                assert!(unique.contains(&(key.to_string(), step)));
            }
        }
        // 2507 metrics → 3 batches
        let batches = requests.iter().filter(|r| json_has(r, "metrics")).count();
        assert_eq!(batches, 3);
        std::fs::remove_dir_all(run.path().parent().unwrap()).unwrap();
    }

    #[test]
    fn epoch_request_is_one_log_batch_of_the_epochs_metrics() {
        let m = EpochMetrics {
            epoch: 3,
            train_mse: 0.5,
            train_l2: 0.25,
            test_l2: 0.125,
            current_lr: 1e-3,
            epoch_time_s: Some(2.5),
        };
        let at = |key: &str, value: f64| serde_json::json!({"key": key, "value": value, "timestamp": 9, "step": 3});
        assert_eq!(
            epoch_request(&m, 9),
            Request {
                method: Method::Post,
                path: "/api/2.0/mlflow/runs/log-batch".into(),
                body: Body::Json(serde_json::json!({
                    "run_id": RUN_ID,
                    "metrics": [
                        at("train_mse", 0.5),
                        at("train_l2", 0.25),
                        at("test_l2", 0.125),
                        at("lr", 1e-3),
                        at("epoch_time_s", 2.5),
                    ],
                })),
            }
        );
    }

    #[test]
    fn finish_requests_close_a_live_run() {
        let run = fresh_run("finish", "poisson_fno_n129_s64_p0_1793188648");
        run.save_configs(&model(), &TrainingConfig::new(), &data(true))
            .unwrap();
        write_epochs(&run, 3);
        write_costs(&run);
        let requests = finish_requests(&run, "5", 2_000).unwrap();
        assert_eq!(
            pairs(&json(&requests[0])["tags"]),
            vec![
                ("sciml.git_commit".to_string(), "deadbeef".to_string()),
                ("backend".to_string(), "flex".to_string())
            ]
        );
        // only the 7 numeric costs, at the last epoch; no per-epoch metrics
        let metrics = metric_keys(&requests);
        assert_eq!(metrics.len(), 7);
        assert!(metrics.iter().all(|(_, step)| *step == 2));
        assert_eq!(
            artifact_names(&requests),
            [
                "costs.json",
                "data_cfg.json",
                "model_cfg.json",
                "train_cfg.json"
            ]
        );
        assert_eq!(
            *json(requests.last().unwrap()),
            serde_json::json!({"run_id": RUN_ID, "status": "FINISHED", "end_time": 2_000})
        );
        std::fs::remove_dir_all(run.path().parent().unwrap()).unwrap();
    }

    #[test]
    fn non_finite_values_become_strings() {
        assert_eq!(number(f64::NAN), "NaN");
        assert_eq!(number(f64::INFINITY), "Infinity");
        assert_eq!(number(f64::NEG_INFINITY), "-Infinity");
        assert_eq!(number(0.5), 0.5);
    }
}

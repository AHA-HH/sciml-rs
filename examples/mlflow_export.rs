//! Exports finished Poisson run directories to a self-hosted MLflow tracking server, one
//! MLflow run per directory (Phase 8 T1).
//!
//! ```sh
//! cargo run --features mlflow --example mlflow_export -- [--replace] <experiment> runs/<run_dir> ...
//! ```
//!
//! The server is `MLFLOW_TRACKING_URI`, default `http://127.0.0.1:5000`; the experiment
//! is created if it does not exist. A directory already in the experiment (tag
//! `sciml.run_dir`) is skipped, or with `--replace` its MLflow run is deleted and the
//! directory exported again. Prints one line per directory (exported / skipped / failed)
//! and exits non-zero if any failed.

use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use sciml_rs::neural_operators::{
    runs::RunDir,
    tracking::{Client, export_requests},
};

const DEFAULT_URI: &str = "http://127.0.0.1:5000";

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let replace = args.first().is_some_and(|a| a == "--replace");
    if replace {
        args.remove(0);
    }
    if args.len() < 2 {
        eprintln!("usage: mlflow_export [--replace] <experiment> <run_dir> ...");
        return ExitCode::FAILURE;
    }
    let uri = std::env::var("MLFLOW_TRACKING_URI").unwrap_or_else(|_| DEFAULT_URI.into());
    let client = Client::new(&uri);
    let experiment = match client.experiment_id(&args[0]) {
        Ok(id) => id,
        Err(e) => {
            eprintln!("experiment {} at {uri}: {e}", args[0]);
            return ExitCode::FAILURE;
        }
    };

    let mut failed = 0;
    for dir in &args[1..] {
        match export(&client, &experiment, dir, replace) {
            Ok(line) => println!("{line}"),
            Err(e) => {
                failed += 1;
                println!("failed   {dir}: {e}");
            }
        }
    }
    if failed > 0 {
        eprintln!("{failed} of {} run(s) failed", args.len() - 1);
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn export(
    client: &Client,
    experiment: &str,
    dir: &str,
    replace: bool,
) -> Result<String, Box<dyn std::error::Error>> {
    let run = RunDir::open(dir)?;
    let name = run
        .path()
        .file_name()
        .ok_or("no directory name")?
        .to_string_lossy()
        .into_owned();
    let existing = client.find_runs(experiment, &name)?;
    if let Some(id) = existing.first().filter(|_| !replace) {
        return Ok(format!(
            "skipped  {dir}: already {}",
            client.run_url(experiment, id)
        ));
    }
    let now_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as i64;
    let requests = export_requests(&run, experiment, now_ms)?;
    for id in &existing {
        client.delete_run(id)?;
    }
    let run_id = client.export(&requests, now_ms)?;
    let verb = if existing.is_empty() {
        "exported"
    } else {
        "replaced"
    };
    Ok(format!(
        "{verb} {dir}: {}",
        client.run_url(experiment, &run_id)
    ))
}

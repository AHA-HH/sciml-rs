//! CSV persistence for per-epoch training metrics.

use crate::neural_operators::{
    metrics::plots::plot_metrics,
    models::fno::FNOConfig,
    training::{metrics::EpochMetrics, trainer::TrainingConfig},
};
use burn::config::Config;
use std::io::BufRead;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Reads a metrics CSV written by `write_metrics_csv` back into `EpochMetrics`.
/// Panics on missing/malformed fields — no partial-row recovery.
pub fn read_metrics_csv(path: &str) -> Vec<EpochMetrics> {
    let file = std::fs::File::open(path).expect("failed to open csv");
    let reader = std::io::BufReader::new(file);
    let mut metrics = Vec::new();

    for line in reader.lines().skip(1) {
        let line = line.expect("failed to read line");
        let fields: Vec<&str> = line.split(',').collect();
        metrics.push(EpochMetrics {
            epoch: fields[0].parse().unwrap(),
            train_mse: fields[1].parse().unwrap(),
            train_l2: fields[2].parse().unwrap(),
            test_l2: fields[3].parse().unwrap(),
            current_lr: fields[4].parse().unwrap(),
        });
    }
    metrics
}

/// Writes per-epoch metrics to CSV, one row per epoch, with a header row.
pub fn write_metrics_csv(path: &PathBuf, metrics: &[EpochMetrics]) {
    let mut file = File::create(path).expect("failed to create metrics csv");
    writeln!(file, "epoch,train_mse,train_l2,test_l2,current_lr")
        .expect("failed to write header");
    for m in metrics {
        writeln!(file, "{},{},{},{},{}",
            m.epoch, m.train_mse, m.train_l2, m.test_l2, m.current_lr)
            .expect("failed to write row");
    }
}

/// Writes `metrics.csv` and `metrics.png` into a fresh timestamped run
/// directory under `runs/`, and returns the directory.
pub fn write_run_artifacts<D: Config>(name: &str, metrics: &[EpochMetrics], model_cfg: &FNOConfig, train_cfg: &TrainingConfig, data_cfg: &D,) -> PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs();

    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("runs")
        .join(format!("{name}_{stamp}"));
    std::fs::create_dir_all(&dir).expect("could not create run directory");

    model_cfg.save(dir.join("model_cfg.json")).expect("could not save model config");
    train_cfg.save(dir.join("train_cfg.json")).expect("could not save training config");
    data_cfg.save(dir.join("data_cfg.json")).expect("could not save dataset config");

    write_metrics_csv(&dir.join("metrics.csv"), metrics);
    plot_metrics(&dir.join("plots.png"), metrics);

    dir
}
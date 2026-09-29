//! CSV persistence for per-epoch training metrics.

use crate::neural_operators::{
    metrics::plots::plot_metrics,
    models::fno::FNOConfig,
    training::{metrics::EpochMetrics, trainer::TrainingConfig},
};
use burn::config::Config;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// Reads a metrics CSV written by [`write_metrics_csv`] back into
/// `EpochMetrics`.
///
/// # Errors
/// I/O errors from opening or reading the file, and
/// [`io::ErrorKind::InvalidData`] naming the line for a row with missing or
/// unparsable fields.
pub fn read_metrics_csv(path: impl AsRef<Path>) -> io::Result<Vec<EpochMetrics>> {
    let reader = BufReader::new(File::open(path)?);
    let mut metrics = Vec::new();

    for (i, line) in reader.lines().enumerate().skip(1) {
        let line = line?;
        let invalid = |what: String| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("line {}: {what}", i + 1),
            )
        };
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 5 {
            return Err(invalid(format!("expected 5 fields, got {}", fields.len())));
        }
        let bad = |name: &str| invalid(format!("could not parse {name}"));
        metrics.push(EpochMetrics {
            epoch: fields[0].parse().map_err(|_| bad("epoch"))?,
            train_mse: fields[1].parse().map_err(|_| bad("train_mse"))?,
            train_l2: fields[2].parse().map_err(|_| bad("train_l2"))?,
            test_l2: fields[3].parse().map_err(|_| bad("test_l2"))?,
            current_lr: fields[4].parse().map_err(|_| bad("current_lr"))?,
        });
    }
    Ok(metrics)
}

/// Writes per-epoch metrics to CSV, one row per epoch, with a header row.
pub fn write_metrics_csv(path: impl AsRef<Path>, metrics: &[EpochMetrics]) -> io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "epoch,train_mse,train_l2,test_l2,current_lr")?;
    for m in metrics {
        writeln!(
            file,
            "{},{},{},{},{}",
            m.epoch, m.train_mse, m.train_l2, m.test_l2, m.current_lr
        )?;
    }
    Ok(())
}

/// Writes the three configs, `metrics.csv` and `plots.png` into a fresh
/// timestamped run directory under `runs/` and returns the directory.
///
/// The plot is optional: if gnuplot is unavailable a warning is printed and
/// the run directory is still returned, so callers can go on to save weights.
///
/// # Errors
/// If the directory, a config file, or the CSV can't be written.
pub fn write_run_artifacts<D: Config>(
    name: &str,
    metrics: &[EpochMetrics],
    model_cfg: &FNOConfig,
    train_cfg: &TrainingConfig,
    data_cfg: &D,
) -> io::Result<PathBuf> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(io::Error::other)?
        .as_secs();

    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("runs")
        .join(format!("{name}_{stamp}"));
    std::fs::create_dir_all(&dir)?;

    model_cfg.save(dir.join("model_cfg.json"))?;
    train_cfg.save(dir.join("train_cfg.json"))?;
    data_cfg.save(dir.join("data_cfg.json"))?;

    write_metrics_csv(dir.join("metrics.csv"), metrics)?;
    if let Err(e) = plot_metrics(dir.join("plots.png"), metrics, name) {
        eprintln!("warning: skipping plots.png ({e})");
    }

    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_round_trips() {
        let metrics = vec![
            EpochMetrics {
                epoch: 0,
                train_mse: 0.5,
                train_l2: 0.3,
                test_l2: 0.4,
                current_lr: 1e-3,
            },
            EpochMetrics {
                epoch: 1,
                train_mse: 0.25,
                train_l2: 0.15,
                test_l2: 0.2,
                current_lr: 5e-4,
            },
        ];

        let path = std::env::temp_dir().join("sciml_rs_metrics_round_trip.csv");
        write_metrics_csv(&path, &metrics).unwrap();
        let read = read_metrics_csv(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(read.len(), 2);
        assert_eq!(read[1].epoch, 1);
        assert!((read[0].train_mse - 0.5).abs() < 1e-9);
        assert!((read[1].current_lr - 5e-4).abs() < 1e-12);
    }

    #[test]
    fn read_metrics_csv_reports_malformed_row_with_line_number() {
        let path = std::env::temp_dir().join("sciml_rs_metrics_malformed.csv");
        std::fs::write(
            &path,
            "epoch,train_mse,train_l2,test_l2,current_lr\n0,0.5,0.3,0.4,0.001\n1,0.25,oops,0.2,0.0005\n",
        )
        .unwrap();
        let err = read_metrics_csv(&path).unwrap_err();
        std::fs::remove_file(&path).ok();

        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert_eq!(err.to_string(), "line 3: could not parse train_l2");
    }

    #[test]
    fn read_metrics_csv_reports_short_row() {
        let path = std::env::temp_dir().join("sciml_rs_metrics_short.csv");
        std::fs::write(
            &path,
            "epoch,train_mse,train_l2,test_l2,current_lr\n0,0.5\n",
        )
        .unwrap();
        let err = read_metrics_csv(&path).unwrap_err();
        std::fs::remove_file(&path).ok();

        assert_eq!(err.to_string(), "line 2: expected 5 fields, got 2");
    }

    #[test]
    fn read_metrics_csv_missing_file_is_not_found() {
        let err = read_metrics_csv("/definitely/not/here/metrics.csv").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }
}

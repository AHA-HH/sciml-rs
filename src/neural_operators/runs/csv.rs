//! `metrics.csv`: per-epoch training metrics, one row per epoch with a header row. Since
//! Phase 5 T4 a sixth column, `epoch_time_s`, follows the original five; files written
//! before it have five columns and still read, with `epoch_time_s` = `None`.

use super::EpochMetrics;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;

/// Reads a metrics CSV written by [`append_metrics_row`] back into
/// `EpochMetrics`: rows of five fields (before Phase 5 T4) or six; a missing or empty sixth
/// field reads as `epoch_time_s` = `None`.
///
/// # Errors
/// I/O errors from opening or reading the file, and
/// [`io::ErrorKind::InvalidData`] naming the line for a row with missing or
/// unparsable fields.
pub(super) fn read_metrics_csv(path: impl AsRef<Path>) -> io::Result<Vec<EpochMetrics>> {
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
        if !(5..=6).contains(&fields.len()) {
            return Err(invalid(format!(
                "expected 5 or 6 fields, got {}",
                fields.len()
            )));
        }
        let bad = |name: &str| invalid(format!("could not parse {name}"));
        metrics.push(EpochMetrics {
            epoch: fields[0].parse().map_err(|_| bad("epoch"))?,
            train_mse: fields[1].parse().map_err(|_| bad("train_mse"))?,
            train_l2: fields[2].parse().map_err(|_| bad("train_l2"))?,
            test_l2: fields[3].parse().map_err(|_| bad("test_l2"))?,
            current_lr: fields[4].parse().map_err(|_| bad("current_lr"))?,
            epoch_time_s: match fields.get(5) {
                Some(t) if !t.is_empty() => Some(t.parse().map_err(|_| bad("epoch_time_s"))?),
                _ => None,
            },
        });
    }
    Ok(metrics)
}

/// Appends one row to a metrics CSV, creating it with the header row if it is missing or
/// empty. The only writer of `metrics.csv`. A row without an epoch time leaves its
/// `epoch_time_s` field empty.
pub(super) fn append_metrics_row(path: impl AsRef<Path>, m: &EpochMetrics) -> io::Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    if file.metadata()?.len() == 0 {
        writeln!(
            file,
            "epoch,train_mse,train_l2,test_l2,current_lr,epoch_time_s"
        )?;
    }
    let epoch_time = m.epoch_time_s.map_or(String::new(), |t| t.to_string());
    writeln!(
        file,
        "{},{},{},{},{},{epoch_time}",
        m.epoch, m.train_mse, m.train_l2, m.test_l2, m.current_lr
    )
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
                epoch_time_s: Some(1.5),
            },
            EpochMetrics {
                epoch: 1,
                train_mse: 0.25,
                train_l2: 0.15,
                test_l2: 0.2,
                current_lr: 5e-4,
                epoch_time_s: None,
            },
        ];

        let path = std::env::temp_dir().join("sciml_rs_metrics_round_trip.csv");
        std::fs::remove_file(&path).ok();
        for m in &metrics {
            append_metrics_row(&path, m).unwrap();
        }
        let read = read_metrics_csv(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(read.len(), 2);
        assert_eq!(read[1].epoch, 1);
        assert!((read[0].train_mse - 0.5).abs() < 1e-9);
        assert!((read[1].current_lr - 5e-4).abs() < 1e-12);
        assert_eq!(read[0].epoch_time_s, Some(1.5));
        assert_eq!(read[1].epoch_time_s, None);
    }

    // phase 5 T4: runs written before the epoch_time_s column still read
    #[test]
    fn read_metrics_csv_reads_old_five_and_new_six_columns() {
        let path = std::env::temp_dir().join("sciml_rs_metrics_five_six.csv");
        std::fs::write(
            &path,
            "epoch,train_mse,train_l2,test_l2,current_lr\n0,0.5,0.3,0.4,0.001\n",
        )
        .unwrap();
        let old = read_metrics_csv(&path).unwrap();
        std::fs::write(
            &path,
            "epoch,train_mse,train_l2,test_l2,current_lr,epoch_time_s\n0,0.5,0.3,0.4,0.001,11.25\n",
        )
        .unwrap();
        let new = read_metrics_csv(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!((old[0].test_l2, old[0].epoch_time_s), (0.4, None));
        assert_eq!((new[0].test_l2, new[0].epoch_time_s), (0.4, Some(11.25)));
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

        assert_eq!(err.to_string(), "line 2: expected 5 or 6 fields, got 2");
    }

    #[test]
    fn read_metrics_csv_missing_file_is_not_found() {
        let err = read_metrics_csv("/definitely/not/here/metrics.csv").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }
}

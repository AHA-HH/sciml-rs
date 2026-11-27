//! `lbfgs_steps.csv`: one row per L-BFGS step, that is per epoch (optimisers CO2.1), with a
//! header row: the epoch, Φ before and after the step, the step's forward and backward
//! passes (one each per closure evaluation, D7) and its wall time. The pass columns are named
//! as in `gn_steps.csv`, so one reader sums both.

use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::Path;

/// One row of `lbfgs_steps.csv`.
#[derive(Clone, Debug, PartialEq)]
pub struct LbfgsStepRecord {
    /// The zero-based epoch; one step per epoch.
    pub epoch: usize,
    /// The objective at the step's incoming parameters (its first closure evaluation).
    pub phi_before: f64,
    /// The objective `LBFGS::step` returns, at the outgoing parameters.
    pub phi_after: f64,
    /// Closure evaluations, each one forward and one backward of the model on the full
    /// batch (D7); the forward and backward pass columns both hold it.
    pub closure_evals: usize,
    /// Wall-clock seconds of the step (not the metrics forward).
    pub wall_time_s: f64,
}

const HEADER: &str = "epoch,phi_before,phi_after,forward_passes,backward_passes,wall_time_s";

/// Appends one row, creating the file with its header if it is missing or empty.
pub(super) fn append_lbfgs_step_row(path: impl AsRef<Path>, r: &LbfgsStepRecord) -> io::Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    if file.metadata()?.len() == 0 {
        writeln!(file, "{HEADER}")?;
    }
    writeln!(
        file,
        "{},{},{},{},{},{}",
        r.epoch, r.phi_before, r.phi_after, r.closure_evals, r.closure_evals, r.wall_time_s
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::runs::{GnPasses, RunDir};

    /// Rows follow the header; the passes sum by `gn_steps`' reader; a resumed run drops
    /// the rows after its checkpoint; a run without the file has no passes.
    #[test]
    fn rows_sum_and_truncate() {
        let root =
            std::env::temp_dir().join(format!("sciml_rs_lbfgs_steps_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let run = RunDir::create(&root, "lbfgs").unwrap();
        assert!(run.lbfgs_passes().unwrap().is_none());
        run.truncate_lbfgs_steps(1).unwrap();
        for epoch in 0..3 {
            let r = LbfgsStepRecord {
                epoch,
                phi_before: 1.5,
                phi_after: 0.25,
                closure_evals: 4 + epoch,
                wall_time_s: 0.75,
            };
            run.append_lbfgs_step(&r).unwrap();
        }
        let text = std::fs::read_to_string(run.path().join("lbfgs_steps.csv")).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], HEADER);
        assert_eq!(lines[1], "0,1.5,0.25,4,4,0.75");
        let all = GnPasses {
            forward: 15,
            backward: 15,
            steps: 3,
        };
        assert_eq!(run.lbfgs_passes().unwrap(), Some(all));
        run.truncate_lbfgs_steps(2).unwrap();
        assert_eq!(run.lbfgs_passes().unwrap().unwrap().forward, 9);
        let _ = std::fs::remove_dir_all(&root);
    }
}

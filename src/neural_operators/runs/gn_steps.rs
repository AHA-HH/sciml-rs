//! `gn_steps.csv`: one row per Gauss–Newton step (Phase 10 T3), with a header row: the
//! epoch, every [`GnStep`] field in its declared order (the Ritz pair as two columns,
//! `ritz_min` and `ritz_max`), then the step's wall time. A `None` is an empty field.

use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;

use crate::neural_operators::optim::gauss_newton::GnStep;

/// One row of `gn_steps.csv`: a step's diagnostics and where and how long it ran.
#[derive(Clone, Debug, PartialEq)]
pub struct GnStepRecord {
    /// The zero-based epoch the step belongs to.
    pub epoch: usize,
    /// What the step did; `step.step` counts steps from the run's start.
    pub step: GnStep,
    /// Wall-clock seconds of the step (the optimiser's step only, not the metrics forward).
    pub wall_time_s: f64,
}

/// The passes `gn_steps.csv` records, summed over its rows
/// ([`RunDir::gn_passes`](super::RunDir::gn_passes)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GnPasses {
    /// Forward passes of the optimiser's steps (each finite-difference J v as two).
    pub forward: usize,
    /// Backward passes of the optimiser's steps.
    pub backward: usize,
    /// Steps, one per row.
    pub steps: usize,
}

const HEADER: &str = "epoch,step,phi_before,phi_after,grad_norm,step_norm,\
predicted_reduction,actual_reduction,rho,lambda,lambda_next,accepted,accepted_iteration,\
cg_iterations,cg_rel_residual,cg_stop,ritz_min,ritz_max,adjoint_error,forward_passes,\
backward_passes,wall_time_s";

fn opt<T: ToString>(x: Option<T>) -> String {
    x.map_or(String::new(), |x| x.to_string())
}

/// Appends one row, creating the file with its header if it is missing or empty.
pub(super) fn append_gn_step_row(path: impl AsRef<Path>, r: &GnStepRecord) -> io::Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    if file.metadata()?.len() == 0 {
        writeln!(file, "{HEADER}")?;
    }
    let s = &r.step;
    let fields = [
        r.epoch.to_string(),
        s.step.to_string(),
        s.phi_before.to_string(),
        s.phi_after.to_string(),
        s.grad_norm.to_string(),
        s.step_norm.to_string(),
        s.predicted_reduction.to_string(),
        s.actual_reduction.to_string(),
        s.rho.to_string(),
        s.lambda.to_string(),
        s.lambda_next.to_string(),
        s.accepted.to_string(),
        opt(s.accepted_iteration),
        s.cg_iterations.to_string(),
        s.cg_rel_residual.to_string(),
        format!("{:?}", s.cg_stop),
        opt(s.ritz.map(|(lo, _)| lo)),
        opt(s.ritz.map(|(_, hi)| hi)),
        opt(s.adjoint_error),
        s.forward_passes.to_string(),
        s.backward_passes.to_string(),
        r.wall_time_s.to_string(),
    ];
    writeln!(file, "{}", fields.join(","))
}

/// The rows' summed `forward_passes` and `backward_passes`, and the row count.
///
/// # Errors
/// I/O errors, and [`io::ErrorKind::InvalidData`] naming the line for a header without
/// those columns or a field that does not parse.
pub(super) fn sum_passes(path: impl AsRef<Path>) -> io::Result<GnPasses> {
    let invalid = |what: String| io::Error::new(io::ErrorKind::InvalidData, what);
    let mut lines = BufReader::new(File::open(path)?).lines();
    let header = lines.next().transpose()?.unwrap_or_default();
    let column = |name: &str| {
        header
            .split(',')
            .position(|c| c == name)
            .ok_or_else(|| invalid(format!("line 1: no {name} column")))
    };
    let (f, b) = (column("forward_passes")?, column("backward_passes")?);
    let mut sum = GnPasses {
        forward: 0,
        backward: 0,
        steps: 0,
    };
    for (i, line) in lines.enumerate() {
        let line = line?;
        let fields: Vec<&str> = line.split(',').collect();
        let parse = |k: usize, name: &str| -> io::Result<usize> {
            fields
                .get(k)
                .and_then(|x| x.parse().ok())
                .ok_or_else(|| invalid(format!("line {}: could not parse {name}", i + 2)))
        };
        sum.forward += parse(f, "forward_passes")?;
        sum.backward += parse(b, "backward_passes")?;
        sum.steps += 1;
    }
    Ok(sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::optim::gauss_newton::CgStop;

    fn record(epoch: usize, step: u64, ritz: Option<(f64, f64)>) -> GnStepRecord {
        GnStepRecord {
            epoch,
            step: GnStep {
                step,
                phi_before: 1.5,
                phi_after: 1.25,
                grad_norm: 2.0,
                step_norm: 0.5,
                predicted_reduction: -0.3,
                actual_reduction: -0.25,
                rho: 0.8333,
                lambda: 1.0,
                lambda_next: 1.0,
                accepted: true,
                accepted_iteration: Some(3),
                cg_iterations: 4,
                cg_rel_residual: 1e-3,
                cg_stop: CgStop::MaxIters,
                ritz,
                adjoint_error: None,
                forward_passes: 12,
                backward_passes: 5,
            },
            wall_time_s: 0.75,
        }
    }

    /// A resumed run drops the rows of the epochs after its checkpoint; an Adam run has no
    /// file, which is no error.
    #[test]
    fn truncate_gn_steps_keeps_the_earlier_epochs() {
        use crate::neural_operators::runs::RunDir;
        let root =
            std::env::temp_dir().join(format!("sciml_rs_gn_steps_truncate_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let run = RunDir::create(&root, "gn").unwrap();
        run.truncate_gn_steps(2).unwrap();
        assert!(run.gn_passes().unwrap().is_none());
        for (i, epoch) in [0, 0, 1, 2].into_iter().enumerate() {
            run.append_gn_step(&record(epoch, i as u64, None)).unwrap();
        }
        run.truncate_gn_steps(2).unwrap();
        assert_eq!(run.gn_passes().unwrap().unwrap().steps, 3);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rows_follow_the_header_and_passes_sum() {
        let path =
            std::env::temp_dir().join(format!("sciml_rs_gn_steps_{}.csv", std::process::id()));
        std::fs::remove_file(&path).ok();
        append_gn_step_row(&path, &record(0, 0, Some((0.5, 4.0)))).unwrap();
        append_gn_step_row(&path, &record(1, 1, None)).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], HEADER);
        assert_eq!(
            lines[1],
            "0,0,1.5,1.25,2,0.5,-0.3,-0.25,0.8333,1,1,true,3,4,0.001,MaxIters,0.5,4,,12,5,0.75"
        );
        let n = HEADER.split(',').count();
        assert!(lines[1..].iter().all(|l| l.split(',').count() == n));
        let expected = GnPasses {
            forward: 24,
            backward: 10,
            steps: 2,
        };
        assert_eq!(sum_passes(&path).unwrap(), expected);

        std::fs::write(&path, format!("{HEADER}\n0,0,x\n")).unwrap();
        let err = sum_passes(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert_eq!(err.to_string(), "line 2: could not parse forward_passes");
    }
}

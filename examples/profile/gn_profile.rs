//! Optimisers CO0.2 (docs/optimisers/tasks/CO0.2.md): where the time of one Gauss–Newton
//! step goes, measured on a trained run's checkpoint without changing any library code.
//!
//! Run with: cargo run --release --example gn_profile -- <out.md> runs/<run_dir> [<cg_cap> ...]
//! (`--no-default-features --features cuda` on the VM; caps default to 50 and 150).
//!
//! The run must be a Gauss–Newton Poisson run with a checkpoint (its model and
//! Gauss–Newton state). On the first batch of its training split, at its batch size and
//! without shuffling:
//! 1. **Unit costs**, each the median of [`REPS`] calls after [`WARMUP`], every call
//!    bracketed by [`Device::sync`] and ending in a host read: Φ's forward without autodiff
//!    (backtracking, and `train_epoch_gn`'s metric forward), the gradient (`vjp` with
//!    u = r: one tracked forward and one backward, as the step's own gradient), the
//!    finite-difference JVP, the Hutchinson refresh and the adjoint check.
//! 2. **CG in place**, per cap: `cg` on the step's system (−g, the checkpoint's warm start
//!    and diagonal preconditioner, its λ and μ), its products timed one by one; CG's own
//!    work per product is the rest of its time.
//! 3. **Real steps**, per cap: [`STEPS`] `GaussNewton::step`s from the checkpoint's state,
//!    cycling the split's batches, each timed. Each step's counts, from its `GnStep`
//!    ([`counts`]), times the unit costs give a predicted time beside the measured one.
//!
//! Writes the tables to the first argument. Timings are of this machine and backend only.
//!
//! Approximations: the backtracking's Φ forward is timed as a forward of the model without
//! autodiff (`Module::valid`); the step's own (`select`) also shifts θ and rebuilds the
//! model on the autodiff device with untracked parameters (`replace`, private). The metric
//! forward of `train_epoch_gn` takes the same unit, though it computes `LpLoss`, not the
//! residual. CG's in-place products are synced between the JVP and the VJP, which the step
//! is not. What these miss lands in the "unexplained" row. The reference run's own passes
//! per second (Phase 10, the fork) come from its `gn_steps.csv` by hand (the task's report).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::prelude::*;
use sciml_rs::neural_operators::{
    data::{
        batcher::Batch,
        catalogue::stage,
        device_batcher::device_loader,
        loaders::poisson::{PoissonConfig, load_poisson_uniform},
        transforms::normalizers::FlatDecoder,
    },
    optim::gauss_newton::{
        CgConfig, CgStop, GaussNewton, GaussNewtonConfig, GaussNewtonState, GnStep, LeastSquares,
        ParamVec, Preconditioner, adjoint_error, cg, hutchinson_diagonal, jvp, params, vjp,
    },
    runs::RunDir,
    training::{config::OptimizerConfig, gn::FnoResidual},
    utils::backend::{backend_features, git_commit},
};

/// Untimed calls before each unit's timed ones.
const WARMUP: usize = 2;
/// Timed calls per unit; the median is reported.
const REPS: usize = 10;
/// Real steps per cap.
const STEPS: usize = 10;
/// CG caps when none is given: D1's two candidates.
const DEFAULT_CAPS: [usize; 2] = [50, 150];
/// A mean gap between predicted and measured step time above this is flagged (the plan's
/// question for the author).
const GAP_FLAG: f64 = 0.15;

/// What one step did, inferred from its `GnStep` and the optimiser's settings.
#[derive(Debug, PartialEq)]
struct Counts {
    /// Curvature products (J v then Jᵀ(J v)): CG's iterations, plus one for the initial
    /// residual when CG starts warm, plus one when it stops on negative curvature (that
    /// product is made but no iteration counted, `cg`).
    products: usize,
    /// Whether the diagonal was re-estimated (`probes` VJPs).
    refresh: bool,
    /// Whether the adjoint check ran (one JVP and one VJP).
    adjoint: bool,
    /// Φ forwards of the backtracking.
    backtrack: usize,
}

/// The counts of `step`, made with `probes` Hutchinson probes per refresh (0 without the
/// preconditioner) and `warm` true when the step's CG started from a warm start (the
/// optimiser's state had one). The step's passes are 1 + products + probes·refresh +
/// adjoint backward and 1 + 3·products + probes·refresh + 3·adjoint + backtrack forward
/// (`GaussNewton::step`).
///
/// # Panics
/// If the backward passes leave anything but 0 or `probes` for the refresh, or the forward
/// passes are too few.
fn counts(step: &GnStep, probes: usize, warm: bool) -> Counts {
    let adjoint = step.adjoint_error.is_some();
    let a = usize::from(adjoint);
    let products = step.cg_iterations
        + usize::from(warm)
        + usize::from(step.cg_stop == CgStop::NegativeCurvature);
    let rest = step
        .backward_passes
        .checked_sub(1 + products + a)
        .expect("too few backward passes for the gradient, CG and the adjoint check");
    assert!(
        rest == 0 || rest == probes,
        "backward passes left unexplained: {rest} (probes {probes})"
    );
    let refresh = probes > 0 && rest == probes;
    let backtrack = step
        .forward_passes
        .checked_sub(1 + 3 * products + rest + 3 * a)
        .expect("too few forward passes");
    Counts {
        products,
        refresh,
        adjoint,
        backtrack,
    }
}

/// Unit costs, in seconds.
struct Units {
    forward: f64,
    gradient: f64,
    jvp: f64,
    hutchinson: f64,
    adjoint: f64,
}

/// CG's products, in place, in seconds per product, and CG's own work per product.
struct CgCosts {
    products: usize,
    jvp: f64,
    vjp: f64,
    own: f64,
}

/// One real step: its counts, its predicted phases ([`phases`]) and its measured time.
struct Measured {
    step: GnStep,
    counts: Counts,
    phases: [f64; 7],
    seconds: f64,
}

/// The time of one step's phases: gradient, refresh, CG's JVPs, CG's VJPs, CG's own
/// work, backtracking, adjoint check.
fn phases(c: &Counts, u: &Units, cg: &CgCosts) -> [f64; 7] {
    let n = c.products as f64;
    [
        u.gradient,
        if c.refresh { u.hutchinson } else { 0.0 },
        n * cg.jvp,
        n * cg.vjp,
        n * cg.own,
        c.backtrack as f64 * u.forward,
        if c.adjoint { u.adjoint } else { 0.0 },
    ]
}

const PHASES: [&str; 7] = [
    "gradient (1 forward, 1 backward)",
    "Hutchinson refresh",
    "CG's JVPs (2 forwards each)",
    "CG's VJPs (1 forward, 1 backward each)",
    "CG's own work (vectors, host reads)",
    "backtracking (Φ forwards)",
    "adjoint check (JVP and VJP)",
];

/// `f`'s result and its wall time, the device synced before and after.
fn timed<T>(device: &Device, f: impl FnOnce() -> T) -> (T, f64) {
    device.sync().expect("device sync");
    let start = Instant::now();
    let out = f();
    device.sync().expect("device sync");
    (out, start.elapsed().as_secs_f64())
}

/// The median time of `f` over [`REPS`] calls after [`WARMUP`].
fn median(device: &Device, mut f: impl FnMut()) -> f64 {
    for _ in 0..WARMUP {
        timed(device, &mut f);
    }
    let mut t: Vec<f64> = (0..REPS).map(|_| timed(device, &mut f).1).collect();
    t.sort_by(f64::total_cmp);
    t[REPS / 2]
}

/// ‖t‖² on the host: the read that makes the device finish the work behind `t`.
fn squared_norm(t: Tensor<1>) -> f64 {
    t.powf_scalar(2.0).sum().into_scalar::<f64>()
}

/// (diag + λ + μ)^exponent, the step's diagonal preconditioner (a copy of
/// `GaussNewton::preconditioner`'s last line, private in `step.rs`; keep them in step).
fn preconditioner(d: &ParamVec, shift: f64, exponent: f64) -> ParamVec {
    d.from_flat_like(d.to_flat().add_scalar(shift).powf_scalar(exponent))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [out, dir, caps @ ..] = args.as_slice() else {
        panic!("usage: gn_profile <out.md> <run_dir> [<cg_cap> ...]");
    };
    let caps: Vec<usize> = if caps.is_empty() {
        DEFAULT_CAPS.to_vec()
    } else {
        caps.iter()
            .map(|c| c.parse().unwrap_or_else(|e| panic!("cap {c:?}: {e}")))
            .collect()
    };
    let device = Device::default().autodiff();
    let (out, dir) = (PathBuf::from(out), PathBuf::from(dir));
    create_output(&out).unwrap_or_else(|e| panic!("create {}: {e}", out.display()));

    let run = RunDir::open(&dir).unwrap_or_else(|e| panic!("open {}: {e}", dir.display()));
    let train_cfg = run
        .training_config()
        .unwrap_or_else(|e| panic!("training config: {e}"));
    let Some(OptimizerConfig::GaussNewton(gn_cfg)) = train_cfg.optimizer.clone() else {
        panic!("{}: not a Gauss–Newton run", dir.display());
    };
    let ck = run
        .checkpoint::<4>(&device)
        .unwrap_or_else(|e| panic!("checkpoint: {e}"))
        .unwrap_or_else(|| panic!("{}: no checkpoint.bin", dir.display()));
    let state = ck.gn.expect("the checkpoint has no Gauss–Newton state");
    let model = ck.model;

    let data_cfg: PoissonConfig = run
        .data_config()
        .unwrap_or_else(|e| panic!("data config: {e}"));
    let set = stage(data_cfg.n).unwrap_or_else(|e| panic!("{e}"));
    let datasets = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/poisson");
    let (train, _, normalizers) = load_poisson_uniform::<f32>(
        datasets.join(set.npz("train")),
        datasets.join(set.npz("test")),
        &data_cfg,
    )
    .unwrap_or_else(|e| panic!("load stage {}: {e}", data_cfg.n));
    let decoder = FlatDecoder::new(&normalizers.y, &device);
    let problem = FnoResidual::<4, 3>::new(Some(&decoder));
    let batches: Vec<Batch<4, 3>> =
        device_loader::<4, 3, f32>(&train, &device, train_cfg.batch_size, None)
            .iter()
            .map(|b| b.expect("dataset error"))
            .collect();
    let batch = &batches[0];

    println!(
        "{}: step {}, λ {:.3e}, batch {} ({} batches), backend {:?}",
        dir.display(),
        state.step,
        state.lambda,
        train_cfg.batch_size,
        batches.len(),
        backend_features()
    );

    let r = problem.residual(&model, batch).inner();
    let m = r.dims()[0];
    let mu = gn_cfg.weight_decay;
    let theta = params(&model);
    let g = vjp(&problem, &model, batch, &r).axpy(mu, &theta);
    let probes = match gn_cfg.preconditioner {
        Preconditioner::HutchinsonDiagonal { probes } => probes,
        Preconditioner::None => 0,
    };
    let units = Units {
        forward: median(&device, || {
            squared_norm(problem.residual(&model.valid(), batch).inner());
        }),
        gradient: median(&device, || {
            vjp(&problem, &model, batch, &r).norm();
        }),
        jvp: median(&device, || {
            squared_norm(jvp(&problem, &model, batch, &g, gn_cfg.jvp_rel_step));
        }),
        hutchinson: if probes > 0 {
            median(&device, || {
                hutchinson_diagonal(&problem, &model, batch, m, probes, 0).norm();
            })
        } else {
            0.0
        },
        adjoint: median(&device, || {
            adjoint_error(&problem, &model, batch, &r, &g, gn_cfg.jvp_rel_step);
        }),
    };
    println!(
        "units (s): forward {:.4}, gradient {:.4}, jvp {:.4}, hutchinson {:.4}, adjoint {:.4}",
        units.forward, units.gradient, units.jvp, units.hutchinson, units.adjoint
    );

    let precond = state
        .diagonal
        .as_ref()
        .map(|(_, d)| preconditioner(d, state.lambda + mu, gn_cfg.precond_exponent));
    let mut report = String::new();
    writeln!(
        report,
        "# Gauss–Newton step profile\n\nRun `{}` at step {} (λ {:.3e}, μ {mu}), batch {} of \
         {m} residual entries; backend {:?}, commit {}. Unit costs: median of {REPS} after \
         {WARMUP} warm-up calls, each synced.\n",
        dir.display(),
        state.step,
        state.lambda,
        train_cfg.batch_size,
        backend_features(),
        git_commit()
    )
    .unwrap();
    writeln!(report, "## Unit costs\n\n| unit | s |\n| --- | --- |").unwrap();
    for (name, t) in [
        ("Φ forward (no autodiff)", units.forward),
        ("gradient (vjp, u = r)", units.gradient),
        ("JVP (finite differences)", units.jvp),
        ("Hutchinson refresh", units.hutchinson),
        ("adjoint check", units.adjoint),
    ] {
        writeln!(report, "| {name} | {t:.4} |").unwrap();
    }

    for &cap in &caps {
        let cfg = GaussNewtonConfig {
            cg_max_iters: cap,
            ..gn_cfg.clone()
        };
        let cg_costs = cg_in_place(&problem, &model, batch, &g, &state, precond.as_ref(), &cfg);
        let warm_start = |gn: &GaussNewton| gn.state().warm_start.is_some();
        let mut gn = GaussNewton::from_state(cfg, state.clone());
        let mut current = model.clone();
        let mut steps = Vec::with_capacity(STEPS);
        for b in batches.iter().cycle().take(STEPS) {
            let warm = warm_start(&gn);
            let ((next, step), seconds) = timed(&device, || gn.step(current, &problem, b));
            current = next;
            let counts = counts(&step, probes, warm);
            let phases = phases(&counts, &units, &cg_costs);
            println!(
                "cap {cap}, step {}: {:.2} s, {} products, {} backtracking",
                step.step, seconds, counts.products, counts.backtrack
            );
            steps.push(Measured {
                step,
                counts,
                phases,
                seconds,
            });
        }
        write_cap(&mut report, cap, &units, &cg_costs, &steps);
    }
    std::fs::write(&out, report).unwrap_or_else(|e| panic!("write {}: {e}", out.display()));
    println!("wrote {}", out.display());
}

/// Creates `out` (empty) and its parent directories, so that a path that cannot be written
/// fails at startup rather than after the measurement (the VM run of 2026-11-25).
fn create_output(out: &Path) -> std::io::Result<()> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(out, "")
}

/// CG on the step's system at the checkpoint's state, its products timed one by one. The
/// curvature closure, the warm start and the `CgConfig` copy `GaussNewton::step`'s
/// (`step.rs`), which keeps them private; keep them in step.
fn cg_in_place<M: Module, P: LeastSquares<M>>(
    problem: &P,
    model: &M,
    batch: &P::Batch,
    g: &ParamVec,
    state: &GaussNewtonState,
    precond: Option<&ParamVec>,
    cfg: &GaussNewtonConfig,
) -> CgCosts {
    let device = g.device();
    let (mut n, mut t_jvp, mut t_vjp) = (0, 0.0, 0.0);
    let damping = state.lambda + cfg.weight_decay;
    let apply = |v: &ParamVec| {
        let (jv, a) = timed(&device, || jvp(problem, model, batch, v, cfg.jvp_rel_step));
        let (jtjv, b) = timed(&device, || vjp(problem, model, batch, &jv));
        n += 1;
        t_jvp += a;
        t_vjp += b;
        jtjv.axpy(damping, v)
    };
    let warm = state
        .warm_start
        .as_ref()
        .map(|w| w.scale(cfg.warm_start_decay));
    let cg_cfg = CgConfig {
        max_iters: cfg.cg_max_iters,
        rel_tol: cfg.cg_rel_tol,
        progress_tol: cfg.cg_progress_tol,
        ..CgConfig::default()
    };
    let (_, total) = timed(&device, || {
        cg(apply, &g.scale(-1.0), warm.as_ref(), precond, &cg_cfg)
    });
    assert!(n > 0, "CG made no products");
    let k = n as f64;
    CgCosts {
        products: n,
        jvp: t_jvp / k,
        vjp: t_vjp / k,
        own: (total - t_jvp - t_vjp) / k,
    }
}

/// Mean of `f` over `steps`.
fn mean(steps: &[Measured], f: impl Fn(&Measured) -> f64) -> f64 {
    steps.iter().map(f).sum::<f64>() / steps.len() as f64
}

/// One cap's tables: CG's in-place costs, the real steps, and the profile.
fn write_cap(report: &mut String, cap: usize, u: &Units, cg: &CgCosts, steps: &[Measured]) {
    writeln!(
        report,
        "\n## CG cap {cap}\n\nCG in place: {} products, {:.4} s per JVP, {:.4} s per VJP, \
         {:.4} s of CG's own work per product.\n\n\
         | step | products | refresh | adjoint | backtracking | forward | backward | measured (s) | predicted (s) | gap |\n\
         | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        cg.products, cg.jvp, cg.vjp, cg.own
    )
    .unwrap();
    for s in steps {
        let predicted: f64 = s.phases.iter().sum();
        writeln!(
            report,
            "| {} | {} | {} | {} | {} | {} | {} | {:.3} | {:.3} | {:+.1} % |",
            s.step.step,
            s.counts.products,
            s.counts.refresh,
            s.counts.adjoint,
            s.counts.backtrack,
            s.step.forward_passes,
            s.step.backward_passes,
            s.seconds,
            predicted,
            100.0 * (s.seconds - predicted) / s.seconds
        )
        .unwrap();
    }
    let measured = mean(steps, |s| s.seconds);
    let per_phase: Vec<f64> = (0..PHASES.len())
        .map(|i| mean(steps, |s| s.phases[i]))
        .collect();
    let passes: f64 = steps
        .iter()
        .map(|s| (s.step.forward_passes + s.step.backward_passes) as f64)
        .sum();
    writeln!(
        report,
        "\n| phase (mean per step) | s | share |\n| --- | --- | --- |"
    )
    .unwrap();
    for (name, t) in PHASES.iter().zip(&per_phase) {
        writeln!(
            report,
            "| {name} | {t:.3} | {:.1} % |",
            100.0 * t / measured
        )
        .unwrap();
    }
    let rest = measured - per_phase.iter().sum::<f64>();
    writeln!(
        report,
        "| unexplained | {rest:.3} | {:.1} % |\n| **step, measured** | **{measured:.3}** | 100 % |\n\
         | metric forward (`train_epoch_gn`, 1 per step, outside the step) | {:.3} | – |\n\n\
         Passes per second of the steps: {:.2} ({} steps, {:.0} passes, {:.1} s).",
        100.0 * rest / measured,
        u.forward,
        passes / (measured * steps.len() as f64),
        steps.len(),
        passes,
        measured * steps.len() as f64
    )
    .unwrap();
    if rest.abs() > GAP_FLAG * measured {
        writeln!(
            report,
            "\n**Flag:** the unit costs explain the mean step only to {:.1} % (above {:.0} %); \
             a question for the author (an in-library timer).",
            100.0 * rest / measured,
            100.0 * GAP_FLAG
        )
        .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The report's missing parent directories are created; a parent that is a file fails.
    #[test]
    fn create_output_makes_missing_parents() {
        let root = std::env::temp_dir().join(format!("gn_profile_{}", std::process::id()));
        let out = root.join("results/a/CO0.2-vm.md");
        create_output(&out).expect("create");
        assert!(out.is_file());
        assert!(create_output(&out.join("below_a_file.md")).is_err());
        std::fs::remove_dir_all(&root).expect("clean up");
    }

    /// A `GnStep` with the given CG iterations, passes and adjoint check; the rest zero.
    fn step(cg_iterations: usize, forward: usize, backward: usize, adjoint: bool) -> GnStep {
        GnStep {
            step: 0,
            phi_before: 0.0,
            phi_after: 0.0,
            grad_norm: 0.0,
            step_norm: 0.0,
            predicted_reduction: 0.0,
            actual_reduction: 0.0,
            rho: 0.0,
            lambda: 1.0,
            lambda_next: 1.0,
            accepted: true,
            accepted_iteration: None,
            cg_iterations,
            cg_rel_residual: 0.0,
            cg_stop: CgStop::MaxIters,
            ritz: None,
            adjoint_error: adjoint.then_some(0.0),
            forward_passes: forward,
            backward_passes: backward,
        }
    }

    /// Rows of Phase 10's GN200-D run at μ = 0, seed 42 (`gn_steps.csv`, 10 probes).
    #[test]
    fn counts_of_phase10_rows() {
        // step 0: cold start, refresh, adjoint check, one backtracking forward
        assert_eq!(
            counts(&step(8, 39, 20, true), 10, false),
            Counts {
                products: 8,
                refresh: true,
                adjoint: true,
                backtrack: 1
            }
        );
        // step 70: warm start, refresh, adjoint check
        assert_eq!(
            counts(&step(50, 168, 63, true), 10, true),
            Counts {
                products: 51,
                refresh: true,
                adjoint: true,
                backtrack: 1
            }
        );
        // step 71: warm start only
        assert_eq!(
            counts(&step(50, 155, 52, false), 10, true),
            Counts {
                products: 51,
                refresh: false,
                adjoint: false,
                backtrack: 1
            }
        );
    }

    /// Without the preconditioner, a step with three backtracking forwards.
    #[test]
    fn counts_without_preconditioner() {
        // 1 + 3·5 + 3 forward, 1 + 5 backward
        assert_eq!(
            counts(&step(4, 19, 6, false), 0, true),
            Counts {
                products: 5,
                refresh: false,
                adjoint: false,
                backtrack: 3
            }
        );
    }

    #[test]
    #[should_panic(expected = "unexplained")]
    fn counts_reject_unexplained_backwards() {
        counts(&step(4, 30, 9, false), 0, true);
    }

    /// An adjoint check without a refresh, and a negative-curvature stop's uncounted
    /// product: 1 + 3·(3 + 1 + 1) + 3 + 2 forward, 1 + 5 + 1 backward.
    #[test]
    fn counts_adjoint_without_refresh_and_negative_curvature() {
        let mut s = step(3, 21, 7, true);
        s.cg_stop = CgStop::NegativeCurvature;
        assert_eq!(
            counts(&s, 10, true),
            Counts {
                products: 5,
                refresh: false,
                adjoint: true,
                backtrack: 2
            }
        );
    }

    /// The phases sum to the predicted time by hand.
    #[test]
    fn phases_by_hand() {
        let u = Units {
            forward: 1.0,
            gradient: 2.0,
            jvp: 0.0,
            hutchinson: 20.0,
            adjoint: 3.0,
        };
        let cg = CgCosts {
            products: 1,
            jvp: 2.0,
            vjp: 2.0,
            own: 0.5,
        };
        let c = Counts {
            products: 10,
            refresh: true,
            adjoint: false,
            backtrack: 2,
        };
        assert_eq!(phases(&c, &u, &cg), [2.0, 20.0, 20.0, 20.0, 5.0, 2.0, 0.0]);
    }
}

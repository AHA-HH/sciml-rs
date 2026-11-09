//! [`GaussNewton`]: one damped (Levenberg–Marquardt) Gauss–Newton step on any
//! [`LeastSquares`] problem, with the diagnostics of [`GnStep`] (Phase 10 T2; design
//! §13.17; Martens 2010 for the CG stopping, warm start and backtracking).

use std::cell::Cell;

use burn::{config::Config, module::Module, prelude::*, tensor::TensorData};
use rand_chacha::{
    ChaCha8Rng,
    rand_core::{Rng, SeedableRng},
};

use super::{
    cg::{CgConfig, CgIterate, CgStop, cg},
    param_vec::{ParamVec, params, replace, with_params},
    products::{LeastSquares, adjoint_error, jvp, residual_and_gradient, vjp},
    state::GaussNewtonState,
};

/// The preconditioner of the CG solve.
#[derive(Config, Debug, PartialEq)]
pub enum Preconditioner {
    /// None: M = I.
    None,
    /// diag(G) estimated by [`hutchinson_diagonal`] with `probes` Rademacher probes (one
    /// VJP each), applied as M = (diag(G) + λ + μ)^`GaussNewtonConfig::precond_exponent`.
    HutchinsonDiagonal {
        /// Probes per estimate.
        probes: usize,
    },
}

/// Settings of [`GaussNewton`], saved with a run.
#[derive(Config, Debug)]
pub struct GaussNewtonConfig {
    /// The most CG iterations per step.
    #[config(default = 50)]
    pub cg_max_iters: usize,
    /// CG's relative-residual tolerance.
    #[config(default = 1e-4)]
    pub cg_rel_tol: f64,
    /// Martens' relative-progress tolerance of CG (`CgConfig::progress_tol`); 0 disables.
    #[config(default = 5e-4)]
    pub cg_progress_tol: f64,
    /// The damping λ of the first step.
    #[config(default = 1.0)]
    pub lambda_init: f64,
    /// The smallest λ.
    #[config(default = 1e-8)]
    pub lambda_min: f64,
    /// The largest λ.
    #[config(default = 1e8)]
    pub lambda_max: f64,
    /// λ is multiplied by this when ρ < `rho_low`, or when the step is rejected.
    #[config(default = 1.5)]
    pub lambda_raise: f64,
    /// λ is multiplied by this when ρ > `rho_high`.
    #[config(default = 0.6666666666666666)]
    pub lambda_lower: f64,
    /// Below this ρ, λ is raised.
    #[config(default = 0.25)]
    pub rho_low: f64,
    /// Above this ρ, λ is lowered.
    #[config(default = 0.75)]
    pub rho_high: f64,
    /// Weight decay μ as a Tikhonov term: Φ gains ½μ‖θ‖² and the solve μI (G5).
    #[config(default = 1e-4)]
    pub weight_decay: f64,
    /// The next step's CG starts from this times the previous CG solution.
    #[config(default = 0.95)]
    pub warm_start_decay: f64,
    /// The relative step h of the finite-difference J v ([`jvp`]); `None` takes
    /// [`default_rel_step`](super::default_rel_step) for the parameters' dtype.
    pub jvp_rel_step: Option<f64>,
    /// The CG preconditioner.
    #[config(default = "Preconditioner::None")]
    pub preconditioner: Preconditioner,
    /// The exponent of the diagonal preconditioner (Martens' 0.75).
    #[config(default = 0.75)]
    pub precond_exponent: f64,
    /// The diagonal preconditioner is re-estimated every this many steps (≥ 1).
    #[config(default = 10)]
    pub precond_every: usize,
    /// [`adjoint_error`] is computed every this many steps; 0 never.
    #[config(default = 0)]
    pub adjoint_check_every: usize,
    /// Seed of the Hutchinson probes; probe draws depend on (seed, step) only.
    #[config(default = 0)]
    pub seed: u64,
}

/// What one [`GaussNewton::step`] did. Φ(θ) = ½‖r(θ)‖² + ½μ‖θ‖², always on the step's
/// batch.
#[derive(Clone, Debug, PartialEq)]
pub struct GnStep {
    /// The step's number, from 0.
    pub step: u64,
    /// Φ at the incoming parameters.
    pub phi_before: f64,
    /// Φ at the outgoing parameters (`phi_before` if the step was rejected).
    pub phi_after: f64,
    /// ‖g‖, g = Jᵀr + μθ.
    pub grad_norm: f64,
    /// ‖p‖ of the judged candidate (the accepted one, else CG's final iterate; 0 if CG
    /// produced none).
    pub step_norm: f64,
    /// m(p) − m(0) of the judged candidate, m the quadratic model of Φ without the damping.
    pub predicted_reduction: f64,
    /// Φ(θ + p) − Φ(θ) of the judged candidate.
    pub actual_reduction: f64,
    /// actual / predicted reduction of the judged candidate (NaN if CG produced none).
    pub rho: f64,
    /// The damping λ this step used.
    pub lambda: f64,
    /// The damping the next step will use.
    pub lambda_next: f64,
    /// Whether the parameters moved.
    pub accepted: bool,
    /// The CG iteration of the accepted iterate.
    pub accepted_iteration: Option<usize>,
    /// CG iterations taken.
    pub cg_iterations: usize,
    /// CG's final relative (recurrence) residual.
    pub cg_rel_residual: f64,
    /// Why CG stopped.
    pub cg_stop: CgStop,
    /// The extreme Ritz values (θ_min, θ_max) of the (preconditioned) damped curvature.
    pub ritz: Option<(f64, f64)>,
    /// [`adjoint_error`] with u = r and v = g, on the steps it is checked.
    pub adjoint_error: Option<f64>,
    /// Forward passes this step, counting each J v as two (finite differences; a problem
    /// with an exact J v is over-counted).
    pub forward_passes: usize,
    /// Backward passes this step.
    pub backward_passes: usize,
}

/// A damped Gauss–Newton optimiser over the parameters of any [`Module`].
///
/// Each [`step`](Self::step) solves (G + (λ + μ)I) p = −g by [`cg`], with G v = Jᵀ(J v)
/// and g = Jᵀr + μθ, judges the step by ρ on the same batch, backtracks over CG's iterates,
/// and adapts λ from ρ (Levenberg–Marquardt). λ is the only step control: there is no
/// learning rate.
#[derive(Clone, Debug)]
pub struct GaussNewton {
    config: GaussNewtonConfig,
    state: GaussNewtonState,
}

/// Counts forward and backward passes inside a step.
#[derive(Default)]
struct Passes {
    forward: Cell<usize>,
    backward: Cell<usize>,
}

impl Passes {
    fn add(&self, forward: usize, backward: usize) {
        self.forward.set(self.forward.get() + forward);
        self.backward.set(self.backward.get() + backward);
    }
}

/// Which of CG's kept iterates a step takes ([`select`]).
#[derive(Debug)]
enum Selection<'a> {
    /// This iterate lowers Φ to `phi`.
    Accepted { it: &'a CgIterate, phi: f64 },
    /// No iterate lowers Φ. `last` is CG's final iterate and Φ there, judged for ρ; `None`
    /// when CG produced no candidate.
    Rejected { last: Option<(&'a CgIterate, f64)> },
}

impl<'a> Selection<'a> {
    /// The candidate ρ judges: the accepted iterate, else CG's final one.
    fn judged(&self) -> Option<(&'a CgIterate, f64)> {
        match *self {
            Selection::Accepted { it, phi } => Some((it, phi)),
            Selection::Rejected { last } => last,
        }
    }
}

/// The step's candidate among `iterates` (in CG's order): the last if it lowers Φ below
/// `phi_before`, else the latest earlier one that does, else none. `phi_at` evaluates Φ
/// at θ + p; it is called only as far back as needed, the last iterate first.
fn select<'a>(
    iterates: &'a [CgIterate],
    phi_before: f64,
    mut phi_at: impl FnMut(&ParamVec) -> f64,
) -> Selection<'a> {
    let mut last = None;
    for it in iterates.iter().rev() {
        let phi = phi_at(&it.x);
        if phi < phi_before {
            return Selection::Accepted { it, phi };
        }
        last.get_or_insert((it, phi));
    }
    Selection::Rejected { last }
}

/// The damping after a step that used `lambda`: raised (× `lambda_raise`) if the step was
/// rejected or ρ is NaN or below `rho_low`, lowered (× `lambda_lower`) if ρ is above
/// `rho_high`, else kept; then clamped to [`lambda_min`, `lambda_max`].
fn next_lambda(cfg: &GaussNewtonConfig, lambda: f64, accepted: bool, rho: f64) -> f64 {
    let factor = if !accepted || rho.is_nan() || rho < cfg.rho_low {
        cfg.lambda_raise
    } else if rho > cfg.rho_high {
        cfg.lambda_lower
    } else {
        1.0
    };
    (lambda * factor).clamp(cfg.lambda_min, cfg.lambda_max)
}

impl GaussNewton {
    /// A fresh optimiser: λ = `lambda_init`, step 0, no warm start.
    ///
    /// # Panics
    /// If λ's bounds or factors are inconsistent, or `precond_every` is 0.
    pub fn new(config: GaussNewtonConfig) -> Self {
        assert!(
            0.0 < config.lambda_min
                && config.lambda_min <= config.lambda_init
                && config.lambda_init <= config.lambda_max,
            "GaussNewton: need 0 < lambda_min <= lambda_init <= lambda_max"
        );
        assert!(
            config.lambda_raise > 1.0 && 0.0 < config.lambda_lower && config.lambda_lower < 1.0,
            "GaussNewton: need lambda_raise > 1 and 0 < lambda_lower < 1"
        );
        assert!(
            config.precond_every >= 1,
            "GaussNewton: precond_every must be >= 1"
        );
        let state = GaussNewtonState {
            lambda: config.lambda_init,
            step: 0,
            warm_start: None,
            diagonal: None,
        };
        Self { config, state }
    }

    /// An optimiser resumed from `state`, as [`state`](Self::state) gave it: the next step
    /// equals the one the saved optimiser would have taken.
    ///
    /// # Panics
    /// As [`new`](Self::new); if the state's λ is not finite or lies outside the config's
    /// bounds, or its diagonal is dated after its step count. A state from another model
    /// panics at the next [`step`](Self::step).
    pub fn from_state(config: GaussNewtonConfig, state: GaussNewtonState) -> Self {
        assert!(
            state.lambda.is_finite()
                && (config.lambda_min..=config.lambda_max).contains(&state.lambda),
            "GaussNewton::from_state: lambda {} is not finite or outside the config's bounds",
            state.lambda
        );
        assert!(
            state
                .diagonal
                .as_ref()
                .is_none_or(|(at, _)| *at <= state.step),
            "GaussNewton::from_state: the diagonal is dated after the step count"
        );
        Self {
            state,
            ..Self::new(config)
        }
    }

    /// The settings.
    pub fn config(&self) -> &GaussNewtonConfig {
        &self.config
    }

    /// The state to checkpoint (λ, the step count, the warm start, the diagonal estimate);
    /// [`from_state`](Self::from_state) resumes from it.
    pub fn state(&self) -> &GaussNewtonState {
        &self.state
    }

    /// The damping the next step will use.
    pub fn lambda(&self) -> f64 {
        self.state.lambda
    }

    /// Steps taken (accepted or not).
    pub fn steps(&self) -> u64 {
        self.state.step
    }

    /// One Gauss–Newton step on `batch`. Returns the model (with new parameters if the step
    /// was accepted, else `model` itself, bit for bit) and the step's diagnostics.
    ///
    /// The candidates are CG's kept iterates. The last is tried first; if it does not lower
    /// Φ, earlier iterates are tried from the last backwards and the first that lowers Φ
    /// is taken. If none does, the step is rejected.
    ///
    /// ρ is that of the taken candidate (or of CG's final iterate, when rejected). λ is
    /// lowered if ρ > `rho_high`, raised if ρ < `rho_low` or the step is rejected, and
    /// clamped to its bounds.
    ///
    /// Two departures from Martens (2010), who keeps stepping back while an earlier
    /// iterate is better still and takes ρ from CG's final iterate: here the last iterate
    /// is kept whenever it lowers Φ (one forward when it does), and ρ judges the step
    /// actually taken. Martens' rules are recorded as an optional item for later
    /// (`docs/optional/README.md`).
    ///
    /// `model` must be on an autodiff device with parameters that require gradients.
    ///
    /// # Panics
    /// If the residual is not tracked ([`vjp`]), or if the model's parameters no longer
    /// match the warm start's or the diagonal's.
    pub fn step<M: Module, P: LeastSquares<M>>(
        &mut self,
        model: M,
        problem: &P,
        batch: &P::Batch,
    ) -> (M, GnStep) {
        let passes = Passes::default();
        let (mu, lambda) = (self.config.weight_decay, self.state.lambda);
        let theta = params(&model);

        let (r, jtr) = residual_and_gradient(problem, &model, batch);
        passes.add(1, 1);
        let phi_before = 0.5 * squared_norm(&r) + 0.5 * mu * theta.dot(&theta);
        let g = jtr.axpy(mu, &theta);

        let precond = self.preconditioner(problem, &model, batch, r.dims()[0], &passes);
        let cfg = &self.config;
        let damping = lambda + mu;
        let curvature = |v: &ParamVec| {
            let jv = jvp(problem, &model, batch, v, cfg.jvp_rel_step);
            passes.add(3, 1);
            vjp(problem, &model, batch, &jv).axpy(damping, v)
        };
        let warm = self
            .state
            .warm_start
            .as_ref()
            .map(|w| w.scale(cfg.warm_start_decay));
        let cg_cfg = CgConfig {
            max_iters: cfg.cg_max_iters,
            rel_tol: cfg.cg_rel_tol,
            progress_tol: cfg.cg_progress_tol,
            ..CgConfig::default()
        };
        let solve = cg(
            curvature,
            &g.scale(-1.0),
            warm.as_ref(),
            precond.as_ref(),
            &cg_cfg,
        );

        // Φ at θ + p, from one forward with parameters that carry no autodiff.
        let choice = select(&solve.iterates, phi_before, |p| {
            let shifted = theta.axpy(1.0, p);
            let r = problem.residual(&replace(model.clone(), &shifted, false), batch);
            passes.add(1, 0);
            0.5 * squared_norm(&r.inner()) + 0.5 * mu * shifted.dot(&shifted)
        });
        // m(p) − m(0) = q(p) − ½λ‖p‖²: CG's q includes the damping, the model of Φ does not.
        let (step_norm, predicted_reduction, actual_reduction, rho) = match choice.judged() {
            Some((it, phi)) => {
                let predicted = it.quadratic - 0.5 * lambda * it.x.dot(&it.x);
                let actual = phi - phi_before;
                (it.x.norm(), predicted, actual, actual / predicted)
            }
            None => (0.0, 0.0, 0.0, f64::NAN),
        };
        let accepted = matches!(choice, Selection::Accepted { .. });
        let lambda_next = next_lambda(cfg, lambda, accepted, rho);

        let adjoint = (cfg.adjoint_check_every > 0
            && self
                .state
                .step
                .is_multiple_of(cfg.adjoint_check_every as u64))
        .then(|| {
            passes.add(3, 1);
            adjoint_error(problem, &model, batch, &r, &g, cfg.jvp_rel_step)
        });

        let (model, phi_after, accepted_iteration) = match choice {
            Selection::Accepted { it, phi } => (
                with_params(model, &theta.axpy(1.0, &it.x)),
                phi,
                Some(it.iteration),
            ),
            Selection::Rejected { .. } => (model, phi_before, None),
        };

        let report = GnStep {
            step: self.state.step,
            phi_before,
            phi_after,
            grad_norm: g.norm(),
            step_norm,
            predicted_reduction,
            actual_reduction,
            rho,
            lambda,
            lambda_next,
            accepted,
            accepted_iteration,
            cg_iterations: solve.iterations,
            cg_rel_residual: solve.rel_residual,
            cg_stop: solve.stop,
            ritz: solve.ritz,
            adjoint_error: adjoint,
            forward_passes: passes.forward.get(),
            backward_passes: passes.backward.get(),
        };
        if solve.stop != CgStop::ZeroRhs {
            self.state.warm_start = Some(solve.x);
        }
        self.state.lambda = lambda_next;
        self.state.step += 1;
        (model, report)
    }

    /// The diagonal preconditioner for this step, or `None` for
    /// [`Preconditioner::None`]: (diag(G) + λ + μ)^`precond_exponent`, re-estimating
    /// diag(G) by [`hutchinson_diagonal`] (residual length `m`, one VJP per probe, counted
    /// in `passes`) when there is none yet or it is `precond_every` steps old.
    fn preconditioner<M: Module, P: LeastSquares<M>>(
        &mut self,
        problem: &P,
        model: &M,
        batch: &P::Batch,
        m: usize,
        passes: &Passes,
    ) -> Option<ParamVec> {
        let cfg = &self.config;
        let Preconditioner::HutchinsonDiagonal { probes } = cfg.preconditioner else {
            return None;
        };
        let step = self.state.step;
        let stale = self
            .state
            .diagonal
            .as_ref()
            .is_none_or(|(at, _)| step.saturating_sub(*at) >= cfg.precond_every as u64);
        if stale {
            let seed = cfg.seed ^ step.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let d = hutchinson_diagonal(problem, model, batch, m, probes, seed);
            passes.add(probes, probes);
            self.state.diagonal = Some((step, d));
        }
        let (_, d) = self.state.diagonal.as_ref().expect("estimated above");
        let (shift, exponent) = (self.state.lambda + cfg.weight_decay, cfg.precond_exponent);
        Some(d.map(|t| t.add_scalar(shift).powf_scalar(exponent)))
    }
}

/// ‖t‖² of a `[m]` tensor, on the host in f64.
fn squared_norm(t: &Tensor<1>) -> f64 {
    t.clone().powf_scalar(2.0).sum().into_scalar::<f64>()
}

/// Hutchinson's estimate of diag(G) = diag(JᵀJ): the mean of (Jᵀz)² over `probes`
/// Rademacher vectors z ∈ {−1, 1}^m, `m` the residual's length, drawn from
/// `ChaCha8Rng::seed_from_u64(seed)`. Unbiased, since E[(Jᵀz)ₖ²] = Σᵢ Jᵢₖ²; each entry's
/// variance is 2[(Σᵢ Jᵢₖ²)² − Σᵢ Jᵢₖ⁴] / probes. One [`vjp`] per probe.
///
/// `model` must be on an autodiff device, as for [`vjp`].
///
/// # Panics
/// If `probes` is 0, or as [`vjp`].
pub fn hutchinson_diagonal<M: Module, P: LeastSquares<M>>(
    problem: &P,
    model: &M,
    batch: &P::Batch,
    m: usize,
    probes: usize,
    seed: u64,
) -> ParamVec {
    assert!(probes > 0, "hutchinson_diagonal: probes must be positive");
    let template = params(model);
    let device = template.device();
    let dtype = template.dtype().expect("the model has float parameters");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut sum = template.zeros_like();
    for _ in 0..probes {
        let z: Vec<f32> = (0..m)
            .map(|_| if rng.next_u32() & 1 == 0 { -1.0 } else { 1.0 })
            .collect();
        let data = TensorData::new(z, [m]).convert_dtype(dtype);
        let z = Tensor::<1>::from_data(data, &device);
        let jtz = vjp(problem, model, batch, &z);
        sum = sum.axpy(1.0, &jtz.mul(&jtz));
    }
    sum.scale(1.0 / probes as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::optim::gauss_newton::products::residual_and_gradient;
    use burn::nn::{Linear, LinearConfig};

    fn wave(n: usize, phase: f64) -> Vec<f64> {
        (0..n).map(|i| (i as f64 * 0.37 + phase).sin()).collect()
    }

    fn tensor(values: &[f64], device: &Device) -> Tensor<1> {
        let v: Vec<f32> = values.iter().map(|x| *x as f32).collect();
        Tensor::from_floats(v.as_slice(), device)
    }

    fn host(t: Tensor<1>) -> Vec<f64> {
        t.into_data().convert::<f64>().try_to_vec::<f64>().unwrap()
    }

    /// `model` with deterministic weights `scale · sin(0.37 i + phase)`.
    fn deterministic<M: Module>(model: M, scale: f64, phase: f64, device: &Device) -> M {
        let p = params(&model);
        let values: Vec<f64> = wave(p.numel(), phase).iter().map(|x| scale * x).collect();
        with_params(model, &p.from_flat_like(tensor(&values, device)))
    }

    /// Inputs `[n, d_in]`, targets `[n, d_out]`.
    struct Data {
        x: Tensor<2>,
        y: Tensor<2>,
    }

    fn flatten(t: Tensor<2>) -> Tensor<1> {
        let [n, c] = t.dims();
        t.reshape([n * c])
    }

    /// r = W x + b − y: linear in θ.
    struct LinearFit;

    impl LeastSquares<Linear> for LinearFit {
        type Batch = Data;
        fn residual(&self, model: &Linear, batch: &Data) -> Tensor<1> {
            flatten(model.forward(batch.x.clone()) - batch.y.clone())
        }
    }

    /// r = tanh(W x + b) − y.
    struct TanhFit;

    impl LeastSquares<Linear> for TanhFit {
        type Batch = Data;
        fn residual(&self, model: &Linear, batch: &Data) -> Tensor<1> {
            flatten(model.forward(batch.x.clone()).tanh() - batch.y.clone())
        }
    }

    /// Inputs sin(0.61 i² + 0.3): unlike a single sinusoid sampled at a fixed step (whose
    /// rows span only sin, cos and the constant), these give a full-rank design matrix.
    fn linear_data(n: usize, device: &Device) -> Data {
        let x: Vec<f64> = (0..3 * n)
            .map(|i| ((i * i) as f64 * 0.61 + 0.3).sin())
            .collect();
        Data {
            x: tensor(&x, device).reshape([n, 3]),
            y: tensor(&wave(2 * n, 2.1), device)
                .reshape([n, 2])
                .mul_scalar(0.5),
        }
    }

    /// J by rows on the host: row i is Jᵀeᵢ.
    fn explicit_jacobian<P: LeastSquares<Linear, Batch = Data>>(
        problem: &P,
        model: &Linear,
        data: &Data,
        m: usize,
    ) -> Vec<Vec<f64>> {
        let device = model.weight.val().device().inner();
        (0..m)
            .map(|i| {
                let mut e = vec![0.0; m];
                e[i] = 1.0;
                host(vjp(problem, model, data, &tensor(&e, &device)).to_flat())
            })
            .collect()
    }

    /// Solves A x = b by Gaussian elimination with partial pivoting (f64, host).
    fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
        let n = b.len();
        for col in 0..n {
            let pivot = (col..n)
                .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
                .unwrap();
            a.swap(col, pivot);
            b.swap(col, pivot);
            for row in col + 1..n {
                let f = a[row][col] / a[col][col];
                let (top, bottom) = a.split_at_mut(row);
                for (x, p) in bottom[0][col..].iter_mut().zip(&top[col][col..]) {
                    *x -= f * p;
                }
                b[row] -= f * b[col];
            }
        }
        let mut x = vec![0.0; n];
        for row in (0..n).rev() {
            let s: f64 = (row + 1..n).map(|k| a[row][k] * x[k]).sum();
            x[row] = (b[row] - s) / a[row][row];
        }
        x
    }

    #[test]
    fn one_step_solves_linear_least_squares() {
        let device = Device::default().autodiff();
        let model = deterministic(LinearConfig::new(3, 2).init(&device), 0.3, 0.2, &device);
        let data = linear_data(6, &device);
        let m = 12;

        // the normal-equations solution θ* = θ₀ − (JᵀJ)⁻¹ Jᵀr₀, on the host in f64
        let j = explicit_jacobian(&LinearFit, &model, &data, m);
        let p = j[0].len();
        let r0 = host(LinearFit.residual(&model, &data).inner());
        let jtj: Vec<Vec<f64>> = (0..p)
            .map(|a| {
                (0..p)
                    .map(|b| (0..m).map(|i| j[i][a] * j[i][b]).sum())
                    .collect()
            })
            .collect();
        let jtr: Vec<f64> = (0..p)
            .map(|a| -(0..m).map(|i| j[i][a] * r0[i]).sum::<f64>())
            .collect();
        let delta = solve(jtj, jtr);
        let theta0 = host(params(&model).to_flat());
        let expected: Vec<f64> = theta0.iter().zip(&delta).map(|(t, d)| t + d).collect();

        let mut gn = GaussNewton::new(
            GaussNewtonConfig::new()
                .with_lambda_init(1e-10)
                .with_lambda_min(1e-12)
                .with_weight_decay(0.0)
                .with_cg_rel_tol(1e-6)
                .with_cg_progress_tol(0.0),
        );
        let (model, report) = gn.step(model, &LinearFit, &data);
        assert!(report.accepted);
        let theta1 = host(params(&model).to_flat());
        let err = theta1
            .iter()
            .zip(&expected)
            .fold(0.0f64, |acc, (a, b)| acc.max((a - b).abs()));
        assert!(err < 1e-4, "max |θ₁ − θ*| = {err}");
        assert!((report.rho - 1.0).abs() < 1e-3, "ρ = {}", report.rho);
        assert_eq!(report.lambda_next, 1e-12f64.max(1e-10 * 2.0 / 3.0));
        assert!(report.forward_passes > 0 && report.backward_passes > 0);
    }

    /// A 1 → 8 → 1 tanh network.
    #[derive(Module, Debug)]
    struct Mlp {
        a: Linear,
        b: Linear,
    }

    impl Mlp {
        fn new(device: &Device) -> Self {
            let model = Self {
                a: LinearConfig::new(1, 8).init(device),
                b: LinearConfig::new(8, 1).init(device),
            };
            deterministic(model, 0.5, 0.4, device)
        }
    }

    struct MlpFit;

    impl LeastSquares<Mlp> for MlpFit {
        type Batch = Data;
        fn residual(&self, model: &Mlp, batch: &Data) -> Tensor<1> {
            flatten(model.b.forward(model.a.forward(batch.x.clone()).tanh()) - batch.y.clone())
        }
    }

    /// y = 0.8 sin(1.5 x) on 16 points in [−2, 2].
    fn curve(device: &Device) -> Data {
        let x: Vec<f64> = (0..16).map(|i| -2.0 + 4.0 * i as f64 / 15.0).collect();
        let y: Vec<f64> = x.iter().map(|v| 0.8 * (1.5 * v).sin()).collect();
        Data {
            x: tensor(&x, device).reshape([16, 1]),
            y: tensor(&y, device).reshape([16, 1]),
        }
    }

    fn phi(model: &Mlp, data: &Data) -> f64 {
        0.5 * squared_norm(&MlpFit.residual(model, data).inner())
    }

    /// Steps of gradient descent θ ← θ − η Jᵀr until Φ ≤ `tol` (`None` if not within
    /// `max_steps`).
    fn gd_steps(eta: f64, tol: f64, max_steps: usize, device: &Device) -> Option<usize> {
        let data = curve(device);
        let mut model = Mlp::new(device);
        for step in 0..max_steps {
            if phi(&model, &data) <= tol {
                return Some(step);
            }
            let (_, g) = residual_and_gradient(&MlpFit, &model, &data);
            let theta = params(&model);
            model = with_params(model, &theta.axpy(-eta, &g));
        }
        None
    }

    /// Measured once with a 30,000-step cap (Phase 10 README, T2 row): GN reaches
    /// Φ ≤ 1e-3 from Φ = 7.5 in 13 steps; gradient descent at its best tested rate
    /// (η = 0.01) in 12,458, and not at all at 0.003 or 0.02 (≥ 0.1 diverges). The test
    /// keeps a 1,000-step cap on gradient descent to stay fast.
    #[test]
    fn gauss_newton_beats_gradient_descent_on_a_nonlinear_fit() {
        let device = Device::default().autodiff();
        let data = curve(&device);
        let tol = 1e-3;
        let mut model = Mlp::new(&device);
        let mut gn = GaussNewton::new(GaussNewtonConfig::new().with_weight_decay(0.0));
        let mut gn_steps = None;
        for step in 0..20 {
            let (next, report) = gn.step(model, &MlpFit, &data);
            model = next;
            assert!(report.accepted, "step {step} rejected");
            assert!(report.phi_after < report.phi_before);
            // ρ → 1 near the optimum, until f32 rounding of Φ dominates (below ~1e-5)
            if (1e-5..=1e-3).contains(&report.phi_before) {
                assert!(
                    (report.rho - 1.0).abs() < 0.2,
                    "step {step}: ρ = {}",
                    report.rho
                );
            }
            if report.phi_after <= tol && gn_steps.is_none() {
                gn_steps = Some(step + 1);
            }
        }
        let gn_steps = gn_steps.expect("GN reaches the tolerance within 20 steps");
        assert!(gn_steps <= 15, "GN took {gn_steps} steps");
        for eta in [0.003, 0.01, 0.02, 0.03, 0.1] {
            assert_eq!(gd_steps(eta, tol, 1000, &device), None, "η = {eta}");
        }
    }

    #[test]
    fn hutchinson_matches_the_explicit_diagonal_within_sampling_error() {
        let device = Device::default().autodiff();
        let model = deterministic(LinearConfig::new(3, 2).init(&device), 0.4, 0.9, &device);
        let data = linear_data(5, &device);
        let m = 10;
        let probes = 2000;
        let j = explicit_jacobian(&TanhFit, &model, &data, m);
        let estimate = host(hutchinson_diagonal(&TanhFit, &model, &data, m, probes, 7).to_flat());
        for (k, est) in estimate.iter().enumerate() {
            let squares: Vec<f64> = (0..m).map(|i| j[i][k] * j[i][k]).collect();
            let exact: f64 = squares.iter().sum();
            let fourth: f64 = squares.iter().map(|s| s * s).sum();
            let sigma = (2.0 * (exact * exact - fourth) / probes as f64).sqrt();
            assert!(
                (est - exact).abs() <= 5.0 * sigma + 1e-6 * exact.max(1.0),
                "entry {k}: {est} vs {exact} (σ {sigma})"
            );
        }
    }

    #[test]
    fn a_step_after_save_and_load_equals_a_step_without() {
        let device = Device::default().autodiff();
        let data = curve(&device);
        let cfg = || {
            GaussNewtonConfig::new()
                .with_preconditioner(Preconditioner::HutchinsonDiagonal { probes: 4 })
                .with_precond_every(2)
                .with_adjoint_check_every(2)
        };
        let mut gn = GaussNewton::new(cfg());
        let mut model = Mlp::new(&device);
        // 3 steps with re-estimation every 2: the diagonal saved (from step 2) is the one
        // step 3 uses, so the saved state's copy of it is exercised
        for _ in 0..3 {
            model = gn.step(model, &MlpFit, &data).0;
        }
        let bytes = gn.state().to_bytes();
        let state = GaussNewtonState::from_bytes(&bytes, &device.clone().inner()).unwrap();
        let mut resumed = GaussNewton::from_state(cfg(), state);
        assert_eq!(
            (resumed.lambda(), resumed.steps()),
            (gn.lambda(), gn.steps())
        );

        let (a, report_a) = gn.step(model.clone(), &MlpFit, &data);
        let (b, report_b) = resumed.step(model, &MlpFit, &data);
        assert_eq!(report_a, report_b);
        assert_eq!(
            params(&a).to_flat().into_data(),
            params(&b).to_flat().into_data()
        );
        assert_eq!(gn.state().to_bytes(), resumed.state().to_bytes());
    }

    /// A problem whose "exact" J v is a quarter of the true one: the curvature is
    /// under-estimated 4×, so on a linear residual every CG iterate is 4× its true value.
    /// For a CG iterate x, gᵀx = −xᵀG x (Galerkin), so Φ(θ + 4x) − Φ(θ) = 4xᵀG x > 0:
    /// every candidate raises Φ.
    struct QuarterJvp;

    impl LeastSquares<Linear> for QuarterJvp {
        type Batch = Data;
        fn residual(&self, model: &Linear, batch: &Data) -> Tensor<1> {
            LinearFit.residual(model, batch)
        }
        fn jvp(&self, model: &Linear, batch: &Data, v: &ParamVec) -> Option<Tensor<1>> {
            Some(jvp(&LinearFit, model, batch, v, None).mul_scalar(0.25))
        }
    }

    #[test]
    fn a_step_is_rejected_when_no_cg_iterate_lowers_phi() {
        let device = Device::default().autodiff();
        let model = deterministic(LinearConfig::new(3, 2).init(&device), 0.3, 0.2, &device);
        let before = params(&model).to_flat().into_data();
        let mut gn = GaussNewton::new(
            GaussNewtonConfig::new()
                .with_lambda_init(1e-10)
                .with_lambda_min(1e-12)
                .with_weight_decay(0.0)
                .with_cg_progress_tol(0.0),
        );
        let (model, report) = gn.step(model, &QuarterJvp, &linear_data(6, &device));
        assert!(report.cg_iterations > 0);
        assert!(!report.accepted && report.actual_reduction > 0.0);
        assert!(report.rho < 0.0, "ρ = {}", report.rho);
        assert_eq!(params(&model).to_flat().into_data(), before);
        assert!(report.lambda_next > report.lambda);
    }

    #[test]
    #[should_panic(expected = "outside the config's bounds")]
    fn from_state_rejects_a_bad_lambda() {
        let mut state = GaussNewton::new(GaussNewtonConfig::new()).state().clone();
        state.lambda = f64::NAN;
        let _ = GaussNewton::from_state(GaussNewtonConfig::new(), state);
    }

    /// Three kept iterates (iterations 1, 2, 3) sharing one vector: `select` sees them only
    /// through `phi_at`, which here returns `phis` in call order and counts the calls.
    fn select_with(phis: &[f64]) -> (Option<usize>, Option<(usize, f64)>, usize) {
        let x = params(&LinearConfig::new(1, 1).init(&Device::default()));
        let iterates: Vec<CgIterate> = (1..=3)
            .map(|iteration| CgIterate {
                iteration,
                x: x.clone(),
                quadratic: -1.0,
            })
            .collect();
        let mut calls = 0;
        let choice = select(&iterates, 1.0, |_| {
            calls += 1;
            phis[calls - 1]
        });
        let accepted = match choice {
            Selection::Accepted { it, .. } => Some(it.iteration),
            Selection::Rejected { .. } => None,
        };
        let judged = choice.judged().map(|(it, phi)| (it.iteration, phi));
        (accepted, judged, calls)
    }

    #[test]
    fn select_takes_the_last_good_iterate_and_backtracks_only_when_needed() {
        // the last iterate lowers Φ: taken at once, one evaluation
        assert_eq!(select_with(&[0.5]), (Some(3), Some((3, 0.5)), 1));
        // only the second-to-last does: taken after one step back
        assert_eq!(select_with(&[1.2, 0.9]), (Some(2), Some((2, 0.9)), 2));
        // none does: rejected, ρ judged at the last iterate, all three evaluated
        assert_eq!(select_with(&[1.2, 1.1, 1.05]), (None, Some((3, 1.2)), 3));
        // no candidate at all
        let choice = select(&[], 1.0, |_| unreachable!("no iterate to evaluate"));
        assert!(matches!(choice, Selection::Rejected { last: None }));
    }

    #[test]
    fn next_lambda_follows_rho_and_stays_in_bounds() {
        let cfg = GaussNewtonConfig::new()
            .with_lambda_min(0.1)
            .with_lambda_max(10.0);
        let raise = 1.5;
        let lower = cfg.lambda_lower;
        assert_eq!(next_lambda(&cfg, 1.0, false, 0.9), raise); // rejected
        assert_eq!(next_lambda(&cfg, 1.0, true, f64::NAN), raise); // ρ undefined
        assert_eq!(next_lambda(&cfg, 1.0, true, 0.1), raise); // ρ < ¼
        assert_eq!(next_lambda(&cfg, 1.0, true, 0.5), 1.0); // ¼ ≤ ρ ≤ ¾
        assert_eq!(next_lambda(&cfg, 1.0, true, 0.9), lower); // ρ > ¾
        assert_eq!(next_lambda(&cfg, 9.0, false, 0.0), 10.0); // clamped above
        assert_eq!(next_lambda(&cfg, 0.12, true, 0.9), 0.1); // clamped below
    }

    #[test]
    fn state_rejects_bad_bytes() {
        let device = Device::default();
        let bytes = GaussNewton::new(GaussNewtonConfig::new())
            .state()
            .to_bytes();
        assert!(GaussNewtonState::from_bytes(&bytes, &device).is_ok());
        assert!(GaussNewtonState::from_bytes(&bytes[..bytes.len() - 1], &device).is_err());
        let mut long = bytes.clone();
        long.push(0);
        assert!(GaussNewtonState::from_bytes(&long, &device).is_err());
        assert!(GaussNewtonState::from_bytes(b"SCIMLCK1", &device).is_err());
    }
}

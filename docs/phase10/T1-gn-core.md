# T1: the generic core (parameter vectors, VJP, JVP, CG)

Phase 10, task 1. Branch `phase10/T1-gn-core`. Depends on Phase 9 (G8). No trainer
change.

## Read first
- This phase's README ("Facts verified", "The method in brief", G1, G2) and design
  §13.17.
- Burn: `burn-core/src/module/base.rs` (`ModuleVisitor`, `ModuleMapper`, `ParamId`),
  `burn-optim/src/optim/grads.rs` (`GradientsParams::from_grads`, `get`),
  `burn-tensor/src/tensor/api/autodiff.rs` (`backward`, its consumed-graph contract),
  `burn-tensor/src/device.rs` (`autodiff`, `inner`, dtype settings).
- `src/neural_operators/training/step.rs` (`forward_loss`, how a residual is formed
  today) and `training/fixtures.rs` (`tiny_dataset_2d`; `tiny_model_cfg` is 1D, so the 2D tiny FNO is built inline as in `step.rs`'s tests).
- `src/neural_operators/pde/poisson/sparse.rs` (the existing f64 CG, for its stopping
  rule and tests only; do not reuse it).

## Goal
The products Gauss–Newton needs, for any Burn `Module`, without forming a Jacobian:
J v, Jᵀ u and the solve of (G + λI) p = b by CG, each checked against an explicit
Jacobian on models small enough to build one.

## Deliverables
1. **`optim/gauss_newton/` module**, new, under `src/neural_operators/optim/`. It imports
   nothing from `models/` or `training/`; a grep for those paths under `optim/` is part
   of the review.
2. **`ParamVec`**: one flattened `Tensor<1>` per float parameter, keyed by `ParamId`, in
   the visitor's order, on the model's device. `dot`, `norm`, `axpy`, `scale`, `zeros_like`
   and elementwise `mul` / `div` (for diagonal preconditioners). Dot products accumulate
   on the device; state the precision in the doc comment.
3. **`params(&M) -> ParamVec`** (a `ModuleVisitor`) and **`with_params(M, &ParamVec) -> M`**
   (a `ModuleMapper`, same `ParamId`s, shapes restored), for any `M: Module`.
4. **`trait LeastSquares<M: Module>`** with `type Batch`, a required
   `residual(&self, &M, &Self::Batch) -> Tensor<1>` (the weighted, flattened residual,
   tracked when the model is on an autodiff device) and an optional
   `jvp(&self, &M, &Self::Batch, &ParamVec) -> Option<Tensor<1>>` defaulting to `None`.
5. **`vjp(problem, model, batch, u) -> ParamVec`**: Jᵀu by one backward of
   Σ r ⊙ u.detach() on the autodiff device. The gradient Jᵀr is the case u = r.
6. **`jvp(problem, model, batch, v, eps) -> Tensor<1>`**: the problem's exact `jvp` if it
   returns `Some`, else the central difference [r(θ + εv) − r(θ − εv)] / 2ε on the inner
   device (no graph), with ε scaled by ‖θ‖ / ‖v‖ (document the rule and its default).
7. **`adjoint_error(problem, model, batch, u, v) -> f64`**:
   |⟨u, Jv⟩ − ⟨Jᵀu, v⟩| / (‖u‖ ‖Jv‖), the runtime check of the JVP against the VJP.
8. **`cg(apply, b, precond, cfg) -> CgResult`**: preconditioned CG on a symmetric positive
   (semi-)definite operator given as a closure on `ParamVec`, with a warm start, an
   iteration cap, a relative-residual tolerance and Martens' relative-progress stop on
   the quadratic model. `CgResult` returns the iterates needed for backtracking (every
   iterate, or a geometric subset; say which), the iteration count, the final relative
   residual, and the Lanczos tridiagonal's extreme Ritz values from CG's α and β (host
   f64, at most a few dozen iterations).

## Tests (flex)
- `params` then `with_params` round-trips every weight bit for bit, on the 2D tiny FNO
  and on a plain `Linear`.
- **Explicit-Jacobian oracle:** for a tiny FNO (P of order 100, a few outputs) and a
  `Linear`, build J column by column with VJPs on unit output vectors; `vjp` and the
  finite-difference `jvp` match J ᵀu and J v within a stated tolerance.
- The adjoint error is below a stated bound for random u, v; and a deliberately wrong JVP
  (sign flipped) is caught.
- CG on a small explicit SPD matrix reaches the known solution; with λI added it matches
  the damped solve; the Ritz values bracket and converge to the true extreme eigenvalues.
- **f64 on flex:** if flex runs the tiny FNO in f64 (FFTs included), the oracle tests run
  in f64 with tight tolerances and in f32 with loose ones; if not, say so in the PR and
  keep f32. Record which.

## Acceptance
- The tests above pass; the T1 Results row filled in with the measured JVP and adjoint
  errors in f32 (and f64 if available) on the tiny FNO.

## Checks
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`. The
tests with `--features metal` if available, saying whether they ran.

## Do not
- Touch `training/`, `models/`, `runs/` or the examples.
- Form a Jacobian outside tests.
- Implement forward-mode autodiff, or an exact FNO JVP (G2: only after T4, if needed).
- Copy the ndarray CG from `pde/poisson/sparse.rs`.

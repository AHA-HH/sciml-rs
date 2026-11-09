# T2: the Gauss–Newton optimiser

Phase 10, task 2. Branch `phase10/T2-gauss-newton`. Depends on T1. No trainer change.

## Read first
- This phase's README (G1, G3, G5) and design §13.17 (the step, ρ, the damping rule).
- T1's module `optim/gauss_newton/` (`ParamVec`, `LeastSquares`, `vjp`, `jvp`, `cg`).
- Martens, "Deep learning via Hessian-free optimization", ICML 2010 (damping, CG
  stopping, warm start, backtracking); Nocedal & Wright, ch. 10 (Levenberg–Marquardt).

## Goal
One damped Gauss–Newton step on any `LeastSquares` problem, with the diagnostics the
experiments need, proven on problems with known answers.

## Deliverables
1. **`GaussNewtonConfig`** (Burn `Config`, saved with runs): CG iteration cap and
   tolerance, initial λ and its bounds, the λ update factors and ρ thresholds (defaults
   ×3/2 below ρ = ¼, ×2/3 above ¾), weight decay μ (default 1e-4, G5), warm-start decay
   (default 0.95), the finite-difference ε rule, the preconditioner, the adjoint-check
   cadence.
2. **`GaussNewton`** with state (λ, warm-start vector, step count) and
   `step(model, problem, batch) -> (model, GnStep)`:
   - g = Jᵀr + μθ; solve (G + (λ + μ)I) p = −g by `cg`, G v = Jᵀ(J v);
   - the quadratic model's predicted reduction; the actual reduction of Φ (plus ½μ‖θ‖²) on
     the **same batch**; ρ;
   - backtracking over the CG iterates when the last is worse; reject the step (model
     unchanged) if no iterate reduces Φ;
   - the λ update from ρ; the warm start for the next step.
3. **Preconditioners:** `None`, and `HutchinsonDiagonal { probes }`: diag(G) estimated as
   the mean of (Jᵀz)² over Rademacher z in output space (one VJP per probe), applied as
   (diag + λ + μ)^0.75 (Martens' exponent; configurable). Re-estimated every k steps
   (configurable).
4. **`GnStep`** diagnostics: Φ before and after, ‖g‖, ‖p‖, predicted and actual
   reduction, ρ, λ, accepted, CG iterations and relative residual, Ritz θ_min and θ_max,
   the adjoint error when checked, and the count of forward and backward passes.
5. **State for checkpoints:** a record of λ, the step count and the warm-start vector,
   with save and load, so T3 can put it in `checkpoint.bin`.

## Tests (flex)
- **Linear least squares through a `Linear` module** (not an FNO): with λ → 0 and enough
  CG iterations, one step reaches the normal-equations solution (computed on the host in
  f64) to round-off; ρ = 1 to round-off.
- **A nonlinear fit** (a tiny MLP or an exponential model on synthetic data): GN reaches a
  stated loss in fewer steps than plain gradient descent at its best tested learning
  rate; ρ → 1 near the optimum.
- **The tiny 2D FNO** on `tiny_dataset_2d`: Φ never rises on an accepted step; a rejected
  step leaves the weights bit for bit unchanged and raises λ.
- The Hutchinson diagonal matches the explicit diag(JᵀJ) of a tiny model within its
  sampling error.
- The state record round-trips: a step after save and load equals a step without it.

## Acceptance
- The tests pass; the T2 Results row filled in (steps to tolerance on the nonlinear fit,
  GN against gradient descent).

## Checks
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`;
`--features metal` if available, saying whether it ran.

## Do not
- Touch `training/`, `models/`, `runs/` or the examples.
- Add a learning-rate schedule to GN: λ is the step control.
- Add preconditioners beyond the two above (block or Fourier-mode ones go through the
  same interface later, as their own task).

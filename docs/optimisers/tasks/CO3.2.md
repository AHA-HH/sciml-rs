# CO3.2 The SOAP optimiser

Started 2026-11-25 on `optimisers/co3-2`. Roadmap: "CO3.2 The SOAP optimiser. `optim/soap/`:
`SoapConfig`, `Soap` as a Burn `Optimizer`, any tensor rank." Proposal: §10; §1.4, §2,
§5 (5.1, 5.2, 5.4, 5.5), §9; D8 and D9 (roadmap).

## Goal
A SOAP optimiser (Vyas et al. 2024, Algorithm 3, with the reference code's handling of
tensors) as Burn's per-parameter `Optimizer`, for tensors of any rank, used by nothing yet.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)
"the SOAP tests of proposal §9 pass (the identity-basis reduction to Adam, the quadratic,
ranks 1–5, the refresh, record round trip)."

Proposal §9: "with every basis fixed at the identity, it equals Adam with the same moments to
f32 round-off; a quadratic with a known Kronecker-structured Hessian; tensors of rank 1 to
5; the refresh against a host f64 eigendecomposition; the state's record round trip and a
resumed run bit for bit equal to an uninterrupted one; small fixtures from the reference
implementation (Python, outside CI, as in the library objective)."

The resumed *training run* is CO3.3's (it needs the trainer arm); here the resume is of the
optimiser: a `ModuleOptimizer` saved to bytes and reloaded takes the next steps bit for bit
like the uninterrupted one. The Python fixtures are not in the roadmap's *Done when* and are
left out (see Open questions).

## Read first
- Reference `soap.py` at nikhilvyas/SOAP@a1e5535 (fetched by plain HTTP): `step`,
  `update_preconditioner`, `project`, `project_back`, `get_orthogonal_matrix`,
  `get_orthogonal_matrix_QR`. Notes: `docs/optimisers/references.md` §1.
- Burn 0.23.0-pre.2: `burn-optim/src/optim/module/base.rs` (`Optimizer`: `step` per tensor,
  `State<D>: Clone + RecordState`, `to_device`), `adam.rs`, `adamw.rs`, `state.rs`
  (`RecordState`, `StateSink`, `StateSource`; the derive emits `crate::join_index`, so it
  only works inside `burn-optim`: SOAP's state implements `RecordState` by hand),
  `burn-linalg/src/functions/svd.rs` (host; square input gives an orthonormal n × n U,
  singular values descending).
- `spikes/linalg/REPORT.md` (D9's evidence); `optim/gauss_newton/` (module style).

## Changes
New module `src/neural_operators/optim/soap/` (imports nothing from `models/` or
`training/`), declared in `optim/mod.rs`; CLAUDE.md's Layout gains its line.

- `SoapConfig` (derives `Config`), the reference defaults: `beta_1` 0.95, `beta_2` 0.95,
  `shampoo_beta: Option<f32>` (`None` = β₂), `epsilon` 1e-8, `weight_decay` 0.01
  (decoupled), `precondition_frequency` 10, `max_precond_dim` 10,000. `init()` gives a
  `ModuleOptimizer`, `build()` the bare `Soap` (as Burn's `AdamConfig`). D6 picks the
  values later; nothing here depends on them.
- `Soap: Optimizer`, `SoapState<D>` (`RecordState` by hand: `step`, `moment_1`,
  `moment_2`, and per dimension an optional factor and basis, `gram.<k>`, `basis.<k>`).
- The algorithm, per tensor G of rank D:
  - a factor Lₖ (dₖ × dₖ) and basis Qₖ for every dimension k with dₖ ≤ `max_precond_dim`,
    when D ≥ 2 (D8, option A: one factor per
    dimension; re and im are separate parameters already). Rank 1 (biases) and dimensions
    above `max_precond_dim` keep Q = I, so a bias gets Adam (D8) by the same code;
  - first call: Lₖ ← (1 − β_s) Gₖ Gₖᵀ, Qₖ from the host `svd` of Lₖ (D9), the parameter
    unchanged (reference: the gradient never enters its own projection);
  - later calls: G′ = G ×ₖ Qₖᵀ (all k); M ← β₁M + (1 − β₁)G in the original space and
    M′ = M ×ₖ Qₖᵀ (the paper's Algorithm 3; the reference keeps M in the basis and rotates
    it on every refresh, the same M up to round-off, proposal §5.1); V ← β₂V + (1 − β₂)G′²
    in the basis; W ← W − lr·√(1 − β₂ᵗ)/(1 − β₁ᵗ) · (M′ / (√V + ε)) ×ₖ Qₖ; then
    W ← W(1 − lr·wd) (the reference's order); then Lₖ ← β_s Lₖ + (1 − β_s) Gₖ Gₖᵀ, and
    every f-th step the refresh;
  - refresh (D9, host for every backend): Lₖ, Qₖ downloaded; estimated eigenvalues
    diag(QₖᵀLₖQₖ), sorted descending, Qₖ's columns and V along k permuted to follow; one
    power step S = LₖQₖ and a Householder QR of S on the host in f64, rounded to f32 and
    uploaded. f64 for a ≤ 144 × 144 host QR costs nothing and is at least as accurate as f32.
  - mode products: unfold at the leading dimension, `matmul`, rotate the dimensions by
    one; D of them restore the order (the reference's `tensordot` loop).
- No convention change, no dependency (`linalg` is already on, CO3.1), no `training/`
  change, nothing calls SOAP yet.

## Tests that prove it
In `optim/soap/` unit tests, on the default device (flex in CI), f32:
1. **Identity basis = Adam.** `max_precond_dim = 0` (every basis the identity, the
   reference's own way to get Adam): SOAP after its first (skipped) call, fed the same
   gradients as Burn's `Adam` (β 0.95, 0.95, ε 1e-8, no decay), on a rank-3 tensor, gives
   the same parameters to f32 round-off (relative 1e-6) over 20 steps. The only
   differences are ε's placement (Burn scales it by √(1 − β₂ᵗ)) at ε = 1e-8 and gradients
   of order 1, below f32 round-off.
2. **Quadratic, Kronecker Hessian.** f(W) = ½⟨W − W*, A (W − W*) B⟩, Hessian B ⊗ A, A and B
   rotated with κ = 100 each. The first basis's factors equal (1 − β_s) GGᵀ and GᵀG
   (host f64, 1e-5 relative); SOAP at the same lr reaches a loss below Adam's after 200
   steps, and below 1e-2 of the start.
3. **Ranks 1–5.** Shapes [7], [5, 6], [4, 3, 5], [3, 4, 2, 3], [2, 3, 2, 3, 2] (with a
   size-1 dimension in one of them): after three steps every factor equals the host f64
   EMA of the mode-k Grams (1e-5), every basis is orthonormal (1e-5), and one more step's
   update equals a host f64 recomputation from the state's Q, M, V (1e-5 relative).
4. **The refresh against f64.** Frequency 2; the first basis against a host f64 Jacobi
   eigendecomposition of the same f32 factor (columns up to sign, 1e-4 at gaps ≥ 1e-2); the
   refreshed basis against a host f64 run of the same sort, power step and QR (1e-5, up to
   sign); V permuted with the columns.
5. **Record round trip.** A `ModuleOptimizer` of SOAP on a small two-layer module (a rank-2
   weight, a rank-1 bias, a rank-3 parameter), five steps, `into_bytes`, `from_bytes` into
   a fresh one, then five more steps across a refresh: parameters bit for bit equal to the
   uninterrupted run. Also `to_device` keeps the state.

## Checks
`/verify` (fmt, clippy lines, `cargo test`, `cargo doc`, datagen clippy and tests); the
soap tests with `--features metal` (backend-sensitive: matmul, select).

## VM runs
None.

## Constraints and risks
- Constraint 1: nothing existing calls the new module; no existing file changes except
  `optim/mod.rs` (one `pub mod`) and docs. Shown by the existing suite unedited.
- Every 2D and 3D output: untouched.
- Risk: Burn's host `svd` may not converge in its sweep budget on a rank-deficient factor
  (e.g. a [3, 32] weight's 32 × 32 factor has rank ≤ 3). The test covers such a shape; if it
  fails, a 1e-30·I shift as in the reference.

## Pending runs
None.

## Open questions
- Python fixtures from the reference implementation (proposal §9's last item): not in the
  roadmap's *Done when*; left for a follow-up if the author wants them.

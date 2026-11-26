# CM2.5 Equation losses (PINO)

Started 2026-11-26 on `operators/cm2-5`. Roadmap: "CM2.5 Equation losses (PINO).
`equation_losses` on CM1.6's derivatives; the crate's Chebyshev differentiation as a second
path. *Needs:* CM1.6a (2.0.0's `equation_losses` imports only `FiniteDiff`)." Proposal: §10;
§1.2 (equation losses in scope), §1.4, §2, §5, §6.4. D1, D2, D3 (as amended 2026-11-25), D8.
Independent of D6 (the M2/M3 order): it needs only CM1.6a, merged.

## Goal

2.0.0's `equation_losses` in `losses/`, matching 2.0.0 on a committed fixture. The equation
residuals take their derivatives from either `FiniteDiff` or a differentiation matrix per
axis, so the crate's Chebyshev matrices (`chebyshev::diff_matrix`, CONVENTIONS §12) give a
second, spectral path on the CGL grid.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)

- The roadmap line has no *Done when*. The proposal's recipe (§4.3, §6.4) and §5 apply:
  output parity against fixtures from 2.0.0, within D3 (max |a − e| / max |e| ≤ 1e-5 in
  f32, ≤ 1e-12 in f64, on the CPU). No FFT is involved, so the f64 tests run, as in CM1.6a.
- The Chebyshev path: no 2.0.0 reference (2.0.0's `PoissonInteriorLoss` has only
  `method="autograd"`; `"finite_difference"` raises `NotImplementedError`). It is checked
  against closed forms instead (below).
- Constraint 1 (§1.4): every existing FNO output bit for bit identical. This task adds a
  module and changes no existing code path.

## Read first

- 2.0.0 `neuralop/losses/equation_losses.py`: `BurgersEqnLoss` (`fdm` only), `ICLoss`,
  `PoissonInteriorLoss` (`autograd` only: ∇·((1 + 0.1u²)∇u) = f expanded as
  Δu (1 + 0.1u²) + 0.2u |∇u|², on point clouds, second derivatives by double backward),
  `PoissonBoundaryLoss`, `PoissonEqnLoss`. Base loss `F.mse_loss`: the mean over every
  entry, not 2.0.0's `MSELoss` (the crate's `pointwise::mse_loss`, summed over the batch).
- `losses/differentiation.rs` (`FiniteDiff`), `losses/pointwise.rs`;
  `chebyshev/differentiation.rs` (`diff_matrix`, `diff2_matrix`, behind `datagen`);
  `tests/parity_losses_common/mod.rs`, `tests/parity_common/mod.rs`;
  `tools/parity/generate_losses.py`.

## Changes

- `src/neural_operators/losses/differentiation.rs`: `GridDiff<D>`, a trait: the partial
  derivative of order 1 or 2 along a spatial axis (the last `D` axes, as `FiniteDiff`), and
  a default `laplacian`. Implemented by `FiniteDiff<D>` (delegating to its own methods) and
  by `MatrixDiff<D>`: one first- and one second-derivative matrix per axis
  (`new(first, second, device)` from `ndarray` arrays), `partial` contracting the axis
  with the matrix. Behind `datagen`, `MatrixDiff::chebyshev(n, device)` takes
  `diff_matrix`/`diff2_matrix` on the ascending CGL nodes of [−1, 1] (CONVENTIONS §12).
  Ungated otherwise, so a run without `datagen` can pass matrices it built itself.
- `losses/pointwise.rs`: `mean_squared_error(a, b)`, `F.mse_loss`, the mean over every
  entry, `[1]`, beside 2.0.0's `MSELoss` (`mse_loss`).
- `src/neural_operators/losses/equation_losses.rs` (new):
  - `BurgersEqnLoss { visc, domain_length: [f64; 2] }`, default 0.01 and (1, 1):
    `loss(u)` with 2.0.0's `fdm` (`FiniteDiff` non-periodic, h = (L_t / (n_t − 1), L_x / n_x))
    and `loss_with(diff, u)` with any `GridDiff<2>`; u `[batch, 1, n_t, n_x]` (any leading
    axes).
  - `ic_loss(y_pred, x)`: 2.0.0's `ICLoss`, the MSE of channel 0 at t = 0.
  - `nonlinear_poisson_lhs(u, grad, lap)`: 2.0.0's left-hand side, any rank, `D`
    gradient components; `poisson_interior_loss(u, grad, lap, f)`: its MSE against f.
  - `poisson_boundary_loss(y_pred, y, num_boundary)`: 2.0.0's `PoissonBoundaryLoss` on a
    point list (`out_sub_level` = 1).
  - `PoissonEqnLoss { interior_weight, boundary_weight }`: `points(u, y, num_boundary,
    grad, lap, f)`, 2.0.0's point-list form with the interior derivatives supplied, and
    `grid(diff, u, g, f)`, on a grid: the interior residual against f on the interior
    nodes and the boundary MSE of u against g on the boundary nodes, with derivatives from
    a `GridDiff<D>`.
- `losses/mod.rs`: the module and its doc line. `CLAUDE.md` "Layout": the losses entry.
- `tools/parity/generate_losses.py`: fixtures `equation_losses_{f32,f64}`; README table.
- `tests/parity_losses_common/mod.rs`: `run_equation`; `tests/parity_losses{,_f64}.rs`.
- Conventions: none changed.

Not in this task, recorded as follow-ups: 2.0.0's `autograd` method on point clouds (no
point-cloud operator exists before CM3.3, and Burn's autodiff is first order, so the
second derivatives with respect to the query points need another route); a Chebyshev
domain other than [−1, 1] (scale the matrices).

## Tests that prove it

- Parity, f32 and f64, fixture `equation_losses_<dtype>` from 2.0.0:
  - `BurgersEqnLoss` at the defaults and at visc 0.05, domain (2, 3), on a seeded u
    (2, 1, 10, 12).
  - `ICLoss` on seeded x, y_pred (2, 1, 10, 12).
  - The Poisson losses on a point cloud of 6 boundary and 20 interior points, u an analytic
    function of the query points (torch autograd for its derivatives, stored in the
    fixture as `poisson.{u,u_x,u_y,lap}` at the interior points): `PoissonInteriorLoss`,
    `PoissonBoundaryLoss` (seeded y), `PoissonEqnLoss(boundary_weight=2, interior_weight=0.5)`.
    The Rust side computes the left-hand side from the stored derivatives.
  - Within D3. A negative check: the Burgers loss with t and x swapped fails parity.
- Unit tests (default device, f32):
  - `MatrixDiff` with the periodic central-difference matrices equals `FiniteDiff`
    periodic (f32 rounding).
  - `PoissonEqnLoss::grid` on u = sin(πx) sin(πy), f its exact left-hand side, g = u:
    finite differences converge at second order (the loss falls about 16× as h halves,
    the residual 4×); boundary part 0.
  - Behind `datagen`: `MatrixDiff::chebyshev` on the same u at n = 17 has a residual
    far below the finite differences' at n = 65 (spectral); the matrices are exact on a
    polynomial of degree < n.
  - A gradient flows through `BurgersEqnLoss` and `MatrixDiff` on the autodiff device.
  - Panics: shapes that differ, `num_boundary` beyond the points.

## Checks

`/verify` (fmt, clippy default / `--examples` / `--all-targets` / `--features datagen
--all-targets`, `cargo test`, `cargo test --features datagen`, `cargo doc --no-deps`, the
FNO bit-identity dumps); `uv run python generate_losses.py --check` (the earlier fixtures
unchanged). `CARGO_TARGET_DIR=~/sciml-rs-target/cm2-5`.

## VM runs

None.

## Constraints and risks

- Constraint 1: no existing code path changes; `FiniteDiff` gains a trait impl only.
- The 2D objective: unaffected.
- f32 Chebyshev second derivatives amplify rounding about as n⁴; the unit tests use small n
  and tolerances measured with a margin.

## Pending runs

None.

## Open questions

None yet.

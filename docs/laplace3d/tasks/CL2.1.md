# CL2.1 Collocation solver A3

Started 2026-11-24 on `laplace3d/cl2-1`. Roadmap: "**CL2.1 Collocation solver A3.** Fast
diagonalisation in 3D (proposal §5). *Done when:* relative error (max norm and 3D CC norm)
at the f64 floor by n = 33 on manufactured solutions and on sine-series closed forms (a
test-only 3D series evaluator, or CL3.1's); time per solve at n = 33 and 65. *Needs:*
CL1.1, CL1.2." Proposal: §10; §5; §2 ("interior D2 … reusable inside that module"); §6
("Extend"); §7 (the A3 flop row); §9.

## Goal
A 3D entry point `CollocationSolver3d` beside the 2D `CollocationSolver` in
`pde/poisson/collocation.rs`: u = −(V ⊗ V ⊗ V)(Λ ⊕ Λ ⊕ Λ)⁻¹(V⁻¹ ⊗ V⁻¹ ⊗ V⁻¹) f on the
interior nodes of an `[n_x, n_y, n_z]` CGL grid stored 'ijk' (CONVENTIONS §12).

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)
- "relative error (max norm and 3D CC norm) at the f64 floor by n = 33 on manufactured
  solutions": relative max and `cc_rel_l2_error_nd` errors ≤ 1e-12 at n = 33 and 65 (the
  2D solver's floor test uses the same bound) for sin(πx) sin(πy) sin(πz) and a product of
  low sine modes, plus a non-polynomial, non-sine one; printed for n = 9, 17, 33, 65.
- "and on sine-series closed forms (… or CL3.1's)": `grf3d` (on `main`) gives f and the
  exact u = Σ c / μ; at K = 8, n = 33, and K = 16, n = 65, the same two errors ≤ 1e-12.
  No new evaluator is needed.
- "time per solve at n = 33 and 65": setup and mean solve time, release build, flex not
  involved (ndarray only), on the Mac; in `docs/laplace3d/results/cl2-1.md`. No timing is
  asserted.
- Proposal §5: six mode products, about 12 (n − 2)⁴ flops; f64; spectral convergence in n.

## Read first
`pde/poisson/collocation.rs` (the 2D solver, `real_eig`, `interior_d2`, its tests),
`chebyshev::apply_nd` (CL1.2: one matrix per axis, the mode products), `chebyshev::
cc_rel_l2_error_nd`, `pde/poisson/grf3d` (`draw_xi`, `forcing_coeffs`, `solution_coeffs`,
`evaluate`).

## Changes
- `pde/poisson/collocation.rs` only: `pub struct CollocationSolver3d` with `new(nx, ny,
  nz)`, `solve(f_interior: ArrayView3)` and `solve_full(f: ArrayView3)`, mirroring the 2D
  API. It reuses `real_eig` and `interior_d2` per axis and `apply_nd` for the two
  three-way mode products. Every existing 2D item is left byte for byte as it is.
- `pde/poisson/manufactured.rs` (test-only): the 3D manufactured pairs and `sample3` beside
  the 2D ones; `rel_max` generic over the rank (code-quality review).
- `real_eig`'s panic texts start `real_eig:`, not `CollocationSolver::new:`, since A3 calls
  it too (messages only; code-quality review).
- The module docs gain a paragraph on the 3D form. `pde/poisson/mod.rs`'s one-line docs
  mention A3. Root `CLAUDE.md`'s layout line for `collocation` names the 3D type.
- Conventions: none.

## Tests that prove it
In `collocation.rs`'s test module, beside the 2D tests (fixtures in `manufactured.rs`):
- `solver_3d_manufactured_solutions_reach_floor`: against closed-form u (the trusted
  path), ≤ 1e-12 at n ≥ 33, and n = 9 → 17 gains ≥ 100× (spectral decay).
- `solver_3d_grf_closed_form_reaches_floor`: against `grf3d`'s exact series, ≤ 1e-12.
- `solver_3d_residual_at_round_off`: the discrete equation −(D_xx ×₁ + D_yy ×₂ + D_zz ×₃) U
  = F, applied by `apply_nd` with identities, scaled residual ≤ 1e-14 at n = 9 … 65. An
  oracle independent of the eigendecomposition.
- `solver_3d_non_cubic_grid`: a field resolved only along the fine axis reaches the floor
  on `[17, 25, 33]`; on the axis-permuted grid it does not (≥ 1e-6). This catches a
  swapped axis in the 'ijk' layout.
- Panics: n < 3 on any axis, wrong `solve` and `solve_full` shapes.
- `time_per_solve_3d` (`#[ignore]`, prints only): setup and solve times at n = 33, 65.
- Constraint 1: the diff to the 2D items is empty (additions only). The 2D tests
  (`manufactured_solutions_reach_floor`, `agrees_with_kronecker_lu`, `residual_at_round_off`
  …) and the whole 2D suite pass unchanged. The 2D solver is still the only 2D code path,
  so its outputs on fixed inputs are those of `main` by construction.

## Checks
`/verify` (fmt, the three CI clippy jobs, `cargo clippy --all-targets`, `cargo test`,
`cargo doc`, datagen clippy and tests). The timing:
`cargo test --release --features datagen time_per_solve_3d -- --ignored --nocapture`.

## VM runs
None.

## Constraints and risks
- Constraint 1 (D2): only additions in `collocation.rs`; no 2D function edited.
- Behind `datagen`, like the 2D solver (RLST's `eig`).
- n = 129 needs 127³ · 8 B ≈ 16 MB per field, which is fine, but is out of scope (proposal
  §7).

## Pending runs
None.

## Open questions
None.

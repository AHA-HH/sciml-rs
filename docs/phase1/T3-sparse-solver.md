# T3: sparse solver (option C) and the A-vs-C check

Phase 1, task 3. Branch `phase1/T3-sparse-solver`. Depends on T1 (nodes, norm) and T2
(solver A, for the cross-check).

## Read first
- Design §4.1, §4.2 (C as validation oracle, the CG growth), §7 (error 1), §11 (the
  sparse-solver row), §12 decision 10.
- `spikes/solver/REPORT.md`: decisions on C, Findings 4, tables §4–§6;
  `spikes/solver/src/fem.rs` (the `Q1` variant: `p1_1d`, assembly, consistent load, CG
  with the iteration counter).
- rlst 0.9.0: `CsrMatrix::from_aij`, `CgIteration` (`run()` returns the residual, not a
  converged flag; `set_callable` runs once per update).

## Goal
An independent, second-order discretisation of the same problem that checks solver A
on the CGL nodes, with no interpolation.

## Deliverables
New module `src/neural_operators/pde/poisson/sparse.rs`, behind
`#[cfg(feature = "chebyshev")]`:
1. Assembly of bilinear Q1 finite elements on the CGL tensor mesh (n_x × n_y), Dirichlet
   nodes eliminated. The matrix is K_x ⊗ M_y + M_x ⊗ K_y from the 1D P1 stiffness and
   consistent mass, assembled with `CsrMatrix::from_aij`. Unknown ordering stated in the
   doc comment.
2. Consistent load: b = (M_x ⊗ M_y) f restricted to the interior rows, with f the nodal
   values on the full grid.
3. A solve with unpreconditioned `CgIteration` at relative residual tolerance 1e-6
   (decision 10), with a maximum iteration count large enough for n = 257 (T4 needed
   2998). It returns u on the full grid (zero boundary), the iteration count from
   `set_callable`, and the final relative residual. It panics, or returns an error (state
   which), if the residual is above the tolerance at the maximum iteration count.

Document shapes and panics of every public item. ndarray in and out.

## Tests (`#[cfg(test)] mod tests`, gated)
- `q1_is_second_order`: the three manufactured solutions of design §4.3, Clenshaw–Curtis
  relative L² error at n ∈ {33, 65, 129}; observed order log₂(e_n / e_{2n−1}) in
  2 ± 0.1 for both steps (T4: 2.00). n = 257 is left out: about 1.5 s per solve in
  release, tens of seconds in debug.
- `cg_meets_tolerance`: the returned relative residual is ≤ 1e-6, and the iteration count
  is positive and below the cap.
- `matrix_is_symmetric`: Kᵀ = K to round-off at n = 9.
- `agrees_with_collocation`: u_A (T2) against u_C on the common CGL grid for three fixed
  smooth sine-series forcings with hand-chosen coefficients (for example a few low
  (k, l) modes, f = 0 on ∂Ω), at n ∈ {33, 65, 129}; order 2 ± 0.1 (T4 on GRF:
  1.97–2.00). This is error 1 of design §7. Keep the forcings in the test; the GRF sampler
  is Phase 2.

## Acceptance
- All tests pass under `cargo test --features chebyshev` on macOS and in CI.
- The Results rows for C's orders, A against C, and the test wall time are filled in
  in `docs/phase1/README.md`. The wall time is the debug `cargo test --features chebyshev`
  run of the new tests, so the next phases know the budget.
- The Phase 1 exit checklist is ready to tick, apart from the CLAUDE.md pointer.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`, then the
same clippy and test commands with `--features chebyshev`. Give the exact test filters and
check the "running N tests" lines.

## Do not
- Add a preconditioner, GMRES (option D) or finite volume; C is Q1 (decision 10).
- Add the `nd` crates: nd 0.4 depends on rlst 0.6 and cannot build alongside rlst 0.9
  (design decision 11). Assemble Q1 by hand as K₁ ⊗ M₁ + M₁ ⊗ K₁.
- Use C to produce labels, or add C to any data path.
- Change solver A beyond a bug fix stated in the PR.

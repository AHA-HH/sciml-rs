# T2: collocation solver (option A)

Phase 1, task 2. Branch `phase1/T2-collocation-solver`. Depends on T1 (nodes, D,
Clenshaw–Curtis, the L² norm).

## Read first
- Design §4.1 (the separable equation), §4.2 (decision), §4.3 (verification), §11 (the
  two collocation-solver rows), §12 decisions 1 and 10.
- `spikes/solver/REPORT.md`: Findings 1–3, Setup, tables §2–§3;
  `spikes/solver/src/colloc.rs` (`FdSolver`, `real_eig`, `kron_lu`).
- rlst 0.9.0: `EigenvalueDecomposition::eig`, `EigMode`, `Inverse`, and LU `Solve`.
- `tests/rlst_smoke.rs` for how rlst is called and linked in this crate.

## Goal
The solver that produces every training label: −Δu = f on [−1, 1]², u = 0 on the
boundary, by Chebyshev collocation, solved by fast diagonalisation of
−D_xx U − U D_yyᵀ = F on the interior nodes.

## Deliverables
New module `src/neural_operators/pde/poisson/collocation.rs` (and the `pde`/`poisson`
module files), behind `#[cfg(feature = "chebyshev")]`:
1. A solver struct built once per (n_x, n_y). At construction:
   - take the interior blocks D_xx, D_yy of T1's D²;
   - eigendecompose each with `eig(EigMode::BothEigenvectors)`, discarding the left
     vectors. `RightEigenvectors` panics in rlst 0.9.0; say so in a comment at the call;
   - keep the real parts of the eigenvalues and eigenvectors, and assert that the
     largest imaginary part is below a stated tolerance (T4 saw exactly 0);
   - store V_x, V_x⁻¹, V_y, V_y⁻¹ and 1 / −(λ_i + μ_j).
2. `solve(&self, f_interior) -> u_interior`: Ĝ = V_x⁻¹ F V_y⁻ᵀ, Û = Ĝ ⊙ (1 / −(λ_i + μ_j)),
   U = V_x Û V_yᵀ. Shapes `[n_x − 2, n_y − 2]` in and out, 'ij'.
3. `solve_full(&self, f) -> u` on the full `[n_x, n_y]` grid: reads the interior of f and
   returns u with the zero boundary. This is what `generate_poisson` (Phase 2) calls.
4. ndarray in and out; `DynArray` conversions stay private.

Document the shapes and panics (wrong shape, n < 3) of every public item, citing design §4
and `CONVENTIONS §12`.

## Tests (`#[cfg(test)] mod tests`, gated with the module)
- `manufactured_solutions_reach_floor`: the three solutions of design §4.3, errors
  relative max and Clenshaw–Curtis relative L² (T1's norm):
  - sin(πx) sin(πy) and (1 − x²)(1 − y²)(x + y²): ≤ 1e-12 by n = 33 (T4: 1.1e-14);
  - all three: ≤ 1e-12 for every n ∈ {33, 65, 129} (T4: at most 4.6e-13 at 129);
  - spectral decay: on sin(πx) sin(πy) and e^(x + 2y), the error at n = 17 is at least 100×
    smaller than at n = 9.
- `agrees_with_kronecker_lu`: option B, the dense (n − 2)² Kronecker-sum system solved by
  LU (test-only helper), on a fixed smooth forcing, ≤ 1e-10 relative for n ∈ {9, 17, 33}
  (T4: 3.4e-14).
- `rectangular_grid`: n_x = 17, n_y = 33 on a manufactured solution, same floor.
- `factorisation_reused`: two solves on one struct give the same result as two fresh
  structs, to round-off.

Keep n ≤ 129 in tests; they run in debug.

## Acceptance
- All tests pass under `cargo test --features chebyshev` on macOS and in the
  `run-tests-chebyshev` CI job; the default build compiles without the module.
- No call to `solve_sylvester` or `trsyl`.
- One line in the Layout section of `CLAUDE.md` names `pde/poisson/`.
- The Results rows "A's measured floor" and "A against B" in `docs/phase1/README.md` are
  filled in.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`, then the
same clippy and test commands with `--features chebyshev`. Give the exact test filters and
check the "running N tests" lines. The solver is CPU-only, so `--features metal` is not
relevant; say so.

## Do not
- Use `solve_sylvester` (ruled out, decision 10).
- Implement the GRF or the series label check (Phase 2 T2), or solver C (T3).
- Expose rlst types publicly, or change T1's functions beyond a bug fix stated in the PR.

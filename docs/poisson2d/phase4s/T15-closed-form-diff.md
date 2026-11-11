# T15: closed-form differentiation matrices; FFTW dropped ⚑

Phase 4s, task 15. Branch `phase4s/T15-closed-form-diff`. Depends on T14. T16 comes after.
Can run in parallel with Group 3.

> **⚑ Decision needed before any code changes: amend design §12 decision 12.** Record the
> answer as a new decision (26 or later) that amends decision 12, and fill the README Results
> rows "Decisions taken" and "Datasets regenerated or kept".
>
> **Refresh at start:** confirm on `main` that T14 demoted `points` / `quadrature` / `norm`,
> and check what is still left behind the `chebyshev` gate.

## Read first
- Design §12 decision 12 in full. It chose RLST's FFTW-backed `chebychev` transforms for
  reuse, not speed, and the closed forms "become the test oracles". Decision 6
  (requirement 3), and design §4 (solver A, fast diagonalisation).
- Review §5 S4 (last bullet), §7 step C3.
- `chebyshev/differentiation.rs`:
  - `diff_matrix` :18 has no caller outside its own tests.
  - `diff2_matrix` :31 is used only by `pde/poisson/collocation.rs:20, 141`.
  - `derivative_matrix` :37.
- `chebyshev/mod.rs:49-54` `unit_coeffs`, used by `quadrature` (now a test oracle) and
  `differentiation`.
- `Cargo.toml`: the `chebyshev` feature at :91-101 (`rlst/fftw`, `rlst/fftw_system`) and
  its comment at :82-90.
- `.github/workflows/run-tests.yml`, the `run-tests-chebyshev` apt step at :114-117.
- `pde/poisson/collocation.rs`: `residual_at_round_off` :365 (scaled residual ≤ 1e-14 for
  n ∈ {9…129}).

## Decision: replace RLST's FFTW-based D and D² with Trefethen's closed form?

| Option | Consequence |
| --- | --- |
| **Accept** (recommended) | `diff_matrix` becomes Trefethen's `cheb` (off-diagonal closed form, diagonal from the negative row sum), and `diff2_matrix` = D·D. Removes the remaining FFTW use: `rlst/fftw`, `rlst/fftw_system` and `unit_coeffs`, so no build links GPL code and CI needs no `libfftw3-dev`. RLST stays for `eig`, BLAS and LAPACK (solver A). Labels change at round-off. |
| Decline | Nothing changes. T16 then ungates everything except `differentiation.rs`. |

**Recommendation: accept**, on the condition that `residual_at_round_off`, the manufactured
1e-12 floor and the 1e-8 label check pass with the closed form. Decision 12's reason was
reuse of tested transforms, and only one matrix still depends on it. The closed form is
already the test oracle.

- **What to regenerate:** stage 1 (n = 33) into a scratch directory, to compare the label
  check and error 2.
- **Keep the existing datasets** if error 2 agrees to about 1e-12 (a round-off difference).
- If it doesn't, stop and report to the author before going further.

## Deliverables (if accepted)
1. **`diff_matrix(n)`** in closed form: Trefethen, *Spectral Methods in MATLAB*, `cheb`,
   adapted to ascending nodes (CONVENTIONS §12).
   - The diagonal comes from the negative row sum, for accuracy.
   - Document the sign convention for ascending nodes.
2. **`diff2_matrix(n)` = D·D.**
3. **Delete** `derivative_matrix` and `unit_coeffs`.
   - If `quadrature.rs` (the T14 test oracle) still needs `unit_coeffs`, move it under
     `#[cfg(test)]` with the oracle, or remove the RLST-weights oracle test and keep
     `clencurt` against its exact-integral tests. State which.
4. **`Cargo.toml`:**
   - Drop `rlst/fftw` and `rlst/fftw_system` from the `chebyshev` feature.
   - Rewrite the comment at :82-90: there is no FFTW and no GPL; RLST is used for dense
     linear algebra.
5. **CI:** drop `libfftw3-dev` from the `run-tests-chebyshev` apt step. Keep `pkg-config`
   only if the OpenBLAS discovery still needs it.
6. **CLAUDE.md:** "behind the `chebyshev` feature (need system libfftw3)" becomes the new
   requirement (OpenBLAS on Linux, Accelerate on macOS).

## Tests
- **Add** `diff_matrix_matches_rlst`: a gated oracle comparing the new closed form against
  the old RLST route, kept for one phase and then deletable. Or keep the existing
  closed-form oracle tests, which already compare against Trefethen; check which exists.
- `residual_at_round_off`, the manufactured floor and the label check all pass.
- Give the test count. Any oracle tests deleted with `unit_coeffs` are listed.

## Acceptance
- Stage 1 regenerated in a scratch directory. Report the label check and error 2 against
  the stored sidecar (agreement expected to about 1e-12). Results rows filled.
- On this Mac, `cargo tree --features chebyshev -e normal | grep -i fftw` finds nothing.
- The chebyshev CI job is green without `libfftw3-dev`.

## Checks to run and report
The six from the README **plus the chebyshev pair**. Give the line delta.

## Do not
- Replace RLST's `eig` or BLAS (that is T25, optional), or change solver A's algorithm.
- Ungate anything (T16).
- Overwrite `datasets/poisson/` without the author's answer to the regeneration question.

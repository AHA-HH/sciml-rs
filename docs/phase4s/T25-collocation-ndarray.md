# T25 (optional): `collocation.rs` on ndarray after `eig`

Phase 4s, task 25, **optional**. Branch `phase4s/T25-collocation-ndarray`. Depends on T16.
It can run any time after that, in parallel with Groups 5–6, and is not needed to close the
phase.

> **Refresh at start:** re-read `collocation.rs` and `chebyshev::apply` on `main`, and check
> the renamed feature (T16).

## Read first
- Review §6 T6-5.
- Design §4 (solver A, fast diagonalisation) and §12 decision 12.
- `pde/poisson/collocation.rs`:
  - the ndarray↔RLST copies `to_dyn` :227 and `from_dyn` :237, and the owned `transpose`
    :244
  - their uses at :85-86, :107, :115, :170 and the test :443
  - the A·F·Bᵀ products at :107 (`vx_inv · F · vy_inv_t`) and :115 (`vx · Û · vy_t`)
- The other A·F·Bᵀ site in production: `pde/poisson/grf.rs:153`
  (`sine_basis(xs,k)·c·sine_basis(ys,k)ᵀ`). `sparse.rs:176` is test-only after T6.
- `chebyshev/transfer.rs:164` (`apply(tx, ty, f) = tx · f · tyᵀ`).

## Goal
Convert solver A's eigenvector matrices to ndarray once, after `eig`, so the per-sample
solve is plain ndarray. Route every A·F·Bᵀ through `chebyshev::apply`. **Do this only if it
is not slower.**

## Deliverables
1. **Time it first:**
   - Measure the current n = 257 generation: one split, or a fixed number of samples (state
     how many), in `--release`, as the median of 3 runs.
   - Write the measurement into the PR before changing code.
   - If, after the change, the ndarray path is slower by more than the noise (about 5%),
     **stop**, report the numbers, and close the task as skipped. The README row says
     "skipped, slower".
2. **In `CollocationSolver::new`:** after `eig`, convert `vx`, `vx_inv`, `vy_t` and
   `vy_inv_t` to `Array2<f64>` once.
   - `solve` uses `chebyshev::apply`, or ndarray `dot` where `apply`'s argument order
     doesn't fit.
   - Delete `to_dyn`, `from_dyn` and `transpose`.
3. **`grf.rs:153`** uses `apply`.

## Tests
- `residual_at_round_off`, the manufactured 1e-12 floor and the label check pass unchanged.
- Add a test that the solution matches the old RLST path to ≤ 1e-14 relative on n ∈ {9, 33}:
  run both paths in the test before deleting the old one, and keep the comparison if it is
  cheap.

## Acceptance
- Report the timings before and after.
- Generate stage 1 (n = 33) into a scratch directory. The label check and error 2 match the
  stored sidecar at round-off.

## Checks to run and report
The six from the README **plus the feature pair**. Give the line delta.

## Do not
- Replace `eig` or the BLAS provider.
- Change solver A's algorithm or tolerances.

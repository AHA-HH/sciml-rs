# T1: Chebyshev toolkit

Phase 1, task 1. Branch `phase1/T1-chebyshev-toolkit`. Depends on nothing in this phase.

## Read first
- `docs/CONVENTIONS.md` §12 (nodes, 'ij' layout, Clenshaw–Curtis norm, its planned test
  names).
- Design §3.2, §3.4, §4.3 (the D^(2) check), §9 (the `chebyshev` feature and FFTW), §11
  (rows for nodes, D/D², Clenshaw–Curtis), §12 decision 12.
- RLST 0.9.0: `src/interpolation.rs` (`chebychev_points`, descending) and
  `src/chebychev.rs` (`ChebychevCoefficientsToData`, `DerivativeOfChebychevSeries1d`;
  both work on data in RLST's descending node order).
- `spikes/solver/REPORT.md` §1 and `spikes/solver/src/cheb.rs`;
  `spikes/transfers/src/grids.rs` (`cheb_nodes`, `clenshaw_curtis`, `wnorm`, `rel_l2`).
  These are now the closed-form test oracles, not the implementation.
- `src/neural_operators/data/grids.rs` for the crate's existing grid style.

## Goal
The 1D building blocks every later Chebyshev task uses, in f64, with ndarray types,
built on RLST's FFTW-backed Chebyshev transforms behind the `chebyshev` feature
(design decision 12).

## Deliverables
- `Cargo.toml`: `chebyshev` also enables `rlst/fftw` and `rlst/fftw_system`; the comment
  above it says FFTW (GPL-2.0+) comes from the system libfftw3 via pkg-config and enters
  only builds with this feature.
- `.github/workflows/run-tests.yml`, job `run-tests-chebyshev`: install
  `libfftw3-dev pkg-config` next to OpenBLAS. The default job is unchanged.
- New module `src/neural_operators/chebyshev/`, registered in `neural_operators/mod.rs`
  behind `#[cfg(feature = "chebyshev")]`:
  1. `nodes(n) -> Array1<f64>`: `chebychev_points(Kind::Second, n)` reversed, so the CGL
     nodes x_j = −cos(πj/(n − 1)) are ascending, with both endpoints (CONVENTIONS §12).
  2. `diff_matrix(n) -> Array2<f64>` and `diff2_matrix(n) -> Array2<f64>`: column j is the
     unit vector e_j (in RLST's descending order) taken through
     `chebychev_coeffs_from_data_second_kind`, `chebychev_derivative_second_kind(order)`
     and `chebychev_data_from_coeffs_second_kind`; the result is then reordered to
     ascending nodes, D_asc[i][j] = D_desc[n−1−i][n−1−j] (no sign change). If the row-sum
     test shows rows of D not summing to zero to round-off, reset the diagonal by the
     negative-sum trick and document it.
  3. `clenshaw_curtis(n) -> Array1<f64>`: w_j = Σ_k c_k(e_j) · I_k with I_k = 2/(1 − k²)
     for even k and 0 for odd k, reordered to ascending nodes. Exact for polynomials of
     degree ≤ n − 1, summing to 2.
  4. A Chebyshev-grid norm, hand-written (RLST has none): `l2_norm(v, wx, wy)` and
     `rel_l2_error(a, b, wx, wy)` for `[n_x, n_y]` fields, tensor-product weights (design
     §3.4). Phase 1 T2–T3 and the Phase 2 label check reuse them.

Every function panics for n < 2 and documents it, with the shape of every array argument
(CLAUDE.md, rustdoc style). Cite `CONVENTIONS §12` in the doc comments. Check n = 2 and
n = 3 through RLST's transforms first; if either fails, raise the documented minimum n
here and in the design §11 row.

## Tests (`#[cfg(test)] mod tests` in each file)
- `cgl_nodes_ascending_match_rlst_reversed`: `nodes(n)` against the closed form
  −cos(πj/(n − 1)), 1e-14 absolute, n ∈ {2, 3, 9, 33, 257}.
- `clenshaw_curtis_exact_on_polynomials`: ∫ x^k for k ≤ n − 1 to 1e-14, n ∈ {3, 9, 33, 65}.
- `diff_matrix_matches_closed_form`: every entry against Trefethen's `cheb` formulas on
  the ascending nodes, max error relative to ‖D‖_max ≤ 1e-10 (design §11),
  n ∈ {2, 3, 9, 33, 65}; and rows of D sum to zero within 1e-10 · ‖D‖_max.
- `d2_exact_on_polynomials`: D² applied to x^k, k < n, n ∈ {9, 17, 33, 65}. Measure the
  error relative to ‖D²‖_∞ · ‖x^k‖_∞, and require ≤ 1e-10. (T4 measured the error relative
  to the value: it grows to 7.2e-10 at n = 65, which a pointwise-relative test would fail
  for no numerical fault.)
- `l2_norm_of_constant_is_area`, and the norm on a separable polynomial against its exact
  integral.

## Acceptance
- All tests pass under `cargo test --features chebyshev`, locally and in the
  `run-tests-chebyshev` CI job.
- Plain `cargo test` (no feature) still builds and runs, and needs no FFTW on the
  machine.
- rlst types appear only inside the module; the public API is ndarray.
- One line in the Layout section of `CLAUDE.md` names the module and its feature gate.
- The Results row "D and D² tolerances used" in `docs/phase1/README.md` is filled in.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`, then
`cargo clippy --no-deps --features chebyshev --all-targets -- -D warnings` and
`cargo test --features chebyshev`. Give the exact test filters with full module paths and
check the "running N tests" lines.

## Do not
- Put rlst types in the public API.
- Enable `rlst/fftw_source`, `fftw_mkl`, `burn` or `mpi`, or make FFTW part of the default
  build.
- Add transfers (Phase 2 T1), the GRF (Phase 2 T2) or any solver (T2, T3).
- Change `data/grids.rs` or CONVENTIONS.

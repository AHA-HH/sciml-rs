# T1: grid transfers

Phase 2, task 1. Branch `phase2/T1-transfers`. Depends on Phase 1 T1 (nodes, for the
oracle tests).

## Read first
- `docs/CONVENTIONS.md` §12: the uniform grid, both transfers, their weights, the
  node-coincidence rule and the planned test names.
- Design §6.2 (transfers), §11 (the two transfer rows), §12 decisions 5, 8 and 13.
- `spikes/transfers/REPORT.md`: Setup, Acceptance checks, tables 1–4 and 6;
  `spikes/transfers/src/{grids,interp}.rs`.
- rlst 0.9.0 `src/interpolation.rs`: `barycentric_chebychev_weights` and
  `barycentric_evaluate_1d`, both in RLST's descending node order.
- `src/neural_operators/chebyshev/mod.rs` and `src/neural_operators/mod.rs` (the
  current feature gating).

## Goal
The two fixed, dense 1D transfer matrices per (n, s) that move fields between the
Chebyshev grid and the model's uniform grid, in f64, available in the default build.

## Deliverables
1. **Gating.**
   - `neural_operators/mod.rs` registers `chebyshev` without a feature gate.
   - In `chebyshev/mod.rs`, `points`, `differentiation`, `quadrature`, `norm`, their
     `pub use` lines and the RLST helpers move behind `#[cfg(feature = "chebyshev")]`.
     Their code does not change.
   - Update the module doc to say which parts need the feature.
   - `cargo build` with no `chebyshev` must still link no RLST, BLAS or FFTW.
2. **New ungated module `src/neural_operators/chebyshev/transfer.rs`**, re-exported from
   `chebyshev`:
   - `uniform_nodes(s) -> Array1<f64>`: x̃_j = −1 + 2j/(s − 1), both endpoints exact.
   - `cheb_to_uniform(n, s) -> Array2<f64>`, `[s, n]`: polynomial barycentric
     interpolation on the CGL nodes, with second-kind weights w_j = (−1)^j δ_j. The CGL
     nodes come from a `pub(crate)` closed form, −cos(πj/(n − 1)), ascending, because
     `chebyshev::nodes` needs RLST. T3's loader uses it to validate node arrays.
   - `uniform_to_cheb(s, n, d) -> Array2<f64>`, `[n, s]`: Floater–Hormann of degree d on
     the uniform nodes, with the weights of CONVENTIONS §12.
   - In both, a target equal to a node (bitwise, as the spike does) takes that node's
     value, so the row is a unit vector.
   - `apply(tx, ty, f) -> Array2<f64>`: Tx F Tyᵀ for `f: [n_x, n_y]`, 'ij'. Rectangular
     inputs work.
   - Production code uses d = 2 and s = n − 1 (design decision 8). The functions take d
     and s as arguments so the tests can vary them; no constant changes d or s.
3. Document the shapes and panics of every public item: n < 2, s < 2, d > s − 1, and
   mismatched shapes in `apply`. Cite `CONVENTIONS §12`.
4. One line in the Layout section of `CLAUDE.md`: `chebyshev/transfer` is ungated.

## Tests (`#[cfg(test)] mod tests` in `transfer.rs`)
Ungated unless marked. The RLST-oracle tests carry `#[cfg(feature = "chebyshev")]`.
- The planned names of CONVENTIONS §12:
  - `uniform_grid_has_endpoints_and_n_minus_1_points`: s = n − 1 points, exact ±1,
    ascending, uniform spacing to 1e-15.
  - `cheb_bary_weights_match_closed_form` (gated): the weights against RLST's
    `barycentric_chebychev_weights(Kind::Second, n)` reversed, after scaling both to
    w_0 = 1, to 1e-14, for n = 2, 3, 9, 33, 257.
  - `cheb_to_uniform_matches_rlst_barycentric` (gated): T_cu f against
    `barycentric_evaluate_1d` at the uniform nodes, for a smooth f, to 1e-12 relative,
    for n = 33, 65, 129.
  - `fh_reproduces_polynomials_to_degree_d`: T_uc reproduces x^k, k ≤ d, to 1e-12, for
    d = 0..4 and s = 16, 32, 64.
  - `transfers_preserve_ij_layout`: on a rectangular grid, `apply` of f(x, y) = x + 10y
    lands x on axis 0 and y on axis 1, in both directions.
- `cgl_closed_form_matches_nodes` (gated): the `pub(crate)` closed-form nodes equal
  `chebyshev::nodes(n)` to 1e-14.
- `cheb_to_uniform_on_analytic_functions`: sin(πx) sin(πy) and
  (1 − x²)(1 − y²)e^(x + 2y) at n = 33, 65, 129, against the exact values on the
  uniform grid, within 1e-12 relative (design §11; Phase 0: ≤ 1.0e-15).
- `fh_degree_two_rate`: on sin(πx) sin(πy), the CC-weighted error of T_uc against the
  exact values, at n = 33, 65, 129. The observed order log₂(e_n / e_{2n−1}) lies in 3 ± 0.5
  for both steps. CC weights need RLST, so either gate this test or use the trapezoidal
  rule on the Chebyshev nodes and say which.
- `round_trip_matches_spike`: T_uc(T_cu u) − u on sin(πx) sin(πy) at n = 65, d = 2. The
  result is within 2× of the Phase 0 table (1.2e-5).
- `lebesgue_constant_bounded`: the maximum row sum of |T_uc| at d = 2 is ≤ 5 for
  n = 33, 65, 129, 257 (Phase 0: ≤ 4.87). Record the values.
- Panics: one `#[should_panic]` test per documented panic.

Keep the tests fast. n = 257 is allowed for the Lebesgue test: matrix construction is
0.1 ms in release.

## Acceptance
- The ungated tests pass under `cargo test` and the gated ones under
  `cargo test --features chebyshev`, on macOS and in both CI jobs.
- `cargo build` and `cargo clippy` with default features compile `chebyshev::transfer`
  and nothing else from `chebyshev`.
- The three T1 rows of Results in `docs/phase2/README.md` are filled in.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`, then the
same clippy and test commands with `--features chebyshev`. Give the exact test filters and
check the "running N tests" lines in both builds. The transfers are CPU-only host code, so
`--features metal` is not relevant; say so.

## Do not
- Change how `nodes`, `diff_matrix`, `diff2_matrix`, `clenshaw_curtis` or the norm
  compute their results. Only their gating changes.
- Change d or s, or add a choice of d at runtime outside the function arguments.
- Add the GRF, the generator or the loader (T2, T3).
- Use RLST in `transfer.rs` outside gated tests.

# T1: Chebyshev toolkit

Phase 1, task 1. Branch `phase1/T1-chebyshev-toolkit`. Depends on nothing in this phase.

## Read first
- `docs/CONVENTIONS.md` §12 (nodes, 'ij' layout, Clenshaw–Curtis norm, its planned test
  names).
- Design §3.2, §3.4, §4.3 (the D^(2) check), §11 (rows for nodes, D/D², Clenshaw–Curtis).
- `spikes/solver/REPORT.md` §1 and `spikes/solver/src/cheb.rs`;
  `spikes/transfers/src/grids.rs` (`cheb_nodes`, `clenshaw_curtis`, `wnorm`, `rel_l2`).
- `src/neural_operators/data/grids.rs` for the crate's existing grid style.

## Goal
The 1D building blocks every later Chebyshev task uses, in f64, with ndarray types and no
feature gate.

## Deliverables
New module `src/neural_operators/chebyshev/`, registered in `neural_operators/mod.rs`:
1. `nodes(n) -> Array1<f64>`: CGL nodes x_j = −cos(πj/(n − 1)), ascending, both endpoints,
   exactly ±1 at the ends and symmetric about 0 to round-off (CONVENTIONS §12).
2. `diff_matrix(n) -> Array2<f64>`: Trefethen's `cheb` on the ascending nodes, with the
   diagonal set by the negative-sum trick (rows sum to zero). A second function, or a
   documented one-liner, gives D² = D · D.
3. `clenshaw_curtis(n) -> Array1<f64>`: the weights on [−1, 1], exact for polynomials of
   degree ≤ n − 1, summing to 2.
4. A Chebyshev-grid norm: `l2_norm(v, wx, wy)` and `rel_l2_error(a, b, wx, wy)` for
   `[n_x, n_y]` fields, tensor-product weights (design §3.4). Phase 1 T2–T3 and the
   Phase 2 label check reuse them.

Every function panics for n < 2 and documents it, with the shape of every array argument
(CLAUDE.md, rustdoc style). Cite `CONVENTIONS §12` in the doc comments.

## Tests (`#[cfg(test)] mod tests` in each file)
- `cgl_nodes_ascending_match_rlst_reversed`: against RLST's
  `chebychev_points(Kind::Second, n)` reversed, 1e-14 absolute, n ∈ {2, 3, 9, 33, 257}.
  This one test is gated `#[cfg(feature = "chebyshev")]`; also add an ungated test against
  the closed form so the default job checks the nodes.
- `clenshaw_curtis_exact_on_polynomials`: ∫ x^k for k ≤ n − 1 to 1e-14, n ∈ {3, 9, 33, 65}.
- `diff_matrix_matches_closed_form`: the off-diagonal and corner entries of Trefethen's
  formulas, max error relative to ‖D‖_max ≤ 1e-10 (design §11), n ∈ {2, 3, 9, 33, 65}.
- `d2_exact_on_polynomials`: D² applied to x^k, k < n, n ∈ {9, 17, 33, 65}. Measure the
  error relative to ‖D²‖_∞ · ‖x^k‖_∞, and require ≤ 1e-10. (T4 measured the error relative
  to the value: it grows to 7.2e-10 at n = 65, which a pointwise-relative test would fail
  for no numerical fault.)
- `l2_norm_of_constant_is_area`, and the norm on a separable polynomial against its exact
  integral.

## Acceptance
- All tests pass under plain `cargo test` (the RLST oracle test is skipped there) and under
  `cargo test --features chebyshev`.
- The module has no rlst import outside the gated test.
- One line in the Layout section of `CLAUDE.md` names the module.
- The Results row "D and D² tolerances used" in `docs/phase1/README.md` is filled in.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`, then
`cargo clippy --no-deps --features chebyshev --all-targets -- -D warnings` and
`cargo test --features chebyshev`. Give the exact test filters with full module paths and
check the "running N tests" lines.

## Do not
- Put rlst types in the public API, or gate the module behind `chebyshev`.
- Add transfers (Phase 2 T1), the GRF (Phase 2 T2) or any solver (T2, T3).
- Change `data/grids.rs` or CONVENTIONS.

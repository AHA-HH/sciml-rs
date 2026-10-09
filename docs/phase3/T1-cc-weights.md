# T1: ungated Clenshaw–Curtis weights

Phase 3, task 1. Branch `phase3/T1-cc-weights`. Depends on Phase 2 only; can run in
parallel with T2.

## Read first
- Design §3.4 (norms), §7 (error 4), §11 (the closed-form Clenshaw–Curtis row), §12
  decisions 12, 13, 15 and 16.
- `docs/CONVENTIONS.md` §12.
- `src/neural_operators/chebyshev/mod.rs` (the gating, `check_n`), `quadrature.rs`
  (`clenshaw_curtis`, gated, the oracle), `norm.rs` (`l2_norm`, `rel_l2_error`, gated) and
  `transfer.rs` (`cgl_nodes`, the ungated model to follow).
- L. N. Trefethen, *Spectral Methods in MATLAB*, ch. 12, program `clencurt`.

## Goal
Clenshaw–Curtis weights and the relative CC-L² error that build without the `chebyshev`
feature, so that T3's evaluation, locally and on HPC, needs no FFTW (decision 15).

## Deliverables
New ungated module `src/neural_operators/chebyshev/clencurt.rs`, registered in
`chebyshev/mod.rs` and re-exported next to the transfers:
1. **`clencurt(n) -> Array1<f64>`**: the CGL quadrature weights on [−1, 1] in
   ascending node order (CONVENTIONS §12), from Trefethen's closed form, O(n²), ndarray
   only. It panics through `check_n` for n < 2.
2. **`cc_rel_l2_error(a, b) -> f64`**: ‖a − b‖ / ‖b‖ for `[n_x, n_y]` fields stored 'ij',
   with tensor-product `clencurt` weights along each axis (design §3.4), `b` the
   reference. It panics if the shapes differ or an axis has fewer than 2 nodes, and
   returns NaN or infinity when ‖b‖ = 0, as `rel_l2_error` does.
3. Doc comments citing design §3.4 and `CONVENTIONS §12`, with shapes and panics, and
   updates to the module doc of `chebyshev/mod.rs` (the "Always built" list). A Layout
   line in `CLAUDE.md`.

The gated `clenshaw_curtis`, `l2_norm` and `rel_l2_error` stay exactly as they are.

## Tests (`#[cfg(test)] mod tests` in `clencurt.rs`)
Ungated:
- `clencurt_integrates_polynomials`: Σ w_j x_j^k equals ∫ x^k on [−1, 1] for every
  k ≤ n − 1 and n = 2, 3, 9, 33, 65, to 1e-14 (design §11).
- `clencurt_is_symmetric_and_sums_to_two`: w_j = w_{n−1−j}, Σ w = 2.
- `cc_rel_l2_error_known_pair`: a = (1 + ε) b gives ε for a smooth b, and identical
  fields give 0.
- Panics for n < 2 and for mismatched shapes (`#[should_panic]` with the message).

Gated on `chebyshev`:
- `clencurt_matches_rlst_weights`: against `clenshaw_curtis(n)` for n = 2..=129, to
  1e-14 absolute (the §11 row).
- `cc_rel_l2_matches_gated`: equals `rel_l2_error` with `clenshaw_curtis` weights on
  a few fields, to 1e-14 relative.

## Acceptance
- All tests pass under `cargo test` and `cargo test --features chebyshev`, on macOS and
  in both CI jobs.
- The T1 rows of Results in `docs/phase3/README.md` are filled in (max difference per n,
  and the test wall times).

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`, then the
same clippy and test commands with `--features chebyshev`. Give the exact test filters
(`neural_operators::chebyshev::clencurt::tests::<name>`) and check the "running N tests"
lines. Host-only f64 code: `--features metal` is not relevant; say so.

## Do not
- Ungate, change or delete `clenshaw_curtis`, `l2_norm` or `rel_l2_error`.
- Depend on RLST, BLAS or FFTW outside the gated tests.
- Touch the loader, the training code or the examples (T2, T3).

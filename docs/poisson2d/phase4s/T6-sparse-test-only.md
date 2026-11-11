# T6: `SparseSolver` test-only; `rlst_smoke` deleted

Phase 4s, task 6. Branch `phase4s/T6-sparse-test-only`. Depends on T1. Can run fully in
parallel with any task. T14 waits for it, because `sparse.rs` uses the gated `nodes` that
T14 demotes to a test oracle.

## Read first
- Review §5 S2 ("Validation solver C", "GPU smoke" rows), §6 T2-9, T2-13, §7 step A5.
- Design §4.2 (why the Sylvester route is ruled out) and §12 decision 12.
- `src/neural_operators/pde/poisson/mod.rs` (`manufactured` is already
  `#[cfg(test)] mod manufactured;`, :11-12), `pde/poisson/sparse.rs`, `tests/rlst_smoke.rs`.

## Goal
Solver C is a validation oracle and nothing in production calls it. Make it test-only, as
`manufactured` already is, and delete a smoke test for a route the design rejected.

## Deliverables
1. **In `pde/poisson/mod.rs`:** `pub mod sparse;` becomes `#[cfg(test)] mod sparse;`.
   - Its public items `CG_TOL` :32, `SparseSolver` :45 (`new`, `max_iter`, `solve`),
     `SparseSolution` :68 and `CgNotConverged` :80 become `pub(super)`, or whatever the
     other test modules need.
   - Nothing outside `sparse.rs` uses them: the `CsrMatrix` in `tests/rlst_smoke.rs:13` is
     rlst's own.
2. **Delete `tests/rlst_smoke.rs`** (103 lines). Its tests are `cg_solves_1d_dirichlet_laplacian`,
   which duplicates `sparse::cg_meets_tolerance`, and `sylvester_route_solves_small_system`,
   which tests the route design §4.2 rules out.
3. **Docs:**
   - Fix doc comments that link to `SparseSolver` from public items. A public doc cannot
     link to a test-only item, so `cargo doc` would warn; check `collocation.rs` and
     `pde/poisson/mod.rs`.
   - CLAUDE.md "Layout": "`sparse` (`SparseSolver`, solver C, … validation oracle only)"
     becomes "test-only".

## Tests
- Every test in `sparse.rs` still runs under `cargo test --features chebyshev`. Give the
  count before and after.
- **Deliberate deletions:** the two `rlst_smoke` tests.

## Acceptance
- `cargo doc --no-deps --features chebyshev` gives no broken-intra-doc-link warnings.
- The prod line count drops by about 237.

## Checks to run and report
The six from the README **plus the chebyshev pair**. Give the line delta.

## Do not
- Change solver C's numerics or tolerances.
- Route `sparse.rs:176` through `chebyshev::apply`. That is T25's job.

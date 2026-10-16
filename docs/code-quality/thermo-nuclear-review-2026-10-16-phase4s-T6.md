# Thermo-nuclear code quality review: `phase4s/T6`

- **Repo:** sciml-rs
- **Branch:** `phase4s/T6` (working tree, uncommitted)
- **Base:** `79f9c62` (merge-base with `main`)
- **Date:** 2026-10-16
- **Diff:** 4 files, +20 / −121 (`tests/rlst_smoke.rs` deleted; `sparse.rs`, `poisson/mod.rs`,
  `CLAUDE.md` edited)
- **Task brief:** `docs/phase4s/T6-sparse-test-only.md`

## Verdict: APPROVE

This is a deletion-first change of the right shape. Solver C becomes `#[cfg(test)]`, the
way `manufactured` already is. That removes 237 lines of production code and a public
surface (`SparseSolver`, `SparseSolution`, `CgNotConverged`, `CG_TOL`) that nothing in
production used. A 103-line integration test for a route the design ruled out (§4.2) is
deleted. The change adds no branching or wrapper, crosses no layer, and pushes no file near
1k lines. All the findings below are advisory.

The one real "code judo" opportunity is collapsing the `Result<_, CgNotConverged>` contract
now that every caller is a test that panics on `Err`. The brief keeps that type explicitly,
so it goes to a follow-up, not a blocker.

## Presumptive blockers

None met.
- No file crosses 1000 lines.
- No ad-hoc branching or scattered feature checks.
- No new abstraction, wrapper or cast.
- No helper duplication or wrong-layer logic: the change only narrows visibility and
  deletes code.

## File sizes

| File | Before | After | Note |
|---|---:|---:|---|
| `src/neural_operators/pde/poisson/sparse.rs` | 439 | 440 | all 440 lines are now test-only |
| `src/neural_operators/pde/poisson/mod.rs` | 13 | 14 | |
| `CLAUDE.md` | 158 | 158 | |
| `tests/rlst_smoke.rs` | 103 | 0 | deleted |

No threshold crossings.

## Coverage after the deletion

`rlst_smoke` also checked that rlst's BLAS/LAPACK provider links on each platform. That
coverage survives:
- `collocation` calls LAPACK `eig` in production code and in its unit tests, which run under
  `cargo test --features chebyshev` on both CI and macOS.
- The `rlst` dependency is still required in production, by `collocation.rs` and by
  `chebyshev/{mod,points,differentiation}.rs`. No `Cargo.toml` change is needed.

The other "smoke test" mentions in `docs/` (design §12 decision 17, Phase 4s T3) refer to
other tests, so no current-state doc goes stale.

## Findings

### 1. Structural regressions

None.

### 2. Missed simplification (advisory, follow-up)

**`sparse.rs:80-101`, `:167`: the error type is now ceremony.**
- `solve` returns `Result<SparseSolution, CgNotConverged>`. That fitted a public API, where
  a caller might recover from non-convergence.
- Now the module is test-only, and every caller panics on `Err`:
  - `solve_c` uses `.unwrap_or_else(|e| panic!(...))`;
  - `cg_meets_tolerance` uses `.expect("CG")`.
- `impl std::error::Error for CgNotConverged` has no user, and no test exercises the `Err`
  path.
- **Remedy:** make `solve` return `SparseSolution` and panic with the existing `Display`
  message when `residual.is_nan() || residual > CG_TOL`. Document it under `# Panics`.
- **What it deletes:** the `CgNotConverged` struct, its `Display` and `Error` impls (about 25
  lines), and the unwrap at both call sites. The tolerance and stopping rule stay
  bit-identical.
- **Why it isn't in this PR:** the T6 brief lists `CgNotConverged` as an item to keep at
  `pub(super)`. Raise it with the author, or fold it into T25, which already reworks
  `sparse.rs`.

### 3. Spaghetti or branching growth

None.

### 4. Boundaries and type contracts (low)

**`sparse.rs:32-167`: `pub(super)` overstates the contract.**
- No sibling module uses any `sparse` item. The only consumer is the child `mod tests`, and
  a child module can already see its parent's private items.
- `pub(super)` copies `manufactured.rs`, but there it is needed: `collocation`'s tests import
  `manufactured`. Here it suggests a sibling user who doesn't exist.
- **Remedy (optional):** drop the qualifiers so the items are private. The brief allows
  either ("or whatever the other test modules need": nothing). Low conviction; keeping the
  pattern consistent with `manufactured` is a fair counter-argument.

### 5. File size and decomposition

None.

### 6. Modularity

None.

### 7. Legibility (nits)

- **`sparse.rs:58-65`: the manual `impl fmt::Debug for SparseSolver` is now dead weight.**
  - It existed so a public type wouldn't dump a CSR matrix in `{:?}` output.
  - No test formats a `SparseSolver`, and the crate sets no `missing_debug_implementations`
    lint.
  - Delete it with finding 2, or leave it: it costs 8 lines and nothing else.
- **`poisson/mod.rs:4` breaks the module's wrap width.** It's 100 characters against about
  92 in the paragraph around it. rustfmt doesn't rewrap comments, so `fmt --check` misses
  it. Rewrap lines 3–6.
- **`CLAUDE.md:62`** is slightly over the file's wrap width after "test-only" was added.
  Rewrap the bullet.

## Next steps

Each step preserves behaviour and stands on its own.

1. **Rewrap the two doc paragraphs** (`poisson/mod.rs:3-6`, `CLAUDE.md:60-62`).
   Fits in this PR. Verify with `cargo fmt -- --check` and
   `cargo doc --no-deps --features chebyshev` (no warnings).
2. **Collapse `CgNotConverged` into a panic** (finding 2). Optionally also delete the manual
   `Debug` impl and make the items private (findings 4 and 7).
   - Needs the author's sign-off, because the brief keeps the type. Best folded into T25,
     which already touches `sparse.rs`.
   - Verify:
     - all 8 `pde::poisson::sparse::tests::*` still pass under `cargo test --features chebyshev`
       (check "running 8 tests" with the `sparse::tests` filter);
     - `cargo clippy --all-targets --features chebyshev -- -D warnings` is clean;
     - the solver's numerics are unchanged (same `CG_TOL` and iteration cap).

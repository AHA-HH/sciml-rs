# T16: `chebyshev/` fully ungated; the feature renamed

Phase 4s, task 16. Branch `phase4s/T16-ungate-chebyshev`. Depends on T15. Must merge before
Phase 6 T3 (the θ-map, which adds code under `chebyshev/`). T25 (optional) comes after.

> **Refresh at start:**
> - Read T15's decision. If T15 was declined, only `differentiation.rs` stays gated
>   (see below).
> - Count the `feature = "chebyshev"` lines left on `main`. At `6eb3212` there were 11 in
>   `chebyshev/mod.rs`, 2 in `clencurt.rs` (tests), 1 in `transfer.rs` (tests), 1 in
>   `neural_operators/mod.rs` (`pub mod pde`) and 3 in `lib.rs` (:38-46).

## Read first
- Review §5 S4, §6 T3-3, §7 step C4.
- Design §12 decision 24 (the Phase 6 θ-map lives in `chebyshev/`).
- `src/lib.rs:38-46`: the `blas_src` / `lapack_src` extern crates and the OS
  `compile_error!` are gated on the feature.
- `Cargo.toml` `[features]` and `[[example]] generate_poisson` (`required-features`,
  :68).
- `.github/workflows/run-tests.yml` (the `run-tests-chebyshev` job).

## Goal
The feature is named after a module, but it actually means "RLST + BLAS/LAPACK + the
reference solvers + the dataset generator". After T14 and T15, the `chebyshev/` module needs
none of that. Ungate it, and name the feature after what it enables.

## Deliverables
1. **Ungate `chebyshev/`.** All production items in it build without features.
   - Any remaining RLST cross-check oracles stay `#[cfg(all(test, feature = "<new>"))]`.
   - If T15 was declined, `differentiation.rs` alone stays gated, with a one-line reason.
2. **Rename the feature** (`datagen` suggested; `reference-solver` is the alternative).
   Pick one with the author when the task starts.
   - It now gates `pde/` (`neural_operators/mod.rs:9`) and the `lib.rs` extern crates and
     OS check.
   - Update every `cfg`, `required-features`, the CI job name and command, and every doc
     (CLAUDE.md "Layout", "Checks", "Build environment", "Commands";
     `docs/phase4s/README.md`'s "Checks for every task").
3. **Consider an alias** `chebyshev = ["<new>"]` for one phase, so old commands still
   work. Recommended: no alias, because nothing outside the repository uses the name.
   Record the choice.
4. **`lib.rs`'s crate docs:** a single line saying what the renamed feature enables. T24
   does the full rewrite.

## Tests
- `cargo test` (no features) now runs every `chebyshev/` test that was gated only because
  of the module gate. The test count **goes up**; list the newly ungated tests.
- `cargo test --features <new>` runs the `pde/` tests and the RLST oracles.

## Acceptance
- `git grep -n 'feature = "chebyshev"'` finds nothing, or exactly the `differentiation.rs`
  gate if T15 was declined.
- The README Results row "`chebyshev` feature: what it gates after the phase, and its new
  name" is filled.
- The renamed CI job is green.

## Checks to run and report
The six from the README, plus `cargo test --features <new>` and
`cargo clippy --all-targets --features <new> -- -D warnings`. Give the line delta.

## Do not
- Move `pde/` out of the crate, or split the crate.
- Change any numerics.

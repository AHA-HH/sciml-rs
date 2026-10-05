# 2d-chebyshev-poisson-fno phase 0 / T2 - optional `chebyshev-data` feature with RLST (C0.2)

Can start once design §9.2 Q5 (the RLST pin) is confirmed. This brief assumes the recommendation,
`rlst = "=0.6.1"`. If the user chose the git rev `3153a14` instead, use
`rlst = { git = "https://codeberg.org/rlst/rlst", rev = "3153a14" }` and otherwise do the same.

Read first:
- `CLAUDE.md`: Checks, Working rules, Build environment.
- `docs/design/2d-chebyshev-poisson-fno.md`: §5.1, §5.4, §6.1, §9.1.
- `Cargo.toml`, `src/neural_operators/mod.rs`, `.github/workflows/run-tests.yml`.
- `~/Code/rlst/rlst/Cargo.toml:49-56`: RLST's own BLAS/LAPACK set-up.

This task adds an optional cargo feature `chebyshev-data` that brings in RLST and a BLAS/LAPACK
provider, plus an empty `chebyshev` module. Phases 1–3 put the solver, transfers and evaluation in
that module.
- The default build must stay exactly as it is.
- CI gains one job that builds and tests the feature on Linux with OpenBLAS.

Do:
- `Cargo.toml`:
  - Add the feature: `chebyshev-data = ["dep:rlst", "dep:blas-src", "dep:lapack-src", "dep:openblas-src"]`.
    `openblas-src` is only declared for Linux. If Cargo rejects `dep:` for a target-only optional
    dependency on macOS, use the form that works on both and record what was needed.
  - `[dependencies]`: `rlst = { version = "=0.6.1", optional = true }`, with default features only.
    No `fftw*`, `burn` or `mpi` features.
  - `[target.'cfg(target_os = "macos")'.dependencies]`: `blas-src = { version = "0.14", features = ["accelerate"], optional = true }` and `lapack-src = { version = "0.13", features = ["accelerate"], optional = true }`.
  - `[target.'cfg(target_os = "linux")'.dependencies]`: `openblas-src = { version = "0.10", features = ["system"], optional = true }`, plus `blas-src` and `lapack-src` with feature `openblas`, both optional.
  - Leave `default = ["flex"]` unchanged.
  - Add a comment above the feature explaining:
    - it is for data generation and evaluation only;
    - it needs a system BLAS/LAPACK on Linux (`libopenblas-dev`);
    - FFTW is deliberately not enabled (GPL, design §6.1).
- `src/neural_operators/mod.rs`: add `#[cfg(feature = "chebyshev-data")] pub mod chebyshev;`, with a
  one-line doc comment.
- `src/neural_operators/chebyshev/mod.rs`:
  - A module doc comment that says what will live here (design §5.1 table) and cites
    CONVENTIONS §9. If T1 is not merged yet, cite design §2.8 instead.
  - Link the providers with `use blas_src as _; use lapack_src as _;` (and `openblas_src` on Linux),
    so the symbols are linked.
  - A `#[cfg(test)] mod tests` with the smoke test below.
- `.github/workflows/run-tests.yml`: add a job `chebyshev-data` on `ubuntu-latest`. It:
  - uses the same toolchain and cache steps as `run-tests`;
  - runs `sudo apt-get update && sudo apt-get install -y libopenblas-dev`;
  - then runs `cargo test --features chebyshev-data`,
    `cargo clippy --no-deps --features chebyshev-data -- -D warnings` and
    `cargo doc --no-deps --features chebyshev-data`.

  Leave the existing jobs untouched.
- `CLAUDE.md`, Checks section: after the CI command block, add a sentence and a block listing the
  three feature commands, and say they run in the `chebyshev-data` CI job. Change nothing else in
  `CLAUDE.md`.
- PR description:
  - list the RLST APIs the design cites (design §6.1, "Used") with their file:line **in the pinned
    0.6.1 source** (`~/.cargo/registry/src/*/rlst-0.6.1/src/...`);
  - flag any that differ from the `3153a14` line numbers or are missing. These go into design §6.1.

Tests that define done (in `chebyshev/mod.rs`, run only with the feature):
- `rlst_chebychev_points_are_descending_cos_form`: `rlst::interpolation::chebychev_points::<f64>(Kind::Second, 5)`.
  Use the actual module path from 0.6.1.
  - Element 0 equals 1.0 exactly and element 4 equals −1.0 exactly.
  - Element 1 is within 2·f64::EPSILON (absolute) of `(PI/4.0).cos()`.
  - Element 2's absolute value is ≤ 1e-16, not required to be 0. This pins the behaviour that
    design §2.2 relies on when it rejects these nodes.
- `rlst_lapack_links`: build the 2×2 real matrix [[2, 1], [1, 3]] as an RLST dense array and call
  `eig` (eigenvalues only). LAPACK does not order the eigenvalues, so sort by real part. Then
  compare with the closed form (5 − √5)/2 and (5 + √5)/2: real parts within 1e-14 relative to the
  closed form, imaginary parts exactly 0. This proves LAPACK is linked.
  - In 0.6.1, `eig` is at `dense/linalg/lapack/eigenvalue_decomposition.rs:84`, not at :116 as in
    `3153a14`. The PR's line re-read must catch differences like this one.
- Default-build invariance: `cargo test` without the feature runs the same number of tests as on
  `main`. Compare the "running N tests" lines and state both counts in the PR.

Must pass: the full checks from `CLAUDE.md`:

```sh
cargo fmt -- --check
cargo clippy --no-deps -- -D warnings
cargo clippy --no-deps --examples -- -D warnings
cargo test
cargo doc --no-deps
cargo clippy --all-targets -- -D warnings
```

plus, on macOS locally and in the new CI job:

```sh
cargo test --features chebyshev-data
cargo clippy --no-deps --features chebyshev-data -- -D warnings
cargo doc --no-deps --features chebyshev-data
```

Report which of these ran where, and whether `--features metal,chebyshev-data` builds. Building it
is enough; there are no Metal-specific tests.

Do not:
- enable any RLST feature, including `fftw`, `fftw_source`, `burn` and `mpi`;
- add solver, nodes, transfer or quadrature code (Phases 1–2);
- change default features, existing dependencies, existing CI jobs, or anything in `CLAUDE.md`
  outside the Checks section;
- commit `Cargo.lock`.

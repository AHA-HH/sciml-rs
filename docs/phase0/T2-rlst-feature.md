# T2: RLST behind an optional feature

Phase 0, task 2. Branch `phase0/T2-rlst-feature`. Depends on nothing.

## Read first
- Design §1.3 (RLST findings), §9 (dependencies), §12 decisions 1 and 6.
- `Cargo.toml`, `.github/workflows/run-tests.yml`.
- rlst 0.9.0 sources (`~/.cargo/registry/src/*/rlst-0.9.0/`): `Cargo.toml:236–254` (how
  RLST's own tests pick a BLAS provider), `src/doc/getting_started.rs:68–87`,
  `src/traits/linalg/decompositions.rs`, `src/operator/algorithms/conjugate_gradients.rs`,
  `src/sparse/csr_mat.rs`.

## Goal
Make `cargo build --features chebyshev` link RLST and a BLAS/LAPACK provider on macOS and
Ubuntu. Leave the default build byte-for-byte as it is. Find out exactly which RLST APIs
the later phases can rely on.

## Deliverables
1. **`Cargo.toml`.**
   - Feature `chebyshev = ["dep:rlst", "dep:blas-src", "dep:lapack-src", "dep:openblas-src"]`.
   - `rlst = { version = "0.9", optional = true, default-features = false }`.
   - Target-specific optional dependencies, following RLST's own dev-dependencies:
     - macOS: `blas-src` 0.14 and `lapack-src` 0.13 with `accelerate`;
     - Linux: `blas-src` 0.14 and `lapack-src` 0.13 with `openblas`, plus
       `openblas-src` 0.10 with `system`.
   - If optional target-specific dependencies cannot be named in one feature, find the
     working form, and say what was done and why in the PR.
2. **Provider linking.** The provider crates only need to be linked. Add
   `#[cfg(feature = "chebyshev")] extern crate blas_src; extern crate lapack_src;` (or the
   equivalent the crates document) in one place, with a comment saying why.
3. **CI.** A new job, `run-tests-chebyshev`, on `ubuntu-latest`:
   - `sudo apt-get install -y libopenblas-dev`;
   - `cargo clippy --no-deps --features chebyshev --all-targets -- -D warnings`;
   - `cargo test --features chebyshev`.

   The existing jobs are unchanged.
4. **Smoke tests, `tests/rlst_smoke.rs`.** The whole file is gated with
   `#![cfg(feature = "chebyshev")]`; it is its own test binary.
   - `cg_solves_1d_dirichlet_laplacian`:
     - assemble the n = 50 matrix tridiag(−1, 2, −1) with `CsrMatrix::from_aij`;
     - solve A x = 1 with `CgIteration`, tolerance 1e-12;
     - compare with the exact discrete solution x_i = (i + 1)(n − i)/2 (i = 0..n − 1);
     - require relative max error ≤ 1e-9;
     - assert on the returned residual, because `run()` returns no converged flag.
   - `sylvester_route_solves_small_system`:
     - for seeded random 8 × 8 matrices A, B and C, solve A X + X B = C through the route
       that exists in the pinned version: `solve_sylvester` if it is public, otherwise
       eigendecompositions of A and B;
     - require ‖A X + X B − C‖_F / ‖C‖_F ≤ 1e-12;
     - check `status()` and `scale()` when the Sylvester routine is used.
5. **API findings.** A section of the PR description listing, with file:line in the
   pinned rlst version:
   - whether `solve_sylvester`, `eig`, `eigh`, LU solve, `CgIteration`, `GmresIteration`,
     `CsrMatrix::from_aij` and the `interpolation` module are public and usable without
     FFTW;
   - their exact signatures;
   - whether any of them needs a feature.

   Copy the two Results rows ("rlst version pinned", "Public Sylvester routine") into
   `docs/phase0/README.md`.

## Acceptance
- Default build: `cargo tree -e normal` shows no rlst, blas or lapack crate without the
  feature; every CI check from `CLAUDE.md` passes unchanged.
- `cargo test --features chebyshev` passes locally on macOS (Accelerate) and in the new CI
  job on Ubuntu (OpenBLAS).
- `cargo clippy --no-deps --features chebyshev --all-targets -- -D warnings` is clean.
- Neither the feature nor the lock resolution enables any rlst feature (`fftw*`, `burn`,
  `mpi`): `cargo tree --features chebyshev -e features -i rlst`.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`, then the
same with `--features chebyshev`. Say which ran on macOS and which only in CI.

## Do not
- Enable any rlst feature, or depend on `rlst-suitesparse`.
- Touch `src/neural_operators/` beyond the provider-linking lines.
- Change existing CI jobs.
- Write any Chebyshev or Poisson code (that is Phase 1).

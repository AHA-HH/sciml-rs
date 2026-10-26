# T2: the restriction in the library, and CONVENTIONS §12

Phase 5, task 2. Branch `phase5/T2-restriction`. Depends on T1 (D1: the chosen R, its
parameters, and P's d).

## Read first
- `spikes/transfer_floor/REPORT.md` (T1) and decision 30's D1 amendment in design §12.
- `docs/CONVENTIONS.md` §9 (versioning) and §12.
- `src/neural_operators/chebyshev/transfer.rs`: the whole file, including its tests.
- `src/lib.rs` (`CONVENTION_VERSION` and its doc-sync test).
- Design §13.3 (restriction, prolongation).

## Goal
Make the restriction T1 chose a library function, and let `Transfers` serve any learning
grid s, with today's s = n − 1 transfers unchanged.

## Deliverables
1. **Convention first** (CONVENTIONS §9: the convention lands before the code that relies
   on it, in the same PR). A diff to CONVENTIONS §12:
   - the uniform size s is chosen per dataset load, s ≥ some minimum fixed by the code
     (at least d + 2), with s = n − 1 the default;
   - the restriction R (the D1 method, its formula and parameters), applied to f and u
     alike;
   - P = T_uc with T1's d (unchanged at d = 2 if T1 kept it);
   - the new tests in the §12 Verification list.

   Bump `CONVENTION_VERSION` to 2 in `CONVENTIONS.md` and `src/lib.rs`. The PR
   description states the change and the bump (CLAUDE.md "Source of truth").
2. **In `chebyshev/transfer.rs`** (ungated, ndarray, f64), only what D1 needs:
   - the closed-form nodal-to-Chebyshev-coefficient matrix, if R is the filtered
     restriction;
   - a builder for the chosen R as an `[s, n]` matrix, documented like
     `cheb_to_uniform` (shapes, panics);
   - a public enum naming the restriction (the chosen R and unfiltered T_cu, which the
     T5 aliasing control needs), serialisable, since T3 stores it in `data_cfg.json`;
   - `Transfers` for an independent (n, s, restriction, d). `Transfers::new(n)` keeps
     its signature and returns exactly today's matrices (s = n − 1, T_cu, d = 2).
3. If T1 changed d, `TRANSFER_DEGREE` stays the s = n − 1 default and the new d is a
   separate named constant; the generator's sidecar `error2.d` is not changed.

## Tests
Unit tests in `transfer.rs`, ungated and synthetic:
- the identity filter reproduces `cheb_to_uniform(n, s)` to round-off (filtered R only);
- R is exact on a resolved mode: a sine mode with k well inside the cut-off, restricted,
  matches its samples at `uniform_nodes(s)` to a stated tolerance;
- no aliasing: an unresolved sine mode (k above the cut-off), restricted, has small
  energy (stated bound) rather than appearing as a resolved mode;
- rows sum to 1 (constants preserved);
- 'ij' layout preserved under `apply` with s ≠ n − 1;
- `Transfers::new(n)` equals its pre-T2 matrices bit for bit;
- panics on s below the minimum, on n below `MIN_N`.

## Acceptance
- Every existing test passes unchanged, including `poisson_output_is_pinned`.
- The CONVENTIONS diff and the code agree, and the `lib.rs` doc-sync test passes.
- The T2 Results row filled in, with the wall time of the new tests.

## Checks to run and report
All of `CLAUDE.md`'s CI checks (the datagen pair too: the `rlst_oracle` tests touch
`transfer.rs`) and `cargo clippy --all-targets -- -D warnings`.

## Do not
- Change the loaders, `poisson_errors` or any example (T3, T6).
- Add the candidates T1 rejected.
- Regenerate datasets: version 1 sidecars stay valid (the loaders do not check the
  version).

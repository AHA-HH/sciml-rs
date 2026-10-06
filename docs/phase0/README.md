# Phase 0: conventions and feasibility

As of 2026-10-05. Design: `docs/design/2d-chebyshev-poisson-fno.md` (signed off
2026-10-05; decisions in its §12). `docs/CONVENTIONS.md` wins on any conflict.

Phase 0 settles everything the later phases build on, before any production code is
written:
- RLST is wired in behind the optional `chebyshev` feature, and its API is verified on the
  pinned version.
- The grid-transfer and solver choices left open by the design are measured.
- The Chebyshev conventions are written into `docs/CONVENTIONS.md` with the measured
  values filled in.

Nothing in this phase changes the model, the training code or the default build.

## Read first

- Design §1.3 (RLST), §3 (problem, GRF), §4 (solver), §5.3 (resolutions), §6
  (transfers), §8 (conventions diff), §9 (dependencies), §12 (decisions).
- `CLAUDE.md`: Working agreement, Checks.
- `docs/CONVENTIONS.md` §4, §8, §9.

## Tasks

| Task | Brief | Delivers | Depends on |
| --- | --- | --- | --- |
| T2 | [T2-rlst-feature.md](T2-rlst-feature.md) | Feature `chebyshev`, rlst 0.9, BLAS selection, CI job, smoke tests, verified API list | – |
| T3 | [T3-transfer-spike.md](T3-transfer-spike.md) | Measured transfer errors; FH degree d; uniform size s; GRF tail table | – (may use T2's dependency setup) |
| T4 | [T4-solver-spike.md](T4-solver-spike.md) | Measured solver accuracy and cost; solve route; FV or Q1; CG tolerance | T2 |
| T1 | [T1-conventions.md](T1-conventions.md) | CONVENTIONS §12 with the measured d and s | T3, T4 |

Order: T2 first, then T3 and T4 in either order or in parallel, and T1 last. T1 keeps
its design number but comes last: §12 is written once, with T3's measured values, so that
filling them in later does not count as a change to §12 (design decision 4).

One task per branch and PR: `phase0/T<k>-<name>`. Each task is worked from its brief
alone, plus the files the brief lists under "Read first".

## Spikes

T3 and T4 are spikes. Each lives in its own directory under `spikes/`, as a standalone
Cargo package with its own `Cargo.toml`. It is not part of the crate, `cargo test` and CI
never build it, and its code is not reused as-is: Phase 1 and Phase 2 rewrite what is kept,
with tests. Each spike ends with a `REPORT.md` holding the measured tables, the exact
commands to reproduce them, the machine, and a recommendation. T3 adds one line about
`spikes/` to the Layout section of `CLAUDE.md`.

## Results

Filled in as tasks merge. Phase 0 is done when every row has a value.

| Item | Value | Source |
| --- | --- | --- |
| rlst version pinned | | T2 |
| Public Sylvester routine in that version? | | T2 |
| Solve route for option A (Sylvester or fast diagonalisation) | | T4 |
| FV or Q1 for option C | | T4 |
| CG tolerance for option C | | T4 |
| Floater–Hormann degree d | d = 2, rule evaluated on the solution u per design §7 (max 3.6e-5 at n = 65, 1D Λ ≤ 4.9); on f no d ≤ 8 meets 1e-4 (design §12, decision 8) | T3 |
| Uniform size rule (s = n − 1 or s = n) | s = n − 1 (s = n gains ≤ 1.05× on u, ≤ 1.04× on f at d = 2; rule needs 10×) | T3 |
| GRF tail table confirmed; tolerance 5e-3 kept? | Confirmed (ε_K = 1.86e-2, 4.92e-3, 1.26e-3, 3.19e-4); kept, ε_32 = 4.92e-3 | T3 |
| CONVENTIONS §12 merged | | T1 |

## Exit checklist

- [ ] T2 merged; the `chebyshev` CI job is green on Ubuntu, and the feature tests pass
      locally on macOS.
- [ ] T3 and T4 merged, each with its `REPORT.md`.
- [ ] Every row of Results filled in.
- [ ] T1 merged: `docs/CONVENTIONS.md` has §12, and `CONVENTION_VERSION` is still 1.
- [ ] Any result that contradicts the design is recorded in the design's §12 under
      "Recorded decisions", dated, before Phase 1 is planned.
- [ ] The "Current phase" bullet in `CLAUDE.md` points to `docs/phase1/README.md`.

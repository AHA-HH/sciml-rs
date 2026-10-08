# Phase 1: verified reference solver

As of 2026-10-08. Design: `docs/design/2d-chebyshev-poisson-fno.md` (signed off
2026-10-05, amended 2026-10-08; decisions in its §12). `docs/CONVENTIONS.md` wins on any
conflict.

Phase 1 builds the numerical base that the datasets stand on:
- a tested Chebyshev toolkit: CGL nodes, differentiation matrices and Clenshaw–Curtis
  weights (CONVENTIONS §12);
- solver A, Chebyshev collocation by fast diagonalisation, the source of every training
  label;
- solver C, Q1 finite elements with CG, the independent cross-check;
- evidence that A and C agree with manufactured solutions and with each other as h → 0.

Nothing in this phase changes the model or the training code. All three tasks use RLST and
sit behind the `chebyshev` feature; the toolkit (T1) wraps RLST's FFTW-backed `chebychev`
module (design decision 12), so it is built and tested only by the `run-tests-chebyshev`
CI job.

## Read first

- Design §3.2 (nodes), §3.4 (norms), §4 (solver options, decision, verification), §11
  (tests), §12 decisions 1, 10 and 12.
- `docs/CONVENTIONS.md` §12.
- `spikes/solver/REPORT.md` (T4) and, for the nodes and weights, `spikes/transfers/REPORT.md`
  (T3).
- `CLAUDE.md`: Working agreement, Checks.

## Inputs from Phase 0

| Item | Value | Source |
| --- | --- | --- |
| Solver A's route | Fast diagonalisation: `eig(EigMode::BothEigenvectors)` once per axis, then two products per side and a pointwise division per solve. `RightEigenvectors` panics in rlst 0.9.0. | T4, decision 10 |
| Solver C | Q1 on the CGL tensor mesh, consistent mass and consistent load; unpreconditioned `CgIteration` at relative residual 1e-6 | T4, decision 10 |
| A's floor | ≤ 1.1e-14 by n = 33 on the first two manufactured solutions; at most 4.6e-13 up to n = 129 and 1.2e-12 at n = 257 | T4 §2 |
| A against B | ≤ 3.4e-14 for n ≤ 33 | T4 §3 |
| C's order | 2.00 on all three manufactured solutions at n = 65, 129, 257 | T4 §4 |
| A against C on GRF | order 1.97–2.00 at n = 65..257 | T4 §6 |
| C's cost | 987 CG iterations, 118 ms per solve at n = 129; 2998, 1.46 s at n = 257 (release) | T4 §5 |

## Tasks

| Task | Brief | Delivers | Depends on |
| --- | --- | --- | --- |
| T1 | [T1-chebyshev-toolkit.md](T1-chebyshev-toolkit.md) | `chebyshev::{nodes, diff_matrix, clenshaw_curtis}` and a Chebyshev-grid L² norm, with tests | – |
| T2 | [T2-collocation-solver.md](T2-collocation-solver.md) | `pde::poisson::collocation` (solver A) and its convergence tests | T1 |
| T3 | [T3-sparse-solver.md](T3-sparse-solver.md) | `pde::poisson::sparse` (solver C), the order-2 tests and the A-vs-C check | T1, T2 |

Task numbers are execution order (design §10). One task per branch and PR:
`phase1/T<k>-<name>`. Each task is worked from its brief alone, plus the files the brief
lists under "Read first".

The spike code (`spikes/solver/src/{cheb,colloc,fem}.rs`, `spikes/transfers/src/grids.rs`)
is the reference for the algorithms, but it is rewritten, documented and tested, not
copied as-is.

## Module layout

- `src/neural_operators/chebyshev/` (T1) and
  `src/neural_operators/pde/poisson/{collocation,sparse}.rs` (T2, T3): behind
  `#[cfg(feature = "chebyshev")]`. The public API takes and returns ndarray types; rlst's
  `DynArray` stays internal.
- Each new module is registered in `src/neural_operators/mod.rs`, and the task that adds
  it adds one line to the Layout section of `CLAUDE.md`.

## Results

Filled in as tasks merge. Phase 1 is done when every row has a value.

| Item | Value | Source |
| --- | --- | --- |
| D and D² tolerances used, and the n tested | D: 1e-10 relative to max\|D\| against Trefethen's `cheb`, n = 2, 3, 9, 33, 65, 129, 257 (measured ≤ 5.0e-13; rows sum to ≤ 9.9e-16 · max\|D\|, so no diagonal reset). D²: 1e-10 · ‖D²‖_∞ · ‖x^k‖_∞ on x^k, k < n, n = 9, 17, 33, 65, 129, 257 (measured ≤ 2.5e-15). Nodes 1e-14 absolute, n = 2, 3, 9, 33, 257 (measured ≤ 3.3e-16). Clenshaw–Curtis 1e-14 absolute on x^k, k < n, n = 2, 3, 9, 33, 65 (measured ≤ 4.4e-16). n = 2 and 3 work through RLST. | T1 |
| A's measured floor per n (manufactured solutions) | Relative max error (CC relative L² within 1.5×), sin · sin / polynomial / e^(x + 2y): n = 9: 1.3e-4 / 2.3e-15 / 3.6e-5; n = 17: 2.8e-12 / 5.1e-15 / 5.7e-14; n = 33: 4.2e-14 / 3.6e-14 / 1.5e-14; n = 65: 1.1e-13 / 2.3e-13 / 3.0e-13 (L²); n = 129: 4.1e-13 / 5.3e-13 / 4.0e-13 (L²). Floor ≤ 1e-12 from n = 33, tested at 1e-12 for n = 33, 65, 129; rectangular: (1 − x²)(1 + x)(1 − y²) sin 8y on [17, 33] 5.8e-15 (3.6e-5 on [33, 17], so a swapped axis fails), its transpose on [33, 17] 5.7e-15, e^(x + 2y) on [33, 65] 1.5e-13 (L²). Eigenpairs of the interior D² exactly real: max \|Im λ\| and max \|Im v\| measured 0 for n = 9 to 129 (tolerance 1e-10). | T2 |
| A against B at n ≤ 33 | Relative max 4.5e-15 (n = 9), 5.4e-15 (17), 5.5e-14 (33) on f = eˣ cos 2y + xy; tested at 1e-10 | T2 |
| C's observed orders (three solutions, n = 33 → 65 → 129) | | T3 |
| A against C on the fixed sine-series forcings, and its order | | T3 |
| Wall time of the new tests under debug `cargo test --features chebyshev` | | T3 |

## Exit checklist

- [ ] T1, T2 and T3 merged.
- [ ] The design §11 rows for CGL nodes, D/D², Clenshaw–Curtis, the collocation solver
      and the sparse solver pass under `cargo test --features chebyshev`, locally on macOS
      and in the `run-tests-chebyshev` CI job.
- [ ] A and C agree at order 2 on the common CGL grid as h → 0 (error 1 of design §7).
- [ ] Every row of Results filled in.
- [ ] Any result that contradicts the design is recorded in the design's §12 under
      "Recorded decisions", dated, before Phase 2 is planned.
- [ ] The "Current phase" bullet in `CLAUDE.md` points to `docs/phase2/README.md`.

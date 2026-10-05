# 2d-chebyshev-poisson-fno phase 0: conventions and feasibility

Phase 0 settles everything the rest of the objective builds on, before any solver or training code
exists:
- the Chebyshev-grid conventions;
- whether RLST can be an optional dependency that builds on macOS and in CI;
- the two numbers the design leaves open: the Floater–Hormann degree d for the uniform → Chebyshev
  transfer, and the model grid size s and padding.

The work is one documentation change, one dependency change and two throwaway spikes, which are
measured but not merged into the library. The phase ends when the four tasks are merged and the
three human sign-offs are recorded:
- CONVENTIONS §9;
- the FH degree d;
- s and the padding.

Companion documents: [docs/design/2d-chebyshev-poisson-fno.md](../../design/2d-chebyshev-poisson-fno.md)
- §2.2–§2.3: grids;
- §2.6: transfers;
- §2.8: the conventions diff;
- §6.1–§6.2: RLST and the Burn FFT;
- §7: Phase 0, components C0.1–C0.4;
- §8.2: tolerances;
- §9.2: open questions.

Also `docs/CONVENTIONS.md` (preamble, §2, §3, §4).

## Scope

In scope:
- `docs/CONVENTIONS.md`: append §9, Chebyshev grids (T1).
- `Cargo.toml`, `src/neural_operators/mod.rs`, `src/neural_operators/chebyshev/mod.rs`,
  `.github/workflows/run-tests.yml`, `CLAUDE.md` (Checks section): the optional `chebyshev-data`
  feature, a smoke test and a CI job (T2).
- `spikes/cheb-transfer/`: transfer-stability spike and report (T3).
- `spikes/fft-sizes/`: FFT size and padding timing spike and report (T4).
- `.gitignore`: ignore `spikes/*/target/` (T3 and T4).

Out of scope:
- Nodes, differentiation matrices, quadrature and the Poisson solver in the library. These are
  Phase 1, C1.1–C1.4.
- Library transfer operators. These are Phase 2, C2.2, which reuses T3's findings, not its code.
- Any change to `FNO`, `SpectralConv`, `LpLoss`, the trainer, or CONVENTIONS §1–§8.
- Datasets, training and examples.

## Design decisions for this phase

These hold for every task, so that no task decides them on its own:

- **Node convention.** Ascending Gauss–Lobatto nodes:
  - Computed as `x_j = (PI * (2j − (n−1)) as f64 / (2(n−1)) as f64).sin()`, in exactly that order
    of operations (design §2.2).
    - Form `2j − (n−1)` as `i64`, not `usize`: it is negative for j < (n−1)/2, and `usize` would
      overflow.
  - This gives exact antisymmetry, x_mid = 0, endpoints exactly ±1, and bitwise nesting of n = 65 in
    n = 129.
  - RLST's `chebychev_points` is used only in T2's smoke test, never for nodes.
- **Uniform grid with endpoints.**
  - Computed as `x_k = 2.0 * (k as f64 / (s − 1) as f64) − 1.0`, per index, with no accumulated
    step (design §2.6).
  - The model coordinate is ξ = (x + 1)/2, which is CONVENTIONS §2 `arange(s)/(s−1)`.
- **Indexing.** 2D fields are `[n_x, n_y]`, 'ij': axis 0 is x, axis 1 is y. RLST arrays are
  column-major, so any copy into or out of RLST goes element by element through `[i, j]`, never
  through a raw slice.
- **RLST version.** `rlst = "=0.6.1"` from crates.io (design §9.2 Q5, recommended). No `fftw`
  features (GPL), and no `burn` feature (it targets crates.io Burn, not our fork).
- **BLAS/LAPACK provider.**
  - macOS: Accelerate (`blas-src` 0.14 and `lapack-src` 0.13 with feature `accelerate`).
  - Linux: OpenBLAS (`openblas-src` 0.10 `system`, plus `blas-src` and `lapack-src` with feature
    `openblas`).
  - These mirror `~/Code/rlst/rlst/Cargo.toml:49-56`.
- **Spikes stay out of the library build.** Each spike crate:
  - has its own `Cargo.toml` with an empty `[workspace]` table;
  - is not a dependency of `sciml-rs`;
  - is not built by CI.
  Their code is evidence for the report, not reusable library code.
- **Error measures.** Every number in a report names one of these:
  - `rel_max(a, b) = max|a − b| / max|b|`, f64, relative to the analytic field b.
  - `rel_cc(a, b) = ‖a − b‖_CC / ‖b‖_CC` on an n × n GL grid, with tensor Clenshaw–Curtis weights
    (Trefethen `clencurt`), f64, relative to the analytic field b.
  - Wall-clock time per epoch: median of 3 runs after 1 warm-up run, in seconds, with the backend,
    machine and `--release` named.

## Exit gate
- T1–T4 merged. For T2, the default checks are unchanged and the new feature job is green in CI.
- T3's report gives `rel_cc` for the round trip T_uc∘T_cu at s ∈ {64, 128}, for d = 3..8, on
  actual Poisson solutions. It recommends the smallest d whose round-trip floor is ≤ 1e-5
  (design §8.2 L6).
- T4's report gives the epoch times for every (s, padding) pair in its grid and a recommendation.
- Human sign-off on CONVENTIONS §9, on d, and on s with padding, recorded in design §9.2 and §11.
- Measured results are in the PRs and copied into design §6.2, §8.3 and §11.

## Tasks

One pull request each.
- All four tasks can start at once, because they touch disjoint files.
- The one exception: T3 and T4 both add the line `/spikes/*/target/` to `.gitignore`. Whichever
  merges second drops its copy.
- T2 needs design §9.2 Q5 (the RLST pin) confirmed before it starts. The brief assumes the
  recommendation, `=0.6.1`.

| Task | Brief | Delivers | Component | Depends on |
| --- | --- | --- | --- | --- |
| T1 | [T1-conventions-chebyshev.md](T1-conventions-chebyshev.md) | CONVENTIONS §9, Chebyshev grids | C0.1 | none |
| T2 | [T2-rlst-feature.md](T2-rlst-feature.md) | `chebyshev-data` feature, RLST + BLAS/LAPACK, smoke test, CI job | C0.2 | §9.2 Q5 confirmed |
| T3 | [T3-transfer-spike.md](T3-transfer-spike.md) | transfer-stability report, recommended d | C0.3 | none |
| T4 | [T4-fft-size-spike.md](T4-fft-size-spike.md) | FFT size/padding timing report, recommended s and padding | C0.4 | none |

Review T1 yourself before Phase 1 starts. Phase 1's tests encode §9.

## How to run a task with Claude Code

In this objective's worktree, start `claude` and say
"/do-task docs/2d-chebyshev-poisson-fno/phase0/T<k>-<name>.md". Review and merge before starting a
task that depends on it.

## Exit checklist
- [ ] T1 merged: CONVENTIONS §9 appended, `CONVENTION_VERSION` unchanged (1)
- [ ] T2 merged: default checks unchanged; `cargo test --features chebyshev-data` green on macOS and in CI
- [ ] T3 merged: report with round-trip `rel_cc` table for d = 3..8 and s ∈ {64, 128}; d recommended
- [ ] T4 merged: report with epoch times; s and padding recommended
- [ ] Reviewer agent pass on each task (implementation done ≠ review passed)
- [ ] **Human sign-off:** CONVENTIONS §9
- [ ] **Human sign-off:** FH degree d
- [ ] **Human sign-off:** s and padding
- [ ] Design document updated:
  - §7 Phase 0 status;
  - §2.6, §6.2 and §8.3 with measured results;
  - §9.1 risks retired: BLAS/LAPACK build, FH transfer floor, non-power-of-two FFT cost;
  - §9.2 Q5 and Q6 answered.
- [ ] `CLAUDE.md` points to `docs/2d-chebyshev-poisson-fno/phase1/README.md`

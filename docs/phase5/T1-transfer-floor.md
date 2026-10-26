# T1: transfer floor and the choice of R

Phase 5, task 1. Branch `phase5/T1-transfer-floor`. Depends on nothing in this phase.
Needs the stage 3 and stage 4 datasets (`datasets/poisson/poisson_n129_K64_*`,
`poisson_n257_K64_*`) from the main checkout, by copy or symlink. Nothing about that is
committed.

## Read first
- Design §3.3 (the sine KL series, K, the tail table), §13.3 (restriction, prolongation,
  the three errors, the reference check), §13.6, §12 decisions 5 and 8.
- `docs/CONVENTIONS.md` §12 (CGL nodes, the uniform grid, T_cu, T_uc, CC weights).
- `spikes/transfers/REPORT.md` (Phase 0 T3, how d = 2 was chosen: the pattern this
  task follows).
- `src/neural_operators/chebyshev/transfer.rs` (`cheb_to_uniform`, `uniform_to_cheb`,
  `apply`, `nodes`, `uniform_nodes`), `chebyshev/clencurt.rs` (`clencurt`,
  `cc_rel_l2_error`), `data/loaders/poisson/cheb.rs` (`load_poisson_cheb_field`).
- `spikes/metal_diag/Cargo.toml` (how a spike joins the workspace).

## Goal
Measure, before any library code exists, how much a coarse learning grid loses with no
model in the loop, for each candidate restriction R and prolongation P. The answer fixes
D1 and gives every later fine-grid error its floor.

## Deliverables
1. **A spike package** `spikes/transfer_floor/`, a non-default workspace member (add it to
   `[workspace] members` in the root `Cargo.toml`, not to `default-members`). It depends on
   `sciml-rs` by path with `default-features = false, features = ["flex"]` and on
   `ndarray`; it uses no GPU. A `main.rs` taking the dataset directory as an argument.
2. **Reference check.** For every test sample, stage 4 u at the even-indexed 257 nodes
   (exactly the 129 nodes, CONVENTIONS §12) against stage 3 u: max relative CC-L² on the
   129 grid, and the same for f. It should sit at the label-check level (about 1e-12). A
   larger value stops the phase: record it in design §12 and stop.
3. **Candidate restrictions,** each a dense `[s, n]` matrix applied by
   `chebyshev::transfer::apply`:
   - **Filtered Chebyshev:** nodal values → Chebyshev coefficients by the closed-form
     DCT-I matrix, multiply by a filter σ_j, evaluate the truncated series at
     `uniform_nodes(s)`. Sweep the cut-off J (a few values tied to s, e.g. J = s − 2,
     ⌊(s − 2)π/2⌋, n − 1) and the taper (sharp; one smooth taper, e.g. raised cosine or
     exponential). σ ≡ 1 must reproduce `cheb_to_uniform(n, s)` to round-off: assert it.
   - **Cell average:** the mean of the fine interpolant over each coarse cell, by
     Clenshaw–Curtis quadrature. Endpoint cells take half-cells.
   - **Unfiltered T_cu:** `cheb_to_uniform(n, s)`, the control.
4. **Candidate prolongations:** `uniform_to_cheb(s, n, d)` for d ∈ {2, 3, 4}.
5. **Measurements,** on the stage 3 test split (200 samples), at s ∈ {16, 32, 64, 128}:
   - transfer-only error ‖P R u_f − u_f‖ / ‖u_f‖ (CC-weighted, mean / max) for every
     (R, d);
   - P's own error: P applied to u_f sampled exactly at `uniform_nodes(s)` (unfiltered
     T_cu, which is near-exact on the smooth u), so P's share is separated from R's;
   - the filtered-out energy of f: ‖f − F_s f‖² / ‖f‖² on the CGL grid, F_s the chosen
     filter, mean / max;
   - aliasing visible in f: ‖R f − (sine series of f truncated to k ≤ s − 2, evaluated
     on the uniform grid)‖ / ‖that series‖, for each R. The exact series comes from the
     sample's seed (`pde::poisson::grf`), or skip this row if that needs `datagen`, and
     say so;
   - time to build and to apply each R per (n, s);
   - dataset generation time from the stage 3 and 4 sidecars (`wall_time_s`), for the
     Results row on solver cost.
6. **`spikes/transfer_floor/REPORT.md`:** the tables above, the chosen R with its
   parameters, the chosen d, and the reasoning. The rule: the lowest mean transfer-only
   error at s = 32 and 64; values within 10% go to the simpler R (unfiltered T_cu, then
   cell average, then filtered).
7. **D1** recorded in design §12 as a dated amendment to decision 30, and the T1 rows of
   `docs/phase5/README.md` Results filled in.

## Tests
The σ ≡ 1 assert and a check that each R matrix has rows summing to 1 (constants are
preserved), both run by `main`. A spike needs no unit tests; T2 writes the library's.

## Acceptance
- The reference check passes, or the phase is stopped and the reason recorded.
- Every (R, s, d) cell of the transfer-only table has a value, and D1 is recorded.

## Checks to run and report
- `cargo run --release -p transfer-floor-spike -- <dataset dir>` and its wall time.
- `cargo clippy -p transfer-floor-spike -- -D warnings`.
- All of `CLAUDE.md`'s CI checks, to show the root build is unaffected (the spike is not
  a default member).

## Do not
- Touch `src/`. The chosen R reaches the library in T2.
- Regenerate or modify any dataset.
- Train a model.

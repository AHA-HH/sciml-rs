# 3D Poisson and Laplace progress log

Newest entries at the bottom. One entry per task, in that task's pull request, using this
template:

```
## YYYY-MM-DD: <ID> <title>
- Done:
- Tests:
- Checks: ran …; pending …
- Decisions: none, or D… signed off; questions for sign-off …
- Review: plan review (blocking items fixed); code quality APPROVE or BLOCKED → APPROVE (rounds, blockers fixed, findings left as follow-ups)
- 2D: unchanged bit for bit (how shown), or not touched
- Follow-ups:
```

## 2026-11-11: Planning
- Done: the proposal, draft 0.1, in `docs/design/3d-chebyshev-laplace-fno.md`; its
  Section 10 as `docs/laplace3d/roadmap.md`; the docs reorganised by objective (the 2D
  phases moved to `docs/poisson2d/`, index `docs/README.md`).
- Tests: none.
- Checks: none run for the proposal; its cost figures are estimates for CL0.1 to measure.
- Decisions: none yet; D0 (the proposal as the plan) comes first.
- 2D: docs moved and their paths rewritten; no code changed except path strings in
  comments.
- Follow-ups: review the draft; D0; then CL0.1, CL1.1.

## 2026-11-11: Draft 0.2
- Done: an independent review of draft 0.1 against the code and CONVENTIONS (verdict:
  revise first); draft 0.2 answers it (proposal Appendix A). Blocking items: the 3D GRF at
  α = 2 misses ε_K ≤ 5e-3 by 20× at K = 32 (ε_K ~ 1/K in 3D), so α = 3 is proposed (D3);
  D2 now names the 2D outputs, excludes metadata and requires in-process checks. Also:
  error 5 in 3D, the solver's sign, the Laplace discrete lifting with face sine series, the
  §4 grid-order conflict dropped, activation memory and the VM's A100 80 GB in §7, 3D
  dataset names, missing roadmap *Needs*.
- Tests: none (docs). The ε_K table was recomputed independently (lattice sums to K = 160
  plus an integral tail) and agrees with the review's.
- Checks: none (no code changed).
- Decisions: none signed off; D0 is asked again on draft 0.2, with D1–D3 recommendations
  revised.
- Review: the review of 0.1 is the input; 0.2 not yet reviewed by the author.
- 2D: not touched.
- Follow-ups: D0 on draft 0.2; then CL0.1 (the zip64 check added to it), CL0.2, CL1.1.

## 2026-11-24: CL1.1 CONVENTIONS 3D additions
- Done: proposal §4 written into `docs/CONVENTIONS.md` §12 as a "Three dimensions
  (version 3)" block ('ijk' storage, transfers along x, then y, then z, the 3D
  Clenshaw–Curtis norm, the 3D boundary norm of proposal §3.5); `CONVENTION_VERSION` 2 → 3
  in the header, §9 (with a History line) and `src/lib.rs`. Plan
  `docs/laplace3d/tasks/CL1.1.md`.
- Tests: none new (the conventions land before their code, CONVENTIONS §9; CL1.2 and
  CL1.3 add theirs). `convention_version_matches_conventions_md` checks the bump.
- Checks: see the PR (fmt, clippy ×4, `cargo test`, `cargo doc`, datagen clippy and tests).
- Decisions: the additions signed off by the author 2026-11-24 on #122, with two answers
  applied: §12 records the 2D boundary norm `metrics::poisson` already computes, and labels
  its 2D-only statements. The Laplace boundary norm's denominator (‖u‖ or ‖g‖ on ∂Ω) is
  deferred to the L4 gate (D6, D7).
- Review: plan review by the `reviewer` agent; no code-quality review (docs and one
  constant).
- 2D: no numerical code changed; newly generated 2D sidecars record
  `convention_version: 3`, metadata outside D2.
- Follow-ups: CL1.2 and CL3.1 once #122 is merged.

## 2026-11-24: CL0.2 Rank-5 backend parity
- Done: three GPU-against-Flex cases in `tests/gpu_parity.rs`: `grid_3d` (`FNO<5>` with no
  Fourier layers, so the grid channels are isolated; non-cubic shapes, one repeated after
  another), `spectral_conv_3d` (`SpectralConv<5>` forward and input gradient; Bluestein on
  both cfft axes, an even rfft half-width) and `replay_3d_poisson` (the 2D replay schedule
  at rank 5, s = 16, modes 6³, padding 0 and 4, with a resolution change). The 2D replay's
  body became a rank-generic `replay<R>` that both replays call.
- Tests: on metal (Mac, release), 9 of 9 pass. Rank 5 worst relative L² 1.5e-6, every
  max-entry error ≤ 7.2e-7, against 1e-5 (the 2D replay's max-entry error, printed and not
  asserted: 9.6e-7). No Burn bug at rank 5, so no workaround. The 2D
  replay's per-call errors at padding 4, 8, 16 are the same before and after the refactor;
  padding 0, the first configuration per test while Metal autotunes, varies run to run.
- Checks: ran fmt, the three clippy jobs, `cargo clippy --features metal --all-targets`,
  `cargo test` (flex), `cargo doc`, `cargo test --release --features metal --test
  gpu_parity`; not run: cuda (Mac-only lane), datagen and mlflow (not touched).
- Decisions: none.
- Review: plan review PASS (advisory: an even rfft half-width, added); code quality round 1
  BLOCKED (the replay harness duplicated, fixed by `replay<R>`) → round 2 APPROVE (its
  advisories taken: the 3D replay asserts the max-entry error too; the rest in the PR).
- 2D: no 2D code touched; the 2D replay is restructured with the same schedule, inputs,
  loss and tolerance, and passes on metal.
- Follow-ups: CL0.2's cases on cuda before CL4.2's first cuda run (a sentence in CL4.2).

## 2026-11-24: CL3.1 3D GRF
- Done: `pde/poisson/grf3d` (behind `datagen`), the 3D GRF of proposal §3.3 beside the 2D
  one: τ = 3, α = 3 (D3), S_K = ⅛ Σ λ_klm, S_∞ = 1.232538e-4 (cube sums to 80 and 160 with
  the (m + ½)⁻³ tail extrapolated), nested draws of 64³ normals on ChaCha8 stream 3 (2D
  and the Phase 7 subsets use stream 0; seeds numbered as in 2D; `grf3d::RNG` is the
  sidecar string), exact solution c / μ, separable evaluation (three mode products). Plan
  `docs/laplace3d/tasks/CL3.1.md`, report `docs/laplace3d/results/cl3-1.md`.
- Tests: 9 in `grf3d`: evaluation against the direct triple sum (1.3e-15 at scale 2.85),
  ε_K at K = 8, 16, 32, 64 = 2.742e-2, 4.265e-3, 5.892e-4, 7.703e-5 (the proposal's row to
  two digits), nesting, the stream distinct from 2D, the draw order, the pins, mean square
  1, zero on all six faces, −Δu = f by a stencil (2.0e-6 at h = 1e-3), the K > K_MAX panic.
- Checks: ran fmt, the three clippy jobs, `cargo clippy --all-targets`, `cargo test`,
  `cargo doc`, datagen clippy and tests (385 passed), datagen `cargo doc -D warnings`; not
  run: metal and gpu_parity (host-only ndarray code), cuda (no VM).
- Decisions: D3 re-proposed on the measured table (question on the PR); unchanged in the
  table until the author answers.
- Review: plan review FAIL on docs only (results file, plan text), fixed; code quality
  APPROVE, 0 blockers; F1 (own module), F2 (pin test), F3 (evaluate without slices), F7
  and F8 taken, plus the plan review's draw-order test; F4–F6 left (in the PR).
- 2D: `grf.rs` differs from `main` only by `sine_basis` becoming `pub(super)`; the 2D
  GRF and dataset tests pass unchanged.
- Follow-ups: none new; CL3.2 uses `grf3d` (needs CL1.2, CL2.1, D5).

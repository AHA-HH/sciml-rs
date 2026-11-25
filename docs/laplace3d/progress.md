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

## 2026-11-24: CL1.2 N-D transfers and quadrature
- Done: rank-generic siblings of the 2D transfers and Clenshaw–Curtis norms in
  `chebyshev/`: `apply_nd` (T₀ along axis 0 as one matrix product, then the rest on each
  slice, so rank 3 is x, then y, then z, CONVENTIONS §12), `Transfers::to_uniform_nd`,
  `to_cheb_nd`, `to_uniform_all_nd`, and `cc_norm_sq_nd`, `cc_rel_l2_error_nd`,
  `cc_rel_l2_error_weighted_nd` (a field read as `[n₀, rest]` with w₀ and the Kronecker
  product of the other weights, so the 2D scaling serves every rank). Rank 2 calls the
  existing 2D functions (D1). Plan `docs/laplace3d/tasks/CL1.2.md`.
- Tests: 18 new. 3D against closed forms on non-cubic grids: the 'ijk' layout both ways,
  separable fields against the 1D transfers (1e-14), the round trip exact on quadratics,
  the restriction exact on a resolved mode, ∫ 1 = 8 and ∫ x² y⁴ z² exact, (1 + ε) b
  against b, extreme magnitudes, permuted views; and the panics. D2: `rank_two_is_bit_identical_to_2d`
  in both modules (`assert_eq!`, a non-square grid and a transposed view).
- Checks: ran fmt, the three CI clippy jobs, `cargo clippy --all-targets`, `cargo test`,
  `cargo doc`, `cargo test --features datagen`; not run: metal (no backend code).
- Decisions: none.
- Review: plan review PASS (advisories in the PR); code quality APPROVE with findings: the
  shared `as_matrix`, the rank assert in `to_uniform_all_nd` and the docs taken; the
  duplicated shape checks (different panic texts) left; the file size a follow-up.
- 2D: unchanged bit for bit: no 2D function or caller edited, rank 2 routed to them, shown
  in process by the D2 tests; the whole 2D suite passes.
- Follow-ups: in CL1.2's roadmap line: the 3D test names in CONVENTIONS §12
  "Verification" (with CL1.3's, for sign-off); split `transfer.rs`'s tests out before its
  next edit.

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
- Decisions: D3 confirmed by the author 2026-11-24 on the measured table, as recommended
  (#132); the 3D seed scheme, ChaCha8 stream 3 with the 2D seed numbering, approved by the
  author the same day, as recommended.
- Review: plan review FAIL on docs only (results file, plan text), fixed; code quality
  APPROVE, 0 blockers; F1 (own module), F2 (pin test), F3 (evaluate without slices), F7
  and F8 taken, plus the plan review's draw-order test; F4–F6 left (in the PR).
- 2D: `grf.rs` differs from `main` only by `sine_basis` becoming `pub(super)`; the 2D
  GRF and dataset tests pass unchanged.
- Follow-ups: none new; CL3.2 uses `grf3d` (needs CL1.2, CL2.1, D5).

## 2026-11-24: CL1.3 N-D errors
- Done: `metrics::poisson::poisson_errors_nd`, errors 2–5 for 2D or 3D fields (axis 0 the
  samples): 2D fields go to `poisson_errors` unchanged (D1); 3D uses CL1.2's
  `to_uniform_nd`, `to_cheb_nd` and `cc_*_nd`, and error 5 sums the six faces, each with
  2D tensor CC weights, edge and corner nodes once per face, 24 against 8 (CONVENTIONS §12).
  It returns the existing `PoissonErrors` (no new fields; `errors.json` untouched).
  `metrics/poisson.rs` became `metrics/poisson/{mod,tests}.rs`. Plan
  `docs/laplace3d/tasks/CL1.3.md`.
- Tests: 12 new. Every 3D error against explicit triple loops (transfers term by term, CC
  sums with w_i w_j w_k, the six faces written out) to 1e-13 at s = n − 1 and s = 5; the
  floor identity in 3D; the 24/8 boundary RMS and ‖u‖ against closed forms; the general
  face formula at rank 2 against `boundary_errors` (1e-14); the panics, including a 4D
  field. D2: `nd_rank_two_is_bit_identical_to_2d` (bits of all six vectors, s = n − 1 and
  s = 8, standard, transposed and IxDyn views).
- Checks: ran fmt, the three CI clippy jobs, `cargo clippy --all-targets`, `cargo test`
  (341 passed), `cargo doc`, datagen clippy and `cargo test --features datagen` (415
  passed); not run: metal and gpu_parity (host ndarray only).
- Decisions: none; for sign-off: the 3D test names of CL1.2 and CL1.3 in CONVENTIONS §12
  "Verification".
- Review: plan review PASS (advisory taken: rank 2 or 3 only, the ranks CONVENTIONS
  defines); code quality round 1 BLOCKED (the file past 1,000 lines, fixed by moving the
  tests to `poisson/tests.rs`; F2, F4 taken) → round 2 APPROVE with findings (N1 taken; F1,
  N2 to CL4.1; F3 declined, a public bound for one `.into_dyn()`).
- 2D: unchanged bit for bit: `poisson_errors`, `boundary_errors` and `evaluate` not edited,
  `rel_l2_equal_weights` generic with the same body, shown in process by the D2 test; the
  whole 2D suite passes.
- Follow-ups: in CL1.3's roadmap line (CONVENTIONS §12 "Verification") and CL4.1's (one
  loop for 2D and 3D, max |u| into `boundary_errors_nd`).

## 2026-11-24: CL2.1 Collocation solver A3
- Done: `CollocationSolver3d` in `pde/poisson/collocation.rs` (behind `datagen`), A3 of
  proposal §5: one interior D² eigendecomposition per axis (`real_eig`, `interior_d2`,
  reused), the two three-way mode products by CL1.2's `apply_nd`, and the division by
  −(λ_i + μ_j + ν_k); `new`, `solve`, `solve_full` as in 2D. Plan
  `docs/laplace3d/tasks/CL2.1.md`, report `docs/laplace3d/results/cl2-1.md`.
- Tests: 7 new (3D fixtures in `manufactured.rs`). Max and 3D CC errors ≤ 1e-12 from
  n = 33 on sin πx sin πy sin πz, sin πx sin 2πy sin 3πz and (1−x²)eˣ(1−y²)(1−z²) (worst
  3.3e-13 at n = 65), 100× decay from n = 9 to 17; `grf3d`'s closed form at K = 8, n = 33
  and K = 16, n = 65 (worst 1.9e-13); the discrete residual, without the
  eigendecomposition, ≤ 6.0e-17; a non-cubic grid [17, 25, 33] at 1.5e-14 and 8.6e-5
  with the axes reversed; the panics. Time per solve (M2 Pro, release, a loaded machine):
  0.38–0.49 ms at n = 33, 5.5–17.6 ms at n = 65; setup 5–17 ms.
- Checks: ran fmt, the three CI clippy jobs, `cargo clippy --all-targets`, `cargo test`,
  `cargo doc`, datagen clippy and tests (384 passed), datagen `cargo doc -D warnings`, the
  timing test in release; not run: metal and gpu_parity (host-only ndarray code), cuda.
- Decisions: none.
- Review: plan review PASS (advisories taken: the residual range in the report, the decay
  assert over all three pairs); code quality APPROVE with findings, all six taken (3D
  fixtures moved to `manufactured.rs` with a rank-generic `rel_max` and `max_abs`, the
  test imports merged, `Debug` for the 3D solver, neutral `real_eig` panic texts, a doc
  reflow); the rank-generic single solver it waived (2D bit identity) is not a follow-up.
- 2D: unchanged bit for bit: no 2D function body edited (the diff to 2D lines is the
  widened imports, `real_eig`'s panic texts and test-only helpers made rank-generic); the
  whole 2D suite passes.
- Follow-ups: none new; CL3.2 uses A3 for the 3D labels.

## 2026-11-24: CL0.1 Rank-5 FNO cost
- Done: spike `spikes/fno3d_cost/` (a non-default workspace member): `fno3d_cost` times
  real `train_epoch` and `evaluate` epochs of `FNO<5>` on synthetic data, `run.sh` reads
  peak memory outside the process; `zip64_roundtrip`. Results
  `docs/laplace3d/results/cl0-1.md`: on the Mac (flex, metal) time per epoch at s = 32 and
  64, modes 8³ and 12³; activation memory per sample 0.15 GB (s = 32) and 1.15–1.19 GB
  (s = 64) on flex, under §7's 0.25 and 1.6–2 GB; a 4.09 GiB npz round trip passes bit for
  bit with the writer and reader as they are. On cuda (A100, 2026-11-25): 11.1 s (s = 32)
  and 64.0 s (s = 64) per epoch at batch 20, modes 12³; peak 30.6 GB at s = 64, batch 20;
  activation 1.44 GB per sample at s = 64 (12³); zip64 passes on Linux too.
- Tests: `param_counts_match_section_7` (16 785 953 and 56 631 841, exact); the zip64
  round trip checks every value.
- Checks: ran fmt, the four clippy runs, `cargo test`, `cargo doc`, the spike's tests and
  clippy (with metal), `cargo tree`; the cuda matrix and the zip64 check on the VM (2026-11-25, run by the
  coordinating session at the author's go).
- Decisions: D5 signed off by the author 2026-11-24; D4 proposed (modes 12³, batch 20 at
  both s), confirmed by the cuda run, for sign-off.
- Review: see the PR.
- 2D: not touched (no change under `src/`, `examples/`, `tests/`).
- Follow-ups: D4 from the author; CL3.2 (needs D5, now signed) and CL4.1 (needs D4).

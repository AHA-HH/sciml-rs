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
- Decisions: D5 signed off by the author 2026-11-24; D4 signed off 2026-11-25 (width 32,
  4 layers, modes 12³, batch 20 at both s), on the cuda run's figures.
- Review: see the PR.
- 2D: not touched (no change under `src/`, `examples/`, `tests/`).
- Follow-ups: CL3.2 (D5 signed) and CL4.1 (D4 signed).

## 2026-11-25: L1 gate
- Done: gate report `docs/laplace3d/gates/L1.md`: CONVENTIONS version 3 (CL1.1), N-D
  transfers, norms and errors (CL1.2, CL1.3), 2D unchanged by the D2 tests; all met. The
  roadmap's L1 gate closed 2026-11-25.
- Tests: none new. `cargo test --lib -- chebyshev:: metrics::poisson::` on `aabd87a`, 80
  passed; `main`'s last five CI runs green.
- Checks: none beyond that (docs only: CONVENTIONS §9 and §12 Verification, roadmap,
  proposal, this log).
- Decisions: signed off by the author 2026-11-25: Q1 the 3D test names in CONVENTIONS §12
  "Verification" without a bump, §9 amended; Q2 the single per-sample loop stays in CL4.1;
  C4 the N-D tensor helpers folded into CL3.3's first commit; C1–C3 and the line count
  applied (proposal Appendix A, draft 0.2 unchanged).
- Review: thermo-nuclear review of the milestone's diff APPROVE, B+, no blockers (report
  git-ignored, `docs/code-quality/thermo-nuclear-review-2026-11-25-laplace3d-L1-gate.md`);
  findings to CL3.3 and CL4.1. The report fact-checked by an independent agent; its
  corrections applied.
- 2D: not touched (no code).
- Follow-ups: the L0 and L2 gates; CL3.2 (D4 and D5 signed off with CL0.1, #123).

## 2026-11-25: L0 gate
- Done: gate report `docs/laplace3d/gates/L0.md`: proposal §7's cost model measured
  (CL0.1: activations 10–43 % under §7, 30.6 GB peak at s = 64, batch 20, modes 12³; 64 s
  per epoch on an A100; zip64 one file per split), rank-5 parity on metal (CL0.2), D4 and
  D5 signed off; all met. The roadmap's L0 gate closed 2026-11-25.
- Tests: none new. `main`'s CI green after #123 (run 45610871692).
- Checks: none (docs only: proposal §7, §8 and Appendix A, roadmap, this log).
- Decisions: signed off by the author 2026-11-25: the gate with C1 (§7's measured figures)
  and C2 (§8's risk status); D8 as proposed (Q1); the `gpu_parity.rs` tidy-up a follow-up
  under CL4.2 (Q2).
- Review: thermo-nuclear review of the union of L0's two commits APPROVE, B+, no blockers
  (report git-ignored,
  `docs/code-quality/thermo-nuclear-review-2026-11-25-laplace3d-L0-gate.md`); findings 1–4 to
  the Q2 follow-up.
- 2D: not touched (no code).
- Follow-ups: the L2 gate; CL3.2.

## 2026-11-25: L2 gate
- Done: gate report `docs/laplace3d/gates/L2.md`: A3 (CL2.1, #136) at the f64 floor on
  manufactured solutions and `grf3d`'s closed form (≤ 4.3e-14 at n = 33, ≤ 3.3e-13 at
  n = 65); met. The roadmap's L2 gate closed 2026-11-25.
- Tests: none new. `cargo test --release --features datagen --lib solver_3d` on `636298b`
  (Mac), 7 passed, every number of `results/cl2-1.md` reproduced; `main`'s CI green.
- Checks: none beyond that (docs only: proposal §7 and Appendix A, roadmap, this log).
- Decisions: signed off by the author 2026-11-25: the gate with C1 (§7's measured solve
  time); Q1 the n = 129 floor measured before labelling (CL3.2); Q2 the 3D label check in
  CL3.2, three `pde/poisson` doc fixes in CL3.3's first commit.
- Review: thermo-nuclear review of CL2.1's commit APPROVE, A−, no blockers (report
  git-ignored, `docs/code-quality/thermo-nuclear-review-2026-11-25-laplace3d-L2-gate.md`);
  the report fact-checked by an independent agent, its corrections applied.
- 2D: not touched (no code).
- Follow-ups: CL3.2.

## 2026-11-25: CL3.2 3D dataset generation
- Done: `pde/poisson/dataset3d.rs` (behind `datagen`): `generate_3d` (3D GRF of CL3.1,
  A3 labels of CL2.1, label check against the closed form, error 2 by the `_nd`
  transfers), `label_check_3d`, `write_split_3d` (one `.npz` per split, f64, `f`, `u`
  `[N, n, n, n]` 'ijk', `x`, `y`, `z`; the 2D `SidecarRecord` with the 3D α, RNG, ε_K);
  `catalogue::Poisson3dSet` (`poisson3d_n{n}_K{K}_{split}`) and `GENERATED_3D`;
  `examples/generate/poisson3d.rs` (`generate_poisson3d [-- <n> ...]`). Plan
  `docs/laplace3d/tasks/CL3.2.md`, report `docs/laplace3d/results/cl3-2.md`.
- Tests: 5 new (`generate_3d_matches_its_parts`, `label_check_3d_at_n65`,
  `label_check_3d_aborts`, `files_round_trip_3d`, `poisson3d_names`). The n = 33 sets
  written on the Mac (1000/200, 575 MB and 115 MB, under 5 s): label check max 7.4e-7
  (recorded, as 2D at n = 33), error 2 max 4.8e-4; read back by Python's stdlib. The n = 65
  sets (1000/200, 4.39 GB and 879 MB) generated on the author's Mac at `075aa3c` in 31 s:
  label check max 6.2e-10 (enforced, ≤ 1e-8), error 2 max 5.4e-5; uploaded to the VM's
  `~/sciml-rs/datasets/poisson/`, SHA-256 identical on both machines.
- Checks: ran fmt, the three CI clippy jobs, `cargo clippy --all-targets`, datagen clippy,
  `cargo test` (354 passed), `cargo test --features datagen` (439 passed), `cargo doc`
  (with and without datagen); not run: metal and gpu_parity (host ndarray only). VM
  attempts 1 and 2 at n = 65 (2026-11-25) failed in the `datagen` build (libclang, then
  FFTW); the VM's FFTW is a separate infrastructure task.
- Decisions: the author chose to generate n = 65 on the Mac and upload it (2026-11-25), as
  the VM cannot build `datagen` yet (D8 signed off at the L0 gate). The plan's first draft enforced
  the label check from n = 33 on a misread CL2.1 figure (K = (n − 1)/4); the 2D rule
  (n ≥ 65) is kept, on measurement.
- Review: plan review PASS (advisories taken: the report, the assert order); code quality
  APPROVE with findings, B+: F3, F4, F5, F8 taken; F1 (one sidecar constructor) to CL3.4;
  F2 (a shared label-check rule) not taken, per the L2 gate's Q2; F6, F7, F9 in the PR.
- 2D: unchanged bit for bit: no 2D file, function body, constant or name edited (additions
  in `catalogue.rs` only); the whole 2D suite passes.
- Follow-ups: CL3.4 (roadmap); `datagen` on the VM (FFTW), outside this objective.

## 2026-11-26: CL3.3 3D loader
- Done: `data/loaders/poisson/uniform3d.rs`: `load_poisson3d_uniform` and
  `load_poisson3d_uniform_test` (the 3D splits, 'ijk', node fields `x`, `y`, `z`, T_cu along
  every axis at load time, the same `PoissonConfig`, rows, validation and normalisation as
  2D); `load_poisson3d_cheb_field` in `cheb.rs`. The 2D and 3D loaders share one body
  (`load_uniform`, `load_uniform_test`, `read_uniform_split`, `read_cheb_field`, taking the
  node fields). First commit, no behaviour change (L1 gate C4, L2 gate Q2):
  `chebyshev/tensor.rs` (`apply_nd`, `as_matrix`, `kron`, `rank2`, `check_per_axis`),
  `transfer.rs`'s tests in `transfer/tests.rs` (973 → 336 lines), "3D D1, D2" citations,
  the `_nd` sentences, the `IMAG_TOL` and `pde/poisson` module docs. Plan
  `docs/laplace3d/tasks/CL3.3.md`, report `docs/laplace3d/results/cl3-3.md`.
- Tests: 5 new (`shapes_and_closed_form_at_any_s`, `transfer_and_normalization_bit_for_bit`,
  `subset_takes_the_selected_rows`, `invalid_files_are_rejected`,
  `cheb_loader_3d_reads_raw_u`) and the ignored `error2_n65_against_2d`: error 2 at n = 65,
  3D 2.08e-5 mean (max 4.66e-5) at s = 64 and 1.99e-4 (4.56e-4) at s = 32, against 2D
  2.03e-5 (7.81e-5) and 2.33e-4 (8.93e-4); the sidecars' s = 64 figures recomputed to
  1e-12; error 3 exactly 0 on the loader's targets.
- Checks: ran fmt, the three CI clippy jobs, `cargo clippy --all-targets`, `cargo test` (468
  passed), `cargo test --features datagen` (553 passed), `cargo doc`, the ignored
  measurement (release, Mac); not run: metal and gpu_parity (host ndarray only).
- Decisions: none. The L2 gate's "115-column line" (`collocation.rs:146`) is 115 bytes, 87
  characters: not edited, recorded in the plan.
- Review: plan review PASS (should-fix: record the 115-byte line; advisories: the sidecar
  check made unskippable, the plan's wording on a missing `z`, the module docs; all
  taken). Code quality BLOCKED (B; F1, the 3D readers copied the 2D ones) → APPROVE (A−,
  r2); F1, F3, F6, F7, F9 fixed; F2 (the measurement in the loader's tests, as
  `stage2_dataset_loads` is), F4, F5, F8 in the PR.
- 2D: unchanged bit for bit: the 2D suite unedited and passing (`normalization_matches_darcy`
  bit for bit against `Transfers::to_uniform_all`, every error message), the chebyshev D2
  tests and the datagen suite after the move; the 2D split reader now calls
  `to_uniform_all_nd`, `to_uniform_all` itself on 2D samples.
- Follow-ups: none new; CL3.4 and CL4.1 (which can use the 3D loader) stay as in the roadmap.

## 2026-11-26: CL3.4 One sidecar constructor for 2D and 3D
- Done: `pde/poisson/dataset.rs`: `SidecarFacts` and `write_sidecar` (`pub(super)`), the one
  `SidecarRecord` literal and JSON write; `write_split` (2D) and `write_split_3d` call it,
  each with its GRF's α, RNG and ε_K function. `dataset3d.rs` builds no sidecar field by
  field. The author's go to edit the 2D `write_split` body was given 2026-11-26. Plan
  `docs/laplace3d/tasks/CL3.4.md`.
- Tests: none new; `files_round_trip_3d` gains `crate_version` and `wall_time_s`. The 2D
  exact-bytes `files_round_trip` unedited and passing.
- Checks: ran fmt, the three CI clippy jobs, `cargo clippy --all-targets`, datagen clippy,
  `cargo test` (499 passed), `cargo test --features datagen` (586 passed), `cargo doc`
  (and with datagen, `-D warnings`); not run: metal and gpu_parity (host code only).
- Decisions: the author's go for the 2D function body (2026-11-26); F2 of CL3.2's review (a
  shared label-check rule) stays not taken.
- Review: plan review PASS (should-fix, the full suites on the final commit: done;
  advisory, the plan's ε_K wording: taken). Code quality APPROVE, A−: F2 (ε_K evaluated
  by the writer at K) and F3 (long doc lines) taken; F1 (one stats struct inside `Split`
  and `Split3d`) left in the PR, as it edits the 2D struct.
- 2D: unchanged bit for bit: the sidecar bytes by `files_round_trip`, the `.npz` writer not
  touched, the whole datagen suite passing.
- Follow-ups: none in the roadmap; the L3 gate is next.

## 2026-11-26: L3 gate
- Done: gate report `docs/laplace3d/gates/L3.md`: the 3D datasets of D5 written (n = 33 and
  65, 1000/200, 0.690 GB and 5.273 GB), label-checked (n = 65 max 6.2e-10, enforced ≤ 1e-8),
  loadable (error 2 at n = 65 within 15 % of the 2D floor's mean at s = 32 and 64); met. The
  roadmap's L3 gate closed 2026-11-26.
- Tests: none new or re-run; the eight files in the main checkout's `datasets/poisson/`
  re-checked for size and sidecar figures, the n = 65 SHA-256 equal to `results/cl3-2.md`;
  `main`'s CI green after #164 (run 45834459796).
- Checks: none beyond that (docs only: proposal §7 and Appendix A, roadmap, this log).
- Decisions: signed off by the author 2026-11-26: the gate with C1 (§7's measured storage
  and generation time, n = 65 generated on the Mac); Q1 "two tasks after L4" (CL3.5,
  CL3.6); Q2 "split as proposed" (CL4.2's first commit, CL3.6, the sidecar key later).
- Review: thermo-nuclear review of the four L3 PRs BLOCKED, B−, on 2D/3D duplication in
  `dataset3d.rs`, `SidecarFacts` and `Poisson3dSet`; no correctness or constraint 1 problem
  (report git-ignored, `docs/code-quality/thermo-nuclear-review-2026-11-26-laplace3d-L3-gate.md`).
  The blocker goes to CL3.5 and CL3.6 (Q1), not fixed here.
- 2D: not touched (no code).
- Follow-ups: CL4.1 (in progress), CL4.2 (its first commit per Q2), then CL3.5 and CL3.6.

## 2026-11-26: CL4.1 3D training and prediction examples
- Done: `examples/train/poisson3d.rs` (`train_poisson3d`: `FNO<5>`, D4's width 32, 4 layers,
  modes min(12, s/2)³, batch 20, Adam with `train_poisson`'s literals; n ∈ {33, 65};
  `EPOCHS` override; checkpoints and `--resume`; `costs.json`) and
  `examples/predict/poisson3d.rs` (`predict_poisson3d`: errors 2–5 in 3D, both
  cross-checks, `errors.json`, inference time). `metrics/poisson`: one private per-sample
  loop behind `poisson_errors` and `poisson_errors_nd` (the rank-3 shortcut and the 2D
  `boundary_errors` removed; `boundary_errors_nd` takes max |u|; CL1.3 review F1, N2, L1
  gate Q2); `evaluate_3d` and `ERROR1_QUOTE_3D` through a private `evaluate_core`;
  `PoissonEvaluation::plot_run_dir` replaces `run_dir`. `training/` untouched (rank-generic
  since CM0.3). Plan `docs/laplace3d/tasks/CL4.1.md`, with the short run.
- Tests: the D2 oracle `poisson_errors_frozen` (d466d46's body, verbatim) in
  `nd_rank_two_is_bit_identical_to_2d`, with a prediction zero on ∂Ω and one zero
  everywhere (asserting û = 0 on ∂Ω), both entry points and layouts;
  `evaluate_3d_differs_only_in_error1_and_the_plot_hint`. Short run on the Mac (metal):
  two epochs at n = 33, s = 32, then `predict_poisson3d` exit 0: error 2 against the sidecar
  1.4e-16, mean error 3 against test_l2 1.7e-7; errors 3, 4 ≈ 0.21 after two epochs (not a
  result). Epoch times (1,600–2,000 s) were on a loaded Mac, not a measurement.
- Checks: ran fmt, the three CI clippy jobs, `cargo clippy --all-targets`, `cargo test`
  (500 passed), `cargo doc` (also with `-D warnings`), datagen clippy, `cargo test
  --features datagen` (587 passed), all on the final code; `cargo clippy --features metal
  --examples` (the plan reviewer, before the review fixes). Not run: `gpu_parity` and metal
  tests (no backend-sensitive code changed).
- Decisions: plan Q1, the 3D errors record reuses the 2D types (the author, 2026-11-26);
  the roadmap line amended.
- Review: plan review PASS (should-fix: a broken doc link, the rank-3 shortcut; advisories:
  the oracle made byte-faithful, the D2 premise asserted, the plan's test names; all
  taken). Code quality APPROVE with findings, B+: F1 (the shortcut), F2 (`evaluate_core`,
  `plot_run_dir`), F5 (test name) fixed; F3 (a 3D set lookup in `catalogue`) to CL4.2's
  first commit; F4 (the examples' copied helpers, the price of leaving `train_poisson`
  alone) left for the objective's repo review.
- 2D: unchanged bit for bit: the frozen-oracle D2 test (the zero-on-∂Ω cases included),
  the 2D `metrics::poisson` tests unedited but for their reference function, the whole
  2D suite; `train_poisson`, `predict_poisson` and the 2D record not edited.
- Follow-ups: CL4.2's first commit (F3, roadmap).

## 2026-11-27: CL4.2 3D baseline, s = 32 pass (runs; the task stays open)
- Done: the s = 32 pass at the author's go (s = 32 only, p = 0): n = 65, seeds 42–44, 500
  epochs, one run per GPU on the VM at `69c05f6`. Error 4 mean 9.68e-3 [9.45e-3–1.01e-2],
  error 2 (transfer floor) 1.99e-4, boundary ratio 0.012–0.013 (no hard-constraint
  ablation); ~5,840 s per run. Results `docs/laplace3d/results/cl4-2.md`, with 2D Phase 5
  at s = 32 (1.33e-2, n = 129) as context.
- Tests: none new.
- Checks: `gpu_parity` 9/9 on cuda with `flex`; `predict_poisson3d` cross-check of error 3
  against `metrics.csv` (≤ 1.3e-7) on each run; run directories distinct.
- Decisions: the author chose s = 32 first and p = 0 (plan Q1, Q2, 2026-11-27); s = 64 open.
- Review: none (results and a plan fix only).
- 2D: not touched.
- Follow-ups: the plan's parity command lacked `flex` (4 of 9 failed: no Flex rfft without
  `burn/flex`); fixed in the plan and the roadmap line. s = 64 on the author's decision.

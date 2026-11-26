# 3D Poisson and Laplace roadmap

The tasks of `docs/design/3d-chebyshev-laplace-fno.md` (draft 0.2), Section 10, in the
order they are taken. The proposal is the full record: where a task's *Done when* and the
proposal differ, the proposal wins. Progress is logged in `docs/laplace3d/progress.md`;
each task's plan is `docs/laplace3d/tasks/<ID>.md`, written when it starts; each
milestone's gate report is `docs/laplace3d/gates/<milestone>.md`.

Status: **L0, L1 and L2 closed 2026-11-25 (`gates/L0.md`, `gates/L1.md`, `gates/L2.md`).** Draft 0.2 of the proposal
is the plan of record (D0–D3 signed off 2026-11-11, #117). Done: CL0.1 (rank-5 cost on the
Mac and cuda, #123; D5 signed off 2026-11-24, D4 2026-11-25 on the cuda figures), CL0.2
(rank-5 parity on metal), CL1.1–CL1.3 (CONVENTIONS version 3, N-D transfers, norms and
errors, 2D bit for bit unchanged), CL2.1 (A3, `CollocationSolver3d`, f64 floor by n = 33),
CL3.1 (the 3D GRF; D3 confirmed 2026-11-24), CL3.2 (the 3D datasets of D5, n = 33 and 65,
written 2026-11-25; n = 65 generated on the Mac and uploaded to the VM). D8 signed off at
the L0 gate. Next: CL3.3.

## How to read this file

- `- [ ]` is open, `- [x]` done. The tick is part of the task's own pull request, so a task
  counts as done once that PR is merged into `main`.
- A task is **runnable** when it is open, has no branch or open PR, and everything in its
  *Needs* is done: the tasks ticked on `main`, the decisions signed off in the table below.
  Every task also needs **D0**.
- `/next-task` (`.claude/skills/next-task`) takes the first runnable task in the order of
  this file unless the author names one; `/milestone-gate` closes a milestone by writing
  its gate report and changing its `**Gate:**` line to `**Gate (closed <YYYY-MM-DD>):**`.
- A task too big for one reviewable PR is split here into lettered sub-tasks (CL2.1a,
  CL2.1b), each with its own *Done when* and *Needs*, before it starts.
- Task IDs are `CL<milestone>.<n>`; milestones are L0–L5. Always name the objective with
  the number in prose elsewhere ("3D CL2.1").
- *Machines* names what a task's runs need besides the Mac. *Sign-off* marks a task whose
  PR ends with questions for the author.
- **Constraint 1** applies to every task: every 2D output stays bit for bit identical
  (proposal §1.4).

## Decisions

The proposal's Section 12. A decision is recorded here when the author takes it:
`signed off <YYYY-MM-DD>: <the answer as given>`.

| ID | Decision | Proposal's recommendation | Prepared by | Needed by | Status |
| --- | --- | --- | --- | --- | --- |
| D0 | Draft 0.2 of the proposal as the plan of record | after the author's review | – | every task | signed off 2026-11-11: draft 0.2 as the plan of record |
| D1 | Generalise the 2D pieces or copy them | generalise, rank 2 routed to the existing code | – | CL1.2 | signed off 2026-11-11: generalise, rank 2 routed to the existing code |
| D2 | What "bit for bit unchanged" covers for 2D, and how it is checked | the numerical outputs (proposal §9), not metadata; in-process old-against-new tests on fixed 2D inputs, plus the whole 2D suite | – | CL1.2 | signed off 2026-11-11: numbers only, in process (the numerical outputs of proposal §9, not metadata; in-process old-against-new tests on fixed 2D inputs, plus the whole 2D suite) |
| D3 | GRF parameters and K in 3D | τ = 3, α = 3, K(n) = (n − 1)/2, ε_K ≤ 5e-3 on f; alternative α = 2 with the tolerance on u | CL3.1 | CL3.2 | signed off 2026-11-11: τ = 3, α = 3; confirmed 2026-11-24 on CL3.1's measured ε_K table (#132), as recommended |
| D4 | Model size and batch at s = 32 and 64 | width 32, 4 layers, modes 8³ unless 12³ fits; batch 20 unless memory says less | CL0.1 | CL4.1 | signed off 2026-11-25: width 32, 4 layers, modes 12³, batch 20 at s = 32 and 64, as proposed and confirmed by the cuda run (#123) |
| D5 | Stages, sizes, names and storage | n = 33 (tests), 65 (training); 129 test-only if needed; 1000/200; `poisson3d_n{n}_K{K}_{split}`; precision and zip64 from CL0.1 | CL0.1 | CL3.2 | signed off 2026-11-24 (#123): n = 33 (tests and Mac runs), 65 (training), 129 test-only if the resolution study needs it; 1000 train / 200 test; `poisson3d_n{n}_K{K}_{split}`; f64, one `.npz` per split (zip64 verified on the Mac) |
| D6 | Laplace input representation | – | L4 gate | CL5.1 | open |
| D7 | Laplace boundary data family | face sine series (proposal §3.4) | L4 gate | CL5.1 | open |
| D8 | Where runs happen | n = 33 on the Mac, n = 65 generation and training on one A100 80 GB | L0 gate | CL4.2 | signed off 2026-11-25 (L0 gate, Q1): as proposed; n = 33 on the Mac for CL4.1's short runs and tests, any n = 33 baseline over seeds and n = 65 generation and training on one A100 80 GB each |

## L0: Sizing

**Gate (closed 2026-11-25):** the cost model of proposal §7 measured; D4 and D5 put to the
author. Report `docs/laplace3d/gates/L0.md`. Signed off by the author 2026-11-25: the gate,
with corrections C1 (proposal §7's measured figures) and C2 (§8's risk status) applied; Q1
D8 as proposed; Q2 the `gpu_parity.rs` tidy-up a follow-up for the next task that edits it
(CL4.2).

- [x] **CL0.1 Rank-5 FNO cost.** Peak memory and time per epoch of `FNO<5>` on random data
  at s = 32 and 64, modes 8³ and 12³, width 32, on flex, metal and cuda, at the batch sizes
  that fit; a zip64 round trip of an npz above 4 GiB. *Done when:* a table of both, with
  parameter counts and activation memory against §7; the zip64 result; D4 (with the batch)
  and D5 proposed. *Needs:* –.
  *Machines:* the VM (cuda). *Sign-off:* D4, D5.
- [x] **CL0.2 Rank-5 backend parity.** `tests/gpu_parity.rs` cases for the 3D spectral
  layer and the 3D grid channels on metal (the 2D cat and fusion bugs). *Done when:* the
  cases pass, or fail with a recorded Burn bug and a workaround. *Needs:* –.
  Report: `docs/laplace3d/tasks/CL0.2.md` and `progress.md` (2026-11-24); all pass on
  metal, worst rank-5 error 1.5e-6 against 1e-5.

## L1: Conventions and N-D shared pieces

**Gate (closed 2026-11-25):** CONVENTIONS has the 3D additions; the shared pieces take any
rank; 2D unchanged. Report `docs/laplace3d/gates/L1.md`. Signed off by the author
2026-11-25: Q1 "add, no bump, amend §9" (the 3D test names in CONVENTIONS §12
"Verification", and §9's clause that a Verification list changes without a bump); Q2 "keep
in CL4.1"; C4 "fold into CL3.3"; corrections C1, C2, C3 and the 973-line count applied.

- [x] **CL1.1 CONVENTIONS 3D additions.** Proposal §4, with a `CONVENTION_VERSION` bump.
  *Done when:* signed off by the author. *Needs:* –. *Sign-off:* the additions.
  Signed off by the author 2026-11-24 (#122, plan `docs/laplace3d/tasks/CL1.1.md`), with
  the 2D boundary norm recorded in §12 and the 2D-only statements labelled. Deferred to the
  L4 gate (D6, D7): whether the Laplace boundary norm is relative to ‖u‖ (as written) or to
  ‖g‖ on ∂Ω.
- [x] **CL1.2 N-D transfers and quadrature.** `transfer::apply`/`Transfers` and
  `cc_rel_l2_error` for any rank. *Done when:* 3D round-trip and CC-norm tests against
  closed forms; the 2D outputs bit for bit unchanged by D2's test. *Needs:* CL1.1, D1, D2.
  Report: `docs/laplace3d/tasks/CL1.2.md` and `progress.md` (2026-11-24): `apply_nd`,
  `Transfers::{to_uniform_nd, to_cheb_nd, to_uniform_all_nd}`, `cc_norm_sq_nd`,
  `cc_rel_l2_error_nd`, `cc_rel_l2_error_weighted_nd`, rank 2 routed to the 2D functions.
  Follow-ups: the 3D test names go into CONVENTIONS §12 "Verification" with CL1.3's (a
  CONVENTIONS edit, for sign-off); `chebyshev/transfer.rs` is 973 lines, so the next task
  that edits it first moves its tests to `transfer/tests.rs`.
- [x] **CL1.3 N-D errors.** `metrics/poisson` errors 2–5 for any rank, error 5 over six
  faces (proposal §3.5). *Done when:* 3D errors on synthetic fields against direct
  computation; 2D unchanged by D2's test. *Needs:* CL1.2.
  Report: `docs/laplace3d/tasks/CL1.3.md` and `progress.md` (2026-11-24):
  `poisson_errors_nd` for 2D or 3D fields (the ranks CONVENTIONS §12 defines), 2D routed
  to `poisson_errors`; `metrics/poisson` is now a folder, its tests in `poisson/tests.rs`.
  Follow-up: the 3D test names in CONVENTIONS §12 "Verification" (with CL1.2's, for
  sign-off).

## L2: 3D reference solver

**Gate (closed 2026-11-25):** A3 reaches the f64 floor on manufactured and closed-form
solutions. Report `docs/laplace3d/gates/L2.md`. Signed off by the author 2026-11-25: the
gate, with C1 (proposal §7's measured solve time) applied; Q1 "note in CL3.2" (n = 129);
Q2 the 3D label check to CL3.2, the three doc fixes to CL3.3's first commit.

- [x] **CL2.1 Collocation solver A3.** Fast diagonalisation in 3D (proposal §5).
  *Done when:* relative error (max norm and 3D CC norm) at the f64 floor by n = 33 on
  manufactured solutions and on sine-series closed forms (a test-only 3D series
  evaluator, or CL3.1's); time per solve at n = 33 and 65. *Needs:* CL1.1, CL1.2. Report:
  `docs/laplace3d/results/cl2-1.md`.

## L3: 3D GRF and datasets

**Gate:** the 3D datasets of D5 written, label-checked, loadable.

- [x] **CL3.1 3D GRF.** Separable evaluation (proposal §3.3), nested draws, the 3D S_K;
  ε_K for K = 8, 16, 32. *Done when:* the evaluation matches a direct sum on a small case;
  the ε_K table; D3 re-proposed. *Needs:* CL1.1. *Sign-off:* D3. Report:
  `docs/laplace3d/results/cl3-1.md`.
- [x] **CL3.2 3D dataset generation.** Generation, label check against the closed form,
  catalogue names carrying the dimension; the n = 65 set generated on the VM. Near the
  three-module limit: split into CL3.2a/b when it starts if it exceeds it.
  The 3D label check is a new function on `cc_rel_l2_error_nd`, sharing only
  `LABEL_CHECK_MIN_N`, `LABEL_CHECK_TOL` and `LabelCheckError` with the 2D `label_check`,
  whose body stays as it is (L2 gate Q2). An n = 129 set (D5, only if the resolution study
  needs it) is labelled only after measuring A3's floor there against `grf3d`'s closed form,
  with the label-check tolerance set from that measurement: the n = 33 → 65 trend puts it at
  about 3e-13 to 2.5e-12 (L2 gate Q1).
  *Needs:* CL1.2, CL2.1, CL3.1, D3, D5. *Machines:* the VM (n = 65). The n = 65 set was
  generated on the Mac and uploaded to the VM (the author's choice, 2026-11-25: the VM's
  `datagen` build lacks FFTW). Report: `docs/laplace3d/results/cl3-2.md`.
- [ ] **CL3.3 3D loader.** The Poisson loader for 3D splits, T_cu at load time.
  *Done when:* also error 2 of the n = 65 test split at s = 32 and 64, against the 2D floor
  at the same s (proposal §9; L1 gate C3). First commit, no behaviour change (L1 gate C4):
  `chebyshev/tensor.rs` for `apply_nd`, `as_matrix`, `kron`, the rank-2 view and one
  per-axis shape check; `transfer.rs`'s tests to `transfer/tests.rs`; "3D D1, D2" in the
  citations at `transfer.rs:189`, `clencurt.rs:130/147/166`, `metrics/poisson/mod.rs:160`;
  one doc sentence per `_nd` contract (any rank in `chebyshev`, 2D or 3D in `metrics`);
  checked by the D2 tests and the datagen suite (CL2.1 uses `apply_nd`).
  The same first commit fixes three docs in `pde/poisson` (L2 gate Q2): `IMAG_TOL`'s
  "D_xx or D_yy" to "an interior D² block" (`collocation.rs:25–28`), the module doc
  (`mod.rs:1`, 2D only), and the 115-column doc line (`collocation.rs:146`).
  *Needs:* CL1.2, CL3.2, D5.
- [ ] **CL3.4 One sidecar constructor for 2D and 3D.** CL3.2 review F1: extract the
  `SidecarRecord` construction into `pde/poisson/dataset.rs`, used by `dataset3d.rs`. (F2,
  a shared label-check rule, is not taken: the L2 gate's Q2 keeps the 2D `label_check`
  body as it is.) *Done when:* `dataset3d.rs` builds no sidecar field by field; the 2D
  exact-bytes sidecar test (`files_round_trip`) passes unedited (constraint 1). Edits a 2D
  function body: the author's go first. *Needs:* CL3.2.

## L4: 3D Poisson training and evaluation

**Gate:** a 3D Poisson baseline with errors 2–5 over seeds; D6 and D7 put to the author.

- [ ] **CL4.1 3D training and prediction examples.** `train_poisson3d` and the 3D predict
  example (`train_poisson` unchanged). *Done when:* a short run at n = 33 on the Mac writes
  a run directory that the 3D predict example evaluates. Near the three-module limit: split
  when it starts if it exceeds it. *Needs:* CL0.2, CL1.3, CL3.3, D4. The 3D `evaluate` and
  errors record are new 3D types (proposal §6). Also one per-sample loop in
  `metrics/poisson` for 2D and 3D, keeping a frozen copy of today's `poisson_errors` as
  the D2 oracle, and passing max |u| to `boundary_errors_nd` (CL1.3 review F1, N2); it
  needs no 3D types (the generic loop already gives the 2D bits, L1 gate Q2), and first
  gains a D2 case with a prediction exactly zero on ∂Ω (a −0.0 sum).
- [ ] **CL4.2 3D baseline.** Training at n = 65 over seeds. *Done when:* the results table
  with errors 2–5 against the 3D floor, and 2D Phase 5 at the same s as context (the
  input families differ, proposal §3.3). Before its first cuda run, CL0.2's cases run on
  cuda: `cargo test --release --no-default-features --features cuda --test gpu_parity`.
  Follow-up (L0 gate Q2), for whichever task next edits `tests/gpu_parity.rs`: one gated
  `mod gpu`, one `compare` helper, a named `Side` struct, and the qualified "zeroed entries
  cannot hide" doc comment; test-only, no behaviour change.
  *Needs:* CL4.1, D8. *Machines:* the VM.

## L5: 3D Laplace

**Gate:** a 3D Laplace baseline.

- [ ] **CL5.1 Boundary data and lifting.** Planned at the L4 gate (D6, D7) and split there
  into sub-tasks (data family, discrete-lifting solver, loader, training, baseline).
  *Needs:* L4, D6, D7.

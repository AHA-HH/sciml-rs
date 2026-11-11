# 3D Poisson and Laplace roadmap

The tasks of `docs/design/3d-chebyshev-laplace-fno.md` (draft 0.2), Section 10, in the
order they are taken. The proposal is the full record: where a task's *Done when* and the
proposal differ, the proposal wins. Progress is logged in `docs/laplace3d/progress.md`;
each task's plan is `docs/laplace3d/tasks/<ID>.md`, written when it starts; each
milestone's gate report is `docs/laplace3d/gates/<milestone>.md`.

Status: **planning.** Draft 0.1 written 2026-11-11 and reviewed the same day; draft 0.2
(2026-11-11) answers the review. D0–D3 signed off 2026-11-11 (on #117); next CL0.1,
then CL0.2 and CL1.1, runnable once #117 is merged.

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
| D3 | GRF parameters and K in 3D | τ = 3, α = 3, K(n) = (n − 1)/2, ε_K ≤ 5e-3 on f; alternative α = 2 with the tolerance on u | CL3.1 | CL3.2 | signed off 2026-11-11: τ = 3, α = 3 |
| D4 | Model size and batch at s = 32 and 64 | width 32, 4 layers, modes 8³ unless 12³ fits; batch 20 unless memory says less | CL0.1 | CL4.1 | open |
| D5 | Stages, sizes, names and storage | n = 33 (tests), 65 (training); 129 test-only if needed; 1000/200; `poisson3d_n{n}_K{K}_{split}`; precision and zip64 from CL0.1 | CL0.1 | CL3.2 | open |
| D6 | Laplace input representation | – | L4 gate | CL5.1 | open |
| D7 | Laplace boundary data family | face sine series (proposal §3.4) | L4 gate | CL5.1 | open |
| D8 | Where runs happen | n = 33 on the Mac, n = 65 generation and training on one A100 80 GB | – | CL4.2 | open |

## L0: Sizing

**Gate:** the cost model of proposal §7 measured; D4 and D5 put to the author.

- [ ] **CL0.1 Rank-5 FNO cost.** Peak memory and time per epoch of `FNO<5>` on random data
  at s = 32 and 64, modes 8³ and 12³, width 32, on flex, metal and cuda, at the batch sizes
  that fit; a zip64 round trip of an npz above 4 GiB. *Done when:* a table of both, with
  parameter counts and activation memory against §7; the zip64 result; D4 (with the batch)
  and D5 proposed. *Needs:* –.
  *Machines:* the VM (cuda). *Sign-off:* D4, D5.
- [ ] **CL0.2 Rank-5 backend parity.** `tests/gpu_parity.rs` cases for the 3D spectral
  layer and the 3D grid channels on metal (the 2D cat and fusion bugs). *Done when:* the
  cases pass, or fail with a recorded Burn bug and a workaround. *Needs:* –.

## L1: Conventions and N-D shared pieces

**Gate:** CONVENTIONS has the 3D additions; the shared pieces take any rank; 2D unchanged.

- [ ] **CL1.1 CONVENTIONS 3D additions.** Proposal §4, with a `CONVENTION_VERSION` bump.
  *Done when:* signed off by the author. *Needs:* –. *Sign-off:* the additions.
- [ ] **CL1.2 N-D transfers and quadrature.** `transfer::apply`/`Transfers` and
  `cc_rel_l2_error` for any rank. *Done when:* 3D round-trip and CC-norm tests against
  closed forms; the 2D outputs bit for bit unchanged by D2's test. *Needs:* CL1.1, D1, D2.
- [ ] **CL1.3 N-D errors.** `metrics/poisson` errors 2–5 for any rank, error 5 over six
  faces (proposal §3.5). *Done when:* 3D errors on synthetic fields against direct
  computation; 2D unchanged by D2's test. *Needs:* CL1.2.

## L2: 3D reference solver

**Gate:** A3 reaches the f64 floor on manufactured and closed-form solutions.

- [ ] **CL2.1 Collocation solver A3.** Fast diagonalisation in 3D (proposal §5).
  *Done when:* relative error (max norm and 3D CC norm) at the f64 floor by n = 33 on
  manufactured solutions and on sine-series closed forms (a test-only 3D series
  evaluator, or CL3.1's); time per solve at n = 33 and 65. *Needs:* CL1.1, CL1.2.

## L3: 3D GRF and datasets

**Gate:** the 3D datasets of D5 written, label-checked, loadable.

- [ ] **CL3.1 3D GRF.** Separable evaluation (proposal §3.3), nested draws, the 3D S_K;
  ε_K for K = 8, 16, 32. *Done when:* the evaluation matches a direct sum on a small case;
  the ε_K table; D3 re-proposed. *Needs:* CL1.1. *Sign-off:* D3.
- [ ] **CL3.2 3D dataset generation.** Generation, label check against the closed form,
  catalogue names carrying the dimension; the n = 65 set generated on the VM. Near the
  three-module limit: split into CL3.2a/b when it starts if it exceeds it.
  *Needs:* CL1.2, CL2.1, CL3.1, D3, D5. *Machines:* the VM (n = 65).
- [ ] **CL3.3 3D loader.** The Poisson loader for 3D splits, T_cu at load time.
  *Needs:* CL1.2, CL3.2, D5.

## L4: 3D Poisson training and evaluation

**Gate:** a 3D Poisson baseline with errors 2–5 over seeds; D6 and D7 put to the author.

- [ ] **CL4.1 3D training and prediction examples.** `train_poisson3d` and the 3D predict
  example (`train_poisson` unchanged). *Done when:* a short run at n = 33 on the Mac writes
  a run directory that the 3D predict example evaluates. Near the three-module limit: split
  when it starts if it exceeds it. *Needs:* CL0.2, CL1.3, CL3.3, D4.
- [ ] **CL4.2 3D baseline.** Training at n = 65 over seeds. *Done when:* the results table
  with errors 2–5 against the 3D floor, and 2D Phase 5 at the same s as context (the
  input families differ, proposal §3.3).
  *Needs:* CL4.1, D8. *Machines:* the VM.

## L5: 3D Laplace

**Gate:** a 3D Laplace baseline.

- [ ] **CL5.1 Boundary data and lifting.** Planned at the L4 gate (D6, D7) and split there
  into sub-tasks (data family, discrete-lifting solver, loader, training, baseline).
  *Needs:* L4, D6, D7.

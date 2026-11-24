# Optimiser benchmarks roadmap

The tasks of `docs/design/optimisers.md` (draft 0.1), Section 10, in the order they are
taken. The proposal is the full record: where a task's *Done when* and the proposal differ,
the proposal wins. Progress is logged in `docs/optimisers/progress.md`; each task's plan is
`docs/optimisers/tasks/<ID>.md`, written when it starts; each milestone's gate report is
`docs/optimisers/gates/<milestone>.md`. Reading notes: `docs/optimisers/references.md`.

Status: **planning.** Draft 0.1 written 2026-11-24; nothing signed off. D0 first.

## How to read this file

- `- [ ]` is open, `- [x]` done. The tick is part of the task's own pull request, so a task
  counts as done once that PR is merged into `main`.
- A task is **runnable** when it is open, has no branch or open PR, and everything in its
  *Needs* is done: the tasks ticked on `main`, the decisions signed off in the table below.
  Every task also needs **D0**.
- `/next-task` (`.claude/skills/next-task`) takes the first runnable task in the order of
  this file unless the author names one; `/milestone-gate` closes a milestone by writing
  its gate report and changing its `**Gate:**` line to `**Gate (closed <YYYY-MM-DD>):**`.
- A task too big for one reviewable PR is split here into lettered sub-tasks (CO2.2a,
  CO2.2b), each with its own *Done when* and *Needs*, before it starts.
- Task IDs are `CO<milestone>.<n>`; milestones are O0–O4. Always name the objective with
  the number in prose elsewhere ("optimisers CO1.1").
- *Needs* may name another objective's task by its ID (`CL4.2` is the 3D objective's); it is
  met when ticked on `main` in that objective's roadmap.
- *Machines* names what a task's runs need besides the Mac. *Sign-off* marks a task whose
  PR ends with questions for the author.
- **Constraint 1** applies to every task: every existing output stays bit for bit identical,
  the Adam and Gauss–Newton paths included (proposal §1.4).

## Decisions

The proposal's Section 12. A decision is recorded here when the author takes it:
`signed off <YYYY-MM-DD>: <the answer as given>`.

| ID | Decision | Proposal's recommendation | Prepared by | Needed by | Status |
| --- | --- | --- | --- | --- | --- |
| D0 | Draft 0.1 of the proposal as the plan of record | after the author's review | – | every task | open |
| D1 | The reference Gauss–Newton arm | GN200-D at μ = 0; CG cap 50 or 150 from CO0.1 | CO0.1 | CO3.1 | open |
| D2 | Seeds and budgets | seeds 42–46; equal time (CO0.1's Adam mean) and 5,400 s, every arm at both | – | CO0.1 | open |
| D3 | L-BFGS's objective and weight decay | Φ at μ = 0 | – | CO1.1 | open |
| D4 | L-BFGS's batch | one batch of 1,000; accumulation if memory requires | – | CO1.1 | open |
| D5 | L-BFGS's settings | strong Wolfe, lr 1; history and `max_iter` from the pilot | CO1.2 | CO3.1 | open |
| D6 | SOAP's hyperparameters | lr 1e-3 cosine, wd 1e-4, β (0.95, 0.95), f = 10; pilot lr ∈ {1e-3, 3e-3}, f ∈ {10, 50} | CO2.3 | CO3.1 | open |
| D7 | Pass accounting for the new arms | one forward and one backward per SOAP step and per L-BFGS closure evaluation; SOAP's preconditioner work as time | – | CO1.1, CO2.3 | open |
| D8 | SOAP's handling of the spectral tensors | one factor per dimension; re and im separate; biases on Adam | – | CO2.2 | open |
| D9 | Where SOAP's linear algebra runs | `linalg` enabled; first basis by host `svd`, refreshes by device `qr` if CO2.1 passes, else host | CO2.1 | CO2.2 | open |

## O0: Re-baseline on Burn 0.23

**Gate:** the reference arms measured on Burn 0.23 with five seeds; the equal-time budget
set; D1 put to the author; where a Gauss–Newton step's time goes.

- [ ] **CO0.1 Reference arms on Burn 0.23.** Adam (Phase 5 settings) on seeds 42–46, 500
  epochs, giving the equal-time budget; Adam at 5,400 s on seeds 42–46; GN200-D at μ = 0
  with CG cap 50 and 150 on seed 42 at both budgets, then the chosen cap on seeds 42–46 at
  both. *Done when:* a results table (`docs/optimisers/results/CO0.1.md`) against Phase 10's
  fork-era numbers, and D1 proposed. *Needs:* D2. *Machines:* the VM. *Sign-off:* D1.
- [ ] **CO0.2 Cost profile of a Gauss–Newton step.** Time per phase of one GN200-D step
  (the gradient, CG's JVPs and VJPs, backtracking, the Hutchinson refresh), passes per
  second on Burn 0.23 against Phase 10's. *Done when:* the profile table and a short list of
  where passes could be saved (for later work; nothing changed here). *Needs:* –.
  *Machines:* the VM.

## O1: L-BFGS

**Gate:** an L-BFGS arm trains, resumes exactly and counts its passes; its settings piloted.

- [ ] **CO1.1 The L-BFGS arm.** `OptimizerConfig::Lbfgs`, `train_epoch_lbfgs` with the
  closure over Φ on the full batch, passes into `costs.json`, the checkpoint section,
  `OPTIMIZER=lbfgs` in `train_poisson`. *Done when:* tests of proposal §9 pass; a flex smoke
  run and its resume; constraint 1 shown. *Needs:* D3, D4, D7.
- [ ] **CO1.2 L-BFGS pilot.** Seed 42 at the equal-time budget: history ∈ {10, 100},
  `max_iter` ∈ {1, 20}, strong Wolfe. *Done when:* the pilot table and D5 proposed.
  *Needs:* CO1.1, CO0.1. *Machines:* the VM. *Sign-off:* D5.

## O2: SOAP

**Gate:** a SOAP optimiser that passes proposal §9's tests trains the FNO; its settings
piloted.

- [ ] **CO2.1 Burn linear algebra for SOAP.** Enable `linalg`; `qr` and `svd` at 12, 32,
  128 (and 144) against an f64 host reference on flex, metal and cuda: error and time.
  *Done when:* the table; constraint 1 shown with the feature on; D9 proposed. *Needs:* –.
  *Machines:* the VM (cuda). *Sign-off:* D9.
- [ ] **CO2.2 The SOAP optimiser.** `optim/soap/`: `SoapConfig`, `Soap` as a Burn
  `Optimizer`, any tensor rank. *Done when:* the SOAP tests of proposal §9 pass (the
  identity-basis reduction to Adam, the quadratic, ranks 1–5, the refresh, record round
  trip). *Needs:* CO2.1, D8, D9.
- [ ] **CO2.3 The SOAP arm and pilot.** `OptimizerConfig::Soap`, `OPTIMIZER=soap`; resume
  exact; then the pilot of D6 on seed 42 at the equal-time budget. *Done when:* a flex smoke
  run and its resume, constraint 1 shown, the pilot table, D6 proposed. *Needs:* CO2.2,
  CO0.1, D7. *Machines:* the VM. *Sign-off:* D6.

## O3: The 2D comparison

**Gate:** the four arms compared over five seeds at both budgets; results written.

- [ ] **CO3.1 The matrix.** Adam, GN (D1), L-BFGS (D5), SOAP (D6), seeds 42–46, both
  budgets (reusing CO0.1's Adam and GN runs); `poisson_matrix --by-optimizer` extended to
  the new arms. *Done when:* `docs/optimisers/results/CO3.1.md` with error 4 against time
  and passes per arm, and the equal-time verdict with its spread. *Needs:* CO0.1, CO1.2,
  CO2.3, D1, D5, D6. *Machines:* the VM.

## O4: The 3D arms

**Gate:** the arms O3 keeps, compared on the 3D Poisson baseline.

- [ ] **CO4.1 3D comparison.** The arms the author keeps after O3, on the 3D objective's
  Poisson baseline, at the protocol of proposal §3 adapted to 3D (budget from its Adam
  baseline). *Done when:* the 3D results table. *Needs:* CO3.1, CL4.2. *Machines:* the VM.

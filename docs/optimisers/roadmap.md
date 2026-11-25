# Optimiser benchmarks roadmap

The tasks of `docs/design/optimisers.md` (draft 0.2), Section 10, in the order they are
taken. The proposal is the full record: where a task's *Done when* and the proposal differ,
the proposal wins. Progress is logged in `docs/optimisers/progress.md`; each task's plan is
`docs/optimisers/tasks/<ID>.md`, written when it starts; each milestone's gate report is
`docs/optimisers/gates/<milestone>.md`. Reading notes: `docs/optimisers/references.md`.

Status: **planning.** Draft 0.1 written 2026-11-24, draft 0.2 (the Burgers replication as
O1; the former O1–O4 renumbered O2–O5) the same day, corrected after an independent check
(#128). D0 and D2 signed off 2026-11-24 (D2 amended 2026-11-25). 2026-11-25: CO0.1's code
and run plan in its PR (draft), its VM runs waiting for the author's go
(`docs/optimisers/tasks/CO0.1.md`); next CO0.2 and CO3.1.

## How to read this file

- `- [ ]` is open, `- [x]` done. The tick is part of the task's own pull request, so a task
  counts as done once that PR is merged into `main`.
- A task is **runnable** when it is open, has no branch or open PR, and everything in its
  *Needs* is done: the tasks ticked on `main`, the decisions signed off in the table below.
  Every task also needs **D0**.
- `/next-task` (`.claude/skills/next-task`) takes the first runnable task in the order of
  this file unless the author names one; `/milestone-gate` closes a milestone by writing
  its gate report and changing its `**Gate:**` line to `**Gate (closed <YYYY-MM-DD>):**`.
- A task too big for one reviewable PR is split here into lettered sub-tasks (CO3.2a,
  CO3.2b), each with its own *Done when* and *Needs*, before it starts.
- Task IDs are `CO<milestone>.<n>`; milestones are O0–O5. Always name the objective with
  the number in prose elsewhere ("optimisers CO2.1").
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
| D0 | Draft 0.2 of the proposal as the plan of record | after the author's review | – | every task | signed off 2026-11-24: draft 0.2, with the corrections of #128, as the plan of record |
| D1 | The reference Gauss–Newton arm | GN200-D at μ = 0; CG cap 50 or 150 from CO0.1 | CO0.1 | CO4.1 | open |
| D2 | Seeds and budgets | seeds 42–46; equal time (CO0.1's Adam mean) and 5,400 s, every arm at both | – | CO0.1 | signed off 2026-11-24: as recommended (seeds 42–46; equal time and 5,400 s, every arm at both); amended 2026-11-25 by the author: seeds 42–44 at 5,400 s, 42–46 at equal time |
| D3 | L-BFGS's objective and weight decay | Φ at μ = 0 | – | CO2.1 | open |
| D4 | L-BFGS's batch | one batch of 1,000; accumulation if memory requires | – | CO2.1 | open |
| D5 | L-BFGS's settings | strong Wolfe, lr 1; history and `max_iter` from the pilot | CO2.2 | CO4.1 | open |
| D6 | SOAP's hyperparameters | lr 1e-3 cosine, wd 1e-4, β (0.95, 0.95), f = 10; pilot lr ∈ {1e-3, 3e-3}, f ∈ {10, 50} | CO3.3 | CO4.1 | open |
| D7 | Pass accounting for the new arms | one forward and one backward per SOAP step and per L-BFGS closure evaluation; SOAP's preconditioner work as time | – | CO2.1, CO3.3 | open |
| D8 | SOAP's handling of the spectral tensors | one factor per dimension; re and im separate; biases on Adam | – | CO3.2 | open |
| D9 | Where SOAP's linear algebra runs | `linalg` enabled; first basis by host `svd`, refreshes by device `qr` if CO3.1 passes, else host | CO3.1 | CO3.2 | open |
| D10 | O1's fidelity to PETScML | their data, model (GELU option), loss (eq. 6, squared-loss option for Adam), metric and reference recipe; f32, our Levenberg–Marquardt GN; reproduced if A1's mean is within a factor 2 of 0.0014 | – | CO1.1 | open |
| D11 | O1's Adam arms | AdamW, lr 1e-3 halved every 100 epochs, batch 20, 500 epochs, ε 1e-8, weight decay 1e-4 (A1) and 0 (A0); our Adam (coupled L2) at 0 and 1e-4 as two more cells | – | CO1.1 | open |
| D12 | O1's Gauss–Newton arm | full batch, CG cap 100, Phase 10's damping, the Hutchinson diagonal (fixed, not D1's), μ = 0 (G0) and 1e-4 (G1); G0-200, G0 at batch 200 | – | CO1.2 | open |
| D13 | Weight decay as a factor in O4 | every arm at weight decay 0 and 1e-4 (D3 and D6 widened); doubles O4's cost | – | CO2.1, CO3.3, CO4.1 | open |
| D14 | Burgers in O4 | the four arms on O1's Burgers setting, equal time, seeds 42–46 | – | CO4.1 | open |

## O0: Re-baseline on Burn 0.23

**Gate:** the reference arms measured on Burn 0.23 with five seeds; the equal-time budget
set; D1 put to the author; where a Gauss–Newton step's time goes.

- [ ] **CO0.1 Reference arms on Burn 0.23.** Adam (Phase 5 settings) on seeds 42–46, 500
  epochs, giving the equal-time budget; Adam at 5,400 s on seeds 42–44; GN200-D at μ = 0
  with CG cap 50 and 150 on seed 42 at both budgets, then the chosen cap on seeds 43–46 at
  equal time and 43–44 at 5,400 s (D2). Timed runs side by side on the two GPUs only after a
  check that the epoch times match a run alone (the step is host-bound, Phase 9 T3). *Done when:* a results table (`docs/optimisers/results/CO0.1.md`) against Phase 10's
  fork-era numbers, and D1 proposed. *Needs:* D2. *Machines:* the VM. *Sign-off:* D1.
- [ ] **CO0.2 Cost profile of a Gauss–Newton step.** Time per phase of one GN200-D step
  (the gradient, CG's JVPs and VJPs, backtracking, the Hutchinson refresh), passes per
  second on Burn 0.23 against Phase 10's. *Done when:* the profile table and a short list of
  where passes could be saved (for later work; nothing changed here). *Needs:* –.
  *Machines:* the VM.

## O1: The Burgers replication and the Darcy extension

**Gate:** PETScML's Burgers reference reproduced (D10's band) or the gap explained; the
2 × 2 {AdamW, Gauss–Newton} × {weight decay 0, 1e-4} on Burgers over five seeds at equal time and
three at the long budget (D2); the same on Darcy at equal time; the verdict on how much of PETScML's gap the
weight decay accounts for.

- [ ] **CO1.1 The reference recipe.** `OptimizerConfig::AdamW` (Burn's `AdamWConfig`, ε
  1e-8), `TrainingConfig.lr_schedule` (`None` = today's cosine; a step variant halving every
  k epochs), an `FNOConfig` GELU option and a squared relative-L² option for the Adam path (both off by default), `OPTIMIZER=adamw`, `WEIGHT_DECAY` and the subsample rate as variables of
  `train_burgers` with n_train = 1,000 (1,024 today), and `costs.json` for Burgers runs;
  s = 128 and the 1,000/100 split checked against Lu et al. 2022 and PETScML's scripts;
  Burn's `StepLrSchedulerConfig` reused for the step schedule. *Done when:* the O1 tests of proposal
  §9 pass (the parameter count 549,569, AdamW at weight decay 0 equal to Adam at 0 under the
  same schedule, the step schedule's halvings); A1 on seed 42 on flex or metal against
  0.0014; constraint 1 shown (`train_burgers` and `train_darcy` with nothing set unchanged
  bit for bit). *Needs:* D10, D11, the library objective's CM0.3 (`training/` changes are
  serial, the author's sequencing of 2026-11-24).
- [ ] **CO1.2 Gauss–Newton on Burgers and Darcy.** `OPTIMIZER=gauss-newton` and the `GN_*`
  variables in `train_burgers` and `train_darcy`, as in `train_poisson`; `FnoResidual` tested
  on a rank-3 batch; `gn_steps.csv` and passes written. *Done when:* a flex smoke run of G0
  and G1 on Burgers and of G0 on Darcy, a resume of one; constraint 1 shown. *Needs:*
  CO1.1, D12.
- [ ] **CO1.3 The Burgers 2 × 2.** A0, A1, G0, G1 and G0-200 (G0 at batch 200, D12) on seeds 42–46 at A1's mean time and
  42–44 at 6 × it (proposal §3.1, D2); the coupled-Adam cells if D11 keeps them. *Done when:*
  `docs/optimisers/results/CO1.3.md`: test L² per cell (mean and range), against epochs,
  passes, PETScML's oracle calls and time; A1 against D10's band; the verdict on A1 → G0
  split into A1 → A0 (weight decay) and A0 → G0 (the optimiser). *Needs:* CO1.2.
  *Machines:* the VM.
- [ ] **CO1.4 The Darcy 2 × 2.** The same four cells on Darcy at `train_darcy`'s setting,
  seeds 42–46, equal time only. *Done when:* `docs/optimisers/results/CO1.4.md` and whether
  the Burgers verdict carries over. *Needs:* CO1.3. *Machines:* the VM.

## O2: L-BFGS

**Gate:** an L-BFGS arm trains, resumes exactly and counts its passes; its settings piloted.

- [ ] **CO2.1 The L-BFGS arm.** `OptimizerConfig::Lbfgs`, `train_epoch_lbfgs` with the
  closure over Φ on the full batch, passes into `costs.json`, the checkpoint section,
  `OPTIMIZER=lbfgs` in `train_poisson`. *Done when:* tests of proposal §9 pass; a flex smoke
  run and its resume; constraint 1 shown. *Needs:* D3, D4, D7, D13, the library
  objective's CM0.3, and CO1.1 merged (both change `training/`; never at once).
- [ ] **CO2.2 L-BFGS pilot.** Seed 42 at the equal-time budget: history ∈ {10, 100},
  `max_iter` ∈ {1, 20}, strong Wolfe. *Done when:* the pilot table and D5 proposed.
  *Needs:* CO2.1, CO0.1. *Machines:* the VM. *Sign-off:* D5.

## O3: SOAP

**Gate:** a SOAP optimiser that passes proposal §9's tests trains the FNO; its settings
piloted.

- [ ] **CO3.1 Burn linear algebra for SOAP.** Enable `linalg`; `qr` and `svd` at 12, 32,
  128 (and 144) against an f64 host reference on flex, metal and cuda: error and time.
  *Done when:* the table; constraint 1 shown with the feature on; D9 proposed. *Needs:* –.
  *Machines:* the VM (cuda). *Sign-off:* D9.
- [ ] **CO3.2 The SOAP optimiser.** `optim/soap/`: `SoapConfig`, `Soap` as a Burn
  `Optimizer`, any tensor rank. *Done when:* the SOAP tests of proposal §9 pass (the
  identity-basis reduction to Adam, the quadratic, ranks 1–5, the refresh, record round
  trip). *Needs:* CO3.1, D8, D9.
- [ ] **CO3.3 The SOAP arm and pilot.** `OptimizerConfig::Soap`, `OPTIMIZER=soap`; resume
  exact; then the pilot of D6 on seed 42 at the equal-time budget. *Done when:* a flex smoke
  run and its resume, constraint 1 shown, the pilot table, D6 proposed. *Needs:* CO3.2,
  CO0.1, D7, D13, the library objective's CM0.3 (`training/` changes are serial; not at
  once with CO1.1, CO1.2 or CO2.1). *Machines:* the VM. *Sign-off:* D6.

## O4: The 2D comparison

**Gate:** the four arms compared over five seeds at equal time and three at the long budget;
results written.

- [ ] **CO4.1 The matrix.** Adam, GN (D1), L-BFGS (D5), SOAP (D6), seeds 42–46 at equal
  time and 42–44 at 5,400 s (D2) (reusing CO0.1's Adam and GN runs); `poisson_matrix --by-optimizer` extended to
  the new arms, and its arm labels to the CG cap and the budget (CO0.1 tabulates them by
  separate calls); each arm at weight decay 0 and 1e-4 if D13 is taken, and the four arms on
  Burgers at equal time if D14 is. *Done when:* `docs/optimisers/results/CO4.1.md` with
  error 4 against time and passes per arm, and the equal-time verdict with its spread; with
  D13, whether weight decay's effect splits by whether the arm reaches the penalised
  minimiser (full-batch GN and L-BFGS) or is stochastic first-order (Adam, SOAP). *Needs:* CO0.1, CO2.2, CO3.3, D1, D5, D6, D13, D14
  (CO1.3 if D14 is taken). *Machines:* the VM.

## O5: The 3D arms

**Gate:** the arms O4 keeps, compared on the 3D Poisson baseline.

- [ ] **CO5.1 3D comparison.** The arms the author keeps after O4, on the 3D objective's
  Poisson baseline, at the protocol of proposal §3 adapted to 3D (budget from its Adam
  baseline). *Done when:* the 3D results table. *Needs:* CO4.1, CL4.2. *Machines:* the VM.

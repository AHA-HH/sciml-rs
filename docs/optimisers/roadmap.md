# Optimiser benchmarks roadmap

The tasks of `docs/design/optimisers.md` (draft 0.2), Section 10, in the order they are
taken. The proposal is the full record: where a task's *Done when* and the proposal differ,
the proposal wins. Progress is logged in `docs/optimisers/progress.md`; each task's plan is
`docs/optimisers/tasks/<ID>.md`, written when it starts; each milestone's gate report is
`docs/optimisers/gates/<milestone>.md`. Reading notes: `docs/optimisers/references.md`.

Status: **planning.** Draft 0.1 written 2026-11-24, draft 0.2 (the Burgers replication as
O1; the former O1–O4 renumbered O2–O5) the same day, corrected after an independent check
(#128). D0 and D2 signed off 2026-11-24. CO3.1 (2026-11-24, cuda 2026-11-25): `linalg` enabled,
constraint 1 shown, flex, metal and cuda measured; D9 signed off 2026-11-25. Next CO0.1 and
CO0.2 (the VM, running at the author's go, 2026-11-25); D8, D10 and D11 signed off
2026-11-25, so CO3.2 (SOAP) and CO1.1 (the Burgers recipe; after CM0.3, merged) are runnable.
D3, D4, D7, D12 and D13 signed off 2026-11-25: CO1.2 runs after CO1.1, CO3.3 after CO3.2
and CO0.1, CO2.1 after CO1.1 (all three change `training/`, one at a time). CO3.2
(2026-11-25): `optim/soap/`, SOAP as Burn's per-parameter `Optimizer`, the §9 tests
passing; next for O3, CO3.3 (its D7 and D13 now signed off). CO0.2
(2026-11-25): the Gauss–Newton step profile on cuda, `docs/optimisers/results/CO0.2.md`.
CO1.1 done 2026-11-25 (A1 seed 42 inside D10's band); next in O1 CO1.2 (D12 signed off), and
the library's CM0.6 may take `training/` once CO1.1 is merged. CO1.2 done 2026-11-26: every
O1 cell runs from `train_burgers` and `train_darcy` (`EPOCHS` added, the author's choice of
2026-11-26); next CO1.3 on the VM, and `training/` free for CO2.1, CO1.5, CM0.6, CO3.3.
CO0.1 done 2026-11-27 (B = 2,952 s, E = 915; D1 cap 150): at equal time GN200-D at μ = 0
is ahead of Adam on average (6.34e-3 against 1.03e-2, 4 of 5 seeds), at 5,400 s on all
three; next in O0 CO0.3 (the Adam step profile). CO0.4 done 2026-11-27: the
training examples read the commit at the start, and `RunDir::create` never reuses a
directory; its leftovers are CO0.5. CO2.1 done 2026-11-27: the L-BFGS arm
(`OPTIMIZER=lbfgs` in `train_poisson`), resumed bit for bit on flex; next in O2 CO2.2
(the pilot, on the VM; its grid now names `max_eval`), and `training/` free again. CO1.5
done 2026-11-27: `trainer.rs`'s tests in `training/trainer/tests.rs`, the same 26 by name;
`training/` free for CM0.6 and CO3.3 (and the optional CO1.6).

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
| D1 | The reference Gauss–Newton arm | GN200-D at μ = 0; CG cap 50 or 150 from CO0.1 | CO0.1 | CO4.1 | signed off 2026-11-26: cap 150 (CO0.1's proposal; given with stage 3's go, through the coordinating session) |
| D2 | Seeds and budgets | seeds 42–46; equal time (CO0.1's Adam mean) and 5,400 s, every arm at both | – | CO0.1 | signed off 2026-11-24: as recommended (seeds 42–46; equal time and 5,400 s, every arm at both); amended 2026-11-25 by the author: seeds 42–44 at 5,400 s, 42–46 at equal time; 2026-11-25, after Adam measured about 3.6× slower on Burn 0.23 (CO0.1): keep 5,400 s |
| D3 | L-BFGS's objective and weight decay | Φ at μ = 0 | – | CO2.1 | signed off 2026-11-25: Φ, Gauss–Newton's objective, at μ ∈ {0, 1e-4} (widened by D13; ½μ‖θ‖² in the closure) |
| D4 | L-BFGS's batch | one batch of 1,000; accumulation if memory requires | – | CO2.1 | signed off 2026-11-25: as recommended |
| D5 | L-BFGS's settings | strong Wolfe, lr 1; history and `max_iter` from the pilot | CO2.2 | CO4.1 | open |
| D6 | SOAP's hyperparameters | lr 1e-3 cosine, wd 1e-4, β (0.95, 0.95), f = 10; pilot lr ∈ {1e-3, 3e-3}, f ∈ {10, 50} | CO3.3 | CO4.1 | open |
| D7 | Pass accounting for the new arms | one forward and one backward per SOAP step and per L-BFGS closure evaluation; SOAP's preconditioner work as time | – | CO2.1, CO3.3 | signed off 2026-11-25: as recommended |
| D8 | SOAP's handling of the spectral tensors | one factor per dimension; re and im separate; biases on Adam | – | CO3.2 | signed off 2026-11-25: option A as recommended (one factor per dimension; re and im separate; biases on Adam) |
| D9 | Where SOAP's linear algebra runs | `linalg` enabled; first basis by host `svd`, refreshes by device `qr` if CO3.1 passes, else host | CO3.1 | CO3.2 | signed off 2026-11-25: host `svd` for the first basis and host QR for the refreshes, on every backend (one code path; CO3.1, `spikes/linalg/REPORT.md`) |
| D10 | O1's fidelity to PETScML | their data, model (GELU option), loss (eq. 6, squared-loss option for Adam), metric and reference recipe; f32, our Levenberg–Marquardt GN; reproduced if A1's mean is within a factor 2 of 0.0014 | – | CO1.1 | signed off 2026-11-25: as recommended |
| D11 | O1's Adam arms | AdamW, lr 1e-3 halved every 100 epochs, batch 20, 500 epochs, ε 1e-8, weight decay 1e-4 (A1) and 0 (A0); our Adam (coupled L2) at 0 and 1e-4 as two more cells | – | CO1.1 | signed off 2026-11-25: as recommended, the coupled-Adam cells included |
| D12 | O1's Gauss–Newton arm | full batch, CG cap 100, Phase 10's damping, the Hutchinson diagonal (fixed, not D1's), μ = 0 (G0) and 1e-4 (G1); G0-200, G0 at batch 200 | – | CO1.2 | signed off 2026-11-25: as recommended, G0-200 included |
| D13 | Weight decay as a factor in O4 | every arm at weight decay 0 and 1e-4 (D3 and D6 widened); doubles O4's cost | – | CO2.1, CO3.3, CO4.1 | signed off 2026-11-25: every arm at weight decay 0 and 1e-4 |
| D14 | Burgers in O4 | the four arms on O1's Burgers setting, equal time, seeds 42–46 | – | CO4.1 | open |

## O0: Re-baseline on Burn 0.23

**Gate:** the reference arms measured on Burn 0.23 with five seeds; the equal-time budget
set; D1 put to the author; where a Gauss–Newton step's time goes.

- [x] **CO0.1 Reference arms on Burn 0.23.** (`docs/optimisers/results/CO0.1.md`) Adam (Phase 5 settings) on seeds 42–46, 500
  epochs, giving the equal-time budget; Adam at 5,400 s on seeds 42–44; GN200-D at μ = 0
  with CG cap 50 and 150 on seed 42 at both budgets, then the chosen cap on seeds 43–46 at
  equal time and 43–44 at 5,400 s (D2). Timed runs side by side on the two GPUs only after a
  check that the epoch times match a run alone (the step is host-bound, Phase 9 T3). *Done when:* a results table (`docs/optimisers/results/CO0.1.md`) against Phase 10's
  fork-era numbers, and D1 proposed. *Needs:* D2. *Machines:* the VM. *Sign-off:* D1.
- [x] **CO0.2 Cost profile of a Gauss–Newton step.** (`docs/optimisers/results/CO0.2.md`) Time per phase of one GN200-D step
  (the gradient, CG's JVPs and VJPs, backtracking, the Hutchinson refresh), passes per
  second on Burn 0.23 against Phase 10's. *Done when:* the profile table and a short list of
  where passes could be saved (for later work; nothing changed here). *Needs:* –.
  *Machines:* the VM.
- [ ] **CO0.3 Cost profile of an Adam step.** Time per phase of one Adam step at batch 20
  (forward, backward, the optimiser update, the loader, the per-epoch evaluation and
  checkpoint) on Burn 0.23 against the fork, to explain why CO0.1's Adam epoch took about
  3.6× the fork's time (3,067 s for 500 epochs against 859.4 s) while a GN200 step slowed
  about 24 %. *Done when:* the profile table and where the extra time goes (nothing changed
  here; a fix, if any, is a later task, since it would move the equal-time budget).
  *Needs:* –. *Machines:* the VM.
- [x] **CO0.4 Run provenance: the commit at the start, one directory per run.** Two
  faults CO0.1 hit. (1) `train_poisson` and `train_burgers` call `git_commit()` (`git
  rev-parse HEAD` in the source checkout) when they write `costs.json` at the end of a run,
  so a checkout moved during a run records the wrong commit (CO0.1's seeds 42 and 44
  recorded CO0.2's): read it once at the start and carry it to `costs.json`. (2)
  `RunDir::create` names a directory `<name>_<unix_secs>` and opens it with
  `create_dir_all`, which reuses an existing one, so two runs of the same name started in
  the same second write into one directory (CO0.1's two long GN runs on seed 42, 2026-11-26,
  both void): create it with `create_dir`, which fails if it exists, and on that failure try
  `<name>_<unix_secs>_<k>`. Names are unchanged unless two collide; no model output moves.
  Follow-ups of CO0.1 (the author, 2026-11-26), outside O0's gate. *Done when:* both
  examples read the commit before training; a test creates two run directories of one name
  in the same second and gets two directories. *Needs:* –. *Machines:* the Mac. Done
  2026-11-27: plan `docs/optimisers/tasks/CO0.4.md`; `train_darcy` fixed with them.
- [ ] **CO0.5 Provenance leftovers of CO0.4.** (1) `tracking::builder::start_ms` reads a run
  directory's last `_` segment as its start second, so a collided `<name>_<secs>_<k>`
  exports to MLflow with a start time k s after 1970; `_<k>` cannot be told from a
  stamp by its text. The author chose `<name>_<k>_<secs>` (2026-11-27, CO0.4's PR,
  question 1; the code-quality review's suggestion), so `start_ms` needs no change.
  (2) `train_poisson3d` (laplace3d) still reads the commit when it writes `costs.json`:
  read it at the start, as CO0.4's examples do, once CL4.2 has merged (it runs that
  example). *Done when:* a test exports a collided directory with its true start time;
  `train_poisson3d` reads the commit before training. *Needs:* CO0.4, CL4.2.
  *Machines:* the Mac.

## O1: The Burgers replication and the Darcy extension

**Gate:** PETScML's Burgers reference reproduced (D10's band) or the gap explained; the
2 × 2 {AdamW, Gauss–Newton} × {weight decay 0, 1e-4} on Burgers over five seeds at equal time and
three at the long budget (D2); the same on Darcy at equal time; the verdict on how much of PETScML's gap the
weight decay accounts for.

- [x] **CO1.1 The reference recipe.** `OptimizerConfig::AdamW` (Burn's `AdamWConfig`, ε
  1e-8), `TrainingConfig.lr_schedule` (`None` = today's cosine; a step variant halving every
  k epochs), an `FNOConfig` GELU option and a squared relative-L² option for the Adam path (both off by default), `OPTIMIZER=adamw`, `WEIGHT_DECAY` and the subsample rate as variables of
  `train_burgers` with n_train = 1,000 (1,024 today), and `costs.json` for Burgers runs;
  s = 128 and the 1,000/100 split checked against Lu et al. 2022 and PETScML's scripts;
  Burn's `StepLrSchedulerConfig` reused for the step schedule. *Done when:* the O1 tests of proposal
  §9 pass (the parameter count 549,569, AdamW at weight decay 0 equal to Adam at 0 under the
  same schedule, the step schedule's halvings); A1 on seed 42 on flex or metal against
  0.0014; constraint 1 shown (`train_burgers` and `train_darcy` with nothing set unchanged
  bit for bit). *Needs:* D10, D11, the library objective's CM0.3 (`training/` changes are
  serial, the author's sequencing of 2026-11-24). Done 2026-11-25:
  `docs/optimisers/results/CO1.1.md` (A1 seed 42: 7.11e-4, inside D10's band at its lower
  edge); the recipe is the preset `RECIPE=petscml` (the author, 2026-11-25), so
  `train_burgers` with nothing set keeps n_train 1,024 and s = 256.
- [x] **CO1.2 Gauss–Newton on Burgers and Darcy.** `OPTIMIZER=gauss-newton` and the `GN_*`
  variables in `train_burgers` and `train_darcy`, as in `train_poisson`; `FnoResidual` tested
  on a rank-3 batch; `gn_steps.csv` and passes written; `SEED` and `TIME_BUDGET_S` in both
  examples, which CO1.3's cells need (CO1.1 added neither); the examples' `env_var` and
  `RunCosts` assembly, then in three examples, shared rather than copied a third time
  (CO1.1's code-quality review, F5). *Done when:* a flex smoke run of G0
  and G1 on Burgers and of G0 on Darcy, a resume of one; constraint 1 shown. *Needs:*
  CO1.1, D12. Done 2026-11-26: plan `docs/optimisers/tasks/CO1.2.md` (no results file: smoke
  runs only, in the PR and `progress.md`); `examples/train/common.rs` shared by the three
  examples; also `EPOCHS` (AdamW halving every `EPOCHS / 5`, the author, 2026-11-26) and
  `OPTIMIZER=adamw` in `train_darcy` (CO1.4's A cells).
- [ ] **CO1.3 The Burgers 2 × 2.** A0, A1, G0, G1 and G0-200 (G0 at batch 200, D12) on seeds 42–46 at A1's mean time and
  42–44 at 6 × it (proposal §3.1, D2); the coupled-Adam cells if D11 keeps them. *Done when:*
  `docs/optimisers/results/CO1.3.md`: test L² per cell (mean and range), against epochs,
  passes, PETScML's oracle calls and time; A1 against D10's band; the verdict on A1 → G0
  split into A1 → A0 (weight decay) and A0 → G0 (the optimiser). *Needs:* CO1.2.
  *Machines:* the VM. From CO1.2: the long budget's A cells are `EPOCHS=3000` (D11: 6 × the
  epochs, not exactly 6 × A1's time); a Gauss–Newton run stops at `EPOCHS` even under
  `TIME_BUDGET_S`, so G0-200 at the long budget needs a large `EPOCHS`; run names do not tell
  the cells apart, so the tables group runs by `train_cfg.json`.
- [ ] **CO1.4 The Darcy 2 × 2.** The same four cells on Darcy at `train_darcy`'s setting,
  seeds 42–46, equal time only. *Done when:* `docs/optimisers/results/CO1.4.md` and whether
  the Burgers verdict carries over. *Needs:* CO1.3. *Machines:* the VM. From CO1.2: G0 and G1
  at D12's full batch (1,000 samples at 85², padded to 94²) need the whole batch's autodiff
  graph for each VJP; check it fits one A100 before the cells, and if not, the batch is a
  question for the author (a departure from D12).

- [x] **CO1.5 Split `trainer.rs`'s tests.** Move `training/trainer.rs`'s `mod tests`
  (about 1,150 of its 1,950 lines after CO1.1) into a test module of its own, unchanged.
  *Done when:* the same tests, by name and count, pass; no code outside the tests changed.
  *Needs:* CO1.1; no other `training/` task open beside it (CO1.1's code-quality review,
  F6).
  Done 2026-11-27: `docs/optimisers/tasks/CO1.5.md`.
- [ ] **CO1.6 Tidy `trainer/tests.rs`.** CO1.5's code-quality review: (F1) its 11 sections,
  named by the phase that added them, become submodules named by what they test, with the
  shared helpers in `trainer/tests/mod.rs`; (F2) `train_built` moves from `trainer.rs`
  into the tests (the test-only `use burn::module::Module` moved there in library CM0.6);
  (F3) the repeated `load_poisson_uniform` closures and temporary-directory setup become
  fixtures. Also from library CM0.6: the training config passed as the data config to
  `save_configs` (CM0.3's F6, four sites), and `train_fno_none_matches_identity_path` named
  for `train`; and, the author's decision of 2026-11-27 (library CM0.6, #180), the
  `M = FNO<R>` default dropped from `TrainingComponents` once `train_built` is in the tests
  (every remaining use then names `FNO<R>`).
  *Done when:* the same tests, by name but for their new module path, and count, pass; no
  code outside `#[cfg(test)]` changed. *Needs:* CO1.5; no other `training/` task open
  beside it. Optional: nothing waits for it.

## O2: L-BFGS

**Gate:** an L-BFGS arm trains, resumes exactly and counts its passes; its settings piloted.

- [x] **CO2.1 The L-BFGS arm.** `OptimizerConfig::Lbfgs`, `train_epoch_lbfgs` with the
  closure over Φ on the full batch, passes into `costs.json`, the checkpoint section,
  `OPTIMIZER=lbfgs` in `train_poisson`. *Done when:* tests of proposal §9 pass; a flex smoke
  run and its resume; constraint 1 shown. *Needs:* D3, D4, D7, D13, the library
  objective's CM0.3, and CO1.1 merged (both change `training/`; never at once). Plan and
  results: `docs/optimisers/tasks/CO2.1.md`.
- [ ] **CO2.2 L-BFGS pilot.** Seed 42 at the equal-time budget: history ∈ {10, 100},
  `max_iter` ∈ {1, 20}, strong Wolfe. `max_eval` is set with each cell (`LBFGS_MAX_EVAL`):
  Burn's default ⌊1.25 `max_iter`⌋ leaves the line search 0 evaluations at `max_iter` 1
  and 1 at 2, and a run with too few stalls for good (CO2.1's smoke run). Burn's `LBFGS`
  does not expose its inner iterations, so "updates" are not counted (CO2.1): the pilot
  says whether `optimisation_steps` (steps = epochs) and the passes suffice or the count
  is needed. The author (2026-11-27, CO2.1's questions): `max_eval` set explicitly with
  every cell, and the uncounted inner iterations accepted (steps and passes are reported;
  updates can be estimated from closure evaluations if ever needed).
  *Done when:* the pilot table and D5 proposed.
  *Needs:* CO2.1, CO0.1. *Machines:* the VM. *Sign-off:* D5.

## O3: SOAP

**Gate:** a SOAP optimiser that passes proposal §9's tests trains the FNO; its settings
piloted.

- [x] **CO3.1 Burn linear algebra for SOAP.** (`spikes/linalg/REPORT.md`) Enable `linalg`; `qr` and `svd` at 12, 32,
  128 (and 144) against an f64 host reference on flex, metal and cuda: error and time.
  *Done when:* the table; constraint 1 shown with the feature on; D9 proposed. *Needs:* –.
  *Machines:* the VM (cuda). *Sign-off:* D9.
- [x] **CO3.2 The SOAP optimiser.** (`docs/optimisers/tasks/CO3.2.md`) `optim/soap/`: `SoapConfig`, `Soap` as a Burn
  `Optimizer`, any tensor rank. *Done when:* the SOAP tests of proposal §9 pass (the
  identity-basis reduction to Adam, the quadratic, ranks 1–5, the refresh, record round
  trip). *Needs:* CO3.1, D8, D9.
- [ ] **CO3.3 The SOAP arm and pilot.** `OptimizerConfig::Soap`, `OPTIMIZER=soap`; resume
  exact; then the pilot of D6 on seed 42 at the equal-time budget. *Done when:* a flex smoke
  run and its resume, constraint 1 shown, the pilot table, D6 proposed. *Needs:* CO3.2,
  CO0.1, D7, D13, the library objective's CM0.3 (`training/` changes are serial; not at
  once with CO1.1, CO1.2 or CO2.1). *Machines:* the VM. *Sign-off:* D6. From CO3.2's
  questions (the author, 2026-11-25): the resume is checked bit for bit on flex; on the GPU
  the drift of a resumed run from an uninterrupted one is recorded, not asserted zero (metal
  reloads diverge in some processes, CO3.2's plan, Outcome); the pilot logs each factor's
  eigenvalue ratio at the first refresh for one seed, and ratios near 1e7–1e8 bring an f64
  host eigenbasis back to the author as an amendment of D9. No reference fixtures (proposal
  §9): skipped, unless the pilot shows SOAP misbehaving. From CO2.1's code-quality review,
  for the next change to `trainer.rs` (SOAP rides `ModuleOptimizer`, so none of these blocks
  it; each changes existing code, so each with constraint 1 shown): the Gauss–Newton and
  L-BFGS states as one enum in `TrainingComponents` and `OperatorCheckpoint` (F1); one
  step-row sink, removing `run_loop`'s wrapper over `run_loop_with` (F3, with the one
  existing test that calls `run_loop`); the metrics forward shared by `train_epoch_gn` and
  `train_epoch_lbfgs` (F4); `GnPasses` renamed for both step files; the trainer's tests in
  their own file (F8, `trainer.rs` 2,180 lines).

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

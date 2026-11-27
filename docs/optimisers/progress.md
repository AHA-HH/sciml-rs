# Optimiser benchmarks progress log

Newest entries at the bottom. One entry per task, in that task's pull request, using this
template:

```
## YYYY-MM-DD: <ID> <title>
- Done:
- Tests:
- Checks: ran …; pending …
- Decisions: none, or D… signed off; questions for sign-off …
- Review: plan review (blocking items fixed); code quality APPROVE or BLOCKED → APPROVE (rounds, blockers fixed, findings left as follow-ups)
- Adam and Gauss–Newton: unchanged bit for bit (how shown), or not touched
- Follow-ups:
```

## 2026-11-24: Planning
- Done: reading notes on SOAP, Shampoo and Khodakarami et al. 2026
  (`docs/optimisers/references.md`); the proposal, draft 0.1, in `docs/design/optimisers.md`;
  its Section 10 as `docs/optimisers/roadmap.md`.
- Tests: none.
- Checks: none run; the proposal's cost figures are estimates for O0–O2 to measure.
- Decisions: none yet; D0 (the proposal as the plan) comes first.
- Adam and Gauss–Newton: not touched.
- Follow-ups: review the draft; D0, D2; then CO0.1, CO1.1 and CO2.1, which can run in
  parallel.

## 2026-11-24: Draft 0.2 (the Burgers replication)
- Done: the proposal's draft 0.2. A new milestone O1 replicates Zampini et al. 2024
  (PETScML, arXiv 2403.12188, §4.1.1): their FNO Burgers setting as a 2 × 2 of {AdamW,
  Gauss–Newton} × {weight decay 0, 1e-4}, then Darcy as an extension (proposal §3.1). The
  former O1–O4 are now O2–O5, and their tasks CO1.x–CO4.x are CO2.x–CO5.x; the entry above
  uses the old numbers. New decisions D10–D14 (D13: weight decay as a factor in O4; D14:
  Burgers in O4); risks R7–R8. The library objective's CM0.3 recorded as a need of CO1.1
  and CO2.1 (`training/` changes serial, the author's sequencing of 2026-11-24).
- Tests: none.
- Checks: none run (documents only).
- Decisions: none yet; D0 now covers draft 0.2.
- Adam and Gauss–Newton: not touched.
- Follow-ups: the author's review of draft 0.2; D0, D2, D10–D12; then CO0.1, CO0.2 and CO3.1
  at once, CO3.2 after CO3.1, and CO1.1 after CM0.3 (CO1.1 before CO2.1).

## 2026-11-24: Draft 0.2 corrections
- Done: draft 0.2 checked by an independent agent against PETScML (arXiv 2403.12188), the
  code and Burn 0.23; nine corrections applied (proposal Appendix A, "Draft 0.2,
  corrections").
- Tests: none.
- Checks: none (docs).
- Decisions: none signed off; D0 on draft 0.2 next.
- Adam and Gauss–Newton: not touched.
- Follow-ups: D0, D2; then CO0.1, CO0.2 and CO3.1.

## 2026-11-24: D0 and D2
- Done: D0 signed off by the author: draft 0.2 of `docs/design/optimisers.md`, with the
  corrections of #128, as the plan of record. D2 signed off as recommended: seeds 42–46;
  the equal-time budget (CO0.1's Adam mean) and 5,400 s, every arm at both.
- Tests: none.
- Checks: none (docs).
- Decisions: D0 and D2 signed off 2026-11-24.
- Adam and Gauss–Newton: not touched.
- Follow-ups: CO0.1 and CO0.2 (the VM, at the author's go per run) and CO3.1; CO3.2 after
  CO3.1; the tasks that change `training/` after the library's CM0.3.

## 2026-11-25: D2 amended
- Done: D2 amended by the author: seeds 42–44 (three) at the 5,400 s budget, 42–46 (five)
  at the equal-time budget, for every arm. CO0.1, CO1.3, CO4.1 and the O1 and O4 gates
  updated; proposal Sections 3, 3.1, 7, 12 and Appendix A.
- Tests: none.
- Checks: none (docs).
- Decisions: D2 amended 2026-11-25.
- Adam and Gauss–Newton: not touched.
- Follow-ups: CO0.1 at about 13 GPU-hours instead of 19; its plan checks that two timed runs
  side by side keep a lone run's epoch times.

## 2026-11-24: CO3.1 Burn linear algebra for SOAP
- Done: Burn's `linalg` feature enabled in the workspace `burn` dependency (pin unchanged).
  Spike `spikes/linalg/` (`REPORT.md`): `linalg_check` measures `qr` (device) and `svd`
  (host) at n = 12, 32, 128, 144 in f32 against f64 host references; `fingerprint` hashes a
  seeded FNO forward, Adam and hybrid Gauss–Newton runs. Both correct on flex and metal;
  device `qr` 18–23× slower than a host QR (round trip included) on metal at n ≥ 128 and
  95–240× on cuda (2026-11-25, A100).
- Tests: no new crate test (spike only); the existing suite unedited with the feature on.
- Checks: ran fmt, the three clippy lines and `--all-targets`, `cargo test`, `cargo doc`,
  datagen clippy and tests, metal `gpu_parity`, the spike's clippy (flex, metal) and runs
  (flex, metal; cuda on the VM 2026-11-25), the fingerprint on this branch and on
  `origin/main`.
- Decisions: D9 signed off 2026-11-25: host `svd` for the first basis and host QR for the
  refreshes, on every backend.
- Review: plan review PASS (wording fixes applied); code quality APPROVE, no blockers
  (plan amended for the svd metrics, like-for-like host timing, comments).
- Adam and Gauss–Newton: unchanged bit for bit: equal fingerprints with `linalg` off
  (`origin/main` 4d681d0) and on, flex; the feature adds no code the crate uses (REPORT).
- Follow-ups: CO3.2 (needs D8).

## 2026-11-25: D8, D10 and D11
- Done: signed off by the author as recommended. D8: SOAP preconditions each tensor with one
  Kronecker factor per dimension, re and im separately, biases on Adam. D10: O1 takes
  PETScML's data, model (a GELU option), loss (eq. 6) and metric, with our Levenberg–Marquardt
  Gauss–Newton in f32; their reference counts as reproduced when A1's mean is within a factor
  2 of 0.0014. D11: AdamW, lr 1e-3 halved every 100 epochs, batch 20, 500 epochs, ε 1e-8,
  at weight decay 1e-4 (A1) and 0 (A0), plus our coupled Adam at both.
- Tests: none.
- Checks: none (docs).
- Decisions: D8, D10, D11 signed off 2026-11-25.
- Adam and Gauss–Newton: not touched.
- Follow-ups: CO3.2 (SOAP) and CO1.1 (the Burgers recipe) are runnable; both change
  different modules, but CO1.1 changes `training/`, so no other `training/` task beside it.

## 2026-11-25: D3, D4, D7, D12 and D13
- Done: signed off by the author, all as recommended. D3: L-BFGS minimises Φ, Gauss–Newton's
  objective, at μ ∈ {0, 1e-4} as D13 widens it. D4: full batch, one batch of 1,000,
  accumulation only if memory requires. D7: one forward and one backward per SOAP step and
  per L-BFGS closure evaluation, SOAP's preconditioner work reported as time. D12: O1's
  Gauss–Newton arm full batch, CG cap 100, Phase 10's damping and backtracking, the
  Hutchinson diagonal, μ = 0 (G0) and 1e-4 (G1), and G0-200. D13: every arm of O4 at weight
  decay 0 and 1e-4 (the long budget's cost about 18 h → 36 h).
- Tests: none.
- Checks: none (docs).
- Decisions: D3, D4, D7, D12, D13 signed off 2026-11-25.
- Adam and Gauss–Newton: not touched.
- Follow-ups: CO1.2 after CO1.1; CO3.3 after CO3.2 and CO0.1; CO2.1 after CO1.1. They all
  change `training/`, so one at a time (with CM0.6).

## 2026-11-25: CO3.2 The SOAP optimiser
- Done: `optim/soap/` (new; imports nothing from `models/` or `training/`): `SoapConfig`
  (the reference defaults; `precondition_frequency` 0 refused), `Soap`, Burn's
  per-parameter `Optimizer` for any rank, `SoapState` with `RecordState` written by hand
  (Burn's derive works only inside `burn-optim`), `Factor`. Follows nikhilvyas/SOAP @a1e5535:
  first call builds the factors and bases and moves nothing; bias correction, ε on √V,
  decoupled decay after the step, factor EMA after it, refresh every f steps (sort by
  diag(QᵀLQ), V permuted, one power step, QR). D8: one factor per dimension, rank 1 on Adam.
  D9: first basis by Burn's host `svd`, refreshes by a host f64 Householder QR (ndarray).
  Departure, declared: M kept in the original space (the paper's Algorithm 3; three
  projections per step against the reference's four).
- Tests: seven in `optim/soap/tests.rs`: identity bases (`max_precond_dim` 0) equal Burn's
  Adam over 20 steps (1e-6); the Kronecker quadratic (first factors to 1e-5; SOAP 1.6e-7
  against Adam 5.1e-2 from 2.0e2 at lr 0.05 · 0.98ⁱ, 400 steps, κ = 10 per factor; at
  κ = 100 the factors are past f32 and the result depends on the backend); ranks 1–5 (factor EMA,
  orthonormal bases, one step against host f64 at 3–7e-6 of the update); the refresh against
  host f64 (first bases against the exact eigenvectors, refreshed bases against a
  Gram–Schmidt QR, V permuted); the record round trip bit for bit across refreshes (flex;
  on metal it fails about one run in six, see the plan's Outcome); a zero
  first gradient; the frequency check. Mutations (Q for Qᵀ in the back-projection, V not
  permuted, decay before the step) each fail a test.
- Checks: see the PR (fmt, the clippy lines, `cargo test`, `cargo doc`, datagen clippy and
  tests, the soap tests on metal).
- Decisions: none new; D8, D9 applied. The author's answers to the PR's questions
  (2026-11-25): no reference fixtures; resume bit for bit on flex, drift recorded on the
  GPU; f32 kept, CO3.3 logs the factors' conditioning (CO3.3's roadmap line).
- Review: plan review FAIL → fixed (`clippy::needless_range_loop` in the tests; advisories
  taken: decay order now tested, zero-gradient test, frequency 0 refused, f64 host read);
  code quality APPROVE, no blockers (F1 ndarray for the host QR, F4, F5, F6, F8 fixed; F2,
  F3, F7 left, PR text).
- Adam and Gauss–Newton: not touched (nothing calls the module; the existing suite unedited).
- Follow-ups: CO3.3 (the arm, resume bit for bit on flex with the GPU drift recorded, the
  factors' conditioning logged at the pilot; its roadmap line).

## 2026-11-25: CO0.2 Cost profile of a Gauss–Newton step
- Done: `gn_profile` (`examples/profile/gn_profile.rs`): from a Gauss–Newton run's
  checkpoint, the unit costs of a step's parts (synced medians), CG in place with its
  products timed, and real steps at each CG cap with predicted against measured time and
  the profile table; no library code. Report `docs/optimisers/results/CO0.2.md`: one
  GN200-D step on Burn 0.23 (A100, cuda) at cap 50 takes 15.3 s, 13.7 passes per second,
  against the fork's 11.88 s and 17.4 (1.29× slower per step); CG's curvature products are
  98 % of it (JVPs 41 %, VJPs 40 %, CG's own work 21 %); predicted and measured agree to
  −3.8 % (cap 50) and −0.6 % (cap 150) on the mean. Plan `docs/optimisers/tasks/CO0.2.md`
  (amended: the fork's passes per second by `awk` over `gn_steps.csv`).
- Tests: the example's 6 unit tests (the step decomposition on three Phase 10 rows, an
  adjoint step without refresh and a negative-curvature stop, no preconditioner,
  unexplained passes rejected, the phases by hand, the output's directories created).
- Checks: ran fmt, the three clippy runs, `cargo test`, `cargo doc` (all pass); metal smoke
  runs; the VM run (cuda, alone, 2026-11-25 13:58–14:06 UTC, `f893620`). Attempt 1
  (12:05 UTC) measured but failed to write its report (missing directory; fixed).
- Decisions: none.
- Review: plan review FAIL → fixed (the `gn_steps.csv` reader replaced by the plan's
  amendment); code quality APPROVE with findings (fixed: exact warm start, the
  negative-curvature product, phases once per step, source comments, the approximations
  documented, the metric row and gap flag).
- Adam and Gauss–Newton: not touched (no library file changed).
- Follow-ups: the savings list of the report (for later work, nothing changed here); the
  1.29× per-step slowdown against the fork goes with CO0.1's bisect of Adam's 3.5×.

## 2026-11-25: CO1.1 The reference recipe
- Done: `OptimizerConfig::AdamW` (Burn's `AdamWConfig`, ε 1e-8); `TrainingConfig.lr_schedule`
  (`LrSchedule::Step`, Burn's `StepLrSchedulerConfig`, γ 0.5) and `.loss`
  (`TrainLoss::SquaredRelL2`, PETScML's eq. 6, the Adam path only, metrics unchanged);
  `FNOConfig.activation` (`Activation::Gelu`, erf form); `train_burgers`: `RECIPE=petscml`
  (n_train 1,000, s = 128, GELU, the squared loss; the author's choice of a preset,
  2026-11-25), `OPTIMIZER=adamw` (with the step schedule every 100 epochs), `WEIGHT_DECAY`,
  `SUBSAMPLE`, and `costs.json`. Source check: s = 128 and the 1,000/100 split confirmed
  (PETScML's text, Lu et al.'s and Li et al.'s code; PETScML's repository has no Burgers
  script). A1, seed 42, metal: test L² 7.11e-4 final, 7.07e-4 best so far, against 1.4e-3
  (`docs/optimisers/results/CO1.1.md`).
- Tests: parameter count 549,569; GELU against a hand-written forward; `Relu` and `None` bit
  for bit; old configs load; AdamW at weight decay 0 equal to Adam (1e-6), different at 0.1;
  the step schedule's rate at every step, bit for bit; the squared loss's closed form, a
  trainer step against one taken by hand on it, and the unsquared metric; the replica guard.
- Checks: see the PR (`/verify`); the constraint 1 runs below.
- Decisions: the GELU form answered by the author 2026-11-25: keep erf.
- Review: plan review PASS with findings (docs, two tests tightened; fixed); code quality
  APPROVE with findings, no blockers (F2–F4, F7 fixed; F1 declined: `train_epoch`'s public
  signature and the existing tests stay unchanged; F5 into CO1.2, F6 as CO1.5; F8 accepted).
- Adam and Gauss–Newton: unchanged bit for bit: `spikes/linalg`'s fingerprint
  `84e63f28a21024db` on `origin/main` and the branch; `train_burgers` and `train_darcy` with
  nothing set, two epochs each, equal `metrics.csv` rows (flex).
- Follow-ups: CO1.2 (the `SEED` and `TIME_BUDGET_S` variables and the shared example
  helpers, added to its line); CO1.5 (split `trainer.rs`'s tests).

## 2026-11-25: CO0.1 Reference arms on Burn 0.23 (code and run plan; runs pending)
- Done: the run plan (`docs/optimisers/tasks/CO0.1.md`): Adam on seeds 42–46 at 500
  epochs (giving the equal-time budget B), Adam on 42–44 at the long budget, GN200-D at
  μ = 0 with CG caps 50 and 150 on seed 42 at both budgets, then the chosen cap on 43–46
  and 43–44; about 13.2 GPU-hours, 8 h wall on two A100s if the side-by-side checks hold.
  `EPOCHS` in `train_poisson` for Adam runs only: Adam's long budget is E = round(5,400 ×
  500 / B) epochs, the cosine schedule spanning them, since a time budget would cut the
  schedule. The run script `docs/optimisers/co0_1_runs.sh` (Phase 7's pattern).
- Tests: with `EPOCHS` unset, `train_cfg.json`, `model_cfg.json` and `data_cfg.json` byte
  for byte equal to `origin/main`'s binary (flex, `-- 0 129 64`); `EPOCHS=3` saved as
  `epochs: 3`; with `OPTIMIZER=gauss-newton` it panics.
- Checks: ran fmt, clippy (lib, examples, all targets), `cargo test`, `cargo doc`, `bash -n`
  on the script; pending: every VM run (stages 1–3 of the plan).
- Decisions: none signed off; D1 is proposed after stage 2.
- Review: plan review FAIL → fixed (the shared build cache could swap the binary between
  specs: the script now runs its own copies); code quality APPROVE, findings 4.1, 4.2, 7.1,
  7.2 fixed, the rest in the PR.
- Adam and Gauss–Newton: unchanged bit for bit (configs identical with `EPOCHS` unset; the
  example's configs are its only input to training).
- Follow-ups: `poisson_matrix --by-optimizer` labels arms by batch and preconditioner only
  (cap and budget merge); CO4.1's extension of the tool covers it.

## 2026-11-26: CO0.1 stage 1 (Adam at equal time)
- Done: Adam on seeds 42–46, 500 epochs, on the VM; `docs/optimisers/results/CO0.1.md`.
  Error 4 1.029e-2 [9.637e-3–1.105e-2] (seeds 42–44: 1.055e-2 against the fork's 1.061e-2);
  training 2,949 s against the fork's 859 s, 3.4× slower, which a bisect puts on the move
  to Burn 0.23 (#119). B = 2,952 s, E = 915. Seed 46 data-parallel on both GPUs: 2.09× the
  one-GPU time.
- Tests: none new.
- Checks: ran `poisson_matrix --by-optimizer` on the five runs (the table); pending stages
  2 and 3.
- Decisions: D2 amended 2026-11-25 by the author (keep 5,400 s); pairing accepted
  2026-11-26 by the author; D1 after stage 2.
- Review: –
- Adam and Gauss–Newton: not touched.
- Follow-ups: CO0.3 (the Adam step profile); `git_commit()` reads HEAD at the end of a run,
  so a checkout moved mid-run records the wrong commit (seeds 42 and 44): a question for the
  PR.

## 2026-11-26: CO0.1 stage 2 (Gauss–Newton on seed 42; Adam at the long budget)
- Done: GN200-D at μ = 0, seed 42, caps 50 and 150, at B and at 5,400 s; Adam at 915
  epochs on seeds 42 and 43; `docs/optimisers/results/CO0.1.md`. Error 4 on seed 42:
  at equal time Adam 9.637e-3, GN cap 50 6.129e-3, cap 150 5.846e-3; at the long budget
  Adam 8.585e-3, cap 50 4.579e-3, cap 150 2.469e-3. The first long GN pair shared a run
  directory (same name, same second) and was void; rerun at the author's go.
- Tests: none new.
- Checks: after merging `origin/main` (CO1.2's `common::training_config`; `EPOCHS` moved
  into `Literals`): fmt, clippy (lib, examples, all targets), `cargo test`, `cargo doc` pass.
- Decisions: D1 proposed (cap 150); the long GN rerun, the author, 2026-11-26.
- Review: –
- Adam and Gauss–Newton: not touched (the merge keeps main's code with `EPOCHS` unset).
- Follow-ups: CO0.4 (the commit read at run start; one directory per run).

## 2026-11-27: CO0.1 stage 3; CO0.1 done
- Done: GN200-D at μ = 0, cap 150 (D1) on seeds 43–46 at B and 43–44 at 5,400 s; Adam
  seed 44 at 915 epochs; the final tables in `docs/optimisers/results/CO0.1.md`. Error 4,
  mean [min–max]: at equal time Adam 1.029e-2 [9.64e-3–1.105e-2], GN 6.341e-3
  [4.48e-3–1.008e-2] (GN lower on 4 of 5 seeds, level on 45); at 5,400 s Adam 9.160e-3,
  GN 4.044e-3 (lower on all three). No NaN in any valid run. Same-seed GN runs on cuda
  agree only to f32's last digits after the first step, and diverge from there (seed 43's
  long run ended above its equal-time run).
- Tests: none new.
- Checks: none since the merge (docs only).
- Decisions: D1 signed off 2026-11-26, cap 150 (given with stage 3's go through the
  coordinating session; the PR asks the author to confirm); GN pairing accepted with it.
- Review: –
- Adam and Gauss–Newton: not touched.
- Follow-ups: CO0.3, CO0.4 (roadmap lines); for CO4.1, GN's run-to-run spread on cuda
  (results, stage 3).

## 2026-11-26: CO1.2 Gauss–Newton on Burgers and Darcy
- Done: `train_burgers` and `train_darcy` take `OPTIMIZER=adam|adamw|gauss-newton`, the
  `GN_*` variables with D12's literals (full batch, CG cap 100, the Hutchinson diagonal, 10
  probes, μ = 1e-4 for G1, `GN_WEIGHT_DECAY=0` for G0, `GN_BATCH_SIZE=200` for G0-200),
  `WEIGHT_DECAY`, `SEED`, `EPOCHS` (AdamW halving every `EPOCHS / 5`; the author,
  2026-11-26) and `TIME_BUDGET_S`; checkpoints every 25 epochs and `--resume`; `costs.json`
  with steps and passes (new for Darcy), `gn_steps.csv`. `examples/train/common.rs`, shared
  by the three examples (CO1.1's F5): `training_config`, `session`, `run_costs`;
  `train_poisson`'s copies replaced. `train_darcy` also gains `adamw`, which CO1.4 needs.
- Tests: `FnoResidual` on a rank-3 batch (½‖r‖² = (1/2B) Σ eᵢ², relative 1e-5).
- Checks: fmt, the three clippy runs, datagen clippy, `cargo test`, `cargo doc` pass, after
  the merge of `origin/main` too. Smoke (flex): Burgers G0 and G1, `RECIPE=petscml`, full
  batch, `TIME_BUDGET_S=1`: one step each, accepted (ρ 0.99), adjoint error 8e-7,
  `costs.json` 1 step, 31 forward, 17 backward passes. Darcy G0 (batch 100, CG cap 10; full
  batch does not fit flex on 16 GB): 10 steps, all accepted, test L² 0.162, 10 steps, 208
  forward, 76 backward passes. Resume (`5f5fa3e`): Burgers G0, `EPOCHS=27`, CG cap 5, killed
  after its epoch-25 checkpoint and resumed with no variable set, against the run
  uninterrupted: `metrics.csv` and `gn_steps.csv` bit for bit, weights' tensor data equal,
  `costs.json` equal but `resumes` (1). The final binary writes the smoke runs' configs (the
  merge's seven `None` `FNOConfig` fields aside).
- Decisions: `EPOCHS` added (the author, 2026-11-26).
- Review: plan review PASS (advisories: plan text amended, probe count fixed, CO1.3/CO1.4
  notes added to their lines, weights compared in constraint 1); code quality APPROVE with
  findings (F2, F3, F5, the minor ones and part of F1 fixed; F1's rest and F4 in the PR).
- Adam and Gauss–Newton: unchanged bit for bit: `train_burgers`, `train_darcy`,
  `train_poisson` (Adam) and `train_poisson OPTIMIZER=gauss-newton` (N = 20), two epochs on
  flex at `6ea636b` and `a87bf41`: `metrics.csv`, `gn_steps.csv`, configs and `costs.json`
  counts equal; `model_weights.bpk` equal but for Burn's random `param_id`s in its header.
- Follow-ups: CO1.3 (long-budget `EPOCHS`, G0-200's `EPOCHS` cap, grouping by config) and
  CO1.4 (full-batch Darcy on one A100), in their roadmap lines.

## 2026-11-27: CO0.3 Cost profile of an Adam step (code; VM runs pending)
- Done: `examples/profile/adam_profile.rs`: CO0.1's Adam setting (stage 3, s = 64, p = 0,
  batch 20), the library's `train_epoch`, `evaluate` and `save_checkpoint` timed, then a
  copy of `train_epoch`'s loop timed by phase (loader, forward and loss, metrics, backward,
  scheduler, optimiser update, the epoch's host reads), unsynced (host) and synced
  (device). One file builds on `main` and at `4541f02` (the fork); the fork build copies it
  in on the VM (plan, "VM runs").
- Tests: `rows_by_hand` (the per-phase table), `create_output_makes_missing_parents`.
- Checks: fmt, the three clippy runs, `cargo test`, `cargo doc` pass. Smoke on the Mac
  (metal; not results): runs to its report on both builds; inline copy −5.3 % (Burn 0.23)
  and −0.0 % (fork) against `train_epoch`.
- Decisions: none.
- Review: plan review FAIL on the VM commands (environment outside tmux; run C's
  `cubek.toml` restored before the run), fixed; code quality APPROVE with findings (the
  faithful line made conditional, phase arrays sized by `N_PHASES`, `Library` built once,
  a comment; named phase indices and a shared `examples/profile/common.rs` left, the
  latter because the fork build would need two files).
- Adam and Gauss–Newton: not touched (no library code).
- Follow-ups: VM runs A (Burn 0.23), B (fork), C (Burn 0.23, TF32 on), then
  `docs/optimisers/results/CO0.3.md` and the roadmap tick.

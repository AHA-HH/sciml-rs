# T4: validation gate, the smallest viable experiment

Phase 10, task 4. Branch `phase10/T4-validation-gate`. Depends on T3, Phase 7's runs and a working
single-GPU run on the VM (one A100, no NCCL; G8, amended 2026-11-09, removed the wait for Phase 9).

## Read first
- This phase's README (G2–G4, G7, "Risks") and design §13.17.
- `docs/phase7/results.md` (the N = 100 rows and per-run table) and
  `docs/phase7/README.md` (subsets: `SubsetConfig`, the fifth and sixth arguments of
  `train_poisson`).
- `CLAUDE.md`: Commands (`train_poisson`, `predict_poisson`).

## Goal
Show on a small, cheap problem that Gauss–Newton is healthy on the real Poisson FNO
(accurate JVPs, sensible ρ, CG making progress) and how it compares with Adam per update
and per second, before the N = 1000 matrix is spent.

## Experiment
- Stage 3, s = 64, p = 0, **N = 100, subset 0, seed 42**, one A100.
- **GN, full batch** (all 100 samples per step; G3), residual (A), no preconditioner;
  then the same with the Hutchinson diagonal. Defaults from T2; any change stated.
- **Comparator:** Phase 7's Adam run at N = 100, subset 0, seed 42 (`runs/` in the main
  checkout), at equal wall-clock: GN trains for that run's training time.
- If T3 found f64 working on cuda, one extra GN run in f64 to measure the JVP's noise
  against f32.

## Gate (go / no-go for T5)
Proposed thresholds, confirmed by the author in this task's PR before the runs:
- median adjoint error ≤ 1e-3 in f32 (≤ 1e-6 in f64);
- at least 80 % of steps accepted, ρ mostly in (¼, 2);
- CG's relative residual falls in its first iterations on most steps (no stagnation
  from the first iteration).

Not part of the gate: whether GN beats Adam. That is the result, recorded either way.
If the gate fails on JVP noise, a task "exact FNO JVP" is inserted before T5 (G2); if it
fails otherwise, the cause and a proposal go to the author.

## Deliverables
1. The runs, evaluated by `predict_poisson`.
2. A short table in this phase's README (T4 Results row and a paragraph): error 4, final
   train L², updates, passes, wall time and peak memory per run; adjoint error, acceptance
   rate, CG iterations, λ and Ritz range over training; plots of Φ and error 4 against
   updates and against seconds (gnuplot, beside the runs; skipped without it).
3. The go / no-go, and the preconditioner T5 uses for GN200 and Hybrid.

## Checks
No code change expected. If any is needed, all of `CLAUDE.md`'s checks for it.

## Do not
- Tune the GN defaults on the test set: any change is chosen on train-set behaviour
  (Φ, ρ, CG) and stated.
- Start T5 before the go is recorded.

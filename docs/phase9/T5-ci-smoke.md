# T5: smoke-run the examples in CI

Phase 9, task 5. Branch `phase9/T5-ci-smoke`. Depends on T2 (so the smoke run also
covers resume). Supplementary task of old Phase 5 (2026-10-16).

**Status: closed 2026-11-02 by the author, folded into T2.** `train_poisson_smoke`
(`training/trainer.rs`) already runs the Poisson pipeline end to end on flex in CI; T2
extends it with a resume. The example `main`s stay compile-checked only. The brief
below is kept as history.

## Read first
- `.github/workflows/run-tests.yml`.
- `examples/train/poisson.rs`, `examples/predict/poisson.rs`.
- `src/neural_operators/data/loaders/poisson/` (`PoissonConfig`, `DatasetSource`) and
  `pde/poisson/dataset` (`write_split`, behind `datagen`).
- Design §10, old Phase 5 "Supplementary task: smoke-run the examples in CI".

## Goal
CI trains a Poisson FNO end to end on flex for one or two epochs, resumes it, and
evaluates it, so a broken example fails a pull request instead of a night on the A100.

## Deliverables
1. **A data source CI can use.** Choose, and say why in the PR: (a) a small committed
   fixture (a few samples at a small n, `.npz` plus sidecar, a few hundred kB at most;
   `datasets/` stays ignored, so it lives under `tests/fixtures/`), or (b) generating one
   in the job with `generate_poisson` behind `datagen`. (a) keeps the job light and
   needs no FFTW; recommended.
2. **A smoke switch** on `train_poisson` (argument or env var) that points at that data,
   uses a tiny model and 2 epochs, and writes a normal run directory under a temporary
   root. No second code path through the trainer.
3. **A CI job** (flex, no GPU): train 1 epoch, resume to 2 (T2), run `predict_poisson`
   on the run directory, and fail if any step fails or `errors.json` is missing.
4. Job time under a few minutes; state it in the PR.

## Acceptance
- The job passes on the PR, and fails on a deliberately broken branch (show it once,
  then revert).
- The T5 Results row filled in.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`; the new
job's local equivalent.

## Do not
- Check any error threshold in the smoke job; it tests that the pipeline runs.
- Commit anything under `datasets/` or `runs/`.
- Smoke Burgers or Darcy (their datasets are external `.mat` files).

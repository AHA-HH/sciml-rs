# T5: the training matrix

Phase 5, task 5. Branch `phase5/T5-matrix`. Depends on T4. Runs on the cuda A100 VM.

## Read first
- Design §13.8 (the Phase 5 matrix, as amended: no mode-control arm), §13.6
  (confounders), decision 30's Phase 5 amendment.
- `docs/phase4/T3-runs-129.md` and the Phase 4 T3 Results row (the cuda workflow,
  `--no-default-features --features cuda`, gnuplot missing on the VM).
- `CLAUDE.md` Commands (`train_poisson` after T3 and T4).

## Goal
Train the current FNO at every learning grid on the same forcings and test set, with
enough seeds to tell real differences from seed noise.

## Deliverables
1. **Cost first.** Time the first epochs of one run per s, extrapolate the matrix to
   500 epochs, and state it in the PR before the full runs start; the author approves
   it.
2. **Main arm:** stage 3 (n = 129, K = 64; 1000 train, 200 test), s ∈ {16, 32, 64, 128},
   p = 0, m = min(12, s/2) (D2), the D1 restriction, seeds 42, 43, 44, 500 epochs,
   Darcy hyperparameters. 12 runs.
3. **Aliasing control:** s = 32, unfiltered T_cu, otherwise as the main arm. 3 runs. If D1
   chose unfiltered T_cu, this arm is the main arm and is not run; say so.
4. **Padding control:** none trained; `runs/poisson_fno_n129_p4_1792037830` is the p = 4
   reference at s = 128 (D3).
5. Every run in the main checkout's `runs/`, with its `costs.json` (peak GPU memory from
   `nvidia-smi`, or "unavailable"). A table in the PR: run directory, arm, s, seed,
   final train / test relative L², s/epoch, wall time.

## Tests
None new. The runs are the deliverable.

## Acceptance
- 15 runs (or 12, per deliverable 3), all 500 epochs, all with metadata.
- The T5 parts of the Results rows filled in (costs; the errors come in T6).
- A run that diverges or is much worse than its neighbours is reported, not rerun
  silently.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`, locally;
the cuda build (`cargo build --release --no-default-features --features cuda --examples`)
on the VM. Say which ran where.

## Do not
- Retune hyperparameters, width, layers or the model.
- Add a mode-control arm (D2) or padded runs (D3).
- Commit runs, datasets or job scripts with account names.

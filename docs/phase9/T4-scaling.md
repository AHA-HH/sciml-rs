# T4: one against two A100s

Phase 9, task 4. Branch `phase9/T4-scaling`. Depends on T3.

## Read first
- The profiling finding (this phase's README, "Closed at planning") and T3's results.
- `docs/phase5/README.md` (the baseline cost) and `docs/phase7/README.md` (the per-epoch
  fit, and the shared-GPU runs it had to set aside).
- Design §13.11.

## Goal
Measure what a second A100 buys on the baseline, and why it is not 2×.

## Deliverables
1. **Timings** of the baseline (s = 64, p = 0, seed 42) on 1 and 2 A100s of the same
   machine, at N = 1000 and one larger per-epoch load chosen in the PR (e.g. batch work
   at a finer s, or the stage 4 set if present), each over enough epochs for a steady
   median s/epoch after warm-up. Full 500-epoch runs are not required.
2. **Speedup and parallel efficiency** (speedup / D), with the split by the profiling method on
   the 2-GPU run: compute, all-reduce, evaluation, host.
3. **The GPUs are unshared** during timing (`nvidia-smi` before and after, as Phase 7
   T2 did); timings from a shared GPU are reported apart, not used.
4. **From the closed profiling task (moved here):** GPU utilisation (`nvidia-smi dmon` or nsys) on the 1- and
   2-GPU runs, and one nsys trace (or flamegraph) of each, attached to the PR. They
   should say which overhead makes a batch cost about 31 ms whatever its size up to
   s = 64 (the profiling finding).
5. A table in `docs/phase9/results.md` and the T4 Results row.

## Acceptance
- Every number names its machine, GPU, driver and commit.
- The cost of the timing runs is stated before they start.

## Checks to run and report
None beyond those of any code change made (normally none).

## Do not
- Use the numbers in any Phase 5–7 result.
- Tune the model, batch size or schedule for the 2-GPU case.

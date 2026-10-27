# T2: POD on the learning grids, and close-out

Phase 6, task 2. Branch `phase6/T2-pod-learning-grids`. Depends on T1 and on Phase 5 T2
(`Transfers::with_s`, merged). Needs the stage 3 dataset, as T1. Builds with `datagen`.

## Read first
- Design §13.3 (the learning grids, R, the transfer-only floor), §13.4.
- `docs/phase6/README.md` and T1's `docs/phase6/results.md`.
- `docs/phase5/README.md` Results (D1, the transfer-only floor per s).
- `src/neural_operators/chebyshev/transfer.rs` (`Transfers::with_s`, `to_uniform_all`),
  `metrics/pod.rs` (T1).

## Goal
Show how much of the samples' structure survives on each Phase 5 learning grid, and how
the energy rank relates to the learning resolution.

## Deliverables
1. **`poisson_pod` gains the learning grids:** for s ∈ {16, 32, 64, 128}, restrict the
   train and test f and u with R = `Transfers::with_s(129, s).to_uniform_all` and fit a
   `Pod` on the restricted train snapshots with equal weights scaled by the cell area
   (2/(s − 1))² (D5). Report, per s and field: the ranks for 90 / 99 / 99.9% energy, the
   leading eigenvalues against T1's CGL ones, and the reconstruction error against rank on
   train and test (as T1), in a new section of `results.md`.
2. **Rank against resolution:** one table and one plot (energy rank against s per field
   and fraction, T1's 129 CGL value as the reference line), and a paragraph relating it
   to the Phase 5 transfer-only floor per s and to the f energy above the grid's
   resolvable band (design §13.3). The paragraph states what it cannot show (§13.4: POD
   modes are not the FNO's Fourier modes).
3. **Optional:** radially averaged Fourier power spectra of f and u on the uniform grids,
   only if Burn's `rfft` or an ndarray path makes it a short addition; otherwise state it
   was skipped.
4. **Close-out:** the T2 rows of `docs/phase6/README.md` Results; the exit checklist
   ticked; any result that contradicts the design recorded in design §12, dated; the
   "Current phase" bullet of `CLAUDE.md` updated.

## Tests
- `metrics/pod` (T1's tests) unchanged; one more: restricting a field set that lies in
  the span of sine modes resolvable on the s grid keeps its rank (same rank before and
  after R, with energies within the transfer's round-off).

## Acceptance
- Every (s, field) cell of the rank table has a value, from train data only.
- The relationship between dataset rank and learning-grid resolution is discussed, and
  labelled dataset-structure analysis, not network performance.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`; the
example's run with its wall time. No metal run (no Burn tensors).

## Do not
- Choose or change the Phase 7 baseline (Phase 5 T7 does).
- Use the test split in any fit.

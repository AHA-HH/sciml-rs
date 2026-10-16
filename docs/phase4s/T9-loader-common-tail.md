# T9: shared loader tail

Phase 4s, task 9. Branch `phase4s/T9-loader-common-tail`. Depends on T4.
- Can run in parallel with the Chebyshev track. It shares `loaders/poisson.rs` with T8, so
  it rebases if T8 merges first.
- T10 needs this task's `Normalizers` type.

> **Refresh at start:** T4 changes the normalizer methods from trait methods to inherent ones
> and renames the readers' API. Re-read `normalizers.rs` and the three loaders on `main`. If
> T8 has merged, the Poisson loader uses `Transfers`.

## Read first
- Review §5 S3 (second bullet), §6 T2-10, T4-7, §7 step B3. CONVENTIONS §10, which names
  `DarcyNormalizers`.
- `data/loaders/darcy.rs`: `DarcyNormalizers` :51-54, `darcy_from_fields` :126-183.
- `data/loaders/poisson.rs`: `PoissonNormalizers` :112-117, `poisson_from_fields` :321-366.
- `data/loaders/burgers.rs`: `burgers_from_fields` :76-145.

## Goal
The Darcy and Poisson loaders end with the same steps:
1. fit x, then encode both splits;
2. fit y, then encode `y_train` only;
3. add the channel axis and convert to `T`.

They share the same `n_train < 2` check and the same `{x, y}` struct. Write the tail once,
and make `LoadError::Invalid` mean only "the data on disk is wrong".

## Deliverables
1. **Test first.** Before any refactor, pin `load_darcy_uniform`'s output on a small
   synthetic `.mat`: shapes, the normalizer statistics and a checksum of the first sample
   (or the exact first-sample values). Do the same for `load_poisson_uniform` if the
   existing fixtures don't already pin it.
2. **`data/loaders/common.rs`:**
   - `pub struct Normalizers { pub x: UnitGaussianNormalizer, pub y: UnitGaussianNormalizer }`,
     documented.
   - `pub(crate) fn normalize_split_pair(...)` implementing the tail, with the `n_train < 2`
     check.
   - `darcy_from_fields` and `poisson_from_fields` call it.
   - `DarcyNormalizers` and `PoissonNormalizers` are deleted. Their users:
     - `train/darcy.rs:86, 96-105`
     - `train/poisson.rs:124, 140-149`
     - `predict/poisson_resolution.rs:86`
     - the `train_poisson_smoke` test (moved by T3)
     - `predict/darcy.rs:50` and `darcy_plot.rs:41`, which discard it with `_`
   - Burgers doesn't normalise. It reuses only the channel-axis step.
3. **Infallible reshapes:**
   - Replace the reshape `Invalid` arms with `insert_axis(Axis(…))` on shapes built locally:
     `burgers.rs:117-130` (four arms), `darcy.rs:162-167`, `poisson.rs:345-350`.
   - Burgers' step 5 reshapes `[n, s]` to `[n, s]`, which does nothing. Delete it.
4. **No library `println!`:**
   - Remove the prints at `burgers.rs:136-141`, `darcy.rs:173` and `poisson.rs:356`.
   - If a caller printed that information, the example prints it from what the loader
     returns (shapes are available on the dataset).
5. **CONVENTIONS §10 prose:** "Darcy (`DarcyNormalizers`)" becomes "Darcy and Poisson
   (`loaders::common::Normalizers`)". This is prose only, so no version bump.
6. CLAUDE.md "Layout" gets a `loaders/common` line.

## Tests
- The new pinning tests pass both before and after the refactor. They are committed in the
  same PR, in a commit before the refactor commit.
- `normalizer_asymmetry_matches_darcy`, `single_train_sample_is_invalid_not_nan` and the
  other loader tests pass unchanged.

## Acceptance
- `predict_darcy` on an existing Darcy run prints the same test L². If none exists locally,
  say so; the pinning test is then the evidence.
- `git grep -n 'println!' src/neural_operators/data` finds nothing outside tests.

## Checks to run and report
The six from the README. The chebyshev pair is needed only if the edits reach
`pde/poisson/` (expected: no).

## Do not
- Change what is encoded: `y_test` stays physical (CONVENTIONS §10). That changes only in
  T17.
- Split `loaders/poisson.rs` (T22), or change `x_normalizer.json` handling (T18).

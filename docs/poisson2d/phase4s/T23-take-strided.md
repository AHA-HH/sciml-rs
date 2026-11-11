# T23: one rank-generic `take_strided`

Phase 4s, task 23. Branch `phase4s/T23-take-strided`. Depends on T9. Can run in parallel
with T22, which shares `loaders/{burgers,darcy}.rs`, so rebase.

> **Refresh at start:** re-read `loaders/burgers.rs` and `darcy.rs` after T9.

## Read first
- Review §5 S2 ("Subsampling" row), §6 T2-15, §7 Wave E4. CONVENTIONS §10, whose last
  bullet names `transforms::subsample`.
- `data/transforms/subsample.rs` (`subsample` :7). Callers: `loaders/burgers.rs:99-100`,
  and the Darcy test `old_path` at `darcy.rs:250-252`.
- `data/split.rs` (`train_test_split` :12). Caller: `loaders/burgers.rs:113-114`.

## Goal
Burgers copies all 2,048 samples twice: once to subsample, once to split. Replace the two
helpers with one strided take that does both in one view.

## Deliverables
1. **`pub fn take_strided<A, D>(field: ArrayView<A, D>, rows: Range<usize>, step: usize) ->
   Array<A, D>`**, or a signature close to it, rank-generic.
   - It selects `rows` on axis 0 and every `step`-th point on each spatial axis, via
     `slice_each_axis`.
   - Document the shapes and panics (step = 0, rows out of range).
   - Put it in `data/transforms/` (e.g. `transforms/strided.rs`).
2. **Burgers and Darcy** use it, and their output is identical.
3. **Delete** `transforms/subsample.rs` and `data/split.rs`, with their `mod` lines.
   - The Darcy `old_path` test oracle switches to an inline ndarray slice, or to
     `take_strided`. If it switches to `take_strided`, it must still compare against
     something independent.
4. **CONVENTIONS §10, last bullet:** `transforms::subsample` becomes the new name. This is
   prose only, so no version bump.

## Tests
- `take_strided` unit tests on rank 2 and 3: the step along every spatial axis, the row
  range, and step = 1 giving the identity.
- The Burgers and Darcy loader tests pass unchanged.
- **Deliberate removals:** the `subsample` and `split` unit tests. Give the counts.

## Acceptance
`predict_burgers` and `predict_darcy` on existing runs print the same test L².

## Checks to run and report
The six from the README. The chebyshev pair is not needed.

## Do not
Change the subsampling semantics: every `rate`-th point, starting at 0.

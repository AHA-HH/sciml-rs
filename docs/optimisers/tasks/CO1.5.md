# CO1.5 Split `trainer.rs`'s tests

Started 2026-11-27 on `optimisers/co1-5`. Roadmap: O1, CO1.5. Proposal: §1.4, §2
(constraint 1).

## Goal
Move `training/trainer.rs`'s `mod tests` into `training/trainer/tests.rs`, unchanged.

## Acceptance (roadmap *Done when*, verbatim)
The same tests, by name and count, pass; no code outside the tests changed.

## Read first
`src/neural_operators/training/trainer.rs` (the `#[cfg(test)] mod tests` block at its end).

## Changes
- `training/trainer.rs`: the inline `mod tests { … }` becomes `#[cfg(test)] mod tests;`.
  The test-only items above it (`train_built`, the `#[cfg(test)] use burn::module::Module`)
  stay where they are: they are code of `trainer`, not of `tests`, and moving them is
  outside "unchanged".
- `training/trainer/tests.rs` (new): the block's body, one indent level out; nothing else.
  A file module `trainer.rs` finds its child `tests` at `trainer/tests.rs`, so the path
  `training::trainer::tests::<name>` and every test name stay the same.
- No public API, no conventions.

## Tests that prove it
- The list of `neural_operators::training::trainer::` tests from `cargo test --lib -- --list`,
  with and without `datagen`, before (at `origin/main`) and after: identical (26 names).
- Those 26 pass.
- `git diff` of `trainer.rs` removes only the block; `tests.rs` equals the removed block,
  dedented (checked by re-indenting it and comparing).

## Checks
`/verify` (fmt, clippy lib/examples/all targets, datagen clippy and tests, `cargo test`,
`cargo doc`). No outputs can change: no non-test code is touched, so constraint 1 holds by
construction.

## VM runs
None.

## Constraints and risks
Constraint 1: no code outside `#[cfg(test)]` changes. Risk: none beyond the move.

## Pending runs
None.

## Open questions
None.

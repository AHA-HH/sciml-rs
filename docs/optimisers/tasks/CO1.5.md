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

## Result
The 26 tests of `neural_operators::training::trainer::tests`, from
`cargo test --lib -- --list`, at `f7e769a` (before) and at the move (after), with and
without `datagen`: the four lists are byte for byte equal (`diff`), and all 26 pass. The list:

```
neural_operators::training::trainer::tests::adamw_without_weight_decay_equals_adam
neural_operators::training::trainer::tests::checkpoint_round_trips_adam_moments_and_cosine_position
neural_operators::training::trainer::tests::components_take_sample_counts_from_the_datasets
neural_operators::training::trainer::tests::default_observer_prints_the_metrics
neural_operators::training::trainer::tests::device_loaders_train_identically_to_host_loaders
neural_operators::training::trainer::tests::empty_training_set_is_rejected
neural_operators::training::trainer::tests::gauss_newton_smoke_run_resumes_exactly
neural_operators::training::trainer::tests::gauss_newton_with_replicas_panics
neural_operators::training::trainer::tests::hybrid_smoke_run_switches_and_resumes_exactly
neural_operators::training::trainer::tests::hybrid_without_gauss_newton_epochs_panics
neural_operators::training::trainer::tests::lbfgs_smoke_run_resumes_exactly
neural_operators::training::trainer::tests::lbfgs_with_replicas_panics
neural_operators::training::trainer::tests::metrics_csv_holds_every_epoch
neural_operators::training::trainer::tests::old_training_config_trains_as_adam
neural_operators::training::trainer::tests::resumed_parallel_run_matches_uninterrupted_run
neural_operators::training::trainer::tests::resumed_run_matches_uninterrupted_run
neural_operators::training::trainer::tests::schedule_and_test_divisor_match_the_data
neural_operators::training::trainer::tests::squared_loss_steps_on_the_squared_objective
neural_operators::training::trainer::tests::squared_loss_trains_differently_and_reports_the_unsquared_l2
neural_operators::training::trainer::tests::squared_loss_with_replicas_panics
neural_operators::training::trainer::tests::step_schedule_halves_at_the_stated_epochs
neural_operators::training::trainer::tests::time_budget_stops_after_the_epoch_that_reaches_it
neural_operators::training::trainer::tests::train_fno_none_matches_identity_path
neural_operators::training::trainer::tests::train_loss_matches_old_asymmetric_path
neural_operators::training::trainer::tests::train_poisson_smoke
neural_operators::training::trainer::tests::two_replicas_train_as_one_device
```

`trainer/tests.rs` equals rustfmt applied to the old block's body dedented by four
spaces, the continuation line of the raw JSON string in `old_training_config_trains_as_adam`
kept at its old indent (its bytes unchanged). rustfmt rejoins nine `let … =` statements
and one `use` that now fit in 100 columns; nothing else differs.

## Pending runs
None.

## Open questions
None.

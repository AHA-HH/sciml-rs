# CM0.7 Retire `Operator::NAME`

Started 2026-11-27 on `operators/cm0-7`. Roadmap: CM0.7 (CM0.4's code-quality F3; the
author's decision of 2026-11-27 on #180). Proposal: §6.1 (the operator trait), §5
(constraint 1).

## Goal
Remove `Operator::NAME`, a second name for `OperatorConfig::KIND`, so that the registry
name exists once.

## Acceptance (roadmap *Done when*, verbatim)
`Operator` has no `NAME`; its readers use `<M::Config as OperatorConfig>::KIND`; root
`CLAUDE.md` and `feature-matrix.md` say so; the CM0.3 bit-identity check passes against
`main`.

## Read first
`src/neural_operators/models/operator.rs` (the trait, the default `NAME`),
`src/neural_operators/models/registry.rs` (`OperatorConfig::KIND`, `kinds_lists_every_variant`).

## Changes
- `models/operator.rs`: drop the `NAME` constant. Its only readers are two test asserts
  (`operator.rs` `fno_through_the_trait_is_the_fno`, `registry.rs`
  `kinds_lists_every_variant`); they read `<<FNO<3> as Operator>::Config as
  OperatorConfig>::KIND` instead. Nothing in `training/`, `runs/` or `examples/` reads
  `NAME` (grep on `origin/main` 9b0337d), so CO3.3's `training/` is not touched.
- `CLAUDE.md` (Layout, `models/operator.rs`) and `feature-matrix.md` (`BaseModel` row):
  `NAME` removed, the name being `OperatorConfig::KIND`.
- No public API besides the removed constant; no convention.

## Tests that prove it
- The two asserts above, rewritten: the trait's config kind is `"fno"` and equals
  `FNOConfig::KIND`.
- A compile-time absence cannot be unit-tested; `grep -rnw NAME src examples tests` shows
  no reader.
- Constraint 1: `tests/fno_bit_identity.rs` on `origin/main` and on the branch, the
  `target/bit-identity/*.bin` files compared (`cmp`); `run_dir` loading is untouched
  (`model_cfg.json` is written and read by `KIND`, unchanged), and the existing
  `runs`/registry tests that load old configs still pass.

## Checks
`/verify` (fmt, clippy ×4, `cargo test`, `cargo test --features datagen`, `cargo doc`,
the bit-identity dumps). `CARGO_TARGET_DIR=~/sciml-rs-target/cm0-7`.

## VM runs
None.

## Constraints and risks
Constraint 1: no output path changes (a const nobody reads at run time). A downstream
crate naming `Operator::NAME` would break; there is none (one crate).

## Pending runs
None.

## Open questions
None.

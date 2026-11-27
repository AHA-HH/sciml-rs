# CM0.6 Tidy after the generic stack

Started 2026-11-27 on `operators/cm0-6`. Roadmap: "CM0.6 Tidy after the generic stack. The
CM0.3 code-quality review's findings (`docs/operators/progress.md`, CM0.3): one name per
concept for the FNO wrappers (`train_fno`, `build_training_components`, `load_fno`,
`checkpoint`, `Checkpoint`) and their generic twins (keep, deprecate or retire);
`FnoResidual` renamed for any operator; the `#[cfg(test)]` imports that keep the old test
modules' `use super::*` compiling moved into those modules; `training/trainer.rs`'s test
module split out. *Done when:* the same bit-identity check as CM0.3 passes against `main`."
Proposal: §10; §1.4, §2, §6.1.

## Goal

One public name per concept in the generic training stack, with every existing FNO output
(Adam, Gauss–Newton and L-BFGS) bit for bit unchanged.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)

- Roadmap: "the same bit-identity check as CM0.3 passes against `main`."
- Constraint 1 (§1.4): every existing FNO output bit for bit identical; the author's brief
  (2026-11-27) names Adam, Gauss–Newton and L-BFGS.

## Read first

- The CM0.3 code-quality review (main checkout, git-ignored:
  `docs/code-quality/thermo-nuclear-review-2026-11-24-operators-cm0-3.md`), F1–F5 and its
  next step 4; CM0.4's F3 (`Operator::NAME`); CM0.5's F1 (`NpzFileReader::names()`,
  assigned to CM0.6 in the roadmap's CM0.5 line).
- `training/trainer.rs`, `training/gn.rs`, `runs/mod.rs`, `runs/checkpoint.rs`,
  `tests/fno_bit_identity.rs`.

## Changes

- **Retire the FNO wrappers (F2).** The generic names are the API; each FNO wrapper goes,
  its callers (`src/`, `examples/`, `tests/`, `spikes/`) moved to the generic one:

  | retired | use |
  |---|---|
  | `train_fno::<R, RM1, _>` | `train::<FNO<R>, R, RM1, _>` |
  | `build_training_components::<R, RM1, _>` | `build_components::<FNO<R>, R, RM1, _>` |
  | `RunDir::load_fno::<R>` | `RunDir::load_model::<FNO<R>>` |
  | `RunDir::checkpoint::<R>` | `RunDir::checkpoint_of::<FNO<R>>` |
  | `Checkpoint<R>` | `OperatorCheckpoint<FNO<R>>` |

  Retire, not deprecate: clippy's `-D warnings` turns a deprecated call into an error, so
  deprecation would mean moving every caller anyway. The generic names keep their spelling
  (no reuse of a retired name for a different signature), so a parallel branch still
  calling a retired name fails to compile with that name in the error.
- **`EpochObserver<'a, M>` (F3):** the dead `R` and the FNO default go.
- **`FnoResidual` → `GridResidual` (F4)**, for any `GridOperator`; module doc updated.
- **`#[cfg(test)]` imports (F1):** the four shims (`training/step.rs`, `training/trainer.rs`,
  `training/data_parallel.rs`, `models/predict.rs`) move into their test modules (the
  author's answer of 2026-11-25: not a test edit).
- **Test module split (F5):** already done by optimisers CO1.5 (#176,
  `training/trainer/tests.rs`); CO1.5's review findings on that file are optional task
  CO1.6, not this task.
- **`NpzFileReader::names()`** (CM0.5's F1): the parity harness reads the names from the
  reader it already holds instead of re-opening the archive.
- Root `CLAUDE.md`'s Layout entry updated for the names.
- Not changed: `Operator::NAME` (CM0.4's F3) stays until the author answers (a question in
  the PR); `TrainingComponents`' `M = FNO<R>` default (not a second name).
- Conventions: none.

## Tests that prove it

- `tests/fno_bit_identity.rs`, moved to the generic names (all on `main` since CM0.3) and
  given a fourth case, `lbfgs_epoch` (a tiny FNO, one L-BFGS epoch through the same
  checkpoint, reload and predict path), so the file runs on `main` unchanged. Run in
  release on an export of `origin/main` and on the branch; the four dumps (`forward`,
  `phase5_epoch`, `gauss_newton_epoch`, `lbfgs_epoch`) `cmp` identical.
- Every existing test passes; the only test edits are the renamed calls, the moved imports
  and the new case.

## Checks

`/verify`: fmt, clippy (default, `--examples`, `--all-targets`, `--features datagen
--all-targets`), `cargo test`, `cargo test --features datagen`, `cargo doc --no-deps`,
`cargo check -p fno3d-cost-spike -p linalg-spike --all-targets` (their callers moved), and
the bit-identity `cmp` above. `CARGO_TARGET_DIR=~/sciml-rs-target/cm0-6`.

## VM runs

None.

## Constraints and risks

- Constraint 1: names only, no arithmetic moves; checked by the dumps.
- Parallel lanes (CM1.5b, CM1.5c, CM3.6, CO0.3) that add a call to a retired name compile
  until merged beside this; the rename table above goes to them and into the PR.
- The 2D objective's examples change only in the names they call.

## Pending runs

None.

## Open questions

1. `Operator::NAME` (CM0.4's F3): always its config's `KIND`; keep it (documented as not to
   be overridden) or retire it in favour of `OperatorConfig::KIND`?

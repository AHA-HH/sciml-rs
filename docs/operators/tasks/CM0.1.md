# CM0.1 Feature matrix at 2.0.0

Started 2026-11-24 on `operators/cm0-1`. Roadmap: "**CM0.1 Feature matrix at 2.0.0.**
Fill `feature-matrix.md` from the tag's source (every public model, layer, loss, training
feature and dataset, with its options and defaults). *Done when:* every row has a status
and a source path, checked by an agent independent of the writer. *Needs:* D1."
Proposal: §10; §1.2 (scope), §3 (the 2.0.0 reference), §6.4 (every operator task sets its
row).

## Goal
Replace the seeded matrix with one read from the source of `neuraloperator` tag `2.0.0`
(commit `b307030`): every public item, its source path, its options and defaults, what
sciml-rs has against it, a status and the task that sets it.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)
- "every row has a status and a source path, checked by an agent independent of the
  writer."
- Proposal §3: "Every row is to be verified and completed by CM0.1
  (`docs/operators/feature-matrix.md`)."

## Read first
- The tag's source: `git clone --depth 1 --branch 2.0.0
  https://github.com/neuraloperator/neuraloperator.git` into a scratch directory (plain
  HTTP; never the author's Chrome). Package `__init__.py` files for what is public.
- The seeded `docs/operators/feature-matrix.md`; proposal §1.2 and §3.
- The sciml-rs side: `models/fno/mod.rs` (`FNOConfig`), `layers/spectral_convolution.rs`
  (`SpectralInit`), `losses/data_losses.rs`, `training/config.rs`, `data/`,
  `docs/CONVENTIONS.md` §2–§6, §10, §11.

## Changes
- `docs/operators/feature-matrix.md`: tables gain a *Source* column (path under
  `neuralop/` at 2.0.0); rows for every public class and function, including the areas
  the seed lacked (`data/transforms`, `mpu`, `utils.py`, the equation-loss classes, the
  differentiation APIs); an appendix "Options and defaults at 2.0.0" listing each item's
  constructor or function signature with its defaults, extracted mechanically with
  Python's `ast` from the tag and then checked.
- Statuses keep the seed's four (**have**, **partial**, **missing**, **not planned**);
  no row is **have**, as no output parity has been shown yet. Items outside proposal §1.2
  (model parallelism, W&B, download helpers) are **not planned**, listed as a question.
- This plan, the roadmap's CM0.1 line, a `progress.md` entry.
- No code, no convention, no dependency. The proposal is not edited: where the source
  shows it wrong or incomplete, the PR asks.

## Tests that prove it
- An independent agent (not the writer) re-reads the tag's source and checks every row:
  the item exists at the stated path, it is public, the options and defaults match the
  signature, and the sciml-rs column and status are true of this repository. It also
  checks completeness: every public class and function of the tag has a row. Mismatches
  are fixed and the check repeated until none remain.

## Checks
- `/verify` (docs-only: the `changes` filter skips CI's jobs; no Rust check applies).
- The independent row check above.

## VM runs
None.

## Constraints and risks
- Constraint 1: no code changes, so every FNO output is untouched.
- Scope: `docs/operators/` only (CM0.2 runs in parallel). D0's row and the roadmap status
  line belong to PR #126 and are not edited here.
- Risk: "public" in Python is a convention; the matrix counts every top-level class and
  function without a leading underscore in a non-test module, and marks internal helpers
  as such.

## Pending runs
None.

## Open questions
See the PR's questions for sign-off.

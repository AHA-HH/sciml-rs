# Neural operator library progress log

Newest entries at the bottom. One entry per task, in that task's pull request, using this
template:

```
## YYYY-MM-DD: <ID> <title>
- Done:
- Tests:
- Checks: ran …; pending …
- Decisions: none, or D… signed off; questions for sign-off …
- Review: plan review (blocking items fixed); code quality APPROVE or BLOCKED → APPROVE (rounds, blockers fixed, findings left as follow-ups)
- FNO: unchanged bit for bit (how shown), or not touched
- Matrix: rows set
- Follow-ups:
```

## 2026-11-11: Planning
- Done: the proposal, draft 0.1, in `docs/design/neural-operator-library.md`, with its
  development wireframes (§4); its Section 10 as `docs/operators/roadmap.md`; the feature
  matrix seeded from the 2.0.0 file tree in `docs/operators/feature-matrix.md`.
- Tests: none.
- Checks: none run; the 2.0.0 facts come from the tag's file tree and source read on
  2026-11-11.
- Decisions: D1 (`neuraloperator` 2.0.0) taken by the author 2026-11-11; D0 comes next.
- FNO: not touched.
- Matrix: seeded, unverified (CM0.1).
- Follow-ups: review the draft; D0; then CM0.1 and CM0.2.

## 2026-11-24: D0 and D2
- Done: D0 signed off by the author: draft 0.1 of `docs/design/neural-operator-library.md`
  as the plan of record. D2 signed off as recommended: the Python side in `tools/parity/`,
  outside CI, with a lock file; fixtures in `tests/fixtures/parity/`.
- Tests: none.
- Checks: none (docs).
- Decisions: D0 and D2 signed off 2026-11-24.
- FNO: not touched.
- Matrix: unchanged.
- Follow-ups: CM0.1 and CM0.2 (no Rust, can run beside the 3D tasks); CM0.3 before the
  optimisers tasks that change `training/`.

## 2026-11-24: CM0.1 Feature matrix at 2.0.0
- Done: `feature-matrix.md` read from the source of tag 2.0.0 (`b307030`): every public
  top-level class and function of `neuralop/` (tests excluded) in a table row with its
  source path, the sciml-rs counterpart, a status and a task; an appendix of 186
  signatures with defaults, extracted with `ast`. New against the seed: `data/transforms`,
  `mpu`, `utils.py`, `BatchNorm`, the equation-loss classes, the differentiation APIs, the
  registry's contents (UNO and CODANO are not in `get_model`). Corrected: 2.0.0's Burgers
  dataset is space-time, so not the crate's Burgers task (missing, not partial).
- Tests: an independent agent re-read its own clone of the tag and checked every row and
  signature: 183 of 185 signatures exact, one item missing from the appendix, an
  inconsistent "exported" rule, two wrong parentheticals and three thin sciml-rs cells;
  all fixed. A second pass found one more (`LocalNO` does need torch_harmonics: the DISCO
  module builds a dict from names it failed to import); fixed, and nothing else remained
  (the "exported" marks match all five `__init__.py` files; 186 signatures exact).
- Checks: ran the independent row check (twice); `/verify` not run: docs only, no Rust
  check applies and CI's `changes` filter skips its jobs. Pending none.
- Decisions: none taken; questions for sign-off in the PR (scope of the not-planned rows,
  owners for the rows without a task, proposal §3 and CM1.3 corrections).
- Review: plan review PASS, no blocker; its findings fixed (this entry's evidence, the
  normaliser's eps, the legend's qualified statuses) or put to the author (owners of the
  "–" rows); code-quality review skipped (docs only).
- FNO: not touched.
- Matrix: every row set.
- Follow-ups: owners for the "–" rows (datasets, transforms, utilities): no roadmap line
  until the author answers the PR's question 2.

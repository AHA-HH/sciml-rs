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

## 2026-11-24: CM0.2 Pinned Python environment and fixture generator
- Done: `tools/parity/`, a uv project (Python 3.12, `uv.lock`) pinning `neuraloperator`
  2.0.0 (PyPI wheel, its Python code identical to the tag's), torch 2.9.0, tensorly 0.9.0,
  tensorly-torch 0.5.0, torch-harmonics 0.7.4 (the tag's `doc` pin), numpy 2.3.4;
  `generate.py` writes 2.0.0's `FNO` (defaults but sizes) as `.npz` (input, output,
  `param.<name>`, complex as a trailing (re, im) axis) with a `.json` sidecar:
  `fno1d` (P = 3,529) and `fno2d` (P = 7,129), each in f32 and f64, 28–103 kB.
- Tests: the generator asserts P ≤ 10⁴, size ≤ 1 MB and f32/f64-only arrays, and reloads
  each file into a zeroed fresh model, reproducing the output bit for bit; `--check`
  regenerates and compares arrays and sidecars bit for bit (passes).
- Checks: ran `uv sync --locked`, `generate.py`, `generate.py --check`, the wheel-against-tag
  diff; the reviewer ran `cargo fmt --check`, clippy (default, `--all-targets`,
  `datagen`), `cargo test`, `cargo test --features datagen`, `cargo doc` (all pass); metal
  not run (no Rust changed).
- Decisions: D8 (fixture size) signed off 2026-11-24 as recommended (P ≲ 10⁴, under about
  1 MB per fixture).
- Review: plan review PASS (advisories fixed: round trip from zeroed weights, strict load,
  sidecars in `--check`); code quality APPROVE with findings (F1–F3 fixed; F4, F5 left in
  the PR text).
- FNO: not touched (no Rust changed).
- Matrix: no rows (environment only).
- Follow-ups: CM0.5 reads these fixtures (the CM0.5 line).

## 2026-11-24: CM0.3 The operator trait and a generic stack
- Done: D4 signed off by the author (option A: a generic trait with associated config and
  input types, static dispatch, no trait objects). `models/operator.rs`: `Operator`
  (`Config`, `Input`, `Output`, `NAME`, `init`, `forward`) and `GridOperator<R>` (rank-R
  tensors in and out, a blanket impl); `FNO<R>` implements it. Generic over the operator:
  `train_epoch`, `training_loop`, `train_components`, the new `train` and
  `build_components`, the data-parallel epoch, `train_epoch_gn` and `FnoResidual`'s
  `LeastSquares` impl, `evaluate`, `predict_host`, `RunDir::save_weights`,
  `save_checkpoint(_gn)`, the new `load_model` and `checkpoint_of`, `OperatorCheckpoint<M>`;
  `EpochObserver` and `TrainingComponents` take `M = FNO<R>`. The FNO-named functions stay
  as wrappers, so every existing caller compiles unchanged; file formats unchanged.
- Tests: `tests/fno_bit_identity.rs` (ignored by default; release): a fixed Phase 5 FNO
  forward, one Phase 5-shaped Adam epoch (two batches, checkpoint, reload, `predict_host`)
  and one Gauss–Newton epoch (with its checkpointed state), dumped as bytes; the same file
  run on an export of `main` and on the branch gives byte-identical dumps (65,536,
  28,481,592 and 22,476 bytes). Files are compared by content: Burnpack and
  `GaussNewtonState::to_bytes` record random `ParamId`s, so even `main` never writes the
  same bytes twice. `models/operator.rs`: the trait's forward is the FNO's; a test-only
  second operator trains, checkpoints, reloads and predicts through the generic stack.
- Checks: ran `cargo fmt --check`, clippy (default, `--examples`, `--all-targets`,
  `--features datagen --all-targets`), `cargo test` (305 lib + integration, all pass),
  `cargo test --features datagen` (379 lib, all pass), `cargo doc --no-deps` (no
  warnings), `cargo test --features metal --test gpu_parity` (9 pass) and `--features metal --lib`
  (305 pass; built at 851b6e4, later commits change only tests and docs); the main-against-
  branch `cmp` (twice, and independently by the plan reviewer). No hunk inside any
  existing test module; `tests/` only gains the new file.
- Decisions: D4 signed off 2026-11-24. Question for the author in the PR: does moving a
  test module's imports count as editing a test (CM0.6)?
- Review: plan review PASS (one blocker, a test comparing two seeded inits under Flex's
  process-wide RNG, fixed; advisories taken: two batches, the GN state, comment fixes, the
  CM0.4 sentence); code quality APPROVE with 7 findings, no blockers (F6 fixed; F7 and the
  checkpoint naming in the plan; F1–F5 to CM0.6).
- FNO: unchanged bit for bit (the dumps above against `main`).
- Matrix: `BaseModel`, `Trainer`, training state set (partial; CM0.3's part done).
- Follow-ups: CM0.4 (also the MLflow export's model config); CM0.6 (the review's F1–F5).

## 2026-11-25: CM0.4 The registry
- Done: `models/registry.rs`: `OperatorConfig` (`KIND`; `FNOConfig` is `"fno"`), which
  `Operator::Config` now requires (`Operator::NAME` defaults to it); `DEFAULT_KIND`,
  `KINDS`; `ModelConfig` (`from_json`, `load`: by `kind`, a missing `kind` reading as the
  FNO, an unknown one an error naming the registered kinds) and `Model<R>` (`init`,
  `forward`): the constructor by name, one enum variant per kind (D4, static dispatch).
  `RunDir::save_configs` writes `kind` first in `model_cfg.json` (callers unchanged);
  `model_config`, `load_model` and `checkpoint_of` check the file's kind against the
  config type; new `RunDir::any_model_config`. The MLflow export reads the model config
  through the registry; its requests are unchanged.
- Tests: a pre-CM0.4 `model_cfg.json` (no `kind`) loads through `model_config`,
  `any_model_config` and `load_model`, the loaded model predicting bit for bit as the
  saved one; a new file starts with `"kind": "fno"` and still parses as a plain
  `FNOConfig` (what a reader built from `main` does); wrong and unknown kinds fail with
  messages naming them; every kind round-trips through `from_json`; a config older than
  `spectral_init`/`padding` parses (at the parse level, not through `RunDir`); `Model::<4>`
  built by name forwards bit for bit as the FNO with the same weights; the MLflow
  `start_requests` of a directory with and without `kind` are equal.
- Checks: ran `cargo fmt --check`, clippy (default, `--examples`, `--all-targets`,
  `--features mlflow --all-targets`, `--features datagen --all-targets`), `cargo test`,
  `cargo test --features datagen`, `cargo test --features mlflow --lib`, `cargo doc
  --no-deps` (all pass); `cargo test --features metal --lib` (326 pass, at 2a01cf2, before
  the JSON-only review fixes); `tests/fno_bit_identity.rs` on the branch and on `main`,
  dumps identical (65,536, 22,476 and 28,481,592 bytes).
- Decisions: none. Question for the author in the PR: keep `Operator::NAME` (now always its
  config's `KIND`) or retire it (CM0.6).
- Review: plan review PASS (advisories: `DEFAULT_KIND` from `FNOConfig::KIND` and one
  reader, fixed; the older-format test is parse-level, stated above); code quality APPROVE
  with findings (F1, F2 fixed; F3 a question in the PR; F4 to CM2.1).
- FNO: unchanged bit for bit (the dumps above against `main`); only new runs'
  `model_cfg.json` gains the `kind` key.
- Matrix: `BaseModel`, `get_model`/`available_models` set to have (infrastructure).
- Follow-ups: CM2.1 (the second kind: its variants, the export's arm, `Model<R>` as a
  `Module` if needed; review F4); CM0.6 (`Operator::NAME`, on the author's answer).

## 2026-11-25: D3 and the CM0.3 test-import question
- Done: D3 signed off by the author as recommended: parity holds when the relative max
  error is ≤ 1e-5 in f32 and ≤ 1e-12 in f64, on the CPU. The author's answer to CM0.3's
  question: moving a test module's `use` lines into that module is not editing a test, so
  CM0.6 moves the four `#[cfg(test)]` imports.
- Tests: none.
- Checks: none (docs).
- Decisions: D3 signed off 2026-11-25.
- FNO: not touched.
- Matrix: unchanged.
- Follow-ups: CM0.5 (the parity harness) and CM0.6 (the tidy-up) are runnable.

## 2026-11-25: CM0.5 The parity harness
- Done: `tests/parity_common/mod.rs`, the harness: `Fixture` (a CM0.2 `.npz`, its precision
  from the name), `Take` (`Whole`, or the `Re`/`Im` half of a complex tensor's trailing
  axis), `load_into` (a hand-written map of 2.0.0 names to Burn paths, applied with
  `ModuleSnapshot::apply`; panics unless every fixture parameter is mapped, complex ones by
  both halves, every entry applies and no module parameter is left out), `rel_max_error`
  (max |a − e| / max |e|; panics on a NaN or infinity) and `assert_parity`, with D3's
  bounds. `tests/parity_common/fno1d.rs`: `Fno1dRef`, 2.0.0's 1D `FNO` at its defaults in
  test code (Burn `Conv1d` and `gelu`, the crate's `SpectralConv`), and its 44-entry map
  of the fixture's 40 arrays.
- Tests: `fno1d_f32_matches_2_0_0` (3.514e-7 ≤ 1e-5, Flex); `fno1d_f64_matches_2_0_0` in
  its own binary, `#[ignore]`d (1.211e-8 > 1e-12: Flex's f64 `rfft` twiddles are f32); a
  1e-3 weight shift fails parity; the load rejects an unmapped array, an unknown path and
  half a complex tensor; the re/im split; `rel_max_error` on a hand-computed pair and on NaN.
- Checks: ran `cargo fmt --check`, clippy (default, `--examples`, `--all-targets`,
  `--features datagen --all-targets`), `cargo test`, `cargo doc --no-deps` (all pass);
  `cargo test --features datagen` (pass, run by the plan review before its NaN fix, which
  changes only `tests/`). Not run: `--features metal` (no backend-sensitive change), the
  bit-identity dumps (nothing under `src/` changes).
- Decisions: none. Questions for the author in the PR: D3's f64 bound on the CPU while the
  Flex bug stands; the reading of "relative max error".
- Review: plan review FAIL on one blocker (`rel_max_error` let NaN through), fixed with a
  test; its advisories (blanket `allow`, unused `d3`, plan text, grid dtype) fixed. Code
  quality APPROVE with findings (F2, F4, F6, F7 and a nit fixed; F1, F3, F5, F8 in the
  roadmap line).
- FNO: not touched (tests only).
- Matrix: unchanged (the mirror is test code; no row reaches **have**).
- Follow-ups: in CM0.5's roadmap line (for CM0.6, CM1.1 and CM1.5).

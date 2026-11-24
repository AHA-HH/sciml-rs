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
- Decisions: D8 (fixture size) asked in the PR, with these sizes.
- Review: plan review PASS (advisories fixed: round trip from zeroed weights, strict load,
  sidecars in `--check`); code quality APPROVE with findings (F1–F3 fixed; F4, F5 left in
  the PR text).
- FNO: not touched (no Rust changed).
- Matrix: no rows (environment only).
- Follow-ups: D8's sign-off; then CM0.5 reads these fixtures (the CM0.5 line).

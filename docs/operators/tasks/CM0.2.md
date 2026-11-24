# CM0.2 Pinned Python environment and fixture generator

Started 2026-11-24 on `operators/cm0-2`. Roadmap: "**CM0.2 Pinned Python environment and
fixture generator.** `tools/parity/`: `neuraloperator` 2.0.0 with torch, tensorly, tltorch
and torch_harmonics pinned in a lock file; a generator writing weights, input and output of
a named configuration to `.npz`. *Done when:* a 1D and a 2D FNO fixture written, sizes
within D8. *Needs:* D2, D8." Proposal: §10; §1.4, §2, §3, §5 (output parity), §6.3 (weight
conversion), §12 (D2, D3, D8).

## Goal
A reproducible, locked Python environment in `tools/parity/` and a generator that writes
2.0.0's FNO weights, an input and its output to `tests/fixtures/parity/`, for a 1D and a 2D
configuration.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)
- Roadmap: "a 1D and a 2D FNO fixture written, sizes within D8."
- D8 (proposal §12, recommendation): "tiny models only (P ≲ 10⁴), so fixtures stay under
  about 1 MB each." Signed off 2026-11-24 as recommended.
- §5: fixtures are "weights, input, output written by 2.0.0 on the CPU in f32 and f64".
- D2 (signed off 2026-11-24): `tools/parity/`, outside CI, with a lock file; fixtures in
  `tests/fixtures/parity/`.

## Read first
- `neuraloperator` 2.0.0 (tag commit `b307030d`): `pyproject.toml` (torch is not a declared
  dependency; `torch_harmonics==0.7.4` in the `doc` extra), `neuralop/models/fno.py`
  (`FNO.__init__` defaults).
- `src/neural_operators/data/io/readers/npz.rs`: `NpzFileReader::read_field` reads `f32` and
  `f64` arrays only, so complex tensors must be stored as real arrays.

## Changes
- `tools/parity/pyproject.toml` (a uv project, not a package), `uv.lock`,
  `.python-version` (3.12). Pins, exact: `neuraloperator==2.0.0` (PyPI wheel, checked
  against the tag's source), `torch==2.9.0`, `tensorly==0.9.0`, `tensorly-torch==0.5.0`,
  `torch-harmonics==0.7.4`, `numpy==2.3.4`. Rule: the tag's own pin where it gives one
  (torch_harmonics, the `doc` extra), otherwise the last release before the tag's commit
  date (2025-10-22). Everything else is pinned by `uv.lock`, with hashes.
- `tools/parity/generate.py`: named configurations (`fno1d`, `fno2d`) and dtypes (`f32`,
  `f64`); for each it seeds torch, builds 2.0.0's `FNO` (its defaults except the sizes),
  draws a seeded input, runs the forward on the CPU in eval mode, and writes
  `tests/fixtures/parity/<config>_<dtype>.npz` and a `.json` sidecar.
  - `.npz` keys: `input`, `output`, and every `state_dict` entry under its 2.0.0 name with
    the prefix `param.`. A complex tensor is stored as `torch.view_as_real` (a trailing
    axis of 2: real, imaginary); the sidecar lists those names. Only `f32`/`f64` arrays.
  - Sidecar: configuration name, `FNO` kwargs, dtype, seeds, input shape, parameter count
    P, the complex parameter names, and the versions of Python, neuraloperator, torch,
    tensorly, tltorch, torch_harmonics and numpy.
  - `--check`: regenerate in memory and compare with the committed fixtures, array for
    array, bit for bit; exit non-zero on a difference.
- `tools/parity/README.md`: setup (`uv sync`), commands, the fixture format, the pin rule.
- No Rust. No `docs/CONVENTIONS.md` change. No dependency of the crate changes.

## Tests that prove it
- The generator asserts, per fixture: P ≤ 10⁴ and file size ≤ 1 MB (D8 as recommended);
  every array `f32` or `f64` (readable by `NpzFileReader`).
- Round trip in Python (in the generator, after writing): read the `.npz` back, load the
  parameters into a freshly built model (complex rebuilt by `view_as_complex`), run the
  stored input: equal to the stored output bit for bit. This shows the weights in the file
  are complete and the complex packing is right.
- `generate.py --check` passes on the committed fixtures (reproducible from the lock).
- The wheel's `neuralop/` is the tag's: diff of the installed package against the shallow
  clone of tag `2.0.0`.

## Checks
- `/verify`: no Rust changed, so the cargo checks have nothing new to see; it decides what
  to run. CI's `changes` job: `tools/` and `tests/fixtures/` — `tests/` is in its list, so
  CI runs; no Rust test reads the fixtures yet (CM0.5 does).
- `cd tools/parity && uv sync --locked && uv run python generate.py && uv run python
  generate.py --check`.

## VM runs
None.

## Constraints and risks
- Constraint 1: no Rust code changes; every FNO output is untouched.
- Python stays in `tools/parity/`, outside CI (D2).
- Risk: torch CPU kernels may differ across machines/OSes in the last bits; the `--check`
  is a same-machine reproducibility check. The fixtures are the reference whatever machine
  wrote them; the sidecar names the platform.

## Pending runs
None.

## Open questions
- D8 (fixture size): signed off 2026-11-24 as recommended (P ≲ 10⁴, under about 1 MB per
  fixture); the fixtures are at most P = 7,129 and 103 kB.

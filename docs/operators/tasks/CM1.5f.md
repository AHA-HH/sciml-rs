# CM1.5f The parity harness

Started 2026-11-27 on `operators/cm1-5f`. Roadmap: CM1.5f (split from CM1.5d). Proposal:
§10 (M1); §6.4 (the recipe); D3.

## Goal
Pay the parity harness's review debts of CM1.2 and CM1.5a, in test code only: one biased
layer, conversions passed to `load_into`, no hand-written reference FNO, no map hook.

## Acceptance (roadmap *Done when*, verbatim)
"every parity test passes at its recorded error, and the dumps equal `main`'s."

## Read first
`tests/parity_common/{mod,legacy_conv,centred_conv,fno,fno1d}.rs`, `tests/parity_fno*.rs`,
`tests/parity_*spectral_conv*.rs`; CM1.2's code-quality review (M1: `BiasedConv`; B1:
`load_into` takes the converted arrays, `Fixture`'s fields private); CM1.5a's (F4, F6).

## Changes
Test code only (`tests/`); nothing in `src/`, `tools/parity/` or the fixtures.
- `tests/parity_common/biased_conv.rs` (new, `#[path]`-included): `BiasedConv<R>`, a
  `SpectralConv<R>` plus a bias `[1, out, 1, …]`, 2.0.0's layer shape; `legacy_conv.rs` and
  `centred_conv.rs` keep their maps, cases and conversions; `biased_conv::run` (load and
  forward) and `check_d3` serve both, the fixture names by `Fixture::open_as` (review F1).
- `parity_common/mod.rs`: `Conversion { from, parts }` (a 2.0.0 parameter and the arrays
  replacing it; review F2);
  `load_into(module, fixture, map, converted)` and `load_into_part(.., converted, filter)`
  apply them for that load only; `Fixture::replace` and its `replaced` map go, `Fixture`'s
  fields become private. New check: a map entry naming a parameter the fixture (after
  conversion) lacks panics by name instead of in the reader; a conversion part may not take
  a remaining parameter's name (review F3).
- `parity_common/fno.rs`: `output(case, precision, hook, device)` becomes
  `load(case, precision, device) -> Load<R>`, whose public `map` and `converted` a control
  edits before `Load::run` loads and runs the FNO. Parity passes no hook.
- `Fno1dRef` retired: `parity_common/fno1d.rs`, `tests/parity_fno1d.rs`,
  `tests/parity_fno1d_f64.rs` deleted. Their checks move to `tests/parity_fno.rs` on the
  library `FNO` (`FNO1D`): the perturbed weight (a converted conv bias + 1e-3), the
  unmapped parameter, the path the module lacks, half a complex parameter, `take`'s split,
  `rel_max_error`'s scale and NaN. `fno1d_{f32,f64}` stay covered by `parity_fno{,_f64}.rs`.

## Tests that prove it
- Every parity binary passes, each fixture's printed error equal to `main`'s (both runs'
  `--nocapture` lines diffed); every control still fails D3 (`swapped_corners`,
  `swapped_batch_statistics`, `padding_left_off`, the legacy 3D corner order, the centred
  halves and odd extent) and every load rejection still panics with its message.
- New: `load_rejects_a_converted_away_parameter` (a map entry still naming a parameter a
  conversion replaced), `load_rejects_a_conversion_of_a_missing_parameter`,
  `load_rejects_a_conversion_part_shadowing_a_parameter`.
- Constraint 1: `fno_bit_identity` dumps on `main` and the branch `cmp` equal (no `src/`
  change, so expected trivially).

## Checks
`/verify` (fmt, clippy default/`--examples`/`--all-targets`/`--features datagen
--all-targets`, `cargo test`, `cargo test --features datagen`, `cargo doc --no-deps`, the
dumps); the f64 parity binaries with `-- --ignored`, errors equal to `main`'s.

## VM runs
None.

## Constraints and risks
Constraint 1 (no FNO output may change): untouched, no `src/` edit. CM1.5e runs in parallel
in `src/` (spectral layer, `utils/fft.rs`); this task stays in `tests/`. Risk: the test
count falls by the deleted `Fno1dRef` parity test (`fno1d_f32` is still checked by
`parity_fno::fno1d_matches_2_0_0`).

## Pending runs
None.

## Open questions
None.

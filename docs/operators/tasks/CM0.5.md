# CM0.5 The parity harness

Started 2026-11-25 on `operators/cm0-5`. Roadmap: "CM0.5 The parity harness. A test helper
that loads a fixture by parameter-name map into a module and compares outputs (and
gradients) within D3. *Done when:* the CM0.2 1D fixture passes through a hand-written map.
*Needs:* CM0.2, D3. *Sign-off:* D3." Proposal: §10; §1.4, §2, §5, §6.4. D2, D3, D8.

## Goal

A test helper that opens a CM0.2 fixture, applies its 2.0.0-named parameters to a Burn
module through a hand-written name map, and checks a tensor against the fixture's within D3.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)

- Roadmap: "the CM0.2 1D fixture passes through a hand-written map."
- D3 (signed off 2026-11-25): "relative max error ≤ 1e-5 in f32 and ≤ 1e-12 in f64, on
  the CPU". The harness reads it as max |a − e| / max |e| over the whole tensor (error
  relative to the reference's scale, so elements near zero do not dominate).
- Proposal §5: "for a fixture (weights, input, output written by 2.0.0 on the CPU in f32
  and f64), the Rust operator loaded with the same weights gives the same output, within a
  tolerance relative to the reference (D3)." Gradient parity: "the gradient of a fixed
  scalar loss against the fixture's", for parts with custom backward work.
- Constraint 1 (§1.4): every existing FNO output bit for bit identical.

## Read first

- `tools/parity/README.md`, `generate.py`: the `.npz` keys (`input`, `output`,
  `param.<2.0.0 name>`), complex tensors as a trailing (re, im) axis.
- `tests/fixtures/parity/fno1d_f32.{npz,json}`: `FNO(n_modes=(8,), in=1, out=1,
  hidden=8)` at 2.0.0's defaults, input (2, 1, 64).
- 2.0.0's source (`neuralop/models/fno.py`, `layers/fno_block.py`,
  `layers/spectral_convolution.py`, `layers/channel_mlp.py`, `layers/skip_connections.py`,
  `layers/embeddings.py`).
- `burn::store::{ModuleSnapshot, bridge}` (`apply` by path, `ApplyResult`);
  `data::io::readers::NpzFileReader`; `layers::spectral_convolution::SpectralConv`.

## Changes

Tests only; no change under `src/`, no dependency, no convention.

- `tests/parity_common/mod.rs`, the harness:
  - `Fixture::open(name)`: the `.npz` in `tests/fixtures/parity/`, its array names;
    `input()`, `output()`, `param(name)` as `ArrayD<f64>` (f32 cast up exactly).
  - `Take::{Whole, Re, Im}` and a map entry `(2.0.0 name, Burn path, Take)`; `Re`/`Im`
    take one half of a complex tensor's trailing (re, im) axis.
  - `load_into(module, fixture, map, dtype)`: builds the tensors under their Burn paths,
    `ModuleSnapshot::apply`, and panics unless every map entry applied, no module
    parameter is missing, no applied tensor is unused, and every `param.*` array of the
    fixture is used by the map (complex ones by both halves). Shapes are checked by `apply`.
  - `rel_max_error(actual, expected)` and `assert_parity(actual, expected, tol, what)`, for
    any tensor: outputs now, gradients when a fixture carries them (CM0.2 writes none; the
    proposal asks for gradient fixtures only for custom backward parts, from CM3).
  - `D3_F32 = 1e-5`, `D3_F64 = 1e-12`.
- `tests/parity_fno1d.rs`: `Fno1dRef`, a test-only module laid out as 2.0.0's 1D `FNO`
  at its defaults, built from Burn's `Conv1d`, the crate's `SpectralConv<3>` (one corner
  in 1D, `modes = [5]` = 8 // 2 + 1) and a `Param` per spectral bias and soft gate: grid
  embedding (x_j = j / n, appended after the data channel), lifting channel MLP (2 → 16 →
  8, GELU), four blocks (linear FNO skip, spectral conv plus bias, GELU, channel MLP 8 → 4 →
  8 with a soft-gating skip, GELU; no GELU after the last block), projection (8 → 16 → 1,
  GELU). The hand-written map from the 40 `param.*` arrays to its paths.

`fft_norm="forward"` (2.0.0) and Burn's unnormalised forward / 1/n inverse give the same
spectral convolution (it is linear, so only where 1/n is applied differs); no rescaling.

## Tests that prove it

- `fno1d_f32_matches_2_0_0`: the f32 fixture through `Fno1dRef` and the map; output against
  the fixture's within D3's f32 bound (1e-5), on Flex. The acceptance test.
- `fno1d_f64_matches_2_0_0`: the same at f64 against 1e-12, `#[ignore]`d for the known Burn
  bug (Flex's f64 `rfft` uses f32 twiddles, as `tests/gauss_newton_f64.rs`); run with
  `--ignored` and its error recorded.
- Harness self-tests (no model): `rel_max_error` on a hand-computed pair; `load_into`
  panics on a fixture array the map leaves out, and on a map path the module lacks;
  `Re`/`Im` split a hand-made complex array correctly.
- A negative control: perturbing one fixture weight by 1e-3 fails the f32 parity check,
  so the test can fail.

## Checks

`/verify` (fmt, clippy default / `--examples` / `--all-targets` / `--features datagen
--all-targets`, `cargo test`, `cargo test --features datagen`, `cargo doc --no-deps`);
`cargo test --test parity_fno1d -- --include-ignored` for the f64 number. No `src/` change,
so the bit-identity dumps are not rerun (nothing they cover changes).

## VM runs

None.

## Constraints and risks

- Constraint 1: no file under `src/` changes, so every FNO output is unchanged by
  construction.
- The mirror module is test code, not the library's 2.0.0 FNO (CM1.3–CM1.5); it shows the
  harness end to end and that `SpectralConv` and Burn's `Conv1d`/`gelu` agree with 2.0.0
  in 1D. It does not set any feature-matrix row to **have**.
- Risk: a forward detail of 2.0.0 misread (grid points, GELU form, block order). Burn's
  `gelu` is the erf form, as torch's `F.gelu`; the rest is read from 2.0.0's source above.

## Pending runs

None.

## Open questions

None.

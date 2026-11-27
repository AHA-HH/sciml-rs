# CM3.6c Periodic factor refresh (opt-in)

Started 2026-11-27 on `operators/cm3-6c` (the author asked for it that day, after #185).
Roadmap: "CM3.6c Periodic factor refresh (opt-in). 2.0.0 computes GaLore's Tucker factors
once and never refreshes them, though its design intends a refresh every k steps; an option
for that refresh, off by default, so the default stays 2.0.0's behaviour and its parity (the
author, 2026-11-27, CM3.6a PR question). *Done when:* a test that the factors change every k
steps when on and never when off, and the CM3.6a parity tests unchanged. *Needs:* CM3.6a."
Proposal: §10; §1.4, §2, §6.4. D1 (2.0.0 the reference), D3.

## Goal

An opt-in refresh of `TensorGaLoreProjector`'s Tucker factors at every `iter` that is a
multiple of `update_proj_gap` (GaLore's intended schedule), with the default unchanged.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)

- "a test that the factors change every k steps when on and never when off": with
  `update_proj_gap` k = 2 and a new gradient at every `iter` 0..=4, the factors with the
  refresh on equal those of `iter` 0 at `iter` 1, change at 2 (equal to `tucker_factors` of
  that step's gradient), stay at 3, change at 4; with it off they never change. Real and
  complex.
- "the CM3.6a parity tests unchanged": `tests/parity_galore*.rs` and
  `tests/parity_galore_common/` not edited and passing (CM3.6b's complex ones too).
- Proposal §1.4, constraint 1: every existing FNO output bit for bit identical. No FNO code
  path is touched (`optim/galore.rs` and its unit tests only).

## Read first

- `src/neural_operators/optim/galore.rs` (`TensorGaLoreConfig`, `TensorGaLoreProjector::project`,
  the module documentation's first bullet), `optim/galore/tests.rs` (`factors_are_not_refreshed`).
- `docs/operators/tasks/CM3.6a.md` (2.0.0's behaviour: factors computed once).

## Changes

- `optim/galore.rs` only:
  - `TensorGaLoreProjector::with_refresh(self, refresh: bool) -> Self` (builder, default off).
    The option lives on the projector, not as a public field of `TensorGaLoreConfig`, so
    every struct literal of the config, the parity harness's included, stays as it is.
  - `project`: with the refresh on, recompute the factors from `grad` (fresh HOOI, SVD
    initialisation, as the first call) whenever `iter` is a multiple of `update_proj_gap`;
    otherwise reuse them. Off: today's code path, unchanged.
  - The module documentation's first bullet names the option.
- Not offered: 2.0.0's `warm_restart` (start the HOOI from the previous factors) and
  `tucker_n_iter_max`; they would change the refreshed factors' values with no reference to
  test them against. A later task if a training arm wants them.
- Conventions: none.

## Tests that prove it

- `factors_refresh_every_gap_when_on` (real) and `complex_factors_refresh_every_gap_when_on`:
  the schedule above; "equal" is bit identity (`max_abs` of the difference 0), "change" a
  difference above 1e-3 and bit identity with `tucker_factors` of the step's gradient.
- `factors_are_not_refreshed` (existing) and `complex_factors_are_not_refreshed`: the same
  0..=4 schedule at gap 2 on a projector from `init()` alone, so "off" is the default.
- Shared by the four: `check_refresh_schedule`. Shown failing for the right reason: with the
  refresh disabled in `project`, the two "on" tests fail (first at the bit identity with `tucker_factors` of iter 2's gradient).
- The parity tests unedited: `cargo test --test parity_galore` and `--test parity_galore_f64`.

## Checks

`/verify` (fmt, clippy default / `--examples` / `--all-targets` / `--features datagen
--all-targets`, `cargo test`, `cargo test --features datagen`, `cargo doc --no-deps`), with
`CARGO_TARGET_DIR=~/sciml-rs-target/CM3.6c`. `--features metal`: not needed (no new device
code; the SVDs run on the host).

## VM runs

None.

## Constraints and risks

- Constraint 1: FNO untouched. The default projector's path is the same code; the parity
  tests guard 2.0.0's behaviour.
- Scope: `optim/galore.rs` and its tests; nothing in `training/` or `models/`.

## Pending runs

None.

## Open questions

None.

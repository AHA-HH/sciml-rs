# T3: a resolution-aware Poisson loader

Phase 5, task 3. Branch `phase5/T3-loader-s`. Depends on T2 (`Transfers` for any s and the
restriction enum).

## Read first
- `src/neural_operators/data/loaders/poisson/mod.rs` (`PoissonConfig`, `s()`) and
  `uniform.rs` (`load_poisson_uniform`, `load_poisson_uniform_test`, their tests).
- `chebyshev/transfer.rs` after T2.
- `examples/train/poisson.rs`, `examples/predict/poisson.rs`,
  `examples/predict/poisson_resolution.rs`, `examples/predict/poisson_plot.rs` (every
  reader of `data_cfg.json`).
- Design §13.7 step 3; CONVENTIONS §10 (normalisers fitted on the train split).

## Goal
Load one fine-grid dataset at any learning size s, so the same forcings and the same test
set appear at every s with no regeneration.

## Deliverables
1. **`PoissonConfig`** gains the learning size `s` and the restriction, both with
   `#[config(default = …)]` so every existing `data_cfg.json` still loads and means
   s = n − 1 with unfiltered T_cu. `s()` returns the configured s (the default when
   unset). Document the fields with CONVENTIONS §12 citations.
2. **`load_poisson_uniform` and `load_poisson_uniform_test`** build `Transfers` from the
   config instead of `Transfers::new(config.n)`. Normalisers are fitted on the restricted
   train split at that s.
3. **`train_poisson -- [<padding>] [<n>] [<s>]`:** s defaults to n − 1; modes follow D2,
   m = min(12, s/2) per axis, computed from s (no constant edit per run). The run name
   carries s when it differs from n − 1 (e.g. `poisson_fno_n129_s32_p0`).
4. The predict examples read s and the restriction from the run's `data_cfg.json`. They
   still evaluate only s = n − 1 runs correctly until T6; at s ≠ n − 1 they fail with a
   clear message rather than a panic deep in `poisson_errors`.
5. `CLAUDE.md`: the `train_poisson` command line and the Layout line for
   `data/loaders/poisson/`.

## Tests
- `poisson_output_is_pinned` and every existing loader test unchanged.
- An old-format `data_cfg.json` (no s, no restriction) deserialises to s = n − 1, T_cu.
- Shapes at s ≠ n − 1: inputs `[N, s, s, 1]`, targets `[N, s, s]`.
- One synthetic sample loaded at two s values equals the T2 restriction of the same
  field (consistency across s).

## Acceptance
- A stage 2 load with no s gives tensors bit-identical to `main`.
- A short stage 3 run at s = 32 starts on flex (a few epochs; not a result).
- The T3 Results row filled in.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`.

## Do not
- Change `PADDING` in code (still 4; D3's p = 0 is passed as the argument in T5) or the
  model.
- Add timing or metadata (T4) or the new errors (T6).

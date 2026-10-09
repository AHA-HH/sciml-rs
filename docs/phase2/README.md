# Phase 2: datasets and transfers

As of 2026-10-08. Design: `docs/design/2d-chebyshev-poisson-fno.md` (signed off
2026-10-05, amended 2026-10-08; decisions in its §12). `docs/CONVENTIONS.md` wins on any
conflict.

Phase 2 turns the verified solver into data the FNO can train on:
- the two grid transfers of CONVENTIONS §12: Chebyshev → uniform (barycentric) and
  uniform → Chebyshev (Floater–Hormann, d = 2);
- the GRF forcing sampler of design §3.3 and the `generate_poisson` example, which writes
  `.npz` datasets labelled by solver A, with a per-sample label check against the exact
  sine-series solution;
- a Poisson loader that reads those files, transfers them to the uniform grid and hands
  the FNO `[N, s, s, 1]` inputs and `[N, s, s]` targets.

Nothing in this phase changes the model or the training code. Datasets for stages 1 and 2
(n = 33, 65) are generated locally. The stage 3–4 sets and the K = 32 evaluation sets are
generated wherever Phase 4 needs them.

## Read first

- Design §3.3 (forcing family), §5 (datasets), §6.2 (transfers), §7 (error 2), §11
  (tests), §12 decisions 5, 8, 10 and 13.
- `docs/CONVENTIONS.md` §10 (normalisation) and §12 (grids, transfers, norms, the
  planned test names).
- `spikes/transfers/REPORT.md` (Phase 0 T3) and `spikes/transfers/src/{grids,interp,grf}.rs`.
- `CLAUDE.md`: Working agreement, Checks.

## Inputs from Phases 0 and 1

| Item | Value | Source |
| --- | --- | --- |
| FH degree and uniform size | d = 2; s = n − 1 (32/64/128/256); s = n is within 5% and not adopted | Phase 0 T3, decisions 5 and 8 |
| Round trip T_uc(T_cu u) − u on GRF u, d = 2 | max 3.6e-5 (n = 65), 4.4e-6 (129), 4.8e-7 (257); 3.2e-4 at n = 33 (stage 1, exempt) | Phase 0 T3 |
| T_cu alone | ≤ 1.0e-15 on sin · sin and (1 − x²)(1 − y²)e^(x + 2y); GRF u ≤ 2.0e-8 for n ≥ 65 | Phase 0 T3, table 1 |
| Λ(T_uc), 1D, d = 2 | 3.32 / 3.91 / 4.38 / 4.87 at n = 33 / 65 / 129 / 257 | Phase 0 T3, table 4 |
| Transfer cost at n = 257 | build 0.10 ms per matrix; apply about 1.4 ms per field | Phase 0 T3, table 6 |
| GRF tail ε_K | 1.86e-2 / 4.92e-3 / 1.26e-3 / 3.19e-4 for K = 16 / 32 / 64 / 128; tolerance 5e-3 (stages 2–4) | Phase 0 T3, decision 8 |
| GRF normalisation | empirical mean square 0.983 ± 0.013 over 1000 samples, at every K | Phase 0 T3 |
| A on GRF against the exact series | ≤ 7.4e-10 relative CC-L² at n = 65, 1.9e-12 at n ≥ 129, 2.5e-7 at n = 33 | Phase 0 T4, design §4.3 |
| Solver A | `pde::poisson::CollocationSolver::{new, solve_full}`; floor ≤ 1e-12 from n = 33 on manufactured solutions | Phase 1 T2 |
| Chebyshev toolkit | `chebyshev::{nodes, clenshaw_curtis, l2_norm, rel_l2_error}` | Phase 1 T1 |
| Test budget | the chebyshev suite took 109.5 s in debug at the end of Phase 1 | Phase 1 T3 |

## Planning decisions

Recorded in design §12, decision 13.

- **The transfers and the loader build without `chebyshev`.** They need only ndarray and
  closed-form nodes, so the Phase 3 and 4 training runs (including HPC) need no FFTW or
  BLAS. The GRF sampler and the generator use solver A and stay gated, so generating on
  HPC needs those libraries there.
- **`rand_chacha` + `rand_distr` + `serde_json`, pinned to exact versions.** `ChaCha8Rng`
  has a documented, portable stream, so a seed gives the same field on every machine.
  That matters because §5.3 regenerates sets on HPC and `Cargo.lock` is git-ignored.
- **Error 2 goes in the generator's sidecar.** Each dataset carries its own transfer
  round-trip error (max and mean per split), next to the label check.

## Tasks

| Task | Brief | Delivers | Depends on |
| --- | --- | --- | --- |
| T1 | [T1-transfers.md](T1-transfers.md) | `chebyshev::transfer` (barycentric, Floater–Hormann), ungated, with oracle tests | Phase 1 T1 |
| T2 | [T2-grf-generator.md](T2-grf-generator.md) | `pde::poisson::grf`, the dataset generator and the `generate_poisson` example; label check and error 2 in the sidecar; stage 1–2 datasets | T1, Phase 1 T2 |
| T3 | [T3-poisson-loader.md](T3-poisson-loader.md) | `data::loaders::poisson` (design §5.2), ungated | T1, T2 |

Task numbers are execution order (design §10). One task per branch and PR:
`phase2/T<k>-<name>`. Each task is worked from its brief alone, plus the files the brief
lists under "Read first".

The spike code (`spikes/transfers/src/{grids,interp,grf}.rs`) is the reference for the
algorithms. It is rewritten, documented and tested, not copied as-is. Its RNG (`StdRng`)
is replaced, so the spike's samples are not reproduced bit for bit.

## Module layout

- `src/neural_operators/chebyshev/` is registered in `neural_operators/mod.rs` without a
  feature gate (T1). Inside it:
  - `points`, `differentiation`, `quadrature`, `norm` and their re-exports stay behind
    `#[cfg(feature = "chebyshev")]`;
  - `transfer.rs` is ungated.
- `src/neural_operators/pde/poisson/grf.rs` and the generation code (T2): behind
  `chebyshev`, like the rest of `pde`.
- `examples/generate_poisson.rs` (T2): `required-features = ["chebyshev"]` in
  `Cargo.toml`.
- `src/neural_operators/data/loaders/poisson.rs` (T3): ungated.
- Every new module is registered in its parent `mod.rs`. The task that adds it adds one
  line to the Layout section of `CLAUDE.md`.
- The public API takes and returns ndarray types; rlst's `DynArray` stays internal.

## Results

Filled in as tasks merge. Phase 2 is done when every row has a value.

| Item | Value | Source |
| --- | --- | --- |
| T_cu on analytic functions, per n (relative, uniform grid) | trapezoidal relative L², s = n − 1: sin(πx) sin(πy) 4.1e-16 / 4.5e-16 / 5.7e-16, (1 − x²)(1 − y²)e^(x + 2y) 3.7e-16 / 4.3e-16 / 5.6e-16 at n = 33 / 65 / 129 | T1 |
| FH d = 2: observed order on sin(πx) sin(πy), n = 33 → 65 → 129; round trip against the Phase 0 table | CC relative L² 1.2e-4 → 1.2e-5 → 1.5e-6, orders 3.32 and 3.05; round trip at n = 65: 1.23e-5 (Phase 0: 1.2e-5) | T1 |
| Λ(T_uc), d = 2, per n | 3.32 / 3.91 / 4.38 / 4.87 at n = 33 / 65 / 129 / 257 (as Phase 0) | T1 |
| A against the exact series on GRF forcings, n = 33, 65, 129 (relative CC-L², max over samples) | seeds 0–4, K(n): 3.9e-7 / 4.8e-10 / 5.9e-13 (bound 1e-8 for n ≥ 65); over the generated splits, 2.2e-6 at n = 33 and 2.2e-9 at n = 65 (1000 train samples) | T2 |
| GRF checks: mean square (N, value ± s.e.), point covariance against the formula, ε_K | mean square at K = 32, N = 1000: 0.991 ± 0.013; covariance at n = 17, K = 8, N = 2000, three node pairs: 1.49, 0.96, 0.67 s.e. from the formula (bound 5); ε_K 1.86e-2 / 4.92e-3 / 1.26e-3 / 3.20e-4 at K = 16 / 32 / 64 / 128, S_∞ recomputed at m = 1000 to 1.0e-6 relative | T2 |
| Stage 1 (n = 33, K = 16) and stage 2 (n = 65, K = 32) datasets: generation time, file sizes, label check max / mean per split | release, Apple M2 Pro; train / test. n = 33: 0.09 s / 0.02 s, 17.4 MB / 3.5 MB, label 2.2e-6 / 3.3e-7 and 1.3e-6 / 3.2e-7 (recorded, not enforced). n = 65: 0.25 s / 0.06 s, 67.6 MB / 13.5 MB, label 2.2e-9 / 4.0e-10 and 1.5e-9 / 3.8e-10 (enforced at 1e-8) | T2 |
| Error 2 (round trip on u, d = 2) per split, stages 1 and 2: max / mean | n = 33: train 8.1e-4 / 1.8e-4, test 7.0e-4 / 1.8e-4. n = 65: train 1.04e-4 / 2.1e-5, test 7.8e-5 / 2.0e-5 | T2 |
| Load time of the stage 2 dataset (release), and the loaded shapes | release, Apple M2 Pro, `T` = f32, read + T_cu + normalise: 0.10 s (0.13 s first run, 3 runs). Train inputs [1000, 64, 64, 1], targets [1000, 64, 64]; test inputs [200, 64, 64, 1], targets [200, 64, 64] (`poisson::tests::stage2_dataset_loads`, ignored by default) | T3 |
| Wall time of the new tests under debug `cargo test --features chebyshev` and under `cargo test` | T1 (`chebyshev::transfer::tests`): 0.15 s, 20 tests, with the feature; 0.10 s, 15 tests, without. T2 (`pde::poisson::{grf, dataset}::tests`, `tests::convention_version_matches_conventions_md`): 3.1 s, 12 tests, with the feature (whole suite 109.5 s, unchanged); under 0.01 s, 1 test, without. T3 (`data::loaders::poisson::tests`): 0.01 s, 9 tests (+1 ignored), with the feature (whole suite 108.8 s); 0.02 s, 9 tests (+1 ignored), without | T1–T3 |

## Exit checklist

- [ ] T1, T2 and T3 merged.
- [ ] The design §11 rows for Chebyshev → uniform, uniform → Chebyshev (FH), the
      collocation solver on GRF, the GRF sampler, the `generate_poisson` label check and
      the loader pass: the ungated ones under `cargo test` and the gated ones under
      `cargo test --features chebyshev`, locally on macOS and in both CI jobs.
- [ ] Stage 1 and 2 datasets (n = 33, 65; 1000 train / 200 test) generated locally, with
      label checks within 1e-8 at n = 65.
- [ ] Error 2 of design §7 measured on the stage 1 and 2 test splits and recorded above.
- [ ] Every row of Results filled in.
- [ ] Any result that contradicts the design is recorded in the design's §12 under
      "Recorded decisions", dated, before Phase 3 is planned.
- [ ] The "Current phase" bullet in `CLAUDE.md` points to `docs/phase3/README.md`.

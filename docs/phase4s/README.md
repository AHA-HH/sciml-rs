# Phase 4s: repository health (side phase)

As of 2026-10-15; planned, not started (decision 25). Design: `docs/design/2d-chebyshev-poisson-fno.md`
(§10 Phase 4s, §12 decision 25). Source: the whole-repository review at `c63eb32`,
`docs/code-quality/thermo-nuclear-repo-review-2026-10-15.md` (cited below as "review"; its
finding IDs, e.g. T2-3 or S1, are quoted per task). `docs/CONVENTIONS.md` wins on any conflict.

Phase 4s is a side phase between Phase 4 and Phases 5 and 6. It answers no research
question. It makes the repository easier to change before Phase 5 (checkpointing,
multi-GPU) and Phase 6 (a new Chebyshev-native model) build on it. The review graded the
repository C:
- The core (FNO, spectral layer, training loop, Chebyshev numerics) is clean and acyclic.
- Three patterns at the edges cost more than they protect:
  - the run directory has no owner (S1);
  - superseded implementations were kept beside their replacements (S2);
  - per-PDE code is copied across trainers, loaders and examples (S3).
- The `chebyshev` feature splits a module that is mostly ungated (S4).
- The repository has drifted from its own rules (S5).

**Behaviour is preserved.** Every task keeps the model's outputs, the training numbers,
the five errors and the dataset format unchanged, except the four tasks marked **⚑**. Each
⚑ task asks the author for a decision when it starts and records the answer in design §12
before any code changes.

## Read first

- The review, sections 1 (verdict), 5 (systemic issues) and 7 (roadmap). The tasks below
  are its roadmap renumbered into execution order.
- `CLAUDE.md`: Working agreement, Working rules, Checks, Layout.
- `docs/CONVENTIONS.md` §2 (grids), §10 (normalisation), §11 (loss), §12 (Chebyshev);
  §9 for when a change bumps `CONVENTION_VERSION`.
- Design §12 decisions 12 (RLST/FFTW toolkit), 13, 15 (ungated CC weights) and 24 (Phase
  5/6 split).

## Planning decisions

Recorded in design §12, decision 25.

- **A side phase, numbered 4s.** It sits between Phase 4 and Phases 5 and 6 and keeps their
  numbers. Its tasks are T1–T24 (plus an optional T25), in execution order (design §10).
- **It goes before Phases 5 and 6, but blocks them only where the code overlaps:**
  - T10 (`RunDir`) and T12 (epoch observer) come before Phase 5 T2 (checkpoint/resume),
    which would otherwise be wedged into the training loop.
  - T16 (one ungated Chebyshev module) comes before Phase 6 T3 (θ-map), which adds code
    under `chebyshev/`.
  - Phase 5 T1 (profiling) and Phase 6 T1–T2 (baseline padding and seeds) do not depend on
    this phase. They can start once the tasks they touch are merged.
- **The Learner example stays (review T2-1, revised).** `train_burgers_learner` is the
  showcase of Burn's `Learner` (TUI, metric logging, summary) and the template for future
  use. T2 removes its one duplicated loss definition and makes its runs readable. It is
  documented as a showcase; `train_epoch` is the canonical loop.
- **Old runs and datasets stay readable.**
  - `RunDir` (T10) reads every `runs/` directory written since Phase 3.
  - No task changes the `.npz` / JSON dataset format.
  - T14 changes generated node coordinates by about 1 ulp, and T15 changes labels at
    round-off; both are ⚑ and decide whether to regenerate.
- **No `CONVENTION_VERSION` bump is expected.**
  - T5 edits CONVENTIONS §2 prose only.
  - T17 changes §10, which is outside the bumping set (§1–§6, §12). It is still proposed in
    its PR description as CLAUDE.md requires.
  - A task that finds it must change §1–§6 or §12 stops and asks.
- **Each task's PR states its line delta** (prod / test) and which checks ran, so the phase
  can show what it removed.

## Checks for every task

The five commands of the `run-tests` and `style-checks` CI jobs (CLAUDE.md "Checks"), plus
`cargo clippy --all-targets -- -D warnings`:

```sh
cargo fmt -- --check
cargo clippy --no-deps -- -D warnings
cargo clippy --no-deps --examples -- -D warnings
cargo test
cargo doc --no-deps
cargo clippy --all-targets -- -D warnings
```

Tasks that touch `chebyshev/` or `pde/` also run `cargo test --features datagen` and
`cargo clippy --all-targets --features datagen -- -D warnings` (the feature was named
`chebyshev` before T16).

- **The test count must not drop** unless the task deletes tests on purpose. Each PR compares
  the "running N tests" lines before and after, and lists any deliberate removals.
- **Example tasks also re-run one example on an existing run** and confirm it prints the
  same numbers as before. The task row names which example.
- **Unrun checks are reported as unrun.** That includes Metal and anything that needs a
  dataset or a real terminal.

## Tasks

Task numbers are execution order. "∥" in Depends on marks tasks that can run in parallel
with their neighbours (see "Parallel work" below). One task per branch and PR:
`phase4s/T<k>-<name>`. Each task's brief is `T<k>-<name>.md` in this directory (linked
from the task number). All briefs were written on 2026-10-16 from each task's row and the
review findings it cites, with the locations re-checked against `6eb3212`. A brief with a
**Refresh at start** block depends on earlier PRs: re-check what it lists against `main`
before starting. A ⚑ brief sets out the options and a recommendation; the author decides
when the task starts.

### Group 1: setup

| Task | Delivers | Review | Depends on |
| --- | --- | --- | --- |
| [T1](T1-ci-manifest-hygiene.md) | CI and manifest hygiene: delete `run-examples.yml` and the commented apt blocks in `run-tests.yml`; fix `Cargo.toml`'s `homepage`/`repository` (GitHub), the invalid category, the empty dev-dependency tables and the truncated feature comment; drop rocm/vulkan from `lib.rs`'s `compile_error!`; make CLAUDE.md's "Checks" list match the three CI jobs | S5 | – |

### Group 2: independent cleanups (start together)

| Task | Delivers | Review | Depends on |
| --- | --- | --- | --- |
| [T2](T2-learner-showcase.md) | Learner showcase tightened. One `pub fn` in `trainer.rs` for forward + `flatten_pair` + summed loss, called by `train_epoch` and `learner_forward`. `burgers_learner` writes a timestamped run under `CARGO_MANIFEST_DIR` with `data_cfg.json`, which `predict_burgers` can evaluate. Showcase line added to the module doc and CLAUDE.md. Epochs comment fixed | T2-1, A1 | T1 ∥ T3–T8 |
| [T3](T3-train-fno.md) | One `train_fno::<R, RM1, T>(…, y_norm: Option<&UnitGaussianNormalizer>, …)` in `trainer.rs`. `training/trainers/` deleted, and `train_poisson_smoke` moved beside the new function | T2-2, S3, A2 | T1 ∥ |
| [T4](T4-dead-data-generality.md) | Unused data generality removed: `GaussianNormalizer`, `RangeNormalizer`, the `Normalizer` trait (→ inherent methods; `mean`/`std`/`eps` accessors), `NpyFileReader`, `FieldReader` (→ inherent `read`), `BaseDatasetConfig`/`HasBaseConfig`. `NpzFileReader` opens once, and reports a wrong dtype as a parse error rather than `FieldNotFound` | T2-6, T2-7, T2-14, A3 | T1 ∥ |
| [T5](T5-delete-grids.md) | `data/grids.rs` deleted; the f64 oracle the grid tests need moves into `fno.rs`'s tests. CONVENTIONS §2's last bullet updated (prose only) | T2-5, A4 | T1 ∥ |
| [T6](T6-sparse-test-only.md) | `pde/poisson/sparse.rs` becomes `#[cfg(test)] mod sparse;`; `tests/rlst_smoke.rs` deleted | T2-9, T2-13, A5 | T1 ∥ |
| [T7](T7-gpu-parity-split.md) | `tests/metal_diag.rs` split. The asserting regression tests (`repro_*`, `replay_2d_poisson`) go into `tests/gpu_parity.rs`, with the `*_flex` twins outside the GPU gate so CI runs them. The exploratory tests move to `spikes/metal_diag/`. `upstream/burn/*` citations updated. States whether Metal was run | T2-12, A6 | T1 ∥ |
| [T8](T8-chebyshev-shared-pieces.md) | One owner for the shared Chebyshev pieces, all ungated. `TRANSFER_DEGREE` and `Transfers::new(n)` (`t_cu`, `t_uc`, `to_uniform`, `to_cheb`, `round_trip_error`) in `chebyshev::transfer`. `cgl_nodes` public as `nodes_closed_form`, or similar until T14 settles the name. `cc_norm_sq` in `clencurt`, one `ErrorStats`. The generator, `poisson_errors`, the Poisson loader and `poisson_plot` use them; `poisson_plot`'s copy of the nodes is deleted | S4, T6-3, T6-4, C1 | T1 ∥ |

### Group 3: run and training track

| Task | Delivers | Review | Depends on |
| --- | --- | --- | --- |
| [T9](T9-loader-common-tail.md) | Shared loader tail: `loaders/common.rs` with `Normalizers { x, y }` and `normalize_split_pair`, used by Darcy and Poisson (Burgers reuses the channel step). Fallible reshapes become `insert_axis`, Burgers' no-op reshape goes, and the library `println!`s go. **Test added first:** Darcy loader output (shapes, a checksum) on a small synthetic `.mat` is unchanged | T2-10, T4-7, S3, B3 | T4 |
| [T10](T10-run-dir.md) | Top-level `runs` module with `RunDir` (create, save configs / metrics / normalizers / weights; open, load model / configs / normalizers / metrics). It absorbs `metrics/io.rs`, `metrics/plots.rs`, `run::load_fno` and `EpochMetrics`. The root is a parameter; configs are `&impl Config`. Every train and predict example migrates, `burgers_learner` included. **Test added first:** a save → open → `load_fno` round trip with bit-identical outputs. `predict_poisson` re-run on an existing run prints the same errors | S1, T4-4, B1 | T2, T3, T4, T9 |
| [T11](T11-one-batching-path.md) | One batching path: predict examples use `device_loader`. `OperatorBatcher` becomes a `#[cfg(test)]` oracle; `DataItem` and the `Dataset` impl are deleted; `dataitem.rs` → `host_float.rs`. `predict_burgers` and `predict_darcy` print the same test L² as before | T2-4, B2 | T10 ∥ T12 |
| [T12](T12-epoch-observer.md) | Epoch observer in `training_loop`. The default prints; a `RunDir` observer appends `metrics.csv` each epoch. The CSV written incrementally equals the end-of-run CSV | T3-2, B5 | T10 ∥ T11 |
| [T13](T13-evaluate-predict-host.md) | Public `evaluate::<R, RM1>` (per-sample mean, owns `LpLoss`) and a rank-generic `predict_host<R>`. `burgers_resolutions` and `burgers_superresolution` merge into one `burgers_resolution` taking rates from argv. `darcy_plot` uses `predict_host`. Each example reproduces its previous numbers | T4-1, T4-2, T2-11, B4 | T11 |

### Group 4: Chebyshev track (runs beside Group 3)

| Task | Delivers | Review | Depends on |
| --- | --- | --- | --- |
| [T14](T14-generator-ungated-stack.md) | ⚑ **Decision: regenerate datasets or keep them** (nodes move by about 1 ulp, within the loader's `NODE_TOL`). The generator uses the ungated nodes, `clencurt` and `cc_rel_l2_error`. The gated `points`/`quadrature`/`norm` become `#[cfg(all(test, feature = "chebyshev"))]` oracles. `label_check` takes no weights, and `Split.y` goes | T2-8, T4-9, C2 | T6, T8 ∥ Group 3 |
| [T15](T15-closed-form-diff.md) | ⚑ **Decision: amend decision 12.** Closed-form `diff_matrix` / `diff2_matrix` (Trefethen `cheb`, negative-sum diagonal, D² = D·D). Drops `rlst/fftw`, `rlst/fftw_system`, `unit_coeffs` and the system libfftw3 (GPL), with the CI apt step. Labels change at round-off: regenerate stage 1 and compare label check and error 2 | S4, C3 | T14 |
| [T16](T16-ungate-chebyshev.md) | `chebyshev/` fully ungated. The feature is renamed for what it enables (e.g. `datagen`) and gates only `pde/`. `lib.rs`, CLAUDE.md and CI follow. If T15 is declined, only `differentiation.rs` stays gated | S4, T3-3, C4 | T15 |

### Group 5: contracts that need a decision

| Task | Delivers | Review | Depends on |
| --- | --- | --- | --- |
| [T17](T17-y-train-physical.md) | ⚑ **Decision: CONVENTIONS §10.** `y_train` is stored in physical units and only the prediction is decoded. `Postprocess`, `identity` and the closure pair are deleted; the hook is one `Option<FlatDecoder>`. f32 round-off only; proposed in the PR description | T3-1, D1 | T13 |
| [T18](T18-x-normalizer.md) | ⚑ **Decision: `x_normalizer.json`.** Either predict reads it (and stops re-reading and re-transferring the train split), or training stops writing it. `poisson_resolution` keeps refitting on purpose | D2 | T9, T10 ∥ T17 |
| [T19](T19-sidecar-and-eval.md) | A typed `SidecarRecord` shared by generator and loader (byte-compatible; **test first:** parse an existing sidecar fixture). `PoissonErrorsRecord` and the worst-sample / cross-check / boundary verdict move into `metrics::poisson::evaluate`, which returns a `PoissonEvaluation` (the record plus the two cross-checks) with `Display`; `predict_poisson` calls it, and `poisson_resolution` keeps its own per-n table and runs `check_error3` at the trained n. The `Config`-derive allows are scoped per file, and `lib.rs:48`'s blanket allow is deleted | T4-5, T6-2, S5, D3 | T8, T10 ∥ T17, T18 |

### Group 6: decomposition and polish

| Task | Delivers | Review | Depends on |
| --- | --- | --- | --- |
| [T20](T20-fno-tests-split.md) | `models/fno.rs` tests split into `models/fno/tests/{reference,padding,grid,checkpoint}.rs`. `reference_forward` deleted (≡ `Pad::Off`); `to_channels_first` / `to_channels_last` helpers in production and tests. Test count unchanged | T5-1, T7-5, E1 | T5 ∥ T21 |
| [T21](T21-spectral-conv-polish.md) | `SpectralConv`: one `with_bc` fixed-rank helper for the five sites, `flat_modes` computed locally, and the Gauss 3-matmul formula in the doc | T7-4, E2 | – ∥ T20 |
| [T22](T22-poisson-loader-split.md) | `loaders/poisson.rs` split into `poisson/{set, cheb, uniform}.rs`. `enum PoissonField { F, U }` replaces the string selector and the wrapper. The dataset catalogue gets its own module, which the generator imports; `stage()` returns `Option`. `BurgersFiles` / `DarcyFiles` replace the hard-coded `.mat` paths | T4-6, T5-3, T6-1, E3 | T9, T19 |
| [T23](T23-take-strided.md) | One rank-generic `take_strided`; `transforms/subsample.rs` and `data/split.rs` deleted | T2-15, E4 | T9 ∥ T22 |
| [T24](T24-docs-pass.md) | Documentation pass, last because it touches many files:<br>• the 19 `REVIEW.md` citations → CONVENTIONS § / PR numbers<br>• rustdoc for every surviving public item<br>• `lib.rs` crate docs and `models/mod.rs`<br>• `fno.rs` convention prose → citations<br>• `FNOConfig::init`'s `out_channels` assert removed (the trainer owns it)<br>• `LpLoss` `d` / `forward` cleanup<br>• CLAUDE.md Layout and Commands updated for the phase | S5, T4-3, T4-8, T7-1–3, E5 | T1–T23 |
| [T25](T25-collocation-ndarray.md) | *Optional.* `collocation.rs` on ndarray after `eig` (deletes `to_dyn`, `from_dyn`, `transpose`), and the A·F·Bᵀ sites routed through `chebyshev::apply`. **Time n = 257 generation first**; skip it if it is slower than now | T6-5 | T16 |

## Parallel work

There are three independent tracks once T1 is in. Each track can be worked by a separate
session in its own worktree.

```
T1 ─┬─ Run/training track:  T2, T3, T4 ─▶ T9 ─▶ T10 ─▶ T11 ─▶ T13 ─▶ T17
    │                                           ├─▶ T12              └─▶ T18, T19
    ├─ Chebyshev track:     T8 (+T6) ─▶ T14 ─▶ T15 ─▶ T16 ─▶ (T25)
    └─ Standalone:          T5 ─▶ T20;  T6;  T7;  T21
```

**Safe to run fully in parallel** (no shared files): T5, T6, T7 and T21 with anything; T8
with T2, T3 and T5–T7.

**Parallel with a rebase:** these pairs share files in different functions, so the second
to merge rebases.

| Pair | Shared files |
| --- | --- |
| T2 and T3 | both edit `trainer.rs` (a new loss step against a new `train_fno`) and `examples/train/burgers*.rs` |
| T3 and T4 | train example imports |
| T4 and T8 | imports in `loaders/poisson.rs`, `pde/poisson/dataset.rs` (a T4 test), `examples/predict/{poisson,poisson_plot}.rs` |
| T1, T2, T6 and T8 | `CLAUDE.md` Layout and Commands lines |
| T5, T11 and T23 | one `pub mod` line each in `data/mod.rs` |
| T11 and T12 | `trainer.rs` |
| T17, T18 and T19 | after T13; small overlaps in examples |

**Must be sequential:** T9 → T10 → T11 → T13 → T17.
- T10 needs T9's `Normalizers` type.
- T11 and T13 edit the examples T10 migrates.
- T17 needs one evaluation entry point.

The Chebyshev chain T14 → T15 → T16 is sequential because each changes what the next
removes.

**Never side by side:**
- T4 and T14 both edit `pde/poisson/dataset.rs`. T4 merges first in the normal order.
- T10 and any other task in the run/training track: T10 edits every train and predict
  example.

**How many sessions:** three at most, one per track, each in its own worktree
(`phase4s/T<k>-<name>`). The code allows more, but the machine doesn't: each worktree
builds its own Burn `target/` (several GB), and parallel `cargo test` runs compete for CPU.
A shared `CARGO_TARGET_DIR` saves disk, but cargo's build lock then serialises the
sessions.

**Suggested schedule** for one person working one or two tasks at a time:

| Week | Tasks | Notes |
| --- | --- | --- |
| 1 | T1, then T3, T4, T5 and T8 together | T6, T7 and T21 fill the gaps |
| 2 | T2, T9, then T10 | T10 is the largest task; give it a focused stretch |
| 3 | T11 and T12 together, then T13; in parallel, T14 → T15 → T16 | T14 and T15 need decisions |
| 4 | T17–T19 (decisions), then T20, T22, T23 | T24 last |

## Results

Filled in as tasks merge. Phase 4s is done when every row has a value.

| Item | Value | Source |
| --- | --- | --- |
| Line delta per task (prod / test) and the test count before / after | T9: +146 / −165 refactor, +112 test (pins); lib 202 → 205<br>T10: +633 / −408 (about 140 of the added lines are tests); lib 205 → 207<br>T11: +129 / −168; lib 207 → 206, doctests 9 → 7 (3 deliberate deletions)<br>T12: +193 / −45 (`trainer.rs` about +47 prod / +73 test); lib 206 → 208<br>T13: +208 / −227 (about +35 test), then the review fixes +667 / −684 (`trainer.rs` 1004 → 732, split into `training/{config, step, trainer}`); lib 208 → 209 (one test deleted with its error path)<br>T14: +153 / −109 in `src` and `examples` (about +62 / −23 of it tests; most of the rest is the `nodes` rename and docs); lib 209 → 209, with `chebyshev` 262 → 264 (2 added), then the review fixes +60 / −212 (`chebyshev/norm.rs` deleted with its 6 self-tests; `generator_nodes_within_node_tol_of_rlst` deleted, as `cgl_closed_form_matches_nodes` covers it; `written_split_loads` folded into `files_round_trip`); lib 209 → 209, with `chebyshev` 264 → 256<br>T16: +37 / −33 in `src` and `examples` after the rebase onto T14's review fixes (+3 / −3 of it test `cfg` lines; the rest is the rename and docs); lib 209 → 209, with `datagen` 256 → 256. No test was newly ungated: T14 had already left only `differentiation.rs` and the RLST oracles behind the feature. Then the review fixes +14 / −19 in `src` (`unit_coeffs` and its `rlst` imports moved from `chebyshev/mod.rs` into `differentiation.rs`, so every gate in `mod.rs` is on a `mod` or `pub use` line; docs and comments); lib 209 → 209, with `datagen` 256 → 256<br>T17: +81 / −77 prod, +93 / −55 test in `src` and `examples` (most of the test side is `train_loss_matches_old_asymmetric_path` with the old path inlined); lib 209 → 209 (`train_fno_some_matches_decoder_path`, whose closure-pair oracle no longer exists, replaced by `train_loss_matches_old_asymmetric_path`; `normalizer_asymmetry_matches_darcy` renamed `normalization_matches_darcy`)<br>T18: +243 / −96 prod, +48 / −0 test in `src` and `examples` (most of the prod side is the two test-split loaders with their docs, and the split readers moved out of closures into functions); lib 209 → 210 (`test_loader_with_saved_normalizers_matches_refit` added; the missing-`x_normalizer.json` check added to `metrics_and_normalizers_round_trip`), then the review fixes +36 / −27 prod, +38 / −0 test in `data/loaders/darcy.rs` (the `subsample_rate == 0` guard moved into `read_split`, one `check_grid` naming the offending shapes, `darcy_test_from_fields` split out of `load_darcy_test` so the test split is tested without `.mat` files); lib 210 → 211 (Darcy's `test_loader_with_saved_normalizers_matches_refit` added), then the suggestions taken +8 / −26 (the four predict examples' missing-normaliser panic in one line naming `y_normalizer.json`, `Transfers::n()` instead of `s() + 1`, two doc fixes); lib 211 → 211<br>T19: +405 / −231 prod, +206 / −0 test in `src`, `examples` and `tests/fixtures` (the test side includes the 30-line sidecar fixture and the old untyped `json!` writer kept in `files_round_trip` as the byte oracle); lib 211 → 216, with `datagen` 258 → 263 (`sidecar_fixture_parses_and_round_trips` and four `evaluate` tests added), then the review fixes +103 / −79 prod, +17 / −18 test in `metrics/poisson.rs` and `predict_poisson` (`Error2Check` and `PoissonEvaluation` with the report's `Display` split from `PoissonErrorsRecord`, which keeps only its persisted fields and goes back to `#[derive(Config)]`; `error2_matches` and the three `#[serde(skip)]` fields deleted; the four `evaluate` tests adapted); lib 216 → 216, with `datagen` 263 → 263<br>T20: +9 / −44 (the `fno.rs` tests, 1,290 lines, are now `fno/tests/{mod, reference, padding, grid, checkpoint}.rs`, 1,246 lines; `reference_forward` deleted for `Pad::Off`; `to_channels_first` / `to_channels_last` added in `fno/mod.rs`, +22 / −13 of it prod); lib 216 → 216, `models::fno` 36 → 36, then the review suggestions +98 / −106 test (one shared `grads_of`, `test_config` renamed from `padded_config`, `copy_weights` reused, the two permutation helpers pinned inside `forward_matches_reference_1d`; CLAUDE.md paths); `models::fno` 36 → 36<br>T21: +31 / −22 in `spectral_convolution.rs` (about +7 / −1 of it test: `with_bc_prepends_and_panics_on_wrong_rank` added; one argument dropped from the `complex_multiplication` test call); lib 216 → 217, then the review suggestions +7 / −9 (`complex_multiplication` reads the modes from the weight dims and drops its `modes` parameter; the `with_bc` test moved below `use super::*;`); lib 217 → 217. `flat_modes` field dropped: the saved run `poisson_fno_p4_1791922918` still loads and `predict_poisson` output and `errors.json` are identical to the baseline | T1–T24 |
| `predict_poisson` on `runs/poisson_fno_p4_1791922918`: the five errors unchanged after T10, T13, T19 | Unchanged after T10, T13, T17 and T18 (error 3 mean 1.0686e-2, error 4 mean 1.0688e-2, error 2 mean 2.0335e-5; stdout and `errors.json` byte-identical after T17 and T18). Wall-clock after T18: 30.66, 31.12, 38.74 s against 30.24 s before, so not measurably faster (the whole train + test load is 0.167 s). After T19 (`metrics::poisson::evaluate` and the typed sidecar): stdout and `errors.json` byte-identical to `main`'s (wall-clock 47.92 s, one run, not compared); `poisson_resolution` at n = 65 and 129 prints the same as on the T18 tip | T10, T13, T19 |
| `predict_burgers` / `predict_darcy` test L² unchanged after T10, T11, T13 | Unchanged after T10, T11 and T13: `burgers_fno_1790881218` 0.0020854936, `darcy_fno_1790729160` 0.0095942197. After T17 and T18, `predict_darcy` unchanged (0.0095942197, stdout identical); `predict_burgers` not re-run | T10, T11, T13 |
| `train_burgers_learner` run evaluated by `predict_burgers` | | T2 |
| Decisions taken on T14, T15, T17, T18 (decision 25 or later) | T14: decision 26, keep the datasets<br>T15: decision 27, declined (decision 12 stands; FFTW-backed D and D² kept)<br>T17: decision 28, accepted (`y_train` stored in physical units; CONVENTIONS §10 prose, no `CONVENTION_VERSION` bump)<br>T18: decision 29, read it (predict reads both saved normalisers and loads only the test split; `poisson_resolution` keeps refitting, decision 18) | T14, T15, T17, T18 |
| Datasets regenerated or kept; label check and error 2 if regenerated | T14: kept. All six stored sets (n33_K16, n65_K32, n129_K64, n129_K32, n257_K64, n257_K32, train and test) load with `load_poisson_uniform` and `load_poisson_cheb`. Stage 1 regenerated into a scratch directory against the stored sidecars, relative difference: error 2 max / mean 7.9e-15 / 1.5e-15 (train), 7.8e-15 / 7.7e-16 (test); label check max / mean 1.1e-11 / 1.6e-12 (train), 4.9e-12 / 5.7e-13 (test), an absolute change of at most 2.4e-17 on a check of about 2e-6. Fields: max \|f new − f old\| 9.1e-15, max \|u new − u old\| 3.9e-16 (max \|u\| 0.63)<br>T15: no change (declined), nothing regenerated | T14, T15 |
| `chebyshev` feature: what it gates after the phase, and its new name | Renamed `datagen`, no alias (decision 27). It gates `pde/` (reference solvers, generator), `diff_matrix` / `diff2_matrix` with `unit_coeffs` (FFTW, since T15 was declined), the RLST test oracles (`points`, `quadrature` and the tests using them), the `blas_src` / `lapack_src` extern crates and OS check in `lib.rs`, and `generate_poisson`. The rest of `chebyshev/` builds without features | T16 |
| Repository LOC at `c63eb32` against the phase's last merge | 15,609 (in-scope files, review header) | T24 |

## Exit checklist

- [ ] T1–T24 merged (T25 optional).
- [ ] Every ⚑ decision recorded in design §12 before its code merged.
- [ ] The example reruns in Results reproduce their numbers.
- [ ] No blanket `#[allow]`, no `REVIEW.md` citations, every public item documented.
- [ ] CLAUDE.md Layout, Commands, Checks and "Current phase" match the code.
- [ ] A short re-review (same skill, same scope) records the new grade in this README.

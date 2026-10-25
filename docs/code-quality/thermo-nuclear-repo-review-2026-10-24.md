# Thermo-nuclear repo review: sciml-rs (Phase 4s closing re-review)

| | |
|---|---|
| Repo | sciml-rs (`sciml_rs`), worktree `stingaree`, branch `phase4s/group6` (identical to `origin/main`) |
| Commit | `bbd605e` (Phase 4s T24, #63) |
| Date | 2026-10-24 |
| Baseline | `docs/code-quality/thermo-nuclear-repo-review-2026-10-15.md` at `c63eb32`, grade **C** |
| Scope | The same as the baseline: `src/`, `examples/`, `tests/`, `.github/workflows/`, `Cargo.toml`, `.cargo/config.toml` |
| Excluded | The same as the baseline: `spikes/` (now including `spikes/metal_diag`, a non-default workspace member that CI never builds), `upstream/burn/*`, `docs/` (read for context, not reviewed), `datasets/` and `runs/` (git-ignored), licences |
| Size | 83 `.rs`/`.toml`/`.yml` files, 14,725 lines (baseline: 15,609, so −884, −5.7 %). `src/`: about 5,540 production and 6,980 test lines. `examples/`: 1,347 (was about 1,800). `tests/`: 642 (was about 2,170). No file reaches 1,000 lines; the largest is `layers/spectral_convolution.rs` at 860. |
| Coverage | Full, with the baseline's method. Four parallel read-only slice reviewers read every in-scope file in full: (A) models, layers, utils, losses, training; (B) data; (C) chebyshev, pde, metrics; (D) runs, lib.rs, examples, tests, CI, manifest. The cross-cutting pass (layering, literals, allows, duplicate helpers) was done by hand. Every tier 1–3 finding below was re-opened and confirmed in the code. No builds or tests were run: the review is read-only, and another job held the build lock. |

---

## 1. Health verdict: **B** (baseline: C)

The three systemic patterns that earned the C are gone or reduced to residue:
- **The run directory has an owner.** `runs::RunDir` writes and reads every run file, and the only run-file literals outside `runs/` are the black-box format check in `tests/run_artifacts_without_gnuplot.rs`.
- **Superseded implementations are deleted or demoted.** The host batcher, `grids.rs`, two normalizers, the Npy reader, `FieldReader`, the base-config traits, `SparseSolver`, `rlst_smoke`, the gated `norm.rs` and the 1,674-line `metal_diag.rs` are all gone from production.
- **Per-PDE copy-paste is down to a few spots.** There is one `train_fno`, one loader tail, one `evaluate`, one `predict_host<R>` and one Burgers resolution example.
- **The `chebyshev` feature split (S4) is gone.** Only `differentiation.rs` is gated, as decision 27 requires, and the feature is renamed `datagen`.
- **The hygiene items (S5) are closed.** There is no blanket `allow`, there are no `REVIEW.md` citations, `#![warn(missing_docs)]` is on, and CI and `CLAUDE.md` agree.

The grade is B, not A, because a second ring of the same problems remains:
- **The training run is not written atomically.** `train_fno` appends `metrics.csv` every epoch, but configs, normalizers and weights are written by each example after training. A stopped run cannot be read back.
- **The dataset-format owner is in the wrong place.** The sidecar schema sits in the Poisson loader, so the generator and `metrics` import the loader.
- **Some seams are only half finished.** The epoch observer cannot be reached from `train_fno`. Error 2 is computed twice. The Darcy and Poisson loader entry points still mirror each other.
- **Migration oracles were kept as permanent tests.** The phase moved tests rather than deleting them: `trainer.rs` tests grew from 203 to 427 lines, and the `fno` tests went from 1,263 to 1,251 lines across five files.

None of these issues is in the hot inner code. Each is a contained, local fix, and none slows Phase 5 or 6 as much as S1 would have. A B on the skill's scale means "local issues only". This repository sits at the top of B: one more wave of the size of Phase 4s Group 6 would make it an A−.

---

## 2. Architecture map

**Intended direction.** This is unchanged in spirit from the baseline. `runs` is new, and `metrics` lost its persistence role.

```
utils (fft, backend) ─▶ layers (SpectralConv) ─▶ models (FNO, predict_host<R>)
losses (LpLoss)
chebyshev (f64 ndarray; only differentiation gated) ─▶ data, metrics, pde
data (io, catalogue, loaders, transforms, device_batcher)
metrics (poisson errors + evaluation, stats) ─▶ chebyshev
runs (RunDir) ─▶ models, data::transforms
training (config, step, trainer, learner) ─▶ data, models, losses, runs
pde/poisson [datagen] ─▶ chebyshev, data::catalogue, metrics::stats
examples ─▶ everything
```

**Actual direction.** There are no cycles. `runs → training` exists only in tests, and so does `data → runs`.

| Edge | Status | Note |
|---|---|---|
| `metrics → training`, `metrics → models` (baseline ⚠) | ✓ fixed | `metrics` imports only `chebyshev`, `stats`, `runs::EpochMetrics` and the sidecar type. |
| `CARGO_MANIFEST_DIR` in library code (baseline ✗) | ✓ fixed | Only examples use it; `RunDir::create` takes the root (`runs/mod.rs:86`). |
| `FNOConfig::init` asserts `out_channels == 1` (baseline ✗) | ✓ fixed | Removed in T24; `flatten_pair` owns it. |
| Duplicate `TRANSFER_DEGREE` (baseline ✗) | ✓ fixed | One definition at `chebyshev/transfer.rs:22`. |
| Two modules named `metrics` (baseline ⚠) | ✓ fixed | `EpochMetrics` is in `runs`. |
| Evaluation policy in an example (baseline ⚠) | ✓ fixed, with surplus | `metrics::poisson::evaluate` (`metrics/poisson.rs:298`). See T6-2 below. |
| `pde::poisson::dataset → data::loaders::poisson` (baseline ⚠, via the catalogue) | ⚠ **moved, not removed** | The catalogue moved to `data/catalogue.rs`, but T19 put `SidecarRecord`/`SidecarError2`/`SidecarLabelCheck` into the loader (`loaders/poisson/mod.rs:66-144`), and the generator imports them (`pde/poisson/dataset.rs:32-34`). |
| `metrics::poisson → data::loaders::poisson` | ⚠ new | The same cause (`metrics/poisson.rs:31`). |
| `metrics::poisson → runs::EpochMetrics` | ⚠ new, minor | Used by `check_error3` and `evaluate` (`metrics/poisson.rs:32, 165, 298`), which only need `last.test_l2`. It stops `RunDir` from owning `errors.json` without creating a cycle. |
| `runs → data::loaders::common::Normalizers` | ⚠ minor | `runs/mod.rs:35`: the run owner reaches into the loader tail for a transforms concept. |

---

## 3. Module scorecard

Production and test lines are split at the first `#[cfg(test)]`. Whole-file test modules (`fno/tests/*`, `sparse.rs`, `manufactured.rs`, `points.rs`, `quadrature.rs`, `host_oracle.rs`, `fixtures.rs`) count as test.

| Module | LOC (prod/test) | Largest file | Best hotspot rank | Grade (baseline) | Diagnosis |
|---|---|---|---|---|---|
| `models/` | ~360 / ~1,325 | `fno/tests/mod.rs` 318 | 6 | A− (B+) | The production code is clean, rank-generic and cites CONVENTIONS. The tests were split but not shrunk: `pre_padding_forward` and the "old grid" shims remain. |
| `layers/` + `utils/` | ~450 / ~500 | `spectral_convolution.rs` 860 | 2 | A (B+) | Cohesive. `with_bc` landed, `flat_modes` is no longer stored, and the Gauss doc is correct. |
| `data/` | ~1,710 / ~1,600 | `loaders/darcy.rs` 422 | 3 | B+ (B−) | The dead generality is gone. Left: the sidecar schema is in the loader, the Darcy and Poisson `load_*`/`load_*_test` pairs mirror each other, `from_record` panics on bad input, and two file names are stale. |
| `training/` | ~615 / ~670 | `trainer.rs` 777 (350 prod) | 1 | B+ (B) | One trainer, a shared step, `evaluate`, physical `y_train`. The observer cannot be reached from `train_fno`, `train_built` is a test-only seam, and migration oracles doubled the test volume. |
| `losses/` | 129 / 360 | `data_losses.rs` 489 | 9 | B+ (B) | `forward` is gone and `d` moved to `abs`. One production configuration; the rest is surface specified by CONVENTIONS §11 (the owner's call). |
| `metrics/` | ~510 / ~330 | `poisson.rs` 756 (462 prod) | 5 | B− (C+) | The persistence role is gone and there is one `ErrorStats`. `poisson.rs` now mixes pure error numerics with policy, schema, provenance and the report. |
| `chebyshev/` | ~486 / ~860 | `transfer.rs` 617 | 8 | A− (B) | One stack, principled gating. `cheb_to_uniform`, `uniform_to_cheb`, `apply` and `diff_matrix` are public with no outside callers. |
| `pde/poisson/` | ~740 / ~1,160 | `dataset.rs` 540 | 7 | B (B) | `sparse` and `manufactured` are test-only. `generate`'s n ≥ 3 guard is stale, `SplitSpec` round-trips K through an `Option`, and error 2 is written out again. |
| `runs/` | ~260 / ~195 | `mod.rs` 395 | — | A− (new) | The real owner of its eight files, tested bit-identically. Its errors are `Box<dyn Error>` strings, and it does not own `errors.json` or `prediction*.png`. |
| `examples/` | 1,347 | `poisson_plot.rs` 194 | 4 | B− (C) | Thin now. 56 `unwrap_or_else(panic!)` wrappers, the predict→decode block copied 4×, run-write order wrong in 3 train examples, CGL cell edges in an example. |
| `tests/` | 642 | `spectral_init.rs` 338 | — | B+ (C) | Every test asserts something. The GPU bodies in `gpu_parity.rs` never compile in CI, and the `PADS` env var can change what a regression test checks. |
| CI + `Cargo.toml` | ~217 | `Cargo.toml` 133 | 1 (churn) | B+ (C) | Clean and matched to CLAUDE.md. `cargo doc` runs without `-D warnings` and never with `datagen`, and the `"neural operators"` keyword is invalid on crates.io. |

---

## 4. Hotspot table

Churn is over all history (squash merges, so one commit per PR). Score ≈ lines × commits.

| # | File | Lines | Commits | Key problem now |
|---|---|---|---|---|
| 1 | `src/neural_operators/training/trainer.rs` | 777 | 35 | Production is 350 lines and cohesive. The observer seam is unreachable from `train_fno`. `train_built` exists for tests. About 150 lines of T3/T17 migration oracles. |
| 2 | `src/neural_operators/layers/spectral_convolution.rs` | 860 | 19 | None of note. Cohesive, about half tests. |
| 3 | `src/neural_operators/data/loaders/darcy.rs` | 422 | 23 | Grew 312 → 422 lines with T18's `load_darcy_test` and `darcy_test_from_fields`, which mirror Poisson's `uniform.rs`. |
| 4 | `examples/predict/{darcy,poisson,poisson_plot,…}.rs` | 60–194 | 12–21 | Repeated open → normalizers → panic-message → decode skeleton. `errors.json`/`prediction.png` named here. |
| 5 | `src/neural_operators/metrics/poisson.rs` | 756 | 10 | Grew from 445 lines. Numerics and evaluation policy in one file. `evaluate` has 7 parameters, 3 of them provenance. |
| 6 | `src/neural_operators/models/fno/**` | 300 + 1,251 | 16 (as `fno.rs`) | The tests carry `pre_padding_forward`, `with_old_grid` and "pre-#11/#13" vocabulary. |
| 7 | `src/neural_operators/data/transforms/normalizers.rs` | 340 | 14 | `from_record` panics on a malformed file. `FlatDecoder` (Burn tensors) still lives in ndarray `transforms/`. |
| 8 | `src/neural_operators/pde/poisson/dataset.rs` | 540 | 7 | Stale `n >= 3` guard. `SplitSpec` `Option<K>` round trip. Error 2 written out again. Imports the loader for the sidecar. |
| 9 | `src/neural_operators/data/loaders/poisson/{mod,uniform,cheb}.rs` | 349 + 421 + 130 | 13 (as `poisson.rs`) | The sidecar schema sits in `mod.rs`. Two minimum-n rules (`cheb.rs` 3, `uniform.rs` `Transfers::MIN_N` = 4). |
| 10 | `Cargo.toml` | 133 | 34 | The `keywords` entry has a space. Otherwise clean. |

---

## 5. Comparison with the 2026-10-15 review

### 5.1 Systemic issues S1–S5

| Issue | Status | Evidence |
|---|---|---|
| **S1** The run directory has no owner | **Resolved (core), partly (edges)** | `runs/mod.rs:40-50` holds the only file-name constants. `.bpk` is symmetric (`WEIGHTS_STEM`, `:48-50`). `CARGO_MANIFEST_DIR` no longer appears in library code. `burgers_learner` writes a readable run (`examples/train/burgers_learner.rs:96-103`). `x_normalizer.json` is read (`runs/mod.rs:218-231`). Still open: `errors.json` (`examples/predict/poisson.rs:101`) and `prediction*.png` (`darcy_plot.rs:86`, `poisson_plot.rs:189`) are named in examples. The `"{}/y_normalizer.json not found"` message is repeated at 4 sites (`predict/darcy.rs:47`, `darcy_plot.rs:38`, `poisson.rs:66`, `poisson_plot.rs:106`). The run lifecycle is split between `train_fno` and each example, which is not atomic (T2-1 below). |
| **S2** Superseded implementations left alive | **Resolved (production), partly (tests)** | All nine rows of the baseline's S2 table are done (T4, T5, T6, T7, T11, T14, T23). Residue: migration oracles kept as permanent tests. These are `train_fno_none_matches_identity_path` (`trainer.rs:529-554`, which compares `train_built(None)` with the `training_loop(None)` it calls), `train_loss_matches_old_asymmetric_path` (`trainer.rs:556-639`), `device_loaders_train_identically_to_host_loaders` (`trainer.rs:435-509`, which overlaps `device_batcher.rs:163-190`), `pre_padding_forward` (`models/fno/tests/mod.rs:114-126`) and `transfers_round_trip_matches_manual_recipe` (`chebyshev/transfer.rs` tests). |
| **S3** Per-PDE copy-paste at the edges | **Partly** | `training/trainers/` is deleted, and one `train_fno` (`trainer.rs:311`) serves all three train examples. `normalize_split_pair` is shared (`loaders/common.rs:47`). One `evaluate` (`step.rs:125`). One `burgers_resolution`. Remaining: the `load_X` / `load_X_test` pairs for Darcy (`darcy.rs:71-107, 144-169`) and Poisson (`uniform.rs:38-85`) are the same shape, with a duplicated `test_loader_with_saved_normalizers_matches_refit` (`darcy.rs:321`, `uniform.rs:256`). The run-write tail is copied in 3 train examples. The `predict_host → decode → into_dimensionality` block is copied 4× (`predict/poisson.rs:78-83`, `poisson_plot.rs:119-126`, `poisson_resolution.rs:95-100`, `darcy_plot.rs:66-69`). |
| **S4** The `chebyshev` feature splits a mostly ungated module | **Resolved** | Only `differentiation` is gated (`chebyshev/mod.rs:24-25, 34-35`). `points` and `quadrature` are `cfg(all(test, datagen))` oracles (`:26-29`). The feature is `datagen` and gates `pde` (`neural_operators/mod.rs:9`). There is one `TRANSFER_DEGREE`, one `ErrorStats` (`metrics/stats.rs`) and one `Transfers` used at all four recipe sites. FFTW stays by decision 27 (T15 declined). The one residue is the error-2 round trip, written out at `pde/poisson/dataset.rs:208-209` and `metrics/poisson.rs:109-118` (the baseline's target named a `Transfers::round_trip_error` that was not added). |
| **S5** Hygiene drift | **Resolved, with a small new residue** | No blanket `allow`: the allows are file-scoped with a reason in the 8 `#[derive(Config)]` files, plus one item-level `derivable_impls` (`spectral_convolution.rs:54`). There are no `REVIEW.md` citations. `#![warn(missing_docs)]` (`lib.rs:41`). The `lib.rs` docs are current. `run-examples.yml` is deleted and there are no commented apt blocks. The `Cargo.toml` URLs and categories are correct. The CLAUDE.md Checks list matches the three jobs. Loaders do not print. New residue: `train_fno` still prints (`trainer.rs:223, 327, 334`). There are history labels ("B2 contract", "T1–T6", "review R1", "pre-#8/#11/#13") in `models/fno/mod.rs:190-193` and the fno tests. `learner.rs:8` cites `trainer::forward_loss`, which is now `step::forward_loss`. `loaders/mod.rs:3` still lists `.npy`. The `"neural operators"` keyword is invalid (`Cargo.toml:11`). |

### 5.2 Roadmap waves

| Wave | Status | Evidence |
|---|---|---|
| **A** pure deletions (A1–A7) | **Resolved** | A1 (T2): `step::forward_loss` (`step.rs:95`) is used by `trainer.rs:90` and `learner.rs:42`. A2 (T3): `train_fno`. A3 (T4): only `UnitGaussianNormalizer` with inherent methods (`normalizers.rs:38-119`), no `FieldReader` or Npy, plain `DatasetConfig`. A4 (T5): `grids.rs` is gone. A5 (T6): `pde/poisson/mod.rs:13-14` makes `sparse` `#[cfg(test)]`, and `rlst_smoke` is gone. A6 (T7): `tests/gpu_parity.rs` is 248 lines with ungated `*_flex` twins. A7 (T1, T24): CI and manifest are clean, and the citations are replaced. |
| **B** one owner per concept (B1–B5) | **Resolved, B5 partly** | B1 (T10): `RunDir`. B2 (T11): `device_loader` only; `host_oracle` is `#[cfg(test)]` (`data/mod.rs:11-12`). B3 (T9): `loaders/common.rs`, `insert_axis` (`common.rs:104-107`). B4 (T13): `evaluate`, `predict_host<R>` (`models/predict.rs:24`). B5 (T12): `EpochObserver` exists (`trainer.rs:218`), but `train_fno` hard-codes print plus CSV (`trainer.rs:324-332`), so Phase 5 checkpointing still has to change `train_fno`'s signature. |
| **C** Chebyshev single stack (C1–C4) | **Resolved (C3 declined by decision)** | C1 (T8): `Transfers`, `cc_norm_sq` (`clencurt.rs:69`), one `ErrorStats`. C2 (T14, decision 26): the generator uses `nodes`, `Transfers` and `cc_rel_l2_error` (`dataset.rs:30`), and `norm.rs` is deleted. C3 (T15): declined (decision 27). C4 (T16): `datagen`. |
| **D** contracts (D1–D3) | **Resolved, D3 partly** | D1 (T17, decision 28): `y_train` is physical, and `Option<&FlatDecoder>` (`step.rs:98`) replaces `Postprocess`/`identity`. D2 (T18, decision 29): predict reads both normalizers and loads only the test split. D3 (T19): the typed `SidecarRecord` is shared and fixture-tested, `evaluate` is in the library, and the blanket allow is deleted. But the record lives in the loader, which created the `pde → loader` edge, and `evaluate` takes provenance arguments. |
| **E** decomposition and polish (E1–E5) | **Resolved, E1 partly** | E1 (T20): split into `fno/tests/*` (all under 320 lines), `reference_forward` deleted, named permutations (`fno/mod.rs:282, 291`). But total test lines barely moved (1,263 → 1,251). E2 (T21): `with_bc` (`spectral_convolution.rs:87`). E3 (T22): `poisson/{mod,cheb,uniform}`, `PoissonField`, `catalogue.rs`, `stage() -> Result<_, NoStage>`. E4 (T23): `take_strided` (`transforms/strided.rs:15`). E5 (T24): docs and `LpLoss`. |
| T25 (optional) | **Not done** | `collocation.rs:107-115, 226-253` still converts between RLST and ndarray. `grf.rs:153` writes A·F·Bᵀ by hand. |

### 5.3 What the baseline review would still flag

Using the baseline's own categories, these are the items it would raise again at `bbd605e`:
1. **S1-type: run files named outside `runs/`.** These are `errors.json`, `prediction*.png`, and the `y_normalizer.json` message at 4 sites. In addition, the run writes are split across `train_fno` and the examples, in the wrong order.
2. **S2-type: history kept alive, now in tests.** About 300 lines of migration oracles across `trainer.rs` and `models/fno/tests/`.
3. **S3-type: copies at the edges.** The Darcy and Poisson `load_*`/`load_*_test` pairs, the predict→decode block (4×), the train run-write tail (3×), and 56 `panic!` wrappers in examples.
4. **Direction leak (the same row as the baseline's catalogue edge).** `pde → data::loaders::poisson` for the sidecar types.
5. **T4-type contract issues.** `evaluate`'s provenance parameters, `UnitGaussianNormalizer::from_record` panicking on disk data, and `generate`'s stale `n >= 3`.
6. **T6-type unused public surface.** `training_loop`, `train_epoch`, `EpochMeans`, `print_epoch`, `IndexDataset`, `DeviceBatcher`, `cheb_to_uniform`, `uniform_to_cheb`, `apply`, `diff_matrix`, and `grf::{mu, lambda, s_k, S_INF}` have no callers outside their own files.
7. **T7-type legibility.** Stale file names (`data/batcher.rs` holds only `Batch`; `loaders/base_dataset.rs` holds only `DatasetConfig`), history labels in the fno tests, and stale module docs.

---

## 6. Systemic issues (current)

### N1. The training run is not one atomic, owned flow (tier 2)

**Locations**
- `examples/train/burgers.rs:106-120`, `darcy.rs:85-110` and `poisson.rs:132-154` create the `RunDir`, call `train_fno` (which appends `metrics.csv` each epoch, `trainer.rs:324-332`), and only then call `save_configs`, `save_normalizers`, `save_plot` and `save_weights`.
- `burgers_learner.rs:96-103` does the opposite, with a comment saying "the configs go in before training so a run stopped early is still described".
- `train_fno` takes `y_norm: Option<&UnitGaussianNormalizer>` while the example saves `&normalizers` separately: one fact, two hand-offs.

**Problem**
- A run that stops at epoch 300 of 500 leaves a `metrics.csv` that no `RunDir` reader can use, because `model_cfg.json` and `data_cfg.json` are missing.
- The observer seam (B5) is fixed inside `train_fno`, so Phase 5's checkpointing has nowhere to plug in without changing the signature anyway.

**Canonical target**
- `train_fno` owns the whole run. It takes `run: &RunDir`, `data_cfg: &impl Config`, `normalizers: Option<&Normalizers>` and `on_epoch: &mut EpochObserver<R>`.
- At the start it writes the configs and normalizers. Each epoch it appends metrics and calls `on_epoch`. At the end it writes the plot and the weights.
- The three example tails collapse to one call. `train_built`, the `decoder` parameter of `training_loop` and the library `println!`s go: the decoder moves into `TrainingComponents`, and timing and printing go to the default observer the examples pass.

### N2. The dataset-format owner is in the loader (tier 3)

**Locations**
- `SidecarRecord`, `SidecarError2` and `SidecarLabelCheck` are in `data/loaders/poisson/mod.rs:66-144`, whose own doc says the loader does not read the sidecar.
- They are imported by `pde/poisson/dataset.rs:32-34` and `metrics/poisson.rs:31`.
- The sidecar name is built by hand as `format!("{stem}.json")` at `pde/poisson/dataset.rs:256` and `examples/predict/poisson.rs:87`, beside an existing `PoissonSet::npz`.
- Splits are `&str` (`catalogue.rs:40-47, 58`).

**Canonical target**
- `data/catalogue` (or `data/catalogue/sidecar.rs`) owns `PoissonSet`, `enum SplitName { Train, Test }`, `npz(split)`, `sidecar(split)` and the three record types, plus the fixture round-trip test.
- `pde` and `metrics` then import only the catalogue, which closes the baseline's direction row for good.

### N3. Migration oracles kept as permanent tests (tier 6)

Phase 4s pinned every refactor with an "old path" test, which was the right call during the migration. The pins were not retired afterwards:
- `trainer.rs:529-554`: tautological, since it compares a function with its own body.
- `trainer.rs:556-639`: the pre-T17 loop inlined.
- `trainer.rs:435-509`: the host batching path in trainer tests, which overlaps `device_batcher.rs:163-190`.
- `trainer.rs:653-724`: a Poisson npz fixture re-implemented in the generic trainer.
- `models/fno/tests/mod.rs:114-126, 257-268`: `pre_padding_forward` is a copy of `forward` minus padding, built from the same private helpers, so it is not an independent oracle.
- `transfer.rs` tests: the `transfers_round_trip_matches_manual_recipe` scaffolding.

**Target**
- Delete the tautology and the T17 oracle.
- Fold the fixtures into one crate-level `#[cfg(test)] test_support` (temp dir, Poisson npz, `tiny_model_cfg`; the four near-identical tiny `FNOConfig` builders collapse into it).
- Replace `pre_padding_forward` with `bits(off) == bits(zero)` plus the independent `reference_forward_padded(.., Pad::Off)`.
- Rename "old grid" to `grid_oracle_f64`.
- Deliberate test deletions are allowed by the Phase 4s rules; list them in the PR.

---

## 7. Findings by tier

### Tier 1: structural regressions

None. No module fights the architecture, the graph is acyclic, and no file is at or near 1,000 lines. The data slice reviewer flagged the sidecar placement as tier 1. It is graded tier 3 here (N2) because it is a single misplaced type set with a mechanical fix.

### Tier 2: missed simplifications (code-judo)

1. **N1, the run lifecycle in `train_fno`** (`trainer.rs:311-349`, the three train examples). This deletes three tails, `train_built`, the `decoder` parameter and the library prints, and makes the B5 seam usable. *Confirmed.*
2. **One error-2 round trip.** `pde/poisson/dataset.rs:208-209` ≡ `metrics/poisson.rs:109-118`: `cc_rel_l2(to_cheb(to_uniform(u)), u)` is written twice, and the `ERROR2_TOL` cross-check compares the two copies. **Move:** `Transfers::round_trip_error(&self, u, w) -> f64` in `chebyshev/transfer.rs`. Both sites call it, and the manual-recipe test becomes a test of it. *Confirmed.*
3. **`SplitSpec` encodes K twice** (`pde/poisson/dataset.rs:46-105, 225-233`). `of_set` turns a `PoissonSet` into `Option<K>` by comparing with `k_default`, `truncation()` turns it back, `Split.k` stores it again, and `split_stem` rebuilds the `PoissonSet`. `of_set` is the only production constructor (`examples/generate/poisson.rs:41`). **Move:** `SplitSpec { set: PoissonSet, base_seed, count }`, with `k_explicit = set.k != k_default(set.n)` computed in `write_split`. This deletes `of_set`, `truncation`, `Split.k` and `split_stem`. *Confirmed.*
4. **One normalised loader skeleton** (`loaders/darcy.rs:71-107, 144-169` vs `loaders/poisson/uniform.rs:38-85`). Both are `check_n_train → read train → read test → normalize_split_pair` and `read test → normalize_test_split`, plus Darcy's two `_from_fields` test-only wrappers. **Move:** each PDE supplies `read_split(path, split, count) -> Result<(x, y), LoadError>`. `common.rs` gets `load_normalized(read, …)` and `load_test(read, …, &Normalizers)`. The four public functions become one-liners, and the duplicated saved-normalizer test moves to `common.rs`. Darcy drops back toward its pre-T18 size.
5. **The `train_built` seam** (`trainer.rs:338-349`). It exists so tests can share an initial model, and it is why `training_loop` takes `decoder`. **Move:** build `Option<FlatDecoder>` inside `build_training_components` and store it in `TrainingComponents`. This is subsumed by item 1.

### Tier 3: spaghetti / branching

1. **N2, the sidecar schema in the loader.** It is a direction leak, and the sidecar name is built ad hoc. *Confirmed* (`pde/poisson/dataset.rs:32-34`, `metrics/poisson.rs:31`).
2. **`examples/predict/burgers.rs:53-59`**: `if run.metrics().is_ok()` exists only for Learner runs. **Move:** the Learner showcase appends `metrics.csv` through `RunDir::append_metrics`, and the branch goes.

### Tier 4: boundary / type-contract

1. **`pde/poisson/dataset.rs:184`** asserts `n >= 3`, and the docs at :63-64, :76-77 and :179 say so. But `Transfers::new(n)` at :188 needs `n >= Transfers::MIN_N` = 4, so n = 3 passes the guard and panics later with another message. **Move:** assert `Transfers::MIN_N` and fix the three docs. *Confirmed.*
2. **`data/transforms/normalizers.rs:109-118`**: `from_record` `expect`s on data read from disk, while `RunDir::normalizers` (`runs/mod.rs:218-231`) promises an `Err` for a file that "can't be read or parsed". **Move:** `from_record -> Result<Self, ShapeError>` (or `TryFrom`), with `?` in `runs`. *Confirmed.*
3. **`metrics/poisson.rs:298-306`**: `evaluate(errors, n, sidecar, last, device: String, backend_features: Vec<String>, run_dir: &Path)`. Three of the seven parameters are provenance and presentation. `run_dir` exists only to print a `poisson_plot` hint. `last: &EpochMetrics` needs only `test_l2`, and it is what forces `metrics → runs`. **Move:** `evaluate(errors, n, sidecar, last_test_l2: f32)`; the example adds provenance (or a `Provenance::current(&device)` in the library) and prints the hint. `RunDir::save_errors(&PoissonErrorsRecord)` then becomes possible without a cycle.
4. **`runs/mod.rs:15` vs `:174`**: the doc says writers return `io::Result`, but `save_weights` returns `Box<dyn Error>`. `normalizers()` checks only `y_normalizer.json`, so a run with `x` but not `y` is silently `None`. **Move:** fix the doc. Treat "both missing" as `None` and "one missing" as an error. Add `require_normalizers()` so the 4 examples stop writing their own message.
5. **Stringly splits** (`catalogue.rs:40-47, 58`, `darcy.rs:79-80`, `uniform.rs:51,56`). Part of N2.

### Tier 5: file size / decomposition

1. **`metrics/poisson.rs`, 756 lines (462 production).** Pure numerics (`poisson_errors`, `rel_l2_equal_weights`, `boundary_errors`, about 165 lines) share a file with policy, schema and report (`ERROR*_TOL`, `BOUNDARY_THRESHOLD`, `ERROR1_QUOTE`, `Error2Check`, `Error3Check`, `PoissonErrorsRecord`, `PoissonEvaluation`, `evaluate`, `Display`, about 300 lines). **Move:** `metrics/poisson/{mod, errors, evaluation}.rs`. `errors.rs` imports only `chebyshev`, and the tests split along the existing seam (errors at :493-612, evaluation at :619-756).
2. **`training/trainer.rs`, 777 lines.** Production (350) is cohesive. Shrink the tests by about 200 lines (N3), with no split needed.

### Tier 6: modularity

1. **N3, migration oracles.**
2. **Unused public surface.**
   - `training_loop`, `train_epoch`, `EpochMeans` and `print_epoch` (`trainer.rs`) have no callers outside the file.
   - `IndexDataset` and `DeviceBatcher` (`device_batcher.rs:28,54`) are public only for a compile_fail doctest.
   - `cheb_to_uniform`, `uniform_to_cheb` and `apply` (`chebyshev/mod.rs:36-38`), `diff_matrix`, and `grf::{mu, lambda, s_k, S_INF}`.

   **Move:** make them `pub(crate)` or private. Alternatively, let `apply` earn its export through T25 (`grf.rs:153`).
3. **`FlatDecoder` in `transforms/normalizers.rs:121-164`.** It is the only Burn-tensor item in an ndarray module, and its only consumers are in `training/`. Move it to `training/step.rs` or `training/decoder.rs`. Also move `Normalizers { x, y }` from `loaders/common.rs:17` to `transforms/normalizers.rs`, so `runs` depends on transforms only.
4. **The predict→decode block is copied 4×** (S3 above). **Move:** `predict_host` takes `Option<&UnitGaussianNormalizer>` and returns physical `ArrayD<f64>`, or `decode` becomes dimension-generic.
5. **`examples/predict/poisson_plot.rs:47-59`**: `cgl_cell_edges` is CGL geometry checked only by `debug_assert`s in an example that CI never runs. Move it to `chebyshev::transfer::cell_edges(n)` with a unit test, and expose `RunDir::name()` instead of recomputing it (`:150-152`).
6. **`examples/*`**: 56 `unwrap_or_else(|e| panic!(…))`. Use `fn main() -> Result<(), Box<dyn Error>>` and `?`. `RunDir` and `LoadError` already return `Error`s.
7. **CI.**
   - `cargo doc --no-deps` runs without `RUSTDOCFLAGS=-D warnings` and never with `--features datagen` (`run-tests.yml:32-33`).
   - The GPU bodies in `tests/gpu_parity.rs` (about 150 lines) never compile in CI. A `cargo check --tests --features cuda` job is worth trying, after verifying the toolkit question on a runner.
   - The style job could run one `cargo clippy --no-deps --all-targets -- -D warnings`, the check CLAUDE.md asks of humans.
8. **`data/loaders/darcy.rs:191-201`**: `take_subsampled` is a second strided wrapper whose only extra is a zero-copy fast path. Put that path in `strided.rs`, so Burgers gets it too.

### Tier 7: legibility

1. **Stale file names.** `data/batcher.rs` holds only `Batch`; rename it `batch.rs`. `loaders/base_dataset.rs` holds only `DatasetConfig`; rename it `loaders/config.rs`.
2. **History labels.**
   - "B2 contract", "T1–T6", "review R1" and "pre-#8/#11/#13" appear in `models/fno/mod.rs:190-193` and in `fno/tests/{grid,checkpoint,mod,padding}.rs`. "T1–T6" now reads as Phase 4s tasks, which they are not. Cite CONVENTIONS § or a commit instead.
   - `pin_permutations()` is a hidden test inside `forward_matches_reference_1d` (`reference.rs:114-118`). Make it its own `#[test]`.
3. **Stale docs.**
   - `learner.rs:8` cites `trainer::forward_loss`; it is now `step::forward_loss`.
   - `loaders/mod.rs:3` lists `.npy` and has a broken backtick.
   - `io/errors.rs:34-35` lists too few `LoadError` producers.
   - The CONVENTIONS §10 paragraph appears 3× (`darcy.rs:11-15`, `common.rs:1-7`, `poisson/mod.rs:27-30`).
4. **`runs/csv.rs:80, 96, 111`**: fixed temp file names with no process id, unlike `fresh_root` in `runs/mod.rs`. `csv_round_trips` duplicates `metrics_and_normalizers_round_trip`.
5. **`train_epoch` style.**
   - `loop { let Some(batch) = … else { break } }` should be `for` (`trainer.rs:77-105`).
   - `last_lr` and `current_lr` name one quantity twice.
6. **The train examples are inconsistent.** Only `train/poisson.rs` keeps its hyperparameters as literals at the top, as CLAUDE.md asks.
7. **`Cargo.toml:11`**: the keyword `"neural operators"` contains a space, which crates.io rejects. `.cargo/config.toml` sets `+fp16`, which CLAUDE.md does not mention.

---

## 8. Refactor roadmap (to A−)

Each step is one PR and behaviour-preserving unless marked ⚑. Each is verified by the CI checks plus `cargo clippy --all-targets -- -D warnings`. Steps touching `pde/` or `chebyshev/` also run `cargo test --features datagen`. Ordered by leverage.

| # | Step | Deletes / fixes | Proof | Depends on |
|---|---|---|---|---|
| R1 | **`train_fno` owns the run (N1).** It takes `data_cfg`, `Option<&Normalizers>` and `on_epoch`, and writes configs and normalizers first and weights and plot last. The decoder moves into `TrainingComponents`. `train_built` goes. `training_loop`, `train_epoch`, `EpochMeans` and `print_epoch` become `pub(crate)`. | Three example tails, one test seam, the library prints. Makes stopped runs readable and gives Phase 5 T2 its hook. | Trainer tests. `tests/run_artifacts_without_gnuplot.rs`. **Add first:** a test that `RunDir::open(...).model_config()` and `data_config()` succeed after epoch 0 of a run whose observer stops it. Re-run `train_poisson` stopped after epoch 0, then `predict_poisson` on a full run, and check the output is unchanged. | — |
| R2 | **Retire the migration oracles (N3)**, with one `#[cfg(test)] test_support` for temp dirs, the Poisson npz fixture and the tiny `FNOConfig`. | About 300 test lines. | The "running N tests" line drops only by the listed deletions. | R1 (shares `trainer.rs`) |
| R3 | **Sidecar and split names in the catalogue (N2).** `SplitName`, `PoissonSet::sidecar`, the record types moved, `write_split` uses `npz`/`sidecar`. | The `pde → loader` and `metrics → loader` edges, two ad hoc names. | `sidecar_fixture_parses_and_round_trips`, `generator_stems_unchanged`, `cargo test --features datagen`. `predict_poisson` output byte-identical. | — |
| R4 | **`Transfers::round_trip_error`, the `SplitSpec { set, … }` rewrite, and the `MIN_N` guard.** | The error-2 duplicate, `of_set`, `truncation`, `Split.k`, `split_stem`, the stale panic contract. | `cargo test --features datagen`. Regenerate stage 1 into a scratch directory and diff it against the stored files (it must be byte-identical). | R3 |
| R5 | **Split `metrics/poisson.rs`** into `errors`/`evaluation`. `evaluate` takes `last_test_l2` and no provenance. `RunDir::save_errors`. | The `metrics → runs` edge, `errors.json` outside `runs`. | The `evaluate` tests. `errors.json` byte-identical on `runs/poisson_fno_p4_1791922918`. | R3 |
| R6 | **One normalised loader skeleton** in `common.rs`. Darcy's `_from_fields` wrappers go. `from_record -> Result`. `Normalizers` and `FlatDecoder` move to their owners. | About 80 production lines, one duplicated test, one panic on disk data. | Loader pins (the T9 checksum tests). `predict_darcy` 0.0095942197 and `predict_poisson` unchanged. | R5 (examples) |
| R7 | **Examples:** `main() -> Result`, a decoding `predict_host`, `RunDir::require_normalizers`, `prediction_png`, `cell_edges` into `chebyshev`, hyperparameters at the top. | About 56 wrappers and the 4× decode block. | `cargo clippy --examples`. Each predict example prints its recorded numbers. | R1, R5, R6 |
| R8 | **Visibility and naming pass.** `pub(crate)` for the unused exports, rename `batcher.rs`/`base_dataset.rs`, history labels, stale docs. CI gets `RUSTDOCFLAGS=-D warnings cargo doc --features datagen`. Fix the keyword. | Public surface; doc drift. | `cargo doc` with `-D warnings`. Clippy. | Last |
| R9 | *Optional (T25)*: collocation on ndarray, and `apply` earns its export. | `to_dyn`/`from_dyn`/`transpose`. | Time n = 257 first. `cargo test --features datagen`. | — |

**Recommended start:** R1, then R3 and R4. R1 is the only remaining item that Phase 5 T2 would otherwise build on. R3 and R4 close the last direction leak and the last S4 residue before Phase 6 adds code under `chebyshev/`.

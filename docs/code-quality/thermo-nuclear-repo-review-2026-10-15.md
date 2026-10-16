# Thermo-nuclear repo review: sciml-rs

| | |
|---|---|
| Repo | sciml-rs (`sciml_rs`), worktree `seapen`, branch `phase4/nuclear-repo-review` |
| Commit | `c63eb320390721b10808533678bd89bcf291cadd` (Phase 4 closed, #36) |
| Date | 2026-10-15 |
| Scope | Whole repository: `src/`, `examples/`, `tests/`, `.github/workflows/`, `Cargo.toml` |
| Excluded | `spikes/` (standalone Phase 0 measurement packages, never built by CI), `upstream/burn/*` (bug-report READMEs), `docs/` (read for context, not reviewed), `datasets/` and `runs/` (git-ignored), licences |
| Size | 66 Rust/TOML/YAML files, 15,609 lines; 173 commits since 2026-08-27 |
| Coverage | Full. Every in-scope file was read in full by one of five parallel slice reviewers (models/layers/utils, data, training/losses/metrics, chebyshev/pde, examples/tests/CI). All tier 1–3 findings were re-checked against the code before inclusion. No builds or tests were run; this review is read-only. |

---

## 1. Health verdict: **C**

The core of the crate is good. `FNO<R>` and `SpectralConv` are rank-generic and have no 1D/2D/3D branching anywhere. `train_epoch`/`training_loop` has no per-PDE branching. The Chebyshev numerics and solvers are careful, well justified and very heavily tested. No dependency cycle exists. None of the problems sits in the hot inner code.

The grade is C because three systemic patterns now cost more than the code they protect, and Phases 5 and 6 would build directly on top of them:

1. **The run directory has no owner.** The run format (`model_cfg.json`, `model_weights(.bpk)`, `*_normalizer.json`, `data_cfg.json`, `metrics.csv`) is spelled out by hand in about 16 files. Writing it is split three ways, and reading it is split between `models/run.rs` and five examples that bypass it. Phase 5's checkpoint/resume task would land in the middle of this.
2. **Superseded implementations are never deleted.** Each phase added a better path and left the old one alive:
   - `OperatorBatcher` beside `device_loader`
   - the RLST/FFTW Chebyshev stack beside the closed-form stack
   - `grids.rs` beside `FNO::grid_cl`
   - three normalizers where one is used
   - two file readers where one is used generically, plus an unused Npy reader

   Roughly 1,200–1,700 lines (production and tests) are generality or history that nothing in production exercises.
3. **Per-PDE copy-paste at the edges.** The library function `train_poisson` is a byte-for-byte copy of `train_darcy`. The Darcy and Poisson loader tails are the same function. The train examples share a persistence tail, and the Burgers/Darcy predict examples share a load/evaluate skeleton.

None of these is hard to fix. Most fixes are deletions, and every step in the roadmap is a behaviour-preserving PR that the existing test suite already guards.

---

## 2. Architecture map

**Intended direction** (from CLAUDE.md, module docs and imports):

```
utils (fft, backend)
  └─▶ layers (SpectralConv)
        └─▶ models (FNO, run: load_fno / predict_host)
chebyshev (pure f64 ndarray numerics)
  ├─▶ pde/poisson (solvers, GRF, dataset generation)       [feature "chebyshev"]
  └─▶ data (io readers, loaders per PDE, normalizers, batchers)
losses (LpLoss)
training (trainer loop, per-PDE trainers, Learner)  ──▶ data, models, losses
metrics (run artifacts I/O, gnuplot, Poisson errors) ──▶ training, models, chebyshev, losses
examples/ (thin entry points: generate / train / predict × burgers, darcy, poisson)
```

**Actual direction:** no cycles. `training` never imports the top-level `metrics`. The violations and leaks are:

| Edge | Status | Note |
|---|---|---|
| `metrics::io → training::{TrainingConfig, EpochMetrics}`, `models::FNOConfig` | ⚠ ownership | "metrics" is really the run-persistence layer, and it imports concrete config types only to call `.save()` on them. It needs a `runs` module (S1). |
| `pde::poisson::dataset → data::loaders::poisson::{PoissonSet, STAGES}` | ⚠ direction | The gated generator reaches into the ungated *loader* for the dataset catalogue. The catalogue is lower-level than both. |
| `models::FNOConfig::init` asserts `out_channels == 1` | ✗ leak | This is a training-layer restriction (`flatten_pair`) enforced inside the model (fno.rs:75-78). |
| `metrics::io::write_run_artifacts` hard-codes `env!("CARGO_MANIFEST_DIR")/runs` | ✗ leak | Example/workspace policy inside a library function (metrics/io.rs:82). |
| `metrics::poisson::TRANSFER_DEGREE` duplicates `pde::poisson::dataset::TRANSFER_DEGREE` | ✗ duplicate caused by the feature gate | The comment admits it "cannot import" the other copy (metrics/poisson.rs:29-32). |
| Two modules named `metrics` (`training::metrics`, top-level `metrics`) | ⚠ naming | Readers have to work out which one an import means. |
| Evaluation policy in `examples/predict/poisson.rs:156-262` | ⚠ wrong layer | The worst sample, the error-2 cross-check, the boundary verdict and the record type belong in `metrics::poisson`. |

---

## 3. Module scorecard

| Module | LOC (prod/test) | Largest file | Best hotspot rank | Grade | Diagnosis |
|---|---|---|---|---|---|
| `models/` | ~470 / ~1,260 | `fno.rs` 1,569 (306 prod) | 2 | B+ | Clean rank-generic core. Tests carry three generations of reference forwards, and the canonical `load_fno` is ignored by five examples. |
| `layers/` + `utils/` | ~500 / ~450 | `spectral_convolution.rs` 853 | 3 | B+ | Single generic FFT/corner path. Five copies of a fixed-rank array idiom, and one stale doc formula. |
| `data/` | ~1,500 / ~1,400 | `loaders/poisson.rs` 748 | 4 | B- | Careful code, but about 400 prod lines of unused generality (grids, two normalizers, Npy, `FieldReader`, base-config traits, host batcher) and a duplicated Darcy/Poisson tail. |
| `training/` | ~550 / ~400 | `trainer.rs` 570 | 1 | B | A good generic loop. The Burn Learner showcase is worth keeping but duplicates the loss line and writes unreadable runs (A1). Two identical per-PDE wrappers, and a `dyn Fn` pair to express one asymmetry. |
| `losses/` | 134 / 360 | `data_losses.rs` 494 | 7 | B | A faithful port of Li et al.'s `LpLoss`. Most of its options (`abs`, `d`, `Mean`/`None`, `forward`) are unused in production. |
| `metrics/` | ~460 / ~220 | `poisson.rs` 445 | — | C+ | Three unrelated concerns (run persistence, gnuplot, Poisson numerics) in one module, plus a duplicated constant and a hand-rolled CC integral. |
| `chebyshev/` | ~550 / ~900 | `transfer.rs` 510 | — | B | Two parallel stacks (closed-form ungated, RLST/FFTW gated). The gated stack has one real production use, `diff2_matrix`. |
| `pde/poisson/` | ~1,100 / ~700 | `dataset.rs`, `collocation.rs` 476 | — | B | Solid solvers. `sparse.rs` is a public production API for a test-only oracle, and the transfer recipe is rebuilt at four sites. |
| `examples/` | ~1,800 | `predict/poisson.rs` 271 | 5 | C | The Burgers/Darcy examples predate `models::run` and copy its body. The run-file literals and dataset paths are repeated, and there are two near-identical Burgers resolution examples. |
| `tests/` | ~2,170 | `metal_diag.rs` 1,674 | — | C | A 1,674-line GPU bisection scratchpad that CI never compiles. `spectral_init.rs` and `run_artifacts_without_gnuplot.rs` are fine. `rlst_smoke.rs` is redundant. |
| CI + `Cargo.toml` | ~280 | `run-tests.yml` 123 | 1 (Cargo.toml churn) | C | One workflow does nothing, there are about 30 lines of commented apt leftovers, and the manifest URLs point at Codeberg. CLAUDE.md's "CI runs exactly" list is stale. |

---

## 4. Hotspot table

The history is short (7 weeks), so churn is measured over all history. Score ≈ lines × commits.

| # | File | Lines | Commits | Key problem |
|---|---|---|---|---|
| 1 | `src/neural_operators/training/trainer.rs` | 570 | 24 | `Postprocess` closure pair plus a public `identity` to express one train/test asymmetry. The eval contract ("returns a sum, caller divides") leaks to five call sites. There is no epoch hook for Phase 5 checkpointing. |
| 2 | `src/neural_operators/models/fno.rs` | 1,569 | 12 | Over 1k lines. Production (306) is cohesive, tests (1,263) are not: there are three reference forwards, one of which is exactly another with `Pad::Off`. |
| 3 | `src/neural_operators/layers/spectral_convolution.rs` | 853 | 17 | Cohesive. A `vec!+extend+try_into` idiom is repeated five times, `flat_modes` is both stored and passed, and the doc formula is stale. |
| 4 | `src/neural_operators/data/loaders/darcy.rs` | 312 | 15 | The `darcy_from_fields` tail is copied into Poisson. `DarcyNormalizers` and `PoissonNormalizers` are the same struct. |
| 5 | `tests/metal_diag.rs` | 1,674 | 1 | Over 1k lines and compiled out of CI (`#![cfg(any(metal, cuda))]`). About 22 of its 33 tests assert nothing. |
| 6 | `src/neural_operators/data/transforms/normalizers.rs` | 507 | 10 | `GaussianNormalizer` and `RangeNormalizer` have zero callers. The `Normalizer` trait has one live implementor. |
| 7 | `src/neural_operators/losses/data_losses.rs` | 494 | 7 | Option surface unused in production, and `LpLoss::new(d, …)` takes a `d` that `rel` ignores. |
| 8 | `src/neural_operators/data/loaders/poisson.rs` | 748 | 4 | Three concerns (dataset catalogue, Chebyshev reader, uniform loader). A stringly-typed field selector, and an identity wrapper. |
| 9 | `src/neural_operators/data/grids.rs` | 361 | 7 | No production caller; it survives only as an f64 test oracle for `grid_cl`. |
| 10 | `src/neural_operators/data/batcher.rs` | 201 | 11 | Superseded by `device_loader`, and kept alive only by four predict examples. |
| 11 | `examples/predict/poisson.rs` | 271 | 4 | Evaluation policy and persistent schemas defined in an example. A local blanket `#[allow]`. |
| 12 | `Cargo.toml` | 128 | 29 | Wrong `homepage`/`repository`, an invalid category, empty dev-dependency tables, a truncated comment. |

---

## 5. Systemic issues

### S1. The run directory is a format with no owner (tier 2)

**Locations**
- **Writers:**
  - `metrics/io.rs:62-97` (configs, CSV, plot)
  - the persistence tails of `examples/train/burgers.rs:98-118`, `darcy.rs:88-112`, `poisson.rs:127-156` and `burgers_learner.rs:96-102`, which write weights and normalizers
- **Readers:**
  - `models/run.rs:27` (`load_fno`)
  - the inline copies of it in `examples/predict/{burgers.rs:35-38, darcy.rs:42-45, darcy_plot.rs:32, burgers_resolutions.rs:41, burgers_superresolution.rs:50}`
- **Literals:** `"model_cfg.json"` in 8 examples, `"model_weights"` in 11, `"y_normalizer.json"` in 6.

**Problems**
- Weights are written without the `.bpk` extension and read with it.
- `x_normalizer.json` is written (train/darcy.rs:99, train/poisson.rs:143) and never read. Every predict path re-reads and re-transfers the entire train split to refit x.
- `burgers_learner` writes to a cwd-relative, untimestamped directory that `predict_burgers` cannot read.

**Canonical target:** a top-level `runs` module that owns one type.

```rust
pub struct RunDir { dir: PathBuf }
impl RunDir {
    pub fn create(root: &Path, name: &str) -> io::Result<Self>;        // root passed by the example
    pub fn save_configs(&self, model: &impl Config, train: &impl Config, data: &impl Config) -> io::Result<()>;
    pub fn save_metrics(&self, m: &[EpochMetrics]) -> io::Result<()>;   // CSV + plot
    pub fn save_normalizers(&self, n: &Normalizers) -> io::Result<()>;
    pub fn save_weights<const R: usize>(&self, model: &FNO<R>) -> io::Result<()>;
    pub fn open(dir: impl Into<PathBuf>) -> Self;
    pub fn load_fno<const R: usize>(&self, device: &Device) -> Result<FNO<R>, …>;
    pub fn config<C: Config>(&self, which: ConfigFile) -> Result<C, …>;
    pub fn normalizers(&self) -> Result<Option<Normalizers>, …>;
    pub fn metrics(&self) -> io::Result<Vec<EpochMetrics>>;
}
```

The file names become private constants of `runs`. The module absorbs `metrics/io.rs`, `metrics/plots.rs` and `models/run.rs::load_fno`, and `EpochMetrics` moves there too, which ends the two-`metrics` name clash. What is left of `metrics/` (`poisson.rs`) becomes `evaluation/`. Taking `&impl Config` removes the `metrics → training/models` imports.

### S2. Superseded implementations left alive (tier 2)

| Concept | Canonical | Superseded (delete or demote to `#[cfg(test)]` oracle) | Evidence |
|---|---|---|---|
| Batching | `device_batcher::device_loader` | `batcher.rs` (`OperatorBatcher`) + `DataItem` + `impl Dataset for OperatorDataset` | `device_batcher` tests already prove bit identity, and only the four Burgers/Darcy predict examples use the old path. |
| Model grid | `FNO::grid_cl` | `data/grids.rs` (361) | No production caller (its own doc says so). `GridPlacement::BeforeData` and `linspace_excluding_start` serve a 3D pipeline that does not exist. |
| Normalizer | `UnitGaussianNormalizer` | `GaussianNormalizer`, `RangeNormalizer`, the `Normalizer` trait | Zero callers. The trait exists only to force `use Normalizer` in five examples. |
| File reading | `MatFileReader`, `NpzFileReader` (inherent methods) | `NpyFileReader`, the `FieldReader` trait | Npy has zero callers, and nothing is generic or `dyn` over the trait. |
| CGL nodes / CC weights / rel-L² | `cgl_nodes`, `clencurt`, `cc_rel_l2_error` (ungated, overflow-safe) | `nodes`, `clenshaw_curtis`, `l2_norm`, `rel_l2_error` (gated RLST) | The only production consumer is `pde/poisson/dataset.rs:164,196,197,226`. A third copy of `cgl_nodes` sits in `examples/predict/poisson_plot.rs:49-54` because the canonical one is `pub(crate)`. |
| Validation solver C | — | `pde/poisson/sparse.rs` as public API | Nothing outside its own tests calls `SparseSolver`. It should be `#[cfg(test)] mod sparse;`, like `manufactured`. |
| Dataset config access | `cfg.base.n_train` | `BaseDatasetConfig` + `HasBaseConfig` + blanket impl | Three items to read two fields. Callers mix `.n_test()` and `.base().n_test`. |
| Subsampling | one strided take (`take_subsampled`) | `transforms/subsample.rs` + `split.rs` | Burgers copies all 2,048 samples twice to get there. |
| GPU smoke | the asserting `repro_*` and `replay_2d_poisson` tests | the rest of `tests/metal_diag.rs`, plus `tests/rlst_smoke.rs` | The bisection, timing and env-var-driven tests belong in `spikes/`. `rlst_smoke` tests a Sylvester route that design §4.2 rules out. |

### S3. Per-PDE copy-paste at the edges (tier 2)

- `training/trainers/darcy.rs:23-46` and `trainers/poisson.rs:26-49` are **identical except for the function name** (verified with `diff`). `trainers/burgers.rs` is the same function with `identity` hooks.

  **Target:** one `train_fno::<R, RM1, T>(train, test, y_norm: Option<&UnitGaussianNormalizer>, model_cfg, train_cfg, device)` in `trainer.rs`, and delete the `trainers/` directory.
- `data/loaders/darcy.rs:126-183` (`darcy_from_fields`) and `loaders/poisson.rs:321-366` (`poisson_from_fields`) follow the same steps: fit x, encode both, fit y, encode y_train only, add the channel axis, convert to `T`. They also share the same `n_train < 2` check and the same `{x, y}` normalizer struct.

  **Target:** `loaders/common.rs` with `Normalizers { x, y }` and `fn normalize_split_pair(...)`.
- The predict examples: the Burgers/Darcy load → dataset → loader → `eval_epoch` → divide-by-n skeleton appears five times. `burgers_resolutions.rs` and `burgers_superresolution.rs` are about 85% the same file.

  **Target:** `RunDir` (S1), a public `evaluate::<R, RM1>(model, loader, decoder) -> f32` that owns `LpLoss` construction and the per-sample mean, and one `burgers_resolution` example that takes its rates from argv like `poisson_resolution`.

### S4. The `chebyshev` feature splits a module that is mostly ungated (tier 3)

There are 11 `#[cfg(feature = "chebyshev")]` lines in `chebyshev/mod.rs`, plus more in `transfer.rs`, `clencurt.rs` and `neural_operators/mod.rs`. The feature is named after a module but actually means "RLST + BLAS/LAPACK + FFTW + reference solvers + dataset generator". The split directly causes:
- the duplicated `TRANSFER_DEGREE`
- the duplicated `ErrorStats` type (`dataset.rs:70-92` returns NaN when empty, `metrics/poisson.rs:36-50` panics)
- the "s = n − 1, T_cu, T_uc(d = 2), apply both axes" recipe rebuilt at four sites (`dataset.rs:199-231`, `metrics/poisson.rs:121-137`, `loaders/poisson.rs:162`, `poisson_plot.rs:129`)

**Target:**
- `chebyshev` becomes entirely ungated:
  - one public `nodes`
  - `clencurt`
  - `cc_rel_l2_error` built on a shared `cc_norm_sq`
  - `TRANSFER_DEGREE`
  - a `Transfers::new(n)` holding `t_cu`/`t_uc` with `to_uniform`, `to_cheb`, `round_trip_error`
  - `diff_matrix`/`diff2_matrix` in closed form (see the decision note in roadmap step 9)
- The feature then gates only `pde/`, and is renamed after what it enables (e.g. `datagen` or `reference-solver`).

### S5. Hygiene drift against the repo's own rules (tier 6/7)

- **Blanket allow:** `src/lib.rs:48` puts `#[allow(clippy::redundant_field_names)]` on the entire crate, which CLAUDE.md forbids. Burn's `#[derive(Config)]` expansion is the cause (burn-derive `config/analyzer_struct.rs:196`). Scope the allow to the files with `Config` derives, with a one-line reason. `examples/predict/poisson.rs:49` has its own copy because it defines persistent schemas locally (see T6-3).
- **Dead references:** 19 comments cite `REVIEW.md` / `REVIEW1.md`, and neither file exists (`trainer.rs:289,384,463`, `fno.rs:149,466,518`, `data_losses.rs:374`, `tests/run_artifacts_without_gnuplot.rs`, …).
- **Undocumented public items** (CLAUDE.md: "document every public item"):
  - `FNOConfig`, which has a `//` comment instead of a doc comment, and its fields
  - `FNO`, `FNOConfig::init` (which has two panics)
  - `identity`, `training_loop`, `TrainingComponents`, the fields of `EpochSums` and `TrainingConfig`
  - `LpLoss::new`, `LpLoss::forward`
- **Stale docs:**
  - `lib.rs:5-18` claims an `FNO<5>`, lists four modules and no Poisson examples.
  - `lib.rs:31`'s `compile_error!` offers rocm/vulkan, which are disabled.
  - `models/mod.rs:3-4` mentions a "shared model trait" that does not exist.
  - `spectral_convolution.rs:251` states the 4-multiply formula while the code uses Gauss's 3-matmul.
  - `normalizers.rs:60` says "above" for methods that are below.
  - `burgers_learner.rs:69` says "fewer epochs" for an identical count.
- **CI and manifest:**
  - `.github/workflows/run-examples.yml` runs a cold `cargo install` on every push and PR, and its run step is commented out.
  - `run-tests.yml` has about 30 lines of commented apt steps (MPI, CMake, …) duplicated across jobs.
  - CLAUDE.md's "CI runs exactly" list omits the `run-tests-chebyshev` job.
  - `Cargo.toml:8-9` points at Codeberg (the project is on GitHub).
  - `"scientific machine learning"` is not a crates.io category.
  - The two `dev-dependencies` target tables are empty, and the feature comment at :72-75 is truncated.
- **Library side effects:** there are `println!`s in loader code (burgers.rs:136, darcy.rs:173, poisson.rs:356) and in `training_loop`. Return counts and metrics, and let callers print.

---

## 6. Findings by tier

### Tier 1: structural regressions

None. No module is fighting the architecture, and the dependency graph is acyclic.

### Tier 2: missed dramatic simplifications (code-judo)

1. **`training/learner.rs` + `examples/train/burgers_learner.rs`**: keep, but tighten. *(Revised 2026-10-15 after discussion. The first draft said delete; the owner keeps it as the showcase of Burn's `Learner` (TUI, metric logging, summary) and as the template for future Learner use.)* It already reuses `build_training_components` and `flatten_pair`. The real costs are small:
   - The loss line `LpLoss::new(1, 2, Reduction::Sum).rel(...)` is duplicated from `train_epoch`, and a 50-line test exists to keep the two in sync.
   - It writes a cwd-relative, untimestamped `runs/burgers_fno_learner` with no `data_cfg.json` or `metrics.csv`, so `predict_burgers` cannot evaluate it.
   - It is not marked as a showcase, so Phase 5 work risks being built twice.
   - `burgers_learner.rs:69`'s comment says "fewer epochs" for the same 500.

   **Move:** one shared `pub fn` in `trainer.rs` for forward + `flatten_pair` + summed loss, called by both `train_epoch` and `learner_forward`. A timestamped run under `CARGO_MANIFEST_DIR` that includes `data_cfg.json` (later via `RunDir`, B1). A module-doc and CLAUDE.md line saying `train_epoch` is canonical and this is the Burgers-only Learner equivalent. Fix or lower the epoch count. Staying `FNO<3>`-only is fine for a demo; the module doc explains why `TrainStep` cannot be generic over `R` here.
2. **`training/trainers/{burgers,darcy,poisson}.rs`**: three files to choose between "no normalizer" and "decode with y". **Move:** one generic `train_fno` with `y_norm: Option<&UnitGaussianNormalizer>`, and delete the directory (S3).
3. **Run directory (S1)**: the biggest single lever. One `RunDir` type ends 48 literals, five copied loaders, three copied persistence tails, and the `.bpk` asymmetry.
4. **`data/batcher.rs` + `dataitem.rs:39-44` + `dataset.rs:48-67`**: a superseded host batching path. **Move:** move the predict examples to `device_loader`, keep `OperatorBatcher` only as the `#[cfg(test)]` oracle in `device_batcher.rs`, delete `DataItem`, and rename `dataitem.rs` → `host_float.rs`.
5. **`data/grids.rs`**: 361 lines with no production caller. **Move:** inline the roughly 25-line f64 oracle into `fno.rs` tests, delete the module, and update CONVENTIONS §2's last bullet.
6. **`data/transforms/normalizers.rs:12-21, 110-207`**: two dead normalizers and a one-implementor trait. **Move:** inherent `fit`/`encode`/`decode` on `UnitGaussianNormalizer`, and rename `mean_ref`/`std_ref`/`eps_val` → `mean`/`std`/`eps`. Optionally move `FlatDecoder` (Burn tensors) to `training/` so `transforms/` stays pure ndarray.
7. **`data/io/traits.rs`, `io/readers/npy.rs`**: an unused abstraction. **Move:** delete both and give the readers inherent `read`. As part of the same change, fix two latent bugs:
   - `NpzFileReader` re-opens the zip on every field read.
   - The f64→f32 fallback (`npz.rs:43-52`) reports a present field with the wrong dtype as `FieldNotFound`.
8. **Chebyshev twin stack (S2/S4)**: switch `pde/poisson/dataset.rs` to the ungated `cgl_nodes`, `clencurt` and `cc_rel_l2_error`. Make `cgl_nodes` public as `nodes`, and delete the copy in `poisson_plot.rs`. Demote `points.rs`, `quadrature.rs` and `norm.rs` to `#[cfg(all(test, feature = …))]` cross-check oracles. Generated `x`/`y` change by about 1 ulp, which the loader's `NODE_TOL = 1e-14` already accepts.
9. **`pde/poisson/sparse.rs`**: make it `#[cfg(test)] mod sparse;`. This removes about 237 prod lines and three public types that nothing outside its own tests uses.
10. **`data/loaders/darcy.rs:126-183` ≡ `loaders/poisson.rs:321-366`**: the shared tail (S3).
11. **`examples/predict/burgers_resolutions.rs` ≈ `burgers_superresolution.rs`**: merge into one example that takes its rates from argv. This also fixes the file-name/example-name mismatch (`burgers_resolutions.rs` registered as `burgers_resolution`).
12. **`tests/metal_diag.rs` (1,674 lines)**: keep the asserting regression set (`repro_fused_cat_broadcast(+_flex)`, `repro_cfft_bluestein(+_flex)`, `repro_fusion_cache_new_shape`, `repro_fused_cat_reused_at_new_shape`, `replay_2d_poisson`; `upstream/burn/*` cites these) in a `tests/gpu_parity.rs` of about 250 lines. Move the `*_flex` twins out of the GPU gate so CI runs them. Move the bisection, timing and env-var tests to `spikes/metal_diag/` or let git history keep them.
13. **`tests/rlst_smoke.rs`**: it tests a ruled-out Sylvester route and duplicates `sparse::cg_meets_tolerance`. Delete it.
14. **`data/loaders/base_dataset.rs`**: delete both traits and read `cfg.base.n_train` directly.
15. **`data/transforms/subsample.rs` + `data/split.rs`**: replace both with one rank-generic `take_strided(field, rows, step)` (via `slice_each_axis`) shared by Burgers and Darcy.

### Tier 3: spaghetti / branching

1. **`training/trainer.rs:48-53, 315-320`**: `Postprocess = dyn Fn(...)`, a public `identity`, and a `train_post`/`eval_post` pair threaded through three functions, all to express one fact: `y_train` is stored encoded and `y_test` is not (CONVENTIONS §10). Every caller must wire it correctly, and `examples/predict/darcy.rs:66-67` builds a third copy. **Move:** store `y_train` in physical units and decode only the prediction. The hook becomes one `Option<FlatDecoder>` applied to `out`, and the closures and `identity` disappear. *This changes CONVENTIONS §10 (f32 round-off only), so it needs your sign-off and a note in the PR.* A bolder variant puts the decoder inside the model so `load_fno`/`predict_host` return physical units.
2. **`training/trainer.rs:315-366` (`training_loop`)**: hard-coded `println!`, and metrics are only returned at the end. Phase 5 T2 (checkpoint/resume) would otherwise add `if epoch % k == 0` branches and `Option` parameters here. **Move, before Phase 5:** an `on_epoch: &mut dyn FnMut(&EpochMetrics, &FNO<R>)` (or a small observer trait). The default prints; a `RunDir` observer appends CSV and checkpoints.
3. **The `chebyshev` feature scattered through a mostly ungated module (S4).** It disappears once T2-8 is done and the feature is renamed.

### Tier 4: boundary / type-contract

1. **`trainer.rs:110-138` (`eval_epoch`)**: it returns a sum the caller must divide, and five call sites each rebuild `LpLoss::new(d, 2, Sum)` with a hard-coded `d`. **Move:** a public `evaluate` that returns the per-sample mean. Keep the sum-returning version private to `training_loop`.
2. **`models/run.rs:45` (`predict_host`)**: the one non-generic API in a rank-generic crate (`FNO<4>`, `Array3`). As a result `darcy_plot.rs:61-75` hand-rolls a one-sample forward. **Move:** `predict_host<const R>(&FNO<R>, ArrayViewD<f32>, batch, device) -> ArrayD<f64>`.
3. **`models/fno.rs:75-78`**: `init` asserts `out_channels == 1` for a trainer-only limitation that `flatten_pair` already enforces. Delete the model-side assert.
4. **`metrics/io.rs:82`**: `CARGO_MANIFEST_DIR` inside a library function means a crates.io consumer would write into the cargo registry. Take the root as a parameter (part of S1).
5. **`examples/predict/poisson.rs:47-111`**: the sidecar's reader schema (`Sidecar`, `SidecarError2`) is typed here, while its writer is an untyped `serde_json::json!` in `pde/poisson/dataset.rs:291`. The two halves are kept in sync by hand. **Move:** one typed `SidecarRecord` in the ungated `data::loaders::poisson`, serialised by the generator and loaded by the example.
6. **`data/loaders/poisson.rs:191-240`**: `field: &str` is checked at runtime for `"f" | "u"`, and `load_poisson_cheb` is an identity wrapper. **Move:** `enum PoissonField { F, U }`, one function.
7. **`data/loaders/{burgers.rs:117-130, darcy.rs:162-167, poisson.rs:345-350}`**: unreachable `LoadError::Invalid("reshape …")` arms on shapes built locally, and Burgers step 5 reshapes `[n,s]` to `[n,s]`. **Move:** `insert_axis` (infallible) and delete the no-op, so `Invalid` means only "the data on disk is wrong".
8. **`losses/data_losses.rs:20-107`**: `LpLoss::new(d, p, reduction)` takes a `d` that `rel` ignores, and `forward`, `abs`, `Mean` and `None` are unused in production. **Move (minimum):** drop `forward` and make `d` an argument of `abs`. It is a documented port (CONVENTIONS §11), so deeper pruning is your call.
9. **`pde/poisson/dataset.rs:157-164`**: `label_check` takes weights only to infer `n`. `Split.y` duplicates `x` by contract. Both go once T2-8 lands.

### Tier 5: file size / decomposition

1. **`models/fno.rs`: 1,569 lines (306 prod / 1,263 test).** Production is cohesive; the tests are the problem. **Move:**
   - Delete `reference_forward` (it is exactly `reference_forward_padded(.., Pad::Off)`).
   - Share the duplicated `perm_in`/`perm_out` closures.
   - Retire the `with_old_grid`/`old_grid_f64` migration shims if the grid tests can assert directly.
   - Split into `models/fno/{mod.rs, tests/{reference,padding,grid,checkpoint}.rs}`, grouped by the CONVENTIONS section each test verifies. No test file should exceed about 450 lines.
2. **`tests/metal_diag.rs`: 1,674 lines.** See T2-12.
3. **`data/loaders/poisson.rs`: 748 lines.** After T2-10 and moving the catalogue, split into `poisson/{set.rs, cheb.rs, uniform.rs}`, each under 250 prod lines.

### Tier 6: modularity

1. **The dataset catalogue lives in the loader.** `PoissonSet`/`STAGES`/`stage()` are experiment configuration that the gated generator imports from the ungated loader. `stage()` returns `LoadError` without loading anything. **Move:** give the catalogue its own module that both sides import, and an `Option` (or its own error) for `stage()`. Mirror it with `BurgersFiles`/`DarcyFiles` for the five hard-coded `.mat` paths in examples.
2. **Poisson evaluation policy in `examples/predict/poisson.rs:156-262`**: the worst sample, the error-2 cross-check, the boundary verdict and `PoissonErrorsRecord`. **Move:** `metrics::poisson::evaluate(...) -> PoissonErrorsRecord` with `Display`, so `poisson_resolution` reuses it and the verdict becomes unit-testable.
3. **Duplicate `TRANSFER_DEGREE` and `ErrorStats`, and the transfer recipe at four sites (S4).** Move them into `chebyshev::transfer` / one `ErrorStats`.
4. **`metrics/poisson.rs:207-231`**: hand-rolls Σ wᵢwⱼu². `cc_rel_l2_error` also recomputes `clencurt(n)` twice per sample when the caller already holds `w`. **Move:** a public `cc_norm_sq(v, wx, wy)` in `clencurt.rs`.
5. **Tensor-product A·F·Bᵀ written four ways** (`grf.rs:153`, `sparse.rs:176`, `collocation.rs:107,115`) next to the canonical `chebyshev::apply`. Route them through `apply`. As a follow-on, `collocation.rs` can convert its RLST `V` matrices to ndarray once after `eig`, which deletes `to_dyn`, `from_dyn` and `transpose`. The ndarray-vs-BLAS speed at n = 257 is unmeasured, so time it first.
6. **The blanket `#[allow]` at `lib.rs:48` (S5).**
7. **`run-examples.yml`**: delete it, or make it `cargo build --examples`.

### Tier 7: legibility

1. The 19 dead `REVIEW.md` citations should be replaced with the CONVENTIONS §n or PR number they stand for.
2. Undocumented public items (S5). Most disappear with T2-1/2 and T3-1, so document what survives.
3. `fno.rs:34-51, 176-251` restate CONVENTIONS §2 and §3 in prose without citing them. Replace the prose with "CONVENTIONS §n" plus the implementation-only notes (the Metal channels-last reason, the 1-ulp contract).
4. `spectral_convolution.rs` has five copies of `vec![a,b]; extend; try_into::<[_; R]>()` (lines 179-181, 281-283, 356-358, 368-370, 384-386). Replace them with one `with_bc(a, b, rest) -> [T; R]` helper. Also compute `flat_modes` locally rather than storing it and passing it, and fix the formula in the doc at :251.
5. `fno.rs:271-275, 298-302`: name the inline permutation closures `to_channels_first()` / `to_channels_last()` and reuse them in tests.
6. `lib.rs` crate docs, `Cargo.toml` metadata, the commented CI blocks, and CLAUDE.md's "Checks" list (S5).

---

## 7. Refactor roadmap

Each step is one PR on its own branch, behaviour-preserving unless marked **⚑ decision**, and verified by the CI five plus `cargo clippy --all-targets -- -D warnings`. Steps that touch the Chebyshev stack also need `cargo test --features chebyshev` and `cargo clippy --all-targets --features chebyshev -- -D warnings`. Steps are ordered by leverage: the most deleted complexity in the hottest code first. Steps in the same wave are independent of each other.

### Wave A: pure deletions (no design decisions, about 1,000+ lines gone)

| # | Step | Deletes | Proof it's safe | Depends on |
|---|---|---|---|---|
| A1 | **Keep and tighten the Learner showcase** (revised; see T2-1): shared loss step in `trainer.rs` used by `train_epoch` and `learner_forward`; timestamped run under `CARGO_MANIFEST_DIR` with `data_cfg.json`; "showcase, `train_epoch` is canonical" in the module doc and CLAUDE.md; fix the epochs comment | One duplicated loss definition; makes Learner runs evaluable | `train_step_gradients_match_hand_written_loop` and the trainer tests. `cargo clippy --examples`. Run `train_burgers_learner` briefly (needs a real terminal for the TUI) and evaluate its run with `predict_burgers`. | — |
| A2 | Collapse `trainers/` into one `train_fno<R,RM1,T>(…, y_norm: Option<&UnitGaussianNormalizer>, …)`. Move `train_poisson_smoke` alongside it. | ~170 | `cargo test` (the trainer and smoke tests). `cargo clippy --examples` proves the three train examples compile. | — |
| A3 | Dead data generality: `GaussianNormalizer`, `RangeNormalizer`, the `Normalizer` trait (→ inherent methods), `NpyFileReader`, `FieldReader`, `BaseDatasetConfig`/`HasBaseConfig` | ~450 incl. tests | `cargo test`, `cargo clippy --examples`. Grep shows zero external callers (verified). | — |
| A4 | Fold `data/grids.rs` into an `fno.rs` test oracle, and update CONVENTIONS §2's bullet (doc-only, no version bump) | ~330 | The `fno.rs` grid tests still pass against the inlined oracle. | — |
| A5 | `#[cfg(test)] mod sparse;`, delete `tests/rlst_smoke.rs` | ~240 prod + 103 | `cargo test --features chebyshev` (the sparse tests still run under cfg(test)). | — |
| A6 | Split `tests/metal_diag.rs` into `tests/gpu_parity.rs` (asserting repros, `*_flex` twins ungated) and move the rest to `spikes/` | ~1,400 out of `tests/` | `cargo test` now runs the flex twins. `cargo test --features metal --test gpu_parity` locally; **say whether Metal was run**. | — |
| A7 | Hygiene: delete `run-examples.yml` and the commented CI blocks, fix the `Cargo.toml` URLs/category/empty tables, update the CLAUDE.md "Checks" list, replace the 19 `REVIEW.md` citations | ~60 | CI green. `cargo doc --no-deps`. | — |

### Wave B: one owner per concept (the main structural lever)

| # | Step | Effect | Proof | Depends on |
|---|---|---|---|---|
| B1 | **`runs` module with `RunDir`** (S1). Move `metrics/io.rs`, `metrics/plots.rs`, `run::load_fno` and `EpochMetrics` into it. Take the root as a parameter and the configs as `&impl Config`. Migrate all train and predict examples. | Ends 48 literals, five inline loaders and three persistence tails. Removes `metrics → training/models` imports and the two-`metrics` clash. | `tests/run_artifacts_without_gnuplot.rs`, and the `metrics/io` round-trip tests moved along. **Add first:** a round-trip test, `RunDir::create` → `save_*` → `open` → `load_fno` gives bit-identical outputs, before migrating the examples. Then load one existing `runs/poisson_fno_*` directory with `predict_poisson` to prove the old runs still read. Migrate `burgers_learner` too. | A2 (A1 optional: if A1 lands first, its run-writing fix moves onto `RunDir` here) |
| B2 | One batching path: the predict examples use `device_loader`, `OperatorBatcher` becomes a test oracle, `DataItem` and the `Dataset` impl are deleted, `dataitem.rs` → `host_float.rs` | ~250 | The `device_batcher` bit-identity tests (already present). `predict_burgers`/`predict_darcy` print the same test L2 on an existing run. | B1 (examples touched once) |
| B3 | Shared loader tail: `loaders/common.rs` (`Normalizers`, `normalize_split_pair`), plus `insert_axis` instead of fallible reshapes, and no library `println!` | ~120. Removes `DarcyNormalizers`/`PoissonNormalizers` duplication. | `normalizer_asymmetry_matches_darcy` and the loader tests. **Add first:** a test asserting `load_darcy_uniform` output (shapes, a checksum of the first sample) is unchanged on a small synthetic `.mat`. | — |
| B4 | Generic `evaluate::<R,RM1>` (mean, owns `LpLoss`), plus rank-generic `predict_host<R>`. Merge the two Burgers resolution examples, and move `darcy_plot` to `predict_host`. | ~150. Five call sites stop dividing by hand. | The `eval_epoch` tests. Each Burgers/Darcy predict example reproduces its previous number on an existing run. | B1, B2 |
| B5 | Epoch observer in `training_loop` (default prints; a `RunDir` observer appends CSV) | **Prepares Phase 5 T2** without branching the loop | The trainer tests. `metrics.csv` written incrementally matches the end-of-run CSV. | B1 |

### Wave C: Chebyshev single stack (needs your decisions)

| # | Step | Effect | Proof | Depends on |
|---|---|---|---|---|
| C1 | `chebyshev::transfer` owns `TRANSFER_DEGREE`, `Transfers::new(n)` and a public `nodes`. `cc_norm_sq` in `clencurt`. One `ErrorStats`. Dataset, metrics, loader and `poisson_plot` all use them. | Removes four recipe copies, the duplicated constant and type, and the `cgl_nodes` copy in the example | The transfer and metrics tests, `error3_matches_lploss`, and `cargo test --features chebyshev`. | — |
| C2 | Generator onto the ungated `nodes`/`clencurt`/`cc_rel_l2_error`. Gated `points`/`quadrature`/`norm` become test oracles. **⚑ decision:** dataset `x`/`y` move by about 1 ulp (within `NODE_TOL`). Regenerate, or accept old files? | ~270 prod | `label_check` at 1e-8, the manufactured floor at 1e-12, `cc_rel_l2_matches_gated`. Regenerate stage 1 and diff the sidecar errors. | C1 |
| C3 | **⚑ decision (amends design decision 12):** closed-form `diff_matrix`/`diff2_matrix` (Trefethen `cheb`, negative-sum diagonal, D² = D·D). Drop `rlst/fftw`, `unit_coeffs` and the system libfftw3 (GPL) dependency. | Removes the FFTW dependency and the CI apt step | `residual_at_round_off`, the manufactured 1e-12 floor, the 1e-8 label check. Labels change at round-off, so regenerate and compare. | C2 |
| C4 | `chebyshev` module fully ungated, and the feature renamed (e.g. `datagen`) to gate only `pde/`. Fix `lib.rs`/CLAUDE.md/CI. | Removes all 14 cfg lines outside `pde/` | CI's chebyshev job with the renamed feature. | C3 (or C2 if C3 is declined: then only `differentiation.rs` stays gated) |

### Wave D: contracts that touch conventions (⚑ decisions)

| # | Step | Note |
|---|---|---|
| D1 | **⚑ CONVENTIONS §10:** store `y_train` physical, decode only the prediction. Delete `Postprocess`/`identity` and the closure pair. | f32 round-off change only. Per CLAUDE.md, propose it in the PR description; §10 is not in the version-bumping set (§1–§6, §12), but check that before merging. Best after A2 and B4, which reduce the callers to one. |
| D2 | `x_normalizer.json`: either read it in predict (and stop re-reading and re-transferring the train split), or stop writing it. | `poisson_resolution` must keep refitting on purpose. |
| D3 | Typed `SidecarRecord` shared by generator and reader. `PoissonErrorsRecord` and `evaluate` move into `metrics::poisson`. Scope the `Config` allows and delete `lib.rs:48`'s blanket allow. | The sidecar's JSON shape must stay byte-compatible with existing datasets. **Add first:** a test that parses an existing sidecar fixture. |

### Wave E: decomposition and polish (any time, low risk)

- **E1:** split the `fno.rs` tests into `models/fno/tests/*` and delete `reference_forward`. The test count must be unchanged; compare the "running N tests" line before and after.
- **E2:** the spectral-conv `with_bc` helper and the doc formula fix. Remove `flat_modes` as stored state.
- **E3:** split `loaders/poisson.rs` into `set`/`cheb`/`uniform`, and add `enum PoissonField`.
- **E4:** a `take_strided` helper; delete `subsample.rs` and `split.rs`.
- **E5:** rustdoc for the public items that survive, the `lib.rs` crate docs, and the `LpLoss` `d` cleanup.

**Recommended start:** A2 → B1, with A1 alongside (it is small and independent). Together they give one loss definition, one trainer entry point and one owner of the run format, which are exactly the surfaces Phase 5 (checkpointing, multi-GPU) would otherwise build on. Wave C is the natural opening of Phase 6. C2 and C3 need your decisions, since C3 amends a signed-off design decision.

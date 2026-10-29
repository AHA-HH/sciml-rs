# sciml-rs

Neural operators in Rust, on [Burn](https://burn.dev). One crate: package `sciml-rs`,
library `sciml_rs`. The Fourier Neural Operator (FNO) is the model implemented so far.
Tensor rank is a compile-time parameter, so the spatial dimension D is fixed by the type:
`FNO<3>` with `modes: vec![16]` is 1D, `FNO<4>` with `modes: vec![12, 12]` is 2D (R = D + 2).

## Source of truth
- `docs/CONVENTIONS.md` defines tensor layouts, coordinate grids, padding, FFT
  normalisation, mode truncation and initialisation. Never change a convention in code;
  propose the change in the PR description and bump `CONVENTION_VERSION` there.
- Cite conventions in doc comments as `CONVENTIONS §n`.
- Current phase: Phase 5, grid-decoupled operator learning (docs/phase5/README.md,
  briefs docs/phase5/T1–T7; design §13.3, decision 30 and its 2026-10-26 planning
  amendment). Phases 0–4 are done (docs/phase0/README.md to
  docs/phase4/README.md), and so is Phase 4s (side phase, repository health,
  docs/phase4s/README.md; design §10, decision 25), closed 2026-10-24 with re-review
  grade B (docs/code-quality/thermo-nuclear-repo-review-2026-10-24.md).
  Next (design §13, decision 30, which supersedes decision 24's Phase 5/6 split):
  Phase 7, sample efficiency. Phase 6 (SVD/POD information content, beside Phase 5) is done
  2026-10-27: docs/phase6/README.md, results docs/phase6/results.md. Phase 8 (θ-map) is optional; Phase 9 (HPC: profiling, checkpointing,
  multi-GPU) is a separate systems track. Their 4s prerequisites (T10/T12 for Phase 9
  T2, T16 for Phase 8) are merged. Each phase gets its README and briefs when it
  starts; 4s briefs are in docs/phase4s/T*.md.
- Background: docs/design/2d-chebyshev-poisson-fno.md (2D Poisson on a Chebyshev grid,
  RLST reference solver, FNO; signed off 2026-10-05, amended 2026-10-08 and 2026-10-09,
  decisions in its §12). CONVENTIONS.md wins on any conflict.
- Roadmap: design §10 (every phase's goal, ordered tasks, dependencies and exit).

## Working agreement
- The author (AHA-HH) is the sole approver of designs, phase plans, briefs and decisions.
  Never ask for or wait on supervisor approval; record approvals as the author's.
- Approval for one action is not approval for the next one of its kind. A request to
  commit, push, delete, publish or open a PR covers the message it appears in and
  nothing after it; ask again, even when the next action looks like an obvious
  continuation.
- Default flow: make the change, run the checks, report exactly which ones ran and what
  they said, and leave the result uncommitted unless the current message asks for a
  commit. Never report an unrun check as passing; say plainly when something could not
  run (no GPU backend, no dataset, no gnuplot).
- One task per branch and PR; stay inside the modules the task names.
- Task numbers within a phase are execution order, from Phase 1 on (design §10). Phase 0
  keeps its signed-off numbers and runs T2, then T3 and T4, then T1.
- Keep changes targeted: no drive-by reformatting or refactors.

## Working rules
- Run `cargo fmt` after every Rust edit, before the other checks or committing. CI
  rejects unformatted code.
- Document every public item; follow the existing rustdoc style (shape of every tensor
  argument, and the panics). `lib.rs` sets `#![warn(missing_docs)]`, so clippy's
  `-D warnings` rejects an undocumented public item.

## Layout
- `src/neural_operators/`: `models/fno/` (`mod.rs`: FNOConfig, FNO; tests in `tests/{reference,padding,grid,checkpoint}.rs`), `models/predict.rs`
  (`predict_host`: batched inference to host f64, any rank), `layers/spectral_convolution.rs`
  (SpectralConv, SpectralInit), `losses/` (LpLoss), `data/` (loaders, `device_loader`, io for
  npz/.mat, transforms: normalizers and `take_strided`; `host_oracle` is the test-only host
  batcher), `metrics/` (Poisson
  errors, `ErrorStats`; `pod`, behind `datagen`: `Pod`, the weighted POD of a snapshot set, by LAPACK `gesvd`
  since Accelerate's `gesdd` returned wrong singular values), `utils/fft.rs` (icfft_full_spectrum), `utils/backend.rs`
  (`backend_features`, `git_commit`).
- `training/`: `config` (`TrainingConfig`), `step` (`flatten_pair`, `forward_loss`,
  `evaluate`: the per-sample mean test loss), `trainer` (components, `train_epoch`,
  `training_loop` with its `EpochObserver`, `train_fno`), `learner` (Burn `Learner` impls).
- `runs` (`RunDir`, `EpochMetrics`, `RunCosts`) owns the run directory:
  `create(root, name)`, `save_configs`, `append_metrics` (the only `metrics.csv` writer,
  one row per epoch, with `epoch_time_s` since Phase 5 T4; old five-column files still
  read), `save_plot`, `save_normalizers`, `save_weights`, `save_costs` (`costs.json`,
  Poisson runs since Phase 5 T4); `open(dir)`, `load_fno`, `model_config`,
  `training_config`, `data_config::<C>`, `normalizers()`, `costs()`, `metrics()`. Run-file names appear
  nowhere else; examples pass `CARGO_MANIFEST_DIR/runs` as the root.
- `src/neural_operators/chebyshev/` builds without features, except `diff_matrix`/
  `diff2_matrix`, behind the `datagen` feature (need system libfftw3; design §12, decision
  27). `points` and `quadrature` (RLST nodes, `clenshaw_curtis`) are
  `#[cfg(all(test, feature = "datagen"))]` oracles for the closed forms.
- `chebyshev/transfer` is ungated (ndarray only): `nodes` (closed-form CGL nodes),
  `uniform_nodes`, `cheb_to_uniform` (T_cu), `uniform_to_cheb` (T_uc, Floater–Hormann),
  `apply`, `Transfers`.
- `chebyshev/clencurt` is ungated (ndarray only): `clencurt` (closed-form Clenshaw–Curtis
  weights, Trefethen) and `cc_rel_l2_error` (relative CC-L² error on the CGL tensor grid).
- `src/neural_operators/pde/poisson/` (behind `datagen`): `collocation`
  (`CollocationSolver`, solver A, fast diagonalisation); `sparse` (`SparseSolver`, solver C,
  Q1 + unpreconditioned CG, validation oracle, test-only); `manufactured` (test-only fixtures).
- `pde/poisson/grf` (GRF forcings, ChaCha8 draws, exact sine series) and `pde/poisson/dataset`
  (`generate`, `label_check`, `write_split`), driven by `examples/generate/poisson.rs`
  (`generate_poisson`), which writes `datasets/poisson/`.
- `data/loaders/common` holds the loader tail shared by Darcy and Poisson: `Normalizers { x, y }`
  and `normalize_split_pair` (fit, encode, channel axis, cast; CONVENTIONS §10).
- `data/catalogue` (ungated, loads nothing) names the dataset files: `BurgersFiles`,
  `DarcyFiles`, and `PoissonSet { n, k }` for a Poisson dataset's split files
  (`poisson_n{n}_K{K}_{split}`); `STAGES` / `stage(n)` are the training stages and
  `GENERATED` the sets `generate_poisson` writes. The loaders' callers and the generator
  take file names from here, not literals.
- `data/loaders/poisson/` is ungated: `mod.rs` (`PoissonConfig` with the learning grid `s`,
  unset meaning n − 1, and since Phase 5 T4 K, d and the split files with their sidecar
  commits; `SidecarRecord`), `uniform` (`load_poisson_uniform` reads those
  `.npz` files and applies T_cu at load time, giving Darcy-shaped datasets on the uniform
  grid of s points, so one dataset serves every s) and `cheb`
  (`load_poisson_cheb_field` reads one split's raw field, `PoissonField::F` or `U`, on the
  CGL grid, untransferred).
- `metrics/poisson` is ungated (ndarray, f64): `poisson_errors` gives design §7's errors 2–5
  on the Chebyshev grid with the run's `Transfers` (any s since Phase 5 T6: error 2 is the
  transfer-only floor, 3 the coarse error, 4 the fine-grid error) and `check_error3` the
  cross-check against the run's metrics, driven by `examples/predict/poisson.rs`
  (`predict_poisson`), which writes `errors.json` (`ERRORS_JSON`) into the run directory;
  `poisson_matrix` tabulates evaluated runs.
- `training/trainer` (`train_fno`, one trainer for every PDE: `Some(y)` decodes with the
  target normaliser, `None` trains in physical units), driven on those datasets by
  `examples/train/poisson.rs` (`train_poisson`), which writes `runs/poisson_fno_p{p}_*/`.
- `spikes/`: standalone packages, never built by CI: Phase 0 measurements, and
  `metal_diag` (op-by-op GPU vs Flex diagnostics behind `upstream/burn/*`; a non-default
  workspace member sharing the root's lock and Burn pin: `cargo test -p metal-diag-spike`).
- Unit tests sit in `#[cfg(test)] mod tests` in each file. `tests/spectral_init.rs` reseeds
  Flex's process-wide RNG and `tests/run_artifacts_without_gnuplot.rs` clears `PATH`; each
  needs its own test binary, so keep such tests out of `src/`. `tests/gpu_parity.rs` holds
  the Burn bug repros and the 2D GPU replay check; only its `*_flex` twins run without a
  GPU feature.
- Examples are registered in `Cargo.toml`; hyperparameters are literals at the top of
  each file. Training writes `runs/<name>_<unix_secs>/`.
- `datasets/` (.mat files and `datasets/poisson/`, see `datasets/README.md`) and `runs/` are
  git-ignored.
- `Cargo.lock` is committed: CI builds from it, so a yanked dependency (e.g. `zip 6.0.0`)
  cannot break a fresh resolve. `.gitignore` covers `*.csv` and `*.png`.

## Build environment
Stable Rust, edition 2024 (Rust 1.85+), with `rustfmt` and `clippy`. `gnuplot` is
optional: plots are skipped with a warning when it is missing. Examples need the datasets;
tests do not. `.cargo/config.toml` links with `lld` on aarch64 Linux. The optional
`datagen` feature (RLST, the reference solvers in `pde/`, `generate_poisson`) needs a system
libfftw3, and OpenBLAS on Linux (Accelerate on macOS).

## Checks
CI (GitHub Actions, `.github/workflows/run-tests.yml`, pushes and pull requests to `main`)
has three jobs:
```sh
# run-tests
cargo test
cargo doc --no-deps
# style-checks
cargo fmt -- --check
cargo clippy --no-deps -- -D warnings
cargo clippy --no-deps --examples -- -D warnings
# run-tests-datagen (installs OpenBLAS, FFTW and libclang first)
cargo clippy --no-deps --features datagen --all-targets -- -D warnings
cargo test --features datagen
```
CI compiles the examples but never runs them (they need `datasets/` and `runs/`).
Before finishing any task, also run `cargo clippy --all-targets -- -D warnings`. Keep
clippy clean without blanket `#[allow]`s. For backend-sensitive changes, also run the
relevant tests with `--features metal` and say whether it was run.

GPU parity, for backend-sensitive changes: `cargo test --features metal --test gpu_parity`.
Its GPU `repro_*` tests are `#[ignore]`d while the Burn bugs remain; `-- --ignored` shows
them failing.

An exact test filter needs the full module path:
`cargo test neural_operators::models::fno::tests::<group>::<name> -- --exact`. A shorter path runs
0 tests and still reports `ok`; check the "running N tests" line.

## Burn
- Burn is a fork pinned in `Cargo.toml` (`[workspace.dependencies]`): `ax1s-x1zz/burn` at rev `ddfa9af`, features
  `autodiff, signal, store, train`. docs.rs describes upstream, not this.
- `signal` provides `rfft`, `irfft` and a forward `cfft` but no inverse complex FFT;
  `utils/fft.rs::icfft_full_spectrum` supplies it. Non-power-of-two lengths go through
  Bluestein, so any grid size works.
- Backends are additive cargo features: `flex` (default, pure-Rust CPU), `metal`, `cuda`,
  `wgpu`. `Device::default()` picks CUDA > Metal > ROCm > Vulkan > wgpu > Flex;
  `BURN_DEVICE` overrides at runtime. `rocm`/`vulkan` are disabled, and `ndarray` is
  unusable (no FFT). Building with no backend feature is a `compile_error!`.
- Metal: a channels-first `cat` after a permute zeroed grid entries, so `FNO::grid_cl`
  builds the grid channels-last (regression test in `models/fno/tests/grid.rs`).
- `device_loader(&data, &device, batch_size, shuffle)` uploads a split once and batches it
  on `device`: pass the `.autodiff()` device for training, or no gradients flow. The trainer
  evaluates on `device.inner()`; predict examples pass their plain `Device::default()`.
  `data::host_oracle::OperatorBatcher` (host batching) is a test-only oracle for its
  bit-identity tests.

## Navigating the code
Prefer LSP tools (go-to-definition, find-references) over `grep` for symbols. Use `grep`
for comments, CI YAML and `Cargo.toml`. Burn's sources are in
`~/.cargo/git/checkouts/burn-*/ddfa9af/` (FFTs: `crates/burn-signal/src/functions/fft.rs`);
other checkouts there are different revisions. Read the implementation and tests rather
than trusting prose when checking how an API behaves.

## Commands
- All tests: `cargo test`
- Tests with the reference solvers and RLST oracles: `cargo test --features datagen`
- Generate the Poisson datasets: `cargo run --release --features datagen --example generate_poisson`
  (writes `datasets/poisson/`)
- Train: `cargo run --release --example train_burgers [--features metal]`
  (also `train_darcy`)
- `train_burgers_learner` is a showcase of Burn's `Learner` and needs a real terminal for its
  TUI: `train_epoch` is the canonical loop; this is the Burgers-only (`FNO<3>`) Learner
  equivalent. It writes `runs/burgers_fno_learner_<unix_secs>/` (configs and weights, no
  `metrics.csv`), which `predict_burgers` evaluates.
- Train on Poisson: `cargo run --release --example train_poisson [--features metal] [-- [<padding>] [<n>] [<s>] [<seed>]]`
  (the first argument overrides `PADDING`, the second picks the stage, n ∈ {65, 129, 257},
  default 65, the third the learning grid s, default n − 1, with modes min(12, s/2), the
  fourth the seed, default 42; needs that stage in `datasets/poisson/`; `predict_poisson`
  evaluates any s, `poisson_resolution` and `poisson_plot` only s = n − 1). On cuda:
  `--no-default-features --features cuda`
- Evaluate a run: `cargo run --release --example predict_burgers -- runs/<run_dir>`
  (also `burgers_resolution -- runs/<run_dir> [<rate> ...]` on Burgers runs, s = 8192 / rate;
  `predict_darcy`, `darcy_plot` on Darcy runs)
- Evaluate a Poisson run (the five errors, the boundary verdict, the worst sample):
  `cargo run --release --example predict_poisson [--features metal] -- runs/<run_dir>`
- Phase 5 comparison table and plots from evaluated Poisson runs (mean and min–max over
  seeds per (padding, s); plots `<stem>_*.png` beside the table, skipped without gnuplot):
  `cargo run --release --example poisson_matrix -- docs/phase5/results.md runs/<run_dir> ...`
- Resolution study (errors 3 and 4 of a Poisson run on finer grids at the run's K, weights
  unchanged, normalisers refit on that n's `_train` split):
  `cargo run --release --example poisson_resolution [--features metal] -- runs/<run_dir> <n> [<n> ...]`
- POD of the stage 3 f and u, on the CGL grid and the learning grids (Phase 6; tables to
  `<out.md>`, plots beside it):
  `cargo run --release --features datagen --example poisson_pod -- docs/phase6/results.md`
- Plot one Poisson test sample (f, u, û, û − u on the CGL nodes, on the run's stage; writes
  `prediction_<sample>.png`):
  `cargo run --release --example poisson_plot [--features metal] -- runs/<run_dir> [<sample>]`

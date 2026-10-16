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
- Current phase: Phase 4s (side phase, repository health), planned in
  docs/phase4s/README.md (design §10, decision 25; from the review in
  docs/code-quality/). Phases 0–4 are done: docs/phase0/README.md to
  docs/phase4/README.md. After 4s (design §10, decision 24), both optional and
  independent: Phase 5 (HPC: profiling, checkpointing, multi-GPU) and Phase 6
  (Chebyshev-native: baseline padding and seeds first, then the θ-map). Phase 5 T2
  waits for 4s T10/T12, Phase 6 T3 for 4s T16. Each phase gets its README and briefs
  when it starts; 4s briefs are in docs/phase4s/T*.md.
- Background: docs/design/2d-chebyshev-poisson-fno.md (2D Poisson on a Chebyshev grid,
  RLST reference solver, FNO; signed off 2026-10-05, amended 2026-10-08 and 2026-10-09,
  decisions in its §12). CONVENTIONS.md wins on any conflict.
- Roadmap: design §10 (every phase's goal, ordered tasks, dependencies and exit).

## Working agreement
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
  argument, and the panics).

## Layout
- `src/neural_operators/`: `models/fno.rs` (FNOConfig, FNO), `models/run.rs` (`load_fno`,
  `predict_host`: a saved run's model, batched inference to host f64),
  `layers/spectral_convolution.rs` (SpectralConv, SpectralInit), `losses/` (LpLoss), `data/`
  (loaders, batchers, io for npy/npz/.mat, transforms), `training/`, `metrics/` (run
  artifacts, gnuplot), `utils/fft.rs` (icfft_full_spectrum), `utils/backend.rs`
  (`backend_features`).
- `src/neural_operators/chebyshev/`: CGL `nodes`, `diff_matrix`/`diff2_matrix`,
  `clenshaw_curtis`, `l2_norm`/`rel_l2_error` are behind the `chebyshev` feature (need
  system libfftw3).
- `chebyshev/transfer` is ungated (ndarray only): `uniform_nodes`, `cheb_to_uniform` (T_cu),
  `uniform_to_cheb` (T_uc, Floater–Hormann), `apply`.
- `chebyshev/clencurt` is ungated (ndarray only): `clencurt` (closed-form Clenshaw–Curtis
  weights, Trefethen) and `cc_rel_l2_error` (relative CC-L² error on the CGL tensor grid).
- `src/neural_operators/pde/poisson/` (behind `chebyshev`): `collocation`
  (`CollocationSolver`, solver A, fast diagonalisation); `sparse` (`SparseSolver`, solver C,
  Q1 + unpreconditioned CG, validation oracle only); `manufactured` (test-only fixtures).
- `pde/poisson/grf` (GRF forcings, ChaCha8 draws, exact sine series) and `pde/poisson/dataset`
  (`generate`, `label_check`, `write_split`), driven by `examples/generate/poisson.rs`
  (`generate_poisson`), which writes `datasets/poisson/`.
- `data/loaders/poisson` is ungated: `load_poisson_uniform` reads those `.npz` files and
  applies T_cu at load time, giving Darcy-shaped datasets on the uniform grid s = n − 1;
  `load_poisson_cheb` reads one split's raw `u` on the CGL grid, untransferred, and
  `load_poisson_cheb_field` either raw field, `f` or `u`. `PoissonSet { n, k }` names a
  dataset's split files (`poisson_n{n}_K{K}_{split}`) and `STAGES` / `stage(n)` are the
  training stages; examples take file names from these, not literals.
- `metrics/poisson` is ungated (ndarray, f64): `poisson_errors` gives design §7's errors 2–5
  on the Chebyshev grid and `check_error3` the cross-check against `metrics.csv`, driven by
  `examples/predict/poisson.rs` (`predict_poisson`), which writes `errors.json` into the
  run directory.
- `training/trainers/poisson` (`train_poisson`, the Darcy trainer on those datasets), driven
  by `examples/train/poisson.rs` (`train_poisson`), which writes `runs/poisson_fno_p{p}_*/`.
- `spikes/`: standalone packages for Phase 0 measurements; never built by CI.
- Unit tests sit in `#[cfg(test)] mod tests` in each file. `tests/spectral_init.rs` reseeds
  Flex's process-wide RNG and `tests/run_artifacts_without_gnuplot.rs` clears `PATH`; each
  needs its own test binary, so keep such tests out of `src/`.
- Examples are registered in `Cargo.toml`; hyperparameters are literals at the top of
  each file. Training writes `runs/<name>_<unix_secs>/`.
- `datasets/` (.mat files, see `datasets/README.md`) and `runs/` are git-ignored.
- `Cargo.lock` is committed: CI builds from it, so a yanked dependency (e.g. `zip 6.0.0`)
  cannot break a fresh resolve. `.gitignore` covers `*.csv` and `*.png`.

## Build environment
Stable Rust, edition 2024 (Rust 1.85+), with `rustfmt` and `clippy`. `gnuplot` is
optional: plots are skipped with a warning when it is missing. Examples need the datasets;
tests do not. `.cargo/config.toml` links with `lld` on aarch64 Linux.

## Checks
CI (GitHub Actions, `.github/workflows/run-tests.yml`, pushes and pull requests to `main`)
runs exactly:
```sh
cargo fmt -- --check
cargo clippy --no-deps -- -D warnings
cargo clippy --no-deps --examples -- -D warnings
cargo test
cargo doc --no-deps
```
Before finishing any task, also run `cargo clippy --all-targets -- -D warnings`. Keep
clippy clean without blanket `#[allow]`s. For backend-sensitive changes, also run the
relevant tests with `--features metal` and say whether it was run.

An exact test filter needs the full module path:
`cargo test neural_operators::models::fno::tests::<name> -- --exact`. A shorter path runs
0 tests and still reports `ok`; check the "running N tests" line.

## Burn
- Burn is a fork pinned in `Cargo.toml`: `ax1s-x1zz/burn` at rev `ddfa9af`, features
  `autodiff, signal, store, train`. docs.rs describes upstream, not this.
- `signal` provides `rfft`, `irfft` and a forward `cfft` but no inverse complex FFT;
  `utils/fft.rs::icfft_full_spectrum` supplies it. Non-power-of-two lengths go through
  Bluestein, so any grid size works.
- Backends are additive cargo features: `flex` (default, pure-Rust CPU), `metal`, `cuda`,
  `wgpu`. `Device::default()` picks CUDA > Metal > ROCm > Vulkan > wgpu > Flex;
  `BURN_DEVICE` overrides at runtime. `rocm`/`vulkan` are disabled, and `ndarray` is
  unusable (no FFT). Building with no backend feature is a `compile_error!`.
- Metal: a channels-first `cat` after a permute zeroed grid entries, so `FNO::grid_cl`
  builds the grid channels-last (regression test in `fno.rs`).
- `OperatorBatcher` does not choose a device: a DataLoader without `set_device` puts
  batches on non-autodiff `Device::default()`, so training gets no gradients unless the
  device is `.autodiff()`. Evaluation runs on `device.inner()`.

## Navigating the code
Prefer LSP tools (go-to-definition, find-references) over `grep` for symbols. Use `grep`
for comments, CI YAML and `Cargo.toml`. Burn's sources are in
`~/.cargo/git/checkouts/burn-*/ddfa9af/` (FFTs: `crates/burn-signal/src/functions/fft.rs`);
other checkouts there are different revisions. Read the implementation and tests rather
than trusting prose when checking how an API behaves.

## Commands
- All tests: `cargo test`
- Train: `cargo run --release --example train_burgers [--features metal]`
  (also `train_darcy`; `train_burgers_learner` needs a real terminal for Burn's TUI)
- Train on Poisson: `cargo run --release --example train_poisson [--features metal] [-- [<padding>] [<n>]]`
  (the first argument overrides `PADDING`, the second picks the stage, n ∈ {65, 129, 257},
  default 65; needs that stage in `datasets/poisson/`). On cuda:
  `--no-default-features --features cuda`
- Evaluate a run: `cargo run --release --example predict_burgers -- runs/<run_dir>`
  (also `burgers_resolution`, `burgers_superresolution` on Burgers runs;
  `predict_darcy`, `darcy_plot` on Darcy runs)
- Evaluate a Poisson run (the five errors, the boundary verdict, the worst sample):
  `cargo run --release --example predict_poisson [--features metal] -- runs/<run_dir>`
- Resolution study (errors 3 and 4 of a Poisson run on finer grids at the run's K, weights
  unchanged, normalisers refit on that n's `_train` split):
  `cargo run --release --example poisson_resolution [--features metal] -- runs/<run_dir> <n> [<n> ...]`
- Plot one Poisson test sample (f, u, û, û − u on the CGL nodes, on the run's stage; writes
  `prediction_<sample>.png`):
  `cargo run --release --example poisson_plot [--features metal] -- runs/<run_dir> [<sample>]`

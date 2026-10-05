# sciml-rs

Neural operators in Rust, on [Burn](https://burn.dev). One crate: package `sciml-rs`,
library `sciml_rs`. The Fourier Neural Operator (FNO) is the model implemented so far.
Tensor rank is a compile-time parameter, so the spatial dimension D is fixed by the type:
`FNO<3>` with `modes: vec![16]` is 1D, `FNO<4>` with `modes: vec![12, 12]` is 2D (R = D + 2).

## Source of truth
- `docs/CONVENTIONS.md` (not yet written) will define tensor layouts, FFT normalisation,
  mode truncation, grid channels, padding and initialisation. Until it exists, the code
  and its doc comments are the reference (`FNO::forward`, `FNO::grid_cl`,
  `SpectralConv::check_modes_fit`, `SpectralInit`); do not change any of them silently.

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
- Keep changes targeted: no drive-by reformatting or refactors.

## Working rules
- Run `cargo fmt` after every Rust edit, before the other checks or committing. CI
  rejects unformatted code.
- Document every public item; follow the existing rustdoc style (shape of every tensor
  argument, and the panics).

## Layout
- `src/neural_operators/`: `models/fno.rs` (FNOConfig, FNO), `layers/spectral_convolution.rs`
  (SpectralConv, SpectralInit), `losses/` (LpLoss), `data/` (loaders, batchers, io for
  npy/npz/.mat, transforms), `training/`, `metrics/` (run artifacts, gnuplot),
  `utils/fft.rs` (icfft_full_spectrum).
- Unit tests sit in `#[cfg(test)] mod tests` in each file. `tests/spectral_init.rs` reseeds
  Flex's process-wide RNG and `tests/run_artifacts_without_gnuplot.rs` clears `PATH`; each
  needs its own test binary, so keep such tests out of `src/`.
- Examples are registered in `Cargo.toml`; hyperparameters are literals at the top of
  each file. Training writes `runs/<name>_<unix_secs>/`.
- `datasets/` (.mat files, see `datasets/README.md`) and `runs/` are git-ignored.
- `Cargo.lock` is git-ignored (library crate). `.gitignore` covers `*.csv` and `*.png`.

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
- Evaluate a run: `cargo run --release --example predict_burgers -- runs/<run_dir>`
  (also `burgers_resolution`, `burgers_superresolution` on Burgers runs;
  `predict_darcy`, `darcy_plot` on Darcy runs)

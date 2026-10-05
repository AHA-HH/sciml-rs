# sciml-rs

Neural operators in Rust, on [Burn](https://burn.dev). One crate (`sciml_rs`); the
Fourier Neural Operator (FNO) is the model implemented so far. Tensor rank is a
compile-time parameter, so the spatial dimension D is fixed by the type: `FNO<3>` with
`modes: vec![16]` is 1D, `FNO<4>` with `modes: vec![12, 12]` is 2D (R = D + 2).

## Source of truth
- Conventions file: `docs/CONVENTIONS.md`. It defines tensor layouts, FFT normalisation, mode truncation,
  grid channels, padding and initialisation. Never change a convention in code;
  propose the change in the PR description and bump `CONVENTION_VERSION` there.
- Cite conventions in doc comments as `CONVENTIONS §n`.
- Design documents: `docs/design/<objective>.md`. Phase and task briefs:
  `docs/<objective>/phase<N>/README.md` and `T<k>-<name>.md`. `CONVENTIONS.md` wins on
  any conflict with a design document.
- Active objectives and their current phase: none yet; the single design document is
  being written.

## Working agreement
- Approval for one action is not approval for the next one of its kind. A request to
  commit, push, delete, publish or open a PR covers the message it appears in and
  nothing after it; ask again, even when the next action looks like an obvious
  continuation.
- Default flow: make the change, run the checks, report exactly which ones ran and what
  they said, and leave the result uncommitted unless the current message asks for a
  commit. Never report an unrun check as passing; say plainly when something could not
  run (no GPU backend, no dataset, no gnuplot).
- One task per branch and PR; stay inside the modules the task brief names.
- Keep changes targeted: no drive-by reformatting or refactors.
- Each objective lives on its own branch and worktree. Do not edit another objective's
  `docs/<objective>/` tree or design document from a worktree that is not its own.

## Working rules
- Run `cargo fmt` after every Rust edit, before the other checks or committing. CI
  rejects unformatted code.
- Tests first: write the acceptance tests from the task brief, then the implementation.
- Every fast or new path is tested against a slower trusted one: a naive DFT or einsum
  in the test, a published PyTorch reference value, or the existing implementation it
  replaces. State the tolerance and what it is relative to.
- Numeric changes must hold on the default `flex` backend; state whether other backends
  were run. Tolerances are f32 unless a test says otherwise.
- Saved checkpoints must keep loading: do not rename or reshape parameters, and add
  new config fields as `Option<_>` so old configs still deserialise (see
  `FNOConfig::spectral_init`, `FNOConfig::padding`).
- Document every public item; follow the existing rustdoc style (explain the shape of
  every tensor argument, and the panics).

## Burn
- Burn is a **fork**, pinned in `Cargo.toml`: `ax1s-x1zz/burn` at rev `ddfa9af`, features
  `autodiff, signal, store, train`. docs.rs describes upstream, not this; read the
  sources in `~/.cargo/git/checkouts/burn-*/ddfa9af/` (FFTs:
  `crates/burn-signal/src/functions/fft.rs`).
- Burn provides `signal::rfft`, `irfft` and a forward `cfft`, but no inverse complex
  FFT; `utils/fft.rs::icfft_full_spectrum` supplies it. Non-power-of-two lengths go
  through Bluestein, so any grid size works.
- Backends are cargo features: `flex` (default, pure-Rust CPU), `metal`, `cuda`, `wgpu`.
  They are additive; `Device::default()` picks CUDA > Metal > ROCm > Vulkan > wgpu >
  Flex, and `BURN_DEVICE` overrides at runtime. The `ndarray` backend is not usable
  (no FFT).
- A known Metal quirk: a channels-first `cat` after a permute zeroed grid entries, which
  is why `FNO::grid_cl` builds the grid channels-last. Prefer the existing data flow over
  re-layouts on hot paths.

## Layout
```
src/neural_operators/
  data/        loaders (burgers, darcy), batchers, dataset, split, grids,
               io (npy/npz/.mat readers), transforms (normalizers, subsample)
  layers/      spectral_convolution.rs (SpectralConv, SpectralInit)
  models/      fno.rs (FNOConfig, FNO)
  losses/      data_losses.rs (LpLoss, Reduction)
  training/    trainer, learner, per-problem trainers, metrics
  metrics/     run artifacts on disk, gnuplot plots
  utils/       fft.rs (icfft_full_spectrum)
examples/      train/ and predict/ entry points; hyperparameters are literals at the top
tests/         integration tests
datasets/      .mat files, not committed (datasets/README.md)
runs/          training output, not committed
docs/          CONVENTIONS.md, design/, <objective>/phase<N>/
```
- `Cargo.lock` is not committed (library crate).
- `.gitignore` covers `*.csv` and `*.png`, so run artifacts stay out.

## Build environment
Stable Rust, edition 2024 (README: Rust 1.85+), with `rustfmt` and `clippy`. `gnuplot` is
optional: plots are skipped when it is missing (`tests/run_artifacts_without_gnuplot.rs`).
Examples need the datasets in `datasets/`; unit tests do not.

## Checks
CI (GitHub Actions, `.github/workflows/run-tests.yml`, pushes and PRs to `main`) runs
exactly:

```sh
cargo fmt -- --check
cargo clippy --no-deps -- -D warnings
cargo clippy --no-deps --examples -- -D warnings
cargo test
cargo doc --no-deps
```

Before finishing any task, also run `cargo clippy --all-targets -- -D warnings`. Keep
clippy clean without blanket `#[allow]`s. For backend-sensitive changes, also run the
relevant tests with `--features metal` (Apple) and say whether it was run.

## Gotchas
- An exact test filter needs the full module path:
  `cargo test neural_operators::models::fno::tests::<name> -- --exact`. A short path
  such as `models::fno::tests::<name> --exact` runs 0 tests and still reports `ok`;
  check the "running N tests" line.
- `runs/` fills up with every training example; it is git-ignored, do not commit it.

## Navigating the code
Prefer LSP tools (go-to-definition, find-references) over `grep` for symbols. Use `grep`
for comments, CI YAML and `Cargo.toml`. Read the implementation and tests rather than
trusting prose when checking how an API behaves, especially Burn's, given the fork.

## Commands
- All tests: `cargo test`
- Train: `cargo run --release --example train_burgers [--features metal]`
  (also `train_darcy`, `train_burgers_learner`)
- Evaluate a run: `cargo run --release --example predict_burgers -- runs/<run_dir>`
  (also `predict_darcy`, `burgers_resolution`, `burgers_superresolution`, `darcy_plot`)

## Agent workflow
- Skills `/design-doc <objective>`, `/phase-plan <objective> <N>` and
  `/do-task <brief>`, and the `reviewer` agent, are user-level, from the agent-workflow
  toolkit; they hold no project knowledge and take it from this file.
- Flow per objective: own branch and worktree → `/design-doc` → review → `/phase-plan`
  → one `/do-task` session per brief, one PR each.

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
- Docs are organised by objective (`docs/README.md`, 2026-11-11): design documents in
  `docs/design/`, one folder per objective. Phase and milestone numbers restart per
  objective, so name the objective with the number ("2D Phase 10 T3", "3D CL2.1").
- Objective 3D (3D Poisson, then 3D Laplace): proposal
  docs/design/3d-chebyshev-laplace-fno.md (draft 0.2, 2026-11-11, nothing signed off);
  milestones L0–L5, tasks `CL<m>.<n>`, decisions D0–D8 in docs/laplace3d/roadmap.md; log
  docs/laplace3d/progress.md; each task's plan docs/laplace3d/tasks/<ID>.md. Constraint 1:
  every 2D output stays bit for bit identical.
- Objective library (neural operators: `neuraloperator` 2.0.0 and beyond): proposal
  docs/design/neural-operator-library.md (draft 0.1, 2026-11-11; D1, the 2.0.0 tag, signed
  off), development wireframes in its §4; milestones M0–M4, tasks `CM<m>.<n>`, decisions
  D0–D8 in docs/operators/roadmap.md; feature matrix docs/operators/feature-matrix.md; log
  docs/operators/progress.md. Constraint 1: every existing FNO output stays bit for bit
  identical. Python (fixtures from 2.0.0) stays outside CI.
- Objective 2D (everything below; phases in docs/poisson2d/).
- Current phase: Phase 10 (Gauss–Newton optimisation), planned 2026-11-09:
  docs/poisson2d/phase10/README.md, briefs T1–T6 (T5 a draft for its runs), design §13.17, decision 31,
  decisions G1–G8 chosen 2026-11-09. T1, the generic core, done (#98); T2, the
  `GaussNewton` optimiser, done (#102); T3, trainer integration, code done (#104); T4,
  the validation gate, code done (#105); T5, the matrix, code done (#106); T6, conditioning, code done (#107); all code done, the VM runs remain, in the order of docs/poisson2d/phase10/runs.md (the run sheet). Order amended
  2026-11-11 (decision 31): task code first, every VM run (T3's acceptance, T4's gate,
  T5, T6) at the end of the phase; T5's runs wait for T4's go. A model-agnostic,
  matrix-free Levenberg–Marquardt Gauss–Newton (finite-difference JVPs: Burn has
  reverse mode only), validated on the Phase 5 baseline. G8, amended 2026-11-09: it
  waits neither for Phase 9 (now done) nor for the θ-map. Code is written on the Mac;
  T3's cuda f64 check and the runs (T4–T6) are on one A100 on the VM (no NCCL needed).
- The θ-map is an optional, unnumbered phase since 2026-11-09 (decision 30):
  docs/poisson2d/optional/README.md, design §13.18. T1, the θ-map model (brief
  docs/poisson2d/optional/T1-theta-map.md approved, decisions P8–P17 chosen 2026-11-09); T2, the
  comparison against the Phase 5 baseline. It runs on the Mac, at any time; nothing waits
  for it. It was Phase 9 T5–T6 for part of 2026-11-09.
- Phase 8 (experiment tracking) is done 2026-11-09: docs/poisson2d/phase8/README.md. T1 (#91):
  MLflow export of run directories to a self-hosted server and opt-in live logging
  (MLflow replaced the W&B task). Its θ-map tasks T2–T3 are now the optional phase. Open
  items (mlflow CI steps, review leftovers, D6, the aliasing row) are in its README.
- Phase 9 (HPC) is done 2026-11-10: docs/poisson2d/phase9/README.md. T1–T4, planned 2026-11-02, renumbered 2026-11-03
  and 2026-11-04: briefs T1–T4 (T1 checkpoint and resume, T2
  multi-GPU, T3 scaling, T4 flex run; three tasks closed at planning are in its "Closed
  at planning"). T1 done (#87); T2 done (code #88, VM acceptance 2026-11-09, criterion
  P18; the pause of 2026-11-06 is lifted); T3 done (results docs/poisson2d/phase9/results.md:
  the step is host-bound, 2 GPUs 0.44× at s = 64); T4 done (flex, test L² 1.021e-2 ≤
  2e-2; decision 21 closed). T2 and T3 run on a
  2× A100 VM: code here, cuda checks and GPU runs there. Phases 5–7, the research programme (design §13.16), are
  done. Phase 7 (training-set size and sample efficiency) is done 2026-10-30:
  docs/poisson2d/phase7/README.md, results docs/poisson2d/phase7/results.md; D7 (mean fine-grid error
  ≤ 2e-2) is met at N = 500 (decision 30, amendment of 2026-10-30). Phases 0–4 are done
  (docs/poisson2d/phase0/README.md to docs/poisson2d/phase4/README.md), and so is Phase 4s (side phase,
  repository health, docs/poisson2d/phase4s/README.md; design §10, decision 25), closed 2026-10-24
  with re-review grade B (docs/code-quality/thermo-nuclear-repo-review-2026-10-24.md).
  Phase 5 (grid-decoupled operator learning) is done 2026-10-29: docs/poisson2d/phase5/README.md,
  results docs/poisson2d/phase5/results.md. Phase 6 (SVD/POD information content, beside Phase 5) is
  done 2026-10-27: docs/poisson2d/phase6/README.md, results docs/poisson2d/phase6/results.md. Phase 8 (MLflow) is optional; Phase 9 (HPC: profiling, checkpointing,
  multi-GPU) is a separate systems track. Their 4s prerequisites
  (T10/T12 for Phase 9 T1, T16 for the optional θ-map T1) are merged. Each phase gets its README and briefs when it
  starts; 4s briefs are in docs/poisson2d/phase4s/T*.md.
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
- Exception for `/next-task` and `/milestone-gate` (`.claude/skills/`, the author,
  2026-11-11): invoking one is a request to create its branch (`<objective>/<id>`), commit
  to it, push it, and open or update its PR (a draft while VM runs, sign-offs or open
  questions are pending). Merging, pushing to `main`, force-pushing, rewriting history,
  deleting branches and closing PRs still need a yes in the current message.
- `main` is protected on GitHub (2026-11-11): changes only by PR, the CI jobs
  `run-tests`, `run-tests-datagen` and `style-checks` green, the branch up to date with
  `main`. Merged branches are deleted. The rules do not bind admins, by the author's
  choice: the author may merge past them with `gh pr merge --admin` (e.g. when an apt
  mirror outage fails `run-tests-datagen` on a PR it cannot judge); never suggest
  changing that. `.claude/settings.json` denies pushing to `main`, force-pushing,
  deleting remote branches, and `gh pr merge` / `gh pr close`, so no session can merge or
  push to `main`: the author runs those (`! <command>` in the prompt). Bring a stale PR branch up to date by merging `origin/main` into it,
  never by rebasing. Branch from `origin/main` with `git switch --no-track -c`, never
  checking out `main` (worktrees).
- Default flow: make the change, run the checks, report exactly which ones ran and what
  they said, and leave the result uncommitted unless the current message asks for a
  commit. Never report an unrun check as passing; say plainly when something could not
  run (no GPU backend, no dataset, no gnuplot).
- One task per branch and PR; stay inside the modules the task names.
- Task numbers within a phase are execution order, from Phase 1 on (design §10). Phase 0
  keeps its signed-off numbers and runs T2, then T3 and T4, then T1.
- Keep changes targeted: no drive-by reformatting or refactors.
- When listing tasks (status reports, summaries of a phase), order them by completion,
  earliest first; tasks still open follow, in execution order.

## Skills
Slash commands in `.claude/skills/` for the objectives with a roadmap (`laplace3d`,
`operators`; the docs are the whole state, no GitHub issues):
- `/next-task`: the default loop. Use it whenever the user says "continue" or "next",
  names a roadmap ID (CL…, CM…) or an objective: it syncs the open task PRs, picks the next
  runnable task from the objective's roadmap, plans it (`tasks/<ID>.md`), implements it
  tests first, verifies, reviews (the `reviewer` agent against the plan, and a
  thermo-nuclear code quality review per task), logs (`progress.md`) and opens the PR. One
  PR per task; a task over about 1,000 lines or three modules is split first.
- `/verify`: the checks below for what a branch changed, with the objective's constraint 1.
  Run before calling anything done.
- `/milestone-gate`: a milestone's gate report (`gates/<milestone>.md`), with a
  thermo-nuclear review of the milestone's whole diff, and at an objective's last
  milestone the thermo-nuclear repo review; then the close.
- Thermo-nuclear reports are git-ignored, always in this checkout's `docs/code-quality/`
  (also from worktrees); PR texts and gate reports carry their verdicts.

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
- `training/`: `config` (`TrainingConfig`; since Phase 10 T3 `optimizer`, an
  `OptimizerConfig`, `None` meaning Adam; since T5 `adam_epochs`, the Hybrid arm's Adam
  epochs before Gauss–Newton), `step` (`flatten_pair`, `forward_loss`,
  `evaluate`: the per-sample mean test loss), `trainer` (components, `train_epoch`,
  `training_loop` with its `EpochObserver`, `train_fno`), `gn` (Phase 10 T3:
  `FnoResidual`, the FNO's Gauss–Newton residual, and `train_epoch_gn`, the epoch the loop
  runs when the config asks for Gauss–Newton), `learner` (Burn `Learner` impls).
- `optim/gauss_newton/` (Phase 10 T1, generic over any Burn `Module`; imports nothing from
  `models/` or `training/`): `ParamVec` (a model's float parameters as one vector, on
  the model's device without autodiff), `params` / `with_params`, `LeastSquares`
  (residual, optional exact J v), `vjp` (Jᵀu, one backward), `jvp` (central finite
  differences, step ∛u · max(‖θ‖, 1) / ‖v‖), `adjoint_error`, and `cg` (preconditioned
  CG, warm start, Martens' stop, Lanczos Ritz values); since T2 `GaussNewton` (`step`:
  one damped step, ρ on the same batch, backtracking over CG's iterates, λ from ρ;
  `GaussNewtonConfig`, `GnStep` diagnostics, `GaussNewtonState` (owned state, bytes for checkpoints))
  and `hutchinson_diagonal` (the `Preconditioner::HutchinsonDiagonal`); since T6 `lanczos`
  (every Ritz value, full reorthogonalisation). Its FNO tests are in
  `tests/gauss_newton_fno.rs` (helpers in `tests/gauss_newton_common/`), and the f64
  ones are `#[ignore]`d in `tests/gauss_newton_f64.rs` (flex's f64 rfft uses f32
  twiddles).
- `runs` (`RunDir`, `EpochMetrics`, `RunCosts`) owns the run directory:
  `create(root, name)`, `save_configs`, `append_metrics` (the only `metrics.csv` writer,
  one row per epoch, with `epoch_time_s` since Phase 5 T4; old five-column files still
  read), `save_plot`, `save_normalizers`, `save_weights`, `save_costs` (`costs.json`,
  Poisson runs since Phase 5 T4, `optimisation_steps` since Phase 7 T1, `resumes` since
  Phase 9 T1; `forward_passes` / `backward_passes` since Phase 10 T3), `save_checkpoint` /
  `checkpoint::<R>(device)` (`checkpoint.bin`, Phase 9 T1; `save_checkpoint_gn` adds the
  Gauss–Newton state, Phase 10 T3), `truncate_metrics`, and since Phase 10 T3
  `append_gn_step` (`gn_steps.csv`, `GnStepRecord`, the only writer), `truncate_gn_steps`
  and `gn_passes`; `open(dir)`, `load_fno`, `model_config`,
  `training_config`, `data_config::<C>`, `normalizers()`, `costs()`, `metrics()`, `artifacts()` (the files the MLflow export uploads). Run-file names appear
  nowhere else; examples pass `CARGO_MANIFEST_DIR/runs` as the root.
- `tracking/` (behind `mlflow`, Phase 8 T1): `export_requests` (pure: a Poisson
  `RunDir` to the MLflow REST requests: tags, params, per-epoch and summary metrics in
  `log-batch`es within its limits, artifact `PUT`s from `RunDir::artifacts`) and `Client`
  (ureq: sends them, threads the run id, finds/deletes runs by the `sciml.run_dir` tag,
  marks a failed export `FAILED`; `MlflowError`), driven by `examples/mlflow_export.rs`;
  `start_requests` / `epoch_request` / `finish_requests` (the same run in three parts)
  and `MlflowObserver` (opt-in live logging, best-effort, off after the first failure),
  which `train_poisson` passes to `train_components` as an extra `EpochObserver`.
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
  commits, and since Phase 7 T1 an optional `subset`; `SidecarRecord`), `subset`
  (`SubsetConfig`, `subset_indices`: nested seeded train subsets by ChaCha8, sorted, with
  an FNV-1a `indices_hash` the loader checks; `rand_chacha` is a default dependency for
  it), `uniform` (`load_poisson_uniform` reads those
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

GPU VM (2× A100, `ssh cricket`, repository at `~/sciml-rs`): `source ~/sciml-gpu.sh`
before any `cargo` command. It activates the Spack environment in `env/vm-a100/` (NCCL
built for a CUDA the driver supports, gnuplot), which the system NCCL is not. Long runs
go in tmux with a `tee` log under `logs/`. Runs of `/next-task` tasks each use a worktree
of their own, `~/sciml-rs-wt/<id>` (`.claude/skills/next-task`, step 6), so parallel runs
never share a checkout.

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
A `changes` job runs first: a pull request that touches none of `src/`, `examples/`,
`tests/`, `benches/`, `spikes/`, `Cargo.toml`, `Cargo.lock`, `.cargo/`, the workflows or
`docs/CONVENTIONS.md` skips the three jobs, which then report success (pushes to `main`
always run them). The apt step retries and times out after 4 minutes.
Before finishing any task, also run `cargo clippy --all-targets -- -D warnings`. Keep
clippy clean without blanket `#[allow]`s. For backend-sensitive changes, also run the
relevant tests with `--features metal` and say whether it was run.

GPU parity, for backend-sensitive changes: `cargo test --features metal --test gpu_parity`.
Its GPU `repro_*` tests (the Burn fusion bugs of the old fork, fixed in 0.23.0-pre.2) run
as regression tests.

An exact test filter needs the full module path:
`cargo test neural_operators::models::fno::tests::<group>::<name> -- --exact`. A shorter path runs
0 tests and still reports `ok`; check the "running N tests" line.

## Burn
- Burn is pinned exactly in `Cargo.toml` (`[workspace.dependencies]`): `burn = "=0.23.0-pre.2"`
  from crates.io, a pre-release, features `autodiff, signal, store, train`. It replaced the
  fork `ax1s-x1zz/burn` at `ddfa9af` on 2026-11-13 (the fork's Bluestein FFT is upstream
  #5757); every 2D Phase 1–10 result was measured on the fork.
- `cubek.toml` (repository root) sets `[matmul] allow_tf32 = false`: Burn 0.23's cuda matmuls
  default to TF32, which made the GPU 50× less accurate than Flex in `replay_2d_poisson`
  (1.6e-4; 1.4e-6 with it off). It is found by walking up from the working directory, so
  run from the repository. cuda f64 matmuls are unsupported in this release (every cubek
  kernel declines f64), so f64 on cuda still fails, now in the matmul rather than the FFT.
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
`~/.cargo/registry/src/*/burn-*-0.23.0-pre.2/` (FFTs: `burn-signal-0.23.0-pre.2/src/functions/fft.rs`);
other versions there are not the pin. Read the implementation and tests rather
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
- Train on Poisson: `cargo run --release --example train_poisson [--features metal] [-- [<padding>] [<n>] [<s>] [<seed>] [<N> <subset_seed>]]`
  (the first argument overrides `PADDING`, the second picks the stage, n ∈ {65, 129, 257},
  default 65, the third the learning grid s, default n − 1, with modes min(12, s/2), the
  fourth the seed, default 42, the fifth and sixth together a training subset of N samples
  and its seed (Phase 7), default the whole split; needs that stage in `datasets/poisson/`; `predict_poisson`
  evaluates any s, `poisson_resolution` and `poisson_plot` only s = n − 1). On cuda:
  `--no-default-features --features cuda`. It checkpoints every 25 epochs and after the
  last; `train_poisson -- --resume runs/<run_dir>` continues an interrupted run from its
  checkpoint with the run's saved configs (Phase 9 T1). `OPTIMIZER=gauss-newton` trains a new run with
  Gauss–Newton (the `GN_*` literals; one device; run name `_gn`; writes `gn_steps.csv`;
  Phase 10 T3). `OPTIMIZER=hybrid` trains `HYBRID_ADAM_EPOCHS` with Adam, then
  Gauss–Newton (run name `_hybrid`); `GN_PRECONDITIONER=none|hutchinson` picks the
  preconditioner (Phase 10 T5)
- Export run directories to a local MLflow server (Phase 8 T1; server: `mlflow server
  --host 127.0.0.1 --port 5001 --backend-store-uri sqlite:///$HOME/mlflow/mlflow.db
  --artifacts-destination $HOME/mlflow/artifacts`, see `README.md`):
  `cargo run --features mlflow --example mlflow_export -- [--replace] <experiment> runs/<run_dir> ...`
  (`MLFLOW_TRACKING_URI`, default `http://127.0.0.1:5001`; skips directories already
  exported, `--replace` re-exports them; exits non-zero if any failed)
- Live MLflow logging while training (opt-in, Q7): `MLFLOW_EXPERIMENT_NAME=<exp> cargo run
  --release --features mlflow --example train_poisson -- ...`; unset, training is
  unchanged. From the GPU VM: `ssh -R 5001:127.0.0.1:5001 <vm>`, then on the VM
  `MLFLOW_EXPERIMENT_NAME=<exp> MLFLOW_TRACKING_URI=http://127.0.0.1:5001 cargo run
  --release --no-default-features --features cuda,mlflow --example train_poisson -- ...`.
  After `--resume` re-trained epochs appear twice; `mlflow_export --replace` gives the
  clean record (and adds `errors.json`)
- Evaluate a run: `cargo run --release --example predict_burgers -- runs/<run_dir>`
  (also `burgers_resolution -- runs/<run_dir> [<rate> ...]` on Burgers runs, s = 8192 / rate;
  `predict_darcy`, `darcy_plot` on Darcy runs)
- Evaluate a Poisson run (the five errors, the boundary verdict, the worst sample):
  `cargo run --release --example predict_poisson [--features metal] -- runs/<run_dir>`
- Phase 5 comparison table and plots from evaluated Poisson runs (mean and min–max over
  seeds per (padding, s); plots `<stem>_*.png` beside the table, skipped without gnuplot):
  `cargo run --release --example poisson_matrix -- docs/poisson2d/phase5/results.md runs/<run_dir> ...`;
  with `--by-n` the Phase 7 table instead (runs at one (padding, s) and one backend,
  grouped by N, whole-split runs as N = 1000; D7 verdict, spread by seed, cost per fall in
  error; plots against N): `poisson_matrix -- --by-n docs/poisson2d/phase7/results.md runs/<run_dir> ...`;
  with `--by-optimizer` the Phase 10 table (runs at one (padding, s) and one backend,
  grouped by optimiser arm; plots of error 4 against time, updates and passes):
  `poisson_matrix -- --by-optimizer docs/poisson2d/phase10/results.md runs/<run_dir> ...`
- Conditioning of trained Poisson runs (Phase 10 T6: Ritz values of G + μI at s = 32, 64,
  128, the tangent kernel in the sine basis, the Laplacians' κ; `datagen` adds the
  Chebyshev κ): `cargo run --release --example poisson_conditioning --
  docs/poisson2d/phase10/conditioning.md runs/<run_dir> ...`
- Resolution study (errors 3 and 4 of a Poisson run on finer grids at the run's K, weights
  unchanged, normalisers refit on that n's `_train` split):
  `cargo run --release --example poisson_resolution [--features metal] -- runs/<run_dir> <n> [<n> ...]`
- POD of the stage 3 f and u, on the CGL grid and the learning grids (Phase 6; tables to
  `<out.md>`, plots beside it):
  `cargo run --release --features datagen --example poisson_pod -- docs/poisson2d/phase6/results.md`
- Plot one Poisson test sample (f, u, û, û − u on the CGL nodes, on the run's stage; writes
  `prediction_<sample>.png`):
  `cargo run --release --example poisson_plot [--features metal] -- runs/<run_dir> [<sample>]`

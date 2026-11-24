# CM0.3 The operator trait and a generic stack

Started 2026-11-24 on `operators/cm0-3`. Roadmap: "CM0.3 The operator trait and a generic
stack. Proposal §6.1: trainer, evaluate, `predict_host`, checkpoints, the epoch observer,
the run directory's weights and the Gauss–Newton residual generic over the trait; `FNO<R>`
implements it." Proposal: §10; §1.4, §2, §4.1, §4.4, §5 (constraint 1), §6.1, §8 (R1, R4),
§12 (D4).

## Goal

One trait that every operator implements, with the training stack (trainer, evaluation,
inference, checkpoints, run-directory weights, Gauss–Newton residual) written against it
instead of `FNO<R>`, and the FNO's outputs unchanged bit for bit.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)

- Roadmap: "every existing test passes unedited, and a fixed FNO forward and one Phase 5
  epoch are bit for bit unchanged." *Needs:* D4. *Sign-off:* D4.
- Proposal §5: "Constraint 1: every existing FNO test, plus bit-identity tests on a fixed
  FNO forward and on one Phase 5 training epoch with the default options."
- Proposal §6.1: "A trait over Burn modules with an associated config type: build from
  config, forward on the input type, a name for the registry, and the run directory's save
  and load. The trainer, `evaluate`, `predict_host`, checkpoints, the epoch observer and the
  Gauss–Newton residual become generic over it. `FNO<R>` implements it first, unchanged."
- Coordinating session (2026-11-24): the Gauss–Newton path is also bit for bit unchanged.

## Read first

- `src/neural_operators/models/fno/mod.rs` (`FNOConfig::init::<R>`, `FNO::forward`).
- `src/neural_operators/training/`: `step.rs` (`forward_loss`, `evaluate`), `trainer.rs`
  (`train_epoch`, `TrainingComponents`, `build_training_components`, `EpochObserver`,
  `print_epoch`, `training_loop`, `run_loop`, `train_fno`, `train_components`), `gn.rs`
  (`FnoResidual`, `train_epoch_gn`), `data_parallel.rs` (`Peers`, `shard_grads`,
  `train_epoch_parallel`; FNO-typed too, not named in the roadmap line).
- `src/neural_operators/runs/mod.rs` (`save_weights`, `load_fno`, `model_config`,
  `save_checkpoint`, `save_checkpoint_gn`, `checkpoint`) and `runs/checkpoint.rs`
  (`Checkpoint<R>`, `encode`, `decode`).
- `src/neural_operators/models/predict.rs` (`predict_host`).
- `src/neural_operators/optim/gauss_newton/` (`LeastSquares<M>`: already generic over any
  `Module`).
- Burn 0.23.0-pre.2: `burn-core/src/module/base.rs` (`Module`: `valid`, `fork`, generic
  `visit`/`map`, so not dyn-compatible); `burn-optim/src/optim/module/module_optimizer.rs`
  (`ModuleOptimizer` is already type-erased over the module);
  `burn-derive/src/module/codegen_enum.rs` (an enum `Module` enters a path segment named
  after the variant).

## What the existing tests pin

"Unedited" fixes these public names and their generic-argument lists (turbofish and type
arguments used in `tests/`, `examples/` and the in-file test modules):
`train_fno::<4, 3, _>`, `build_training_components::<R, RM1, T>`, `predict_host::<4>`,
`load_fno::<R>`, `RunDir::checkpoint::<4>`, `model_config()` returning `FNOConfig`,
`FnoResidual::<4, 3>::new`, `LeastSquares<FNO<4>>`, `EpochObserver<'_, 4>`,
`Checkpoint<4>`, `TrainingComponents<R, RM1>`. Rust gives two ways to keep them while
adding a model parameter:

- a function taking the model as `&impl Trait` keeps its explicit generic list (explicit
  arguments beside argument-position `impl Trait` are allowed since Rust 1.63), so
  `predict_host::<4>(&fno, …)` still compiles;
- a struct or type alias takes a defaulted parameter, `M = FNO<R>`, so `Checkpoint<4>` and
  `TrainingComponents<4, 3>` still mean the FNO.

Where neither works (a function that returns the model, so it must name `M`), the
FNO-named function stays as a one-line wrapper over a new generic one.

## D4: the shape of the operator trait (for sign-off)

### Option A (proposal's recommendation): a generic trait with associated config and input types, static dispatch

```rust
// models/operator.rs
pub trait Operator: Module {
    /// Hyperparameters, saved as `model_cfg.json`.
    type Config: Config;
    /// What `forward` takes and returns: `Tensor<R>` for grid operators.
    type Input;
    type Output;
    /// The registry's name (CM0.4 reads it).
    const NAME: &'static str;
    fn init(config: &Self::Config, device: &Device) -> Self;
    fn forward(&self, x: Self::Input) -> Self::Output;
}
/// The grid operators the current stack trains: rank-R in, rank-R out.
pub trait GridOperator<const R: usize>: Operator<Input = Tensor<R>, Output = Tensor<R>> {}
impl<const R: usize, M: Operator<Input = Tensor<R>, Output = Tensor<R>>> GridOperator<R> for M {}

impl<const R: usize> Operator for FNO<R> {
    type Config = FNOConfig; type Input = Tensor<R>; type Output = Tensor<R>;
    const NAME: &'static str = "fno";
    fn init(c: &FNOConfig, d: &Device) -> Self { c.init::<R>(d) }
    fn forward(&self, x: Tensor<R>) -> Tensor<R> { FNO::forward(self, x) }
}
```

| Part | Effect |
| --- | --- |
| Trainer | `train_epoch`, `run_loop`, `training_loop`, `train_components` and the data-parallel epoch take `M: GridOperator<R>`; `TrainingComponents<R, RM1, M = FNO<R>>`; a generic `build_components::<M, …>(&M::Config, …)` beside `build_training_components` (FNO wrapper) and `train::<M, …>` beside `train_fno` (FNO wrapper). `Batch<R, RM1>` and the loaders are unchanged. `ModuleOptimizer` and the scheduler need nothing (already module-agnostic). The epoch observer becomes `dyn FnMut(&EpochMetrics, &M)`, `EpochObserver<'a, R, M = FNO<R>>`. |
| Checkpoints | `Checkpoint<R, M = FNO<R>>`; `encode`/`decode` and `save_checkpoint(_gn)` take any `Module` (Burnpack already is); `decode` builds through `M::init(&M::Config)`. File format, magic and blob order unchanged, so every `checkpoint.bin` on disk reads as before. |
| `predict_host` | `predict_host<const R>(model: &impl GridOperator<R>, …)`; body unchanged. |
| Run directory | `save_weights(&impl Module)`; new `load_model::<M>()` reads `model_cfg.json` as `M::Config`, `M::init`, loads the Burnpack weights; `load_fno::<R>` becomes `load_model::<FNO<R>>`. File names and contents unchanged. `kind` in `model_cfg.json` stays CM0.4's. |
| Gauss–Newton residual | `impl<M: GridOperator<R>> LeastSquares<M> for FnoResidual<R, RM1>`: one generic impl, the name kept (renaming it is a later, separate change); `train_epoch_gn` generic. `optim::gauss_newton` itself is untouched. |
| Later tasks | CM3.3 (GINO, point clouds) adds an operator with another `Input` and the stack's bounds generalise from `GridOperator<R>` to `Operator` where needed; nothing in this task's trait changes (R4). |
| Cost | Static dispatch, monomorphised per operator and rank, as the FNO is today: no runtime cost, longer builds per added operator. |

### Option B: a grid-only trait, no associated input type

`trait GridOperator<const R: usize>: Module { type Config: Config; const NAME; fn init; fn
forward(&self, Tensor<R>) -> Tensor<R>; }`. Same effects on the trainer, checkpoints,
`predict_host`, the run directory and the residual as A, a little shorter (no
`Input = Tensor<R>` bound). CM3.3's point-cloud operators then need a second trait or a
change to this one, and every bound in the stack is revisited then (R4).

### Option C: an enum of operators

`#[derive(Module)] enum AnyOperator<const R> { Fno(FNO<R>), … }` and the stack concrete on
it. The registry is a `match`. But Burn's enum derive prefixes parameter paths with the
variant (`Fno.fc0.weight` against today's `fc0.weight`), so every existing
`model_weights.bpk` and `checkpoint.bin` needs a path remap on load (CM0.4's "every existing
run directory loads unchanged" at risk), every new operator edits the central enum, and
the trainer's FNO path gains a `match` per forward (constraint 1 still provable, but the
saved files differ in their paths).

### Option D: trait objects

`Box<dyn Operator>` with a runtime registry. Burn's `Module` is not dyn-compatible (generic
`visit`/`map`) and `ModuleOptimizer::step` takes the module by value, so this needs a
hand-written type-erasing wrapper over every `Module` method, plus a dynamic call per
forward. Proposal §12 recommends against ("no trait objects").

**Recommendation:** A. It is the proposal's, it costs the same as B now, and it leaves
CM3.3 an extension rather than a rewrite.

## Changes (option A, signed off)

- `models/operator.rs` (new): `Operator`, `GridOperator<R>` with its blanket impl;
  `impl Operator for FNO<R>` (in `models/fno/mod.rs` or `operator.rs`). `models/mod.rs`
  exports it. CLAUDE.md "Layout" entry.
- `models/predict.rs`: `predict_host` over `&impl GridOperator<R>`.
- `training/step.rs`: `forward_loss`, `evaluate` over `&impl GridOperator<R>`.
- `training/trainer.rs`: as in the table; `print_epoch` over any model.
- `training/gn.rs`: the generic `LeastSquares` impl; `train_epoch_gn` generic.
- `training/data_parallel.rs`: `Peers`, `shard_grads`, `train_epoch_parallel` generic.
- `runs/mod.rs`, `runs/checkpoint.rs`: as in the table.
- Not touched: `optim/`, `layers/`, `losses/`, `data/`, the FNO's own code, the file formats,
  `training/learner.rs` (the Burgers-only `FNO<3>` showcase of Burn's `Learner`), the
  MLflow export (reads `model_cfg.json` as `FNOConfig`; generalised with the registry in
  CM0.4, a sentence to add to that roadmap line), examples (they keep calling the FNO
  names).
- Public API: additions only (`Operator`, `GridOperator`, `train`, `build_components`,
  `load_model`, a generic parameter with an FNO default on `TrainingComponents`,
  `Checkpoint`, `EpochObserver`). No conventions change.
- Size: three modules (`models`, `training`, `runs`) and one test file; estimated 400–700
  changed lines.

## Tests that prove it

- **Every existing test, unedited:** `git diff origin/main -- tests/` empty, and no hunk in
  the `#[cfg(test)]` modules of changed files; `cargo test`, `cargo test --features
  datagen`.
- **Bit identity against `main`** (`tests/fno_bit_identity.rs`, new, flex): it uses only
  names that exist on `main` too, so the same file runs on both. Each case writes its
  result as raw little-endian bytes under `target/bit-identity/<case>.bin`:
  1. *fixed forward:* `FNOConfig` of the Phase 5 baseline (modes 12×12, width 32, 4
     layers, padding as the Phase 5 baseline), seeded init, fixed seeded input
     `[4, 64, 64, 1]`: the output.
  2. *one Phase 5 epoch:* `train_fno::<4, 3, _>` with the Phase 5 `TrainingConfig`
     defaults and `epochs = 1`, on a seeded synthetic 2D split shaped as Phase 5's
     (s = 64; small N so it runs in seconds), into a temporary `RunDir`: the trained
     weights, `metrics.csv`'s row minus `epoch_time_s`, and `model_weights` and
     `checkpoint.bin` byte for byte.
  3. *one Gauss–Newton epoch:* the same with `OPTIMIZER` Gauss–Newton (the Phase 10
     `GN_*` defaults) on a tiny model: weights, the `gn_steps.csv` rows minus times.
  `/verify` runs it on a detached `origin/main` worktree (the file copied in) and on the
  branch, then `cmp`s each `.bin`. Trusted path: `main`'s code. Tolerance: none (bytes).
  In CI the test only checks the cases run; the comparison is the `/verify` step, since
  flex's bits need not agree across the Mac and CI's x86.
- **The trait:** a unit test in `models/operator.rs` that `<FNO<4> as Operator>::init`
  and `FNOConfig::init::<4>` from the same seed give the same weights, and that
  `Operator::forward` equals `FNO::forward` bit for bit.
- **Generic for real:** a test-only second operator (a one-`Linear` pointwise model,
  `#[cfg(test)]`) trains one epoch through `train`, saves and resumes from a checkpoint and
  reloads with `load_model`, so the generic path is exercised by something that is not the
  FNO.

## Checks

`/verify` (operators, constraint 1 including the `main`-against-branch `cmp`); `cargo fmt
--check`; `cargo clippy --no-deps -- -D warnings`; `--examples`; `--all-targets`;
`--features datagen --all-targets`; `cargo test`; `cargo test --features datagen`;
`cargo doc --no-deps`; `cargo test --features metal --test gpu_parity` and the bit-identity
test with `--features metal` (backend-sensitive: the stack runs every forward).

## VM runs

None. cuda is not needed to show the change: the code paths are backend-generic and the
bit identity is shown on flex against `main`.

## Constraints and risks

- Constraint 1 (proposal §1.4): the only risk is R1, a changed FNO output; the FNO's own
  code and the files' formats are untouched, and the `cmp` against `main` shows it.
- Every existing test unedited, so the FNO names stay (see "What the existing tests pin").
- Nothing else touches `training/` while this task is open (optimisers tasks wait for it).
  Parallel lanes touch `metrics/poisson`, `pde/poisson/` and `Cargo.toml`, not these files.
- 2D objective: no output changes (the same claim as constraint 1); no 3D effect.
- Risk: a type alias with a defaulted parameter (`EpochObserver<'a, R, M = FNO<R>>`) may be
  refused if `R` is only used in the default; the fallback is a new alias name beside the
  unchanged `EpochObserver<'a, R>`.

## Pending runs

None.

## Open questions

- D4: signed off 2026-11-24 by the author, option A as recommended.

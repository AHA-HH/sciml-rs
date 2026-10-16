# T3: one `train_fno`

Phase 4s, task 3. Branch `phase4s/T3-train-fno`. Depends on T1.
- Can run in parallel with T4–T8.
- Shares `training/trainer.rs` with T2, so whichever merges second rebases. Suggested order:
  T3 first.
- Shares the imports in `examples/train/*.rs` with T4: a small rebase.

## Read first
- Review §5 S3 (first bullet), §6 T2-2, §7 step A2.
- `src/neural_operators/training/trainers/{burgers,darcy,poisson}.rs` and
  `training/trainers/mod.rs`.
- `trainer.rs`: `Postprocess` :49, `identity` :51, `training_loop` :315-366.
- `data/transforms/normalizers.rs`: `FlatDecoder` :217-244.

## Goal
Three per-PDE trainer functions choose between "no normaliser" and "decode with y". Replace
them with one rank-generic function and delete the directory.

## Deliverables
1. **`pub fn train_fno::<const R, const RM1, T: HostFloat>`** in `trainer.rs`:
   - Arguments: `(train, test, y_norm: Option<&UnitGaussianNormalizer>, model_cfg:
     &FNOConfig, train_cfg: &TrainingConfig, device: &Device)`.
   - Returns `(FNO<R>, Vec<EpochMetrics>)`.
   - `None` reproduces `train_burgers`: `identity` on both sides.
   - `Some(y)` reproduces `train_darcy` / `train_poisson`, whose bodies are identical apart
     from the name: two `FlatDecoder`s, a train hook that decodes both sides, and an eval
     hook that decodes the prediction only (CONVENTIONS §10).
   - Document the arguments and tensor shapes (rustdoc style), and the panics inherited from
     `build_training_components` / `flatten_pair`.
2. **Delete `training/trainers/`** (`mod.rs`, `burgers.rs`, `darcy.rs`, `poisson.rs`) and
   its `pub mod` line in `training/mod.rs`.
3. **Move the test `train_poisson_smoke`** (trainers/poisson.rs:89-134, and its helpers in
   the same test module :51-135) into `trainer.rs`'s tests, beside `train_fno`. It uses
   `chebyshev::cgl_nodes`, `load_poisson_uniform`, `PoissonConfig` and `NpzWriter`; keep the
   imports it needs.
4. **Migrate the callers:** `examples/train/burgers.rs:101`, `darcy.rs:83` and
   `poisson.rs:121`, to `train_fno::<3, 2, _>(…, None, …)` and `train_fno::<4, 3, _>(…,
   Some(&normalizers.y), …)`.
5. CLAUDE.md "Layout": the line about `training/trainers/poisson` (`train_poisson`) now
   points at `train_fno`.

## Tests
- `train_poisson_smoke` keeps its name under `training::trainer::tests`. If the exact-filter
  path changes, give the new path in the PR.
- Add one test: `train_fno` with `None` gives bit-identical metrics to the old Burgers path on
  a tiny synthetic set. A second test does the same with `Some` against the old Darcy/Poisson
  path. Write both before deleting the old files, then keep them as `None`/`Some` regression
  tests.
- The test count rises by 2 at most. Nothing is deleted.

## Acceptance
- `cargo clippy --no-deps --examples -- -D warnings` shows that all three train examples
  compile.
- Training numbers are unchanged. The bit-identity tests above prove it; a full training run
  is not required.

## Checks to run and report
The six from the README. The chebyshev pair is not needed (the moved smoke test is
ungated). Give the line delta.

## Do not
- Change `Postprocess`, `identity` or the train/eval asymmetry. That is T17's job.
- Change `training_loop`'s printing (T12) or `eval_epoch`'s contract (T13).
- Touch `learner.rs` beyond an import path, if one breaks.

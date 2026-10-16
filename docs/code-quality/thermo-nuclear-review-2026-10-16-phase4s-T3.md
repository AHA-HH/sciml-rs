# Thermo-nuclear code quality review: phase4s/T3 (second pass)

- **Repo:** sciml-rs
- **Branch:** `phase4s/T3` (uncommitted working tree)
- **Base:** `79f9c62` (merge-base with `main`)
- **Date:** 2026-10-16
- **Diff:** 10 files, +260 / −247

This pass replaces the first pass from the same day. That pass approved the change with
three in-PR fixes and three follow-ups; the fixes are in and are checked again below.

## Verdict: APPROVE

The production change is as small as this task allows. Three trainer files become one
`train_fno` with a single `Option`, the hook choice sits in one private function, and the
callers change only their import and call. All three first-pass fixes landed correctly:
the `# Panics` section now names `FNOConfig::init`, the device-loader test uses
`metric_bits`, and the Burgers comment is current. What remains is test-only repetition
(the "build twice, share the model" setup now exists three times) and two cosmetic
import issues. None of them meets a blocker condition.

## Presumptive blockers

None met.

- Code-judo move left unused: deleting the `Some` path is T17's, and the brief forbids it.
- File crossing 1k lines: no (`trainer.rs` is 570 → 800).
- Ad-hoc branching, scattered checks, wrong layer, duplicated canonical helper: no.
- Unnecessary wrapper: `train_built` is the test seam the Flex RNG requires (first pass, 4.1).

## File sizes

| File | Before | After |
|---|---:|---:|
| `src/neural_operators/training/trainer.rs` | 570 | 800 |
| `src/neural_operators/training/mod.rs` | 10 | 9 |
| `src/neural_operators/training/trainers/*.rs` (4 files) | 218 | 0 (deleted) |
| `examples/train/burgers.rs` | 120 | 121 |
| `examples/train/darcy.rs` | 113 | 113 |
| `examples/train/poisson.rs` | 157 | 157 |
| `CLAUDE.md` | 158 | 159 |

No threshold crossings.

## Findings

### 1. Structural regressions
None.

### 2. Missed dramatic simplifications
**2.1 (carried over) The `Some` path is T17's.** `trainer.rs:404-426`. Storing `y_train`
physical removes `Postprocess`, `identity`, the two decoders and `train_built`. Name both
`train_built` and `train_fno_some_matches_decoder_path` in the PR so T17 finds them.

### 3. Spaghetti or branching growth
None.

### 4. Boundary, abstraction and type contracts
None new. The `# Panics` fix (`trainer.rs:383-387`) is accurate: both assertions are in
`FNOConfig::init` (`models/fno.rs:67-78`), which `build_training_components` calls.

### 5. File size and decomposition
**5.1 (carried over) `trainer.rs` is 800 lines, about 46% tests** (`mod tests` from :429).
If T12/T13 push it towards 1k, move the tests to `training/trainer/tests.rs`.

### 6. Modularity and reuse
**6.1 "Build twice, share one model" is now written three times.** `trainer.rs:557-571`
(device-loader test), `:650-661` (`None`) and `:682-693` (`Some`) each write a `build`
closure, call it twice, then copy `.model` across. The two new tests are also identical
apart from rank, datasets, `y_norm` and the reference hooks. One test helper does the work:

```rust
/// Two copies of the same components with one initial model (the RNG is shared across tests).
fn twin_components<const R: usize, const RM1: usize>(
    model_cfg: &FNOConfig, cfg: &TrainingConfig,
    data: impl Fn() -> (OperatorDataset<f64>, OperatorDataset<f64>), device: &Device,
) -> (TrainingComponents<R, RM1>, TrainingComponents<R, RM1>) {
    let build = || { let (tr, te) = data(); build_training_components(model_cfg, cfg, tr, te, device) };
    let a = build();
    let mut b = build();
    b.model = a.model.clone();
    (a, b)
}
```

Each regression test then shrinks to: twins, `train_built(a, …)`,
`training_loop(b, reference hooks)`, and `assert_eq!(metric_bits(..), metric_bits(..))`,
about 10 lines instead of 30. The device-loader test can use it too (it replaces
`host.model = …`). The test count stays the same.

**6.2 (carried over) Poisson `.npz` fixture duplication** with `data/loaders/poisson.rs:396-425`
→ T22.

### 7. Legibility
**7.1 Two `use crate::neural_operators…` statements in the test module.** `trainer.rs:432`
and `:435` should be one tree:
`use crate::neural_operators::{chebyshev::cgl_nodes, data::{batcher::OperatorBatcher, loaders::{…}, transforms::normalizers::Normalizer}};`.

**7.2 `tiny_dataset_2d` spells out `ndarray::ArrayD` although `ArrayD` is now imported.**
`trainer.rs:632,635`. This copies `tiny_dataset` (:463-466), which was written before the
import existed. Use `ArrayD::from_shape_fn(IxDyn(…))` in both, with `ndarray::IxDyn`
added to the import.

## Next steps

1. **Extract `twin_components` and use it in all three tests** (6.1). This removes the most
   test code. Verify: run `cargo test neural_operators::training::trainer::tests:: -- --list`
   before and after (same names), then the three tests by exact path ("running 1 test"
   each) and the full `cargo test` (210 lib tests).
2. **Merge the test imports and use the short `ArrayD`/`IxDyn` paths** (7.1, 7.2). Verify:
   `cargo fmt -- --check`, `cargo clippy --all-targets -- -D warnings`.
3. **Follow-ups, not in this PR:** T17 (2.1), T22 (6.2), and splitting out the tests if
   T12/T13 push the file towards 1k lines (5.1).

Steps 1 and 2 touch only `trainer.rs`'s test module and can land as one commit.

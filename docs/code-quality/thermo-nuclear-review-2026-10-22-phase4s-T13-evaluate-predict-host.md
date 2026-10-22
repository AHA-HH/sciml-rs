# Thermo-nuclear code quality review: Phase 4s Group 3 (T10–T13)

- **Repo:** sciml-rs (`AHA-HH/sciml-rs`)
- **Branch:** `phase4s/T13-evaluate-predict-host` (HEAD `dcbfa1f`)
- **Base:** `1161faf` (T9 merged). The range covers T10 `RunDir` (#48), T11 one batching path (#49), T12 epoch observer (#50) and T13 `evaluate` / `predict_host` (#51, open).
- **Date:** 2026-10-22
- **Diff size:** 38 files, +1119 / −800

## Verdict: BLOCKED

Group 3 did what the phase asked: the run directory has one owner, there is one batching path, the loop has a hook, and evaluation has one entry point. Behaviour is pinned by bit-identity tests and example re-runs. Two presumptive blockers remain:
- **`training/trainer.rs` crossed 1000 lines** (860 → 1004). The growth is in its tests, but the production half is also the file that every later phase edits.
- **`evaluate` and `training_loop` each divide the same private `eval_epoch` sum by a different count** (`loader.num_items()` and `components.n_test`). That count is already on the loader.

A behaviour-preserving restructure deletes `eval_epoch`, the `n_train` / `n_test` fields, and the second writer of `metrics.csv`. That is the code-judo move this range missed. None of it is risky. It should land before T17 rewrites the same functions.

## Presumptive blockers

1. **A file crossing 1k lines:** `src/neural_operators/training/trainer.rs` went 860 → 1004.
   - Production is lines 1–507, tests are 508–1004.
   - T12 added the observer plus about 73 test lines; T13 added `evaluate` plus 14.
2. **Incidental complexity with a visible code-judo move:**
   - `trainer.rs:162-168`: `evaluate` = `eval_epoch / loader.num_items()`.
   - `trainer.rs:392` and `:412-420`: `training_loop` divides the same `eval_epoch` sum by `components.n_test`, and the train sums by `n_train`.
   - `n_train` and `n_test` exist as fields of `TrainingComponents` (`trainer.rs:262-266`) only to carry these divisors, but both loaders already know their counts.

## File-size table

| File | Before → after | Note |
| --- | --- | --- |
| `src/neural_operators/training/trainer.rs` | 860 → **1004** | **crosses 1k** |
| `src/neural_operators/models/fno.rs` | 1595 → 1595 | already over 1k; only an import line changed (T20 splits it) |
| `src/neural_operators/data/loaders/poisson.rs` | 719 → 719 | import only |
| `src/neural_operators/runs/mod.rs` | 0 → 398 | new; about 140 lines of tests |
| `src/neural_operators/metrics/poisson.rs` | 411 → 396 | |
| `src/neural_operators/data/batcher.rs` | 201 → 217 | about 15 production lines; the rest is a test oracle |
| `src/neural_operators/runs/csv.rs` | 0 → 144 (from `metrics/io.rs`) | |
| `src/neural_operators/models/run.rs` | 157 → 131 | |
| `examples/predict/burgers_resolution.rs` | 0 → 106 | replaces 82 + 98 |
| All other touched files | < 300 | |

## Findings

### 1. Structural regressions

**1.1 `trainer.rs` is now a 1004-line file holding five concerns** (`trainer.rs:1-507`). It holds:
- the config: `TrainingConfig`;
- the loss step: `Postprocess`, `identity`, `flatten_pair`, `StepLoss`, `forward_loss`;
- evaluation: `evaluate`, `eval_epoch`;
- the training loop: `EpochSums`, `train_epoch`, `TrainingComponents`, `build_training_components`, `EpochObserver`, `print_epoch`, `training_loop`;
- the entry point: `train_fno`, `train_built`.

It also has a 500-line test module that covers all of them.

This pushes the file past 1k lines. Can we decompose it first? Phase 5 (checkpointing, multi-GPU) and T17 (the `Postprocess` rewrite) both edit this file, so the split pays off immediately. Target structure:

```
training/
  config.rs     TrainingConfig
  step.rs       Postprocess, identity, flatten_pair, StepLoss, forward_loss, evaluate   (+ tests)
  trainer.rs    TrainingComponents, build_training_components, train_epoch,
                EpochObserver, print_epoch, training_loop, train_fno               (+ tests)
  learner.rs    (unchanged; imports step::forward_loss)
```

`evaluate` sits beside `forward_loss` because that is all it is: a loop of `forward_loss` over a loader. The Learner needs only `step`. Each file lands well under 600 lines.

### 2. Missed code-judo simplifications

**2.1 The divisors live in two places. Let the loader own them, and `eval_epoch` and two struct fields disappear.**
- `trainer.rs:160-190`: `evaluate` divides `eval_epoch` by `loader.num_items()`.
- `trainer.rs:392`, `:412-420`: `training_loop` divides the same sum by `components.n_test`, and the train sums by `components.n_train`.
- `trainer.rs:262-266`, `:304`: `n_train` / `n_test` are copied out of the datasets into `TrainingComponents` just to be carried to those divisions.

After T13, `evaluate` *is* the per-sample test loss, so the loop should call it:

```rust
let test_l2 = evaluate(&valid_model, &components.test_loader, eval_post);
// train side, same idea
train_l2: sums.l2_sum / components.train_loader.num_items() as f32,
```

What this deletes:
- `eval_epoch` (its body becomes `evaluate`'s);
- the `n_train` and `n_test` fields and their `(n_train, n_test)` plumbing;
- the `evaluate_is_mean_of_eval_epoch` test, which then tests a tautology.

Behaviour is bit-identical: the same f32 sum is divided by the same integer, because `num_items()` is `IndexDataset(len)`'s length. The existing `schedule_and_test_divisor_match_the_data` and `device_loaders_train_identically_to_host_loaders` tests pin it.

`EpochSums` (`trainer.rs:194-198`) is the same pattern one level down:
- `train_epoch` returns sums "because the caller owns the denominators", but the denominators are `loader.num_items()` and the batch count, both known inside `train_epoch`.
- It could return means directly, and `steps_per_epoch` would go from `TrainingComponents` too. It is only needed by the scheduler length, which `build_training_components` computes locally anyway.

**2.2 `metrics.csv` has two writers. Pick one.**
- `runs/mod.rs:166` (`append_metrics`, per epoch, from `train_fno` at `trainer.rs:469`) and `runs/mod.rs:177` (`save_metrics`, a full rewrite, from all three train examples) both write the file.
- T12 kept the rewrite as a "repair" for failed appends.
- That makes the file's final content depend on which writer ran last, and needs a byte-equality test (`trainer.rs:953-980`) to prove the two writers agree.

The direct structure is one writer:
- `append_metrics` is the CSV writer;
- `save_metrics` becomes `save_plot(&[EpochMetrics])` (plot only);
- a failed append aborts or warns loudly, and is not silently repaired later.

This deletes `write_metrics_csv`'s second caller and the byte-equality test. The incremental test then checks the rows directly. If the repair semantics really matter, say so in `RunDir`'s docs as the contract, not in a comment on the observer.

### 3. Spaghetti / branching growth

**3.1 `data/batcher.rs` is 15 lines of production code wrapped in ten `#[cfg(test)]` attributes** (`batcher.rs:7,10,12,14,31,34,65,68,84,133`). After T11, the file's public content is the `Batch` struct, and everything else is a test oracle gated item by item. These are cfg checks scattered through a production module. Remedy:
- Move `HostItems`, `OperatorBatcher` and their tests into one `#[cfg(test)] pub(crate) mod host_oracle;` (`data/host_oracle.rs`), with one gate on the `mod` line.
- Move `Batch` to `device_batcher.rs`, the module that produces every batch now, or leave `batcher.rs` holding only `Batch`.

Either way the gating is one line, and the production file reads as production code.

**3.2 `burgers_resolution.rs:92-103` branches twice on `rate == trained_rate`** (to reuse the baseline and to pick the marker). This is minor. Evaluate every requested rate with `test_l2(rate)` and compute the baseline once, so only the marker branch remains. It costs one extra evaluation at most, in an example.

### 4. Boundary, abstraction and type-contract problems

**4.1 `check_error3` returns `io::Result` for a condition that is not I/O** (`metrics/poisson.rs:158`). Since T10 it takes `&[EpochMetrics]`, and its only error is "no epochs", reported as `io::ErrorKind::InvalidData`. Callers stack two `expect`s (`examples/predict/poisson.rs:232`, `poisson_resolution.rs:118`). Remedy:
- Take the epoch it compares against: `check_error3(last: &EpochMetrics, mean_error3) -> Error3Check`, which is infallible.
- The caller's `metrics.last().expect("metrics.csv has epochs")` states the invariant where it is known.

This deletes the error path, `error3_check_rejects_no_epochs`, and the `io` import.

**4.2 `ConfigFile` plus `config::<C>(ConfigFile)` is a generic mechanism over three fixed shapes** (`runs/mod.rs:73-91`, `:224`).
- `Model` is always `FNOConfig` and `Train` is always `TrainingConfig`, yet nothing stops `config::<DarcyConfig>(ConfigFile::Model)`.
- The usages: `Data` in 7 examples, `Model` once (`burgers_resolution.rs:63`) plus internally in `load_fno`, and `Train` only in a test.

Remedy: typed accessors `model_config() -> FNOConfig`, `train_config() -> TrainingConfig` and `data_config<C: Config>() -> C`. The enum and its `file_name` match disappear, and the type is tied to the file.

**4.3 `models::run` vs `runs`** (`models/run.rs`; imported by `darcy_plot.rs:12`, `poisson.rs:29`, `poisson_plot.rs:27`, `poisson_resolution.rs:27`, `runs/mod.rs:264`). After T10, `models/run.rs` holds only `predict_host`, which has nothing to do with runs, beside a top-level `runs` module that owns run directories. Rename it to `models::predict` (or `models::inference`) so the names stop suggesting a relationship that doesn't exist.

**4.4 `train_fno` now requires a `&RunDir`** (`trainer.rs:451`), so the generic entry point of `training` depends on `runs`. That is acceptable for the one high-level entry point, and it is what T12 asked for. But it makes `train_fno` unusable without touching the filesystem. Tests now create temp run directories to call it (`trainer.rs:901`, `:960`).

If 2.2 lands, consider `train_fno(…, on_epoch: &mut EpochObserver)`, with `runs` providing the observer (`run.metrics_observer()`). Then `training` stops importing `runs`, and the examples choose `print_epoch` plus the run observer explicitly. That is two lines per example, and the dependency points the right way.

### 5. File size and decomposition

See 1.1. `fno.rs` (1595) is already scheduled for T20. No new file crosses 1k.

### 6. Modularity

**6.1 The accumulate-into-`Option<Tensor>` pattern appears three times** (`trainer.rs:178-185`, `:226-234`). `Some(match acc { Some(s) => s + x, None => x })` is written out for `mse_sum`, `l2_sum` and the eval sum. A two-line private `fn add(acc: Option<Tensor<1>>, x: Tensor<1>) -> Option<Tensor<1>>`, or folding over `loader.iter()`, removes the repetition. With 2.1, `train_epoch` and `evaluate` both shrink to a fold.

### 7. Legibility

**7.1 Examples repeat the same run-opening boilerplate.**
- `RunDir::open(...).unwrap_or_else(|e| panic!("open the run: {e}"))`, `load_fno`, `config(Data)` and `normalizers()…expect("a … run has normalizers")` appear in all eight predict examples (11 sites).
- `Path::new(env!("CARGO_MANIFEST_DIR")).join("runs")` appears in all four train examples.

This is acceptable for examples that are meant to be read standalone, so no action is required. If T22 introduces a shared example module for dataset paths, put `runs_root()` there too.

**7.2 `print_epoch` takes an unused `_model`** (`trainer.rs:356`) only to match `EpochObserver`, and `epoch_line` exists only so a test can see the string. Fine as is. If 4.4 lands, `print_epoch` can be `|m, t, _| println!(…)` at the call site and `epoch_line` stays the tested unit.

## Next steps

Each step preserves behaviour and can land as its own PR. Order is by complexity deleted.

1. **Let the loader own the divisors** (2.1, 6.1).
   - `training_loop` calls `evaluate`; `eval_epoch` folds into `evaluate`.
   - Train metrics divide by `train_loader.num_items()`.
   - Delete the `n_train` / `n_test` fields (and optionally `steps_per_epoch`, with `train_epoch` returning means).
   - Factor out the sum accumulation.
   - Verify: `cargo test` with the bit-identity tests `schedule_and_test_divisor_match_the_data`, `device_loaders_train_identically_to_host_loaders` and `train_fno_*_matches_*` unchanged and passing; `predict_burgers` / `predict_darcy` / `predict_poisson` print the Results-table numbers.
2. **Split `trainer.rs`** (1.1) into `config.rs`, `step.rs` (+`evaluate`) and `trainer.rs`, with the tests split alongside.
   - Depends on 1 (it moves less code afterwards).
   - Verify: the test count is unchanged and the six README checks pass.
   - Do it before T17, which rewrites `Postprocess` in `step.rs`.
3. **One writer for `metrics.csv`** (2.2, optionally 4.4).
   - `save_metrics` → `save_plot`; `append_metrics` is the only CSV writer.
   - Optionally, `train_fno` takes the observer and `runs` provides `metrics_observer()`.
   - Verify: replace the byte-equality test with a direct check of the appended rows, then do a short `train_burgers` run (epochs lowered locally) and check `wc -l metrics.csv` and that `plots.png` exists.
4. **Gate the batcher oracle once** (3.1): move it to `data/host_oracle.rs` behind one `#[cfg(test)] mod`, and put `Batch` beside `device_loader`. Verify: `device_batcher` and trainer tests pass with an unchanged count.
5. **Tighten the contracts** (4.1, 4.2, 4.3):
   - `check_error3(&EpochMetrics, f64) -> Error3Check`;
   - typed `model_config` / `train_config` / `data_config<C>` replace `ConfigFile`;
   - `models::run` → `models::predict`.
   - Verify: `predict_poisson` and `poisson_resolution` print the same error 3 cross-check, all examples compile (`clippy --examples`), and `cargo test` passes.
6. **Optional:** simplify the `burgers_resolution` branches (3.2). Verify: the same table as in #51's PR description.

Steps 1–2 overlap T17's files, so schedule them as a small Phase 4s add-on before T17, or fold them into T17's brief. Steps 3–5 are independent of each other.

## Resolution (applied in #51, 2026-10-22)

Steps 1–6 were applied on `phase4s/T13-evaluate-predict-host`. Optional 4.4 (the observer passed into `train_fno`) was not applied: `train_fno` still takes `&RunDir`.

1. **Divisors (2.1, 6.1):**
   - `eval_epoch` is deleted; `evaluate` folds `forward_loss` over the loader and divides by `num_items()`.
   - `train_epoch` returns `EpochMeans`, dividing by its own batch count and by `loader.num_items()`.
   - `TrainingComponents` loses `n_train`, `n_test` and `steps_per_epoch`.
   - The sum accumulation is one private `accumulate`.
2. **Split (1.1):** `training/{config, step, trainer}.rs` plus the shared test fixtures `training/fixtures.rs`. `trainer.rs` is 1004 → 732 lines and `step.rs` is 239.
3. **One writer (2.2):**
   - `append_metrics` is the only `metrics.csv` writer.
   - `save_metrics` → `save_plot` (infallible, warns without gnuplot).
   - `write_metrics_csv` is deleted.
   - The byte-equality test is replaced by `metrics_csv_holds_every_epoch`, which reads the rows back and compares them bit for bit with the returned metrics.
4. **Oracle (3.1):** `HostItems` / `OperatorBatcher` moved to `data/host_oracle.rs` behind one `#[cfg(test)]`; `batcher.rs` holds only `Batch`.
5. **Contracts (4.1–4.3):**
   - `check_error3(&EpochMetrics, f64) -> Error3Check` is infallible.
   - `RunDir::{model_config, data_config::<C>}` replace `ConfigFile` / `config`; no reader for `train_cfg.json` (nothing read it outside a test).
   - `models::run` → `models::predict`.
6. **`burgers_resolution` (3.2):** one evaluation path per rate; only the marker branch remains.

**Checks:**
- The six README checks and `clippy --features chebyshev --all-targets` pass.
- Lib tests go 210 → 209: `error3_check_rejects_no_epochs` was deleted with the error path; `evaluate_is_mean_of_eval_epoch` became `evaluate_is_the_per_sample_mean_of_the_summed_loss`; the CSV test was renamed.
- The bit-identity tests (`device_loaders_train_identically_to_host_loaders`, `train_fno_*_matches_*`) pass unchanged.

**Example re-runs:**
- `predict_darcy`, `predict_burgers`, `predict_poisson`, `burgers_resolution` and `darcy_plot` (on a scratch copy; PNG byte-identical) match the T13 baselines.
- `poisson_resolution 65` reproduces `predict_poisson`'s errors.
- A 3-epoch `train_burgers` (epochs lowered locally, not committed, into a local `runs/`) wrote `metrics.csv` (header + 3 rows) and `plots.png`.

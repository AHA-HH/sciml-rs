# T10: `runs` module with `RunDir`

Phase 4s, task 10. Branch `phase4s/T10-run-dir`. Depends on T2, T3, T4 and T9. It is the
largest task in the phase, so give it a focused stretch.
- Must merge before T11, T12, T13, T18 and T19, and before Phase 5 T2
  (checkpoint/resume).
- Can run in parallel with the Chebyshev track only. It edits every train and predict
  example.

> **Refresh at start:** re-read on `main`:
> - T2's Learner run writing in `examples/train/burgers_learner.rs`
> - T3's `train_fno`
> - T9's `loaders::common::Normalizers`
> - T8's `Transfers`, if merged: the Poisson predict examples will have changed.
>
> Re-grep the file-name literals below, since earlier tasks may have moved them.

## Read first
- Review §5 S1 (the whole section, including the sketched API), §6 T4-4, §7 step B1.
- `metrics/io.rs` (`write_run_artifacts` :70-97, `read_metrics_csv` :20,
  `write_metrics_csv` :49), `metrics/plots.rs`.
- `models/run.rs` (`load_fno` :27-32, `predict_host` :45-73).
- `training/metrics.rs` (`EpochMetrics` :9-15).
- `tests/run_artifacts_without_gnuplot.rs` (it clears `PATH`, so it needs its own test
  binary).

## Goal
The run directory is a file format with no owner. Its file names are spelled out by hand in
about 20 files, writing it is split three ways, and reading it is split between `load_fno`
and five examples that bypass it. Make one type own it before Phase 5 adds checkpoints.

## Facts verified at `6eb3212`
- **Weights are saved under one name and read under another.** They are saved as
  `model_weights` (`BurnpackStore` appends `.bpk`) in `train/{burgers:114, darcy:107,
  poisson:151, burgers_learner:100}`, and read as `model_weights.bpk` in `run.rs:30` and
  `predict/{burgers:38, darcy:44, burgers_resolutions:43, burgers_superresolution:52,
  darcy_plot:34}`.
- **File-name literals appear in about 20 files:**
  - `model_cfg.json`: 9 sites
  - `data_cfg.json`: 10 sites
  - `y_normalizer.json`: 6 sites
  - `metrics.csv`: 5 sites
  - `train_cfg.json`: 3 sites
  - `x_normalizer.json`: 2 sites, written and never read (T18)
- **`CARGO_MANIFEST_DIR`** appears inside the library at `metrics/io.rs:82`, and in 12
  examples.
- **Five predict examples load the model inline** instead of calling `load_fno`:
  `predict/{burgers, darcy, burgers_resolutions, burgers_superresolution, darcy_plot}`.

## Deliverables
1. **Test first.** Add a round trip: `RunDir::create` → `save_configs` / `save_weights` →
   `RunDir::open` → `load_fno`, whose outputs must be bit-identical to the original model on
   a fixed input. Write it against the planned API before migrating any example.
2. **A top-level `runs` module** (`src/neural_operators/runs/` or `src/runs.rs`; follow
   the crate's layout) owning `pub struct RunDir`.
   - **Creating:**
     - `create(root: &Path, name: &str) -> io::Result<Self>`. The root is a parameter;
       examples pass `Path::new(env!("CARGO_MANIFEST_DIR")).join("runs")`. The unix-seconds
       suffix is added as today.
     - `save_configs(&impl Config, &impl Config, &impl Config)`
     - `save_metrics(&[EpochMetrics])`: CSV, plus the plot if gnuplot is installed; warn
       otherwise, as today.
     - `save_normalizers(&Normalizers)`
     - `save_weights::<R>(&FNO<R>)`
   - **Opening:**
     - `open(dir)` and `path()`
     - `load_fno::<R>(&Device)`
     - `config::<C: Config>(ConfigFile)`, where `ConfigFile` is an enum
       `Model | Train | Data`
     - `normalizers() -> Result<Option<Normalizers>, …>`: `None` for Burgers runs
     - `metrics() -> io::Result<Vec<EpochMetrics>>`
   - **File names:** private constants, one for the weights stem. Store `model_weights`
     and let Burn append `.bpk`; reading uses the same stem, so the mismatch can't recur.
   - **Errors:** a small error enum, or `Box<dyn Error>` as `load_fno` uses today. Pick
     one and document it.
3. **What moves in:**
   - `metrics/io.rs` and `metrics/plots.rs`
   - `models/run.rs::load_fno`, which becomes a method. `predict_host` stays in
     `models/run.rs`; T13 makes it rank-generic.
   - `EpochMetrics`, from `training/metrics.rs`. That ends the two-`metrics` name clash.
     If `training/metrics.rs` is left empty, delete it.
   - The `metrics → training/models` imports disappear, because the configs come in as
     `&impl Config`.
   - Leave `metrics/poisson.rs` where it is. Renaming `metrics/` to `evaluation/` is
     optional: do it only if the import churn is small, and note the choice.
4. **Migrate every example:**
   - train: `burgers`, `darcy`, `poisson`, `burgers_learner` (T2's run writing moves onto
     `RunDir`)
   - predict: `burgers`, `darcy`, `burgers_resolutions`, `burgers_superresolution`,
     `darcy_plot`, `poisson`, `poisson_plot`, `poisson_resolution`

   After this, no example contains a run-file literal.
5. **Old runs stay readable:** every `runs/` directory written since Phase 3 opens.
   `RunDir::open` must not require files that older runs lack (for example, Learner runs
   from before T2).
6. **Docs:** CLAUDE.md "Layout" gets a `runs` line, and the `models/run.rs` line is
   updated. `tests/run_artifacts_without_gnuplot.rs` moves to the new API.

## Tests
- The round trip above.
- The `metrics/io` round-trip tests, moved along with the code.
- `tests/run_artifacts_without_gnuplot.rs`, still in its own binary.
- `normalizers()` returns `None` on a run without normalizer files.
- The test count does not drop.

## Acceptance
- `predict_poisson -- runs/poisson_fno_p4_1791922918` prints the same five errors as
  before. Fill the README Results row for T10.
- `predict_burgers` and `predict_darcy` print the same test L² on existing runs, so record
  them before starting.
- If a run or dataset is missing locally, say which and why.
- `git grep -nE '"(model_cfg|data_cfg|train_cfg|metrics)\.(json|csv)"|model_weights|_normalizer\.json' -- examples src`
  finds hits only inside `runs`.

## Checks to run and report
The six from the README. The chebyshev pair is needed only if `pde/` is touched (expected:
no). Report the line delta, and the example re-runs with the backend each one used.

## Do not
- Add checkpointing or resume. That is Phase 5 T2; this task only makes room for it.
- Change the on-disk format (file names, JSON shapes, CSV columns).
- Merge the batching paths (T11) or add `evaluate` (T13).
- Change what `x_normalizer.json` is used for (T18). It is still written.

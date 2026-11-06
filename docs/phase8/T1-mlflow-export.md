# T1: MLflow export of run directories

Phase 8, task 1. Branch `phase8/T1-mlflow-export`. Depends on Phase 5 T4 (run files).
Replaces the W&B export of design §13.10 (decision 30, amendment of 2026-11-06).

## Read first
- This phase's README (Q1–Q6, "Facts verified").
- `src/neural_operators/runs/mod.rs` (`RunDir::open`, `model_config`, `training_config`,
  `data_config::<C>`, `metrics`, `costs`) and `runs/csv.rs`.
- `examples/predict/poisson.rs` (`errors.json`, `ERRORS_JSON`) and
  `data/loaders/poisson/mod.rs` (`PoissonConfig`: n, s, K, d, subset).
- MLflow's REST reference for the pinned server version (`docs/api_reference/source/
  rest-api.rst` in the MLflow repository at that tag): `experiments/get-by-name`,
  `experiments/create`, `runs/create`, `runs/log-batch` and its limits, `runs/update`,
  `runs/search`, `runs/delete`, and the `mlflow-artifacts` `PUT`.
- `Cargo.toml` (`[features]`, the `datagen` comment style).

## Goal
`cargo run --features mlflow --example mlflow_export -- <experiment> runs/<dir> ...`
puts each finished run directory into a local MLflow server as one MLflow run. Configs
become params, `metrics.csv` becomes per-epoch metrics, `errors.json`/`costs.json`
become summary metrics, and the plots become artifacts. Training is not changed.

## Server setup (part of the deliverable, documented in the PR and `README.md`)
- Install a pinned MLflow outside the repo, e.g. `uv tool install mlflow==<X.Y.Z>` (or
  `pipx`); record the version in the PR.
- Run `mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri
  sqlite:///$HOME/mlflow/mlflow.db --artifacts-destination $HOME/mlflow/artifacts`.
  Bind to localhost only (Q6). Proxied artifacts are the default, so the client never
  writes the artifact store directly.
- The exporter reads `MLFLOW_TRACKING_URI` (default `http://127.0.0.1:5000`). No URI,
  account or key is committed.
- Runs trained on the VM: `rsync` the run directories back to `runs/` first.

## Deliverables
1. **Feature `mlflow`** in `Cargo.toml`, with `ureq` (rustls) and `serde_json` as
   optional dependencies, pinned with `=` like the others, and a comment in the
   `datagen` style. The default build and CI's default jobs are unchanged.
2. **`tracking` module** (`src/neural_operators/tracking/`, `#[cfg(feature =
   "mlflow")]`), split in two:
   - a **pure builder**: from a `RunDir` and experiment id, the list of requests to send
     (method, path, JSON body or file path). No I/O beyond reading the run directory, so
     it can be tested without a network;
   - a **client**: sends the requests in order, threads the `run_id` returned by
     `runs/create` into the rest, and maps HTTP and MLflow errors (`error_code`,
     `message`) into one error type. Any failed request stops the export of that run,
     marks the run `FAILED` through `runs/update`, and the example exits non-zero.
3. **Mapping** (one MLflow run per run directory):
   - run name: the directory name; tags: `sciml.run_dir` (Q5), `sciml.git_commit` (if
     the configs record it), `padding`, `s`, `n`, `K`, `seed`, `N` (subset size, or the
     whole split), `backend`, and `group = p{padding}_s{s}` so a cell's seeds sit together;
   - params: the three configs flattened to dotted keys (`model.modes`, `train.lr`,
     `data.n`, ...), values as JSON text;
   - metrics, one step per epoch from `metrics.csv`: `train_mse`, `train_l2`,
     `test_l2`, `lr`, and `epoch_time_s` when present. Batched within `log-batch`'s
     limits (per-request entity counts from the pinned reference; split as needed);
   - summary metrics, when the files exist: errors 2–5 from `errors.json` and the
     scalar fields of `costs.json`, logged at the last step;
   - artifacts, when present: `plots.png`, `prediction_*.png`, `errors.json`,
     `costs.json`, the three configs. Not the weights or `checkpoint.bin` (size, and the
     run directory stays the record);
   - status `FINISHED` with `end_time` after the last request.
4. **Idempotency (Q5):** before creating a run, `runs/search` on the experiment with
   `tags.sciml.run_dir = '<name>'`. If one exists, skip and say so; with `--replace`,
   delete it and export again.
5. **Example `mlflow_export`** registered in `Cargo.toml` with `required-features =
   ["mlflow"]`: `mlflow_export [--replace] <experiment> <run_dir> ...`. It creates the
   experiment if it does not exist, prints one line per run (exported / skipped /
   failed and the MLflow run URL), and exits non-zero if any run failed.
6. **Docs:** the server setup and the command in `README.md` and `CLAUDE.md`
   (Commands, Layout); README "Results" row filled in.

## Tests
No test needs a network or an MLflow server.
- Builder, against a run-directory fixture written in the test (temp dir): the exact
  request list for a run with every file, and for an old run (no `costs.json`, no
  `errors.json`, five-column `metrics.csv`): the summary metrics and their artifacts
  absent, no panic.
- Params flattening: nested config keys and vector values (modes) round-trip as
  expected strings.
- Metric batching: a 500-epoch fixture is split into requests within the per-request
  limits, with every (key, step) present exactly once.
- Client error mapping: an MLflow error body (`{"error_code": ..., "message": ...}`)
  becomes the module's error with both fields, against a canned response, not a socket.

## Acceptance (manual, recorded in the PR)
- With the local server running: export one Phase 5 baseline run and one pre-Phase-5
  run into a scratch experiment. In the UI, the run has its params, the `test_l2`
  curve with 500 steps, the summary errors and `plots.png`. The old run shows its
  curve and no summary errors.
- Re-running the same command skips both runs; `--replace` replaces them (run count
  unchanged).
- Export the Phase 5 matrix and group by `group`: each (padding, s) cell shows its seeds.
- State the MLflow server version used.

## Checks
`cargo fmt`, the CI set in `CLAUDE.md`, plus
`cargo clippy --no-deps --features mlflow --all-targets -- -D warnings` and
`cargo test --features mlflow`. Say whether CI should gain an `mlflow` clippy/test step
(proposed: add both to the style-checks and run-tests jobs. They need no network.)

## Do not
- Change training, `train_epoch`, `EpochObserver` or any run-file writer. Live logging
  is out of scope (design §13.10).
- Upload weights, checkpoints or datasets.
- Hard-code a tracking URI other than the localhost default, or commit credentials.
- Add an async runtime or the unofficial MLflow/W&B crates.
- Cite MLflow in any Results row. The run directories are the record.

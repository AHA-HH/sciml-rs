# Phase 8 (optional): experiment tracking and the θ-map

As of 2026-11-06: T1 (MLflow export and opt-in live logging) **done** (#91); T2 (the
θ-map model) is next, its brief not yet written; T3 follows T2. Phase 9 is paused (`docs/phase9/README.md`, "Paused")
and resumes independently when the GPU VM is back. Design:
`docs/design/2d-chebyshev-poisson-fno.md` (§10 Phase 8, §13.10, decision 30 and its
amendments of 2026-10-27 and 2026-11-06). `docs/CONVENTIONS.md` wins on any conflict.

Phase 8 holds two independent pieces of work:

1. **Experiment tracking (T1).** Export finished run directories to a self-hosted MLflow
   tracking server, so runs can be browsed, grouped and compared in a UI. It replaces the
   Weights & Biases export of the amendment of 2026-10-27 (planning decision Q1).
2. **The θ-map (T2, T3).** Old Phase 6 T3–T4, unchanged in substance: option 2 of §6
   with its own convention diff, compared against the Phase 5 baseline (decision 24's
   gate) at 65², 129² and across resolutions, on the same seeds and test sets.

The run directories stay the record of every result. MLflow is a view of them: no
Results row in any phase cites MLflow alone (design §13.10).

## Read first

- Design §13.10 (this phase), §13.9 (run-directory metadata), §6 (option 2, the θ-map),
  decisions 24 and 30.
- `src/neural_operators/runs/mod.rs` (`RunDir`: the run-file names and their readers).
- `CLAUDE.md`: Working agreement, Checks, Commands.

## Inputs from earlier phases

| Item | Value | Source |
| --- | --- | --- |
| Run directory | `model_cfg.json`, `train_cfg.json`, `data_cfg.json`, `metrics.csv` (one row per epoch), `model_weights.bpk`, `x_normalizer.json`, `y_normalizer.json`; `plots.png` when gnuplot ran | Phase 4s T10/T12 |
| Files added later | `costs.json` (Phase 5 T4; `optimisation_steps` Phase 7 T1, `resumes` Phase 9 T1), `errors.json` (`predict_poisson`, Phase 5), `prediction_<k>.png` (`poisson_plot`), `checkpoint.bin` (Phase 9 T1) | as listed |
| Older runs | Runs from before Phase 5 T4 have no `costs.json` and, unless evaluated, no `errors.json` (e.g. `runs/poisson_fno_p0_1791906496`); `metrics.csv` may have five columns (no `epoch_time_s`) | checked 2026-11-06 |
| Baseline for the θ-map | stage 3 (n = 129, K = 64), s = 64, p = 0, modes [12, 12], width 32, 4 layers, batch 20, 500 epochs, several seeds | Phase 5 T7 |

## Facts verified at `bfe62c5`

- **`RunDir` owns every run-file name** (`runs/mod.rs:52-66`), with typed readers:
  `model_config`, `training_config`, `data_config::<C>`, `metrics()` (old five-column
  files still read), `costs()`. `errors.json` is written by `examples/predict/poisson.rs`
  (`ERRORS_JSON`), not by `RunDir`. T1 reads through these, not by file name.
- **MLflow's tracking API is documented REST** (unlike W&B's GraphQL and file stream):
  `GET /api/2.0/mlflow/experiments/get-by-name`, `POST .../experiments/create`,
  `POST .../runs/create`, `POST .../runs/log-batch` (metrics with `step`, params, tags),
  `POST .../runs/update` (status, end time). Artifacts go through the server's proxy,
  `PUT /api/2.0/mlflow-artifacts/artifacts/<experiment_id>/<run_id>/artifacts/<path>`,
  when the server runs with `--artifacts-destination` (proxied artifact access).
  T1 pins the server version and checks these against its `rest-api` reference.
- **No HTTP client is a dependency today.** `serde_json` is optional, behind `datagen`.
  T1 adds both behind the new `mlflow` feature, so the default build, CI's default jobs
  and the licence are unchanged.
- **The training loop is untouched.** The exporter reads finished run directories.
  Live logging (Q7) is opt-in: `train_components` takes one optional extra
  `EpochObserver` (`training/trainer.rs`), which `train_poisson` fills only when built
  with `mlflow` and run with `MLFLOW_EXPERIMENT_NAME` set; `training_loop` and the
  run-file writers are unchanged.

## Planning decisions

Q1 to Q3 and Q7 were chosen by the author (AHA-HH) on 2026-11-06 and are recorded in design §12, decision 30
(amendment of 2026-11-06, Phase 8 planning). Q4 to Q6 were proposed in the plan, built
as proposed in T1 (#91) and confirmed by the author on 2026-11-06.

- **Q1, tracker: MLflow, not W&B.** W&B has no Rust SDK and an undocumented upload
  protocol. MLflow's tracking server has a documented REST API, is open source and is
  self-hostable, so no third-party account or key is needed. The W&B task is dropped.
- **Q2, scope and order:** T1 is the MLflow export; the θ-map follows as T2 (model and
  convention diff) and T3 (comparison). Only T1 has a brief now.
- **Q3, hosting: the author's Mac**, `mlflow server` bound to `127.0.0.1`, SQLite backend
  store and a local artifact directory, both outside the repository. The GPU VM needs no
  network access to the server; run directories are copied back (`rsync`) before export,
  as design §13.10 had for W&B. Setup is in T1's brief.
- **Q4 (confirmed 2026-11-06), HTTP client: `ureq`** (blocking, small, rustls), with `serde_json`,
  both behind `mlflow`. The exporter is a short sequential batch job, so it needs no async
  runtime.
- **Q5 (confirmed 2026-11-06), idempotency:** each MLflow run is tagged with its run directory's name
  (`sciml.run_dir`). Re-exporting a directory that is already in the experiment skips it,
  unless `--replace` is given, which deletes the old run first. Without this, an export
  repeated by mistake would double the runs in every comparison view.
- **Q6 (confirmed 2026-11-06), no auth in T1:** the server is bound to localhost only. MLflow's basic
  auth (`--app-name basic-auth`) and remote hosting are later work, needed only if the
  server ever leaves the Mac.
- **Q7 (chosen by the author, 2026-11-06), opt-in live logging in T1:** with the `mlflow`
  feature and `MLFLOW_EXPERIMENT_NAME` (MLflow's own variable) set, `train_poisson` logs
  the run as it trains: the run with its tags and params at the start, each epoch's
  metrics, and the costs, artifacts and `FINISHED` at the end, through the exporter's
  request builder. Never on by default. Best-effort: the first failed request turns it off
  for the run with one warning, and training goes on. A `--resume` continues the same
  MLflow run (found by `sciml.run_dir`), so epochs trained again after the checkpoint
  appear twice. The export stays the authoritative record: `mlflow_export --replace`
  rewrites the run from the finished directory, `errors.json` included. GPU-VM runs can
  log live through a reverse tunnel (`ssh -R 5001:127.0.0.1:5001 <vm>`).

## Tasks

| Task | Brief | Delivers | Depends on |
| --- | --- | --- | --- |
| T1 | [T1-mlflow-export.md](T1-mlflow-export.md) | `mlflow` feature: a Rust exporter of run directories to a self-hosted MLflow server, the `mlflow_export` example, and the server setup | Phase 5 T4 (run files) |
| T2 | to be written (T1 merged) | The θ-map model (option 2 of §6) and its convention diff (`CONVENTION_VERSION` bump) | Phase 5 T7, Phase 4s T16 |
| T3 | written when T2 merges | θ-map against the Phase 5 baseline at 65², 129² and across resolutions, same seeds and test sets | T2 |

Task numbers are execution order (design §10). One task per branch and PR:
`phase8/T<k>-<name>`. Each task is worked from its brief alone, plus the files the brief
lists under "Read first".

## Module layout

- `src/neural_operators/tracking/` (T1, behind `mlflow`): the request builder (pure:
  run directory → list of MLflow requests), the client that sends them, and
  `MlflowObserver` (live logging, Q7).
- `training/trainer.rs` (T1): `train_components` takes an optional extra
  `EpochObserver`; `examples/train/poisson.rs` passes the `MlflowObserver` in `mlflow`
  builds.
- `examples/mlflow_export.rs` (T1): the command line.
- `runs/` (T1): read-only use; at most one public reader added if a file has none.
- `models/`, `chebyshev/`, CONVENTIONS (T2, T3): to be set by T2's brief.

## Results

Filled in as tasks merge. Phase 8 is done when every row has a value, or a task is
dropped with the reason recorded in design §12.

| Item | Value | Source |
| --- | --- | --- |
| Run directories exported to MLflow, and logged live during training | Done (#91), MLflow 3.18.0 on `127.0.0.1:5001`, 2026-11-06. **Export:** the Phase 5 baseline run `poisson_fno_n129_p0_1793189507` arrived `FINISHED` with 10 tags (`group = p0_s128`), 25 params, `test_l2` over 500 steps (0.280 → 0.01045), errors 2–5 and the costs as summary metrics, and its JSON artifacts; shown in the UI; a repeat export skipped, `--replace` kept one run. **Live:** `train_poisson 0 65 16 42 25 7` (flex, 500 epochs) logged epoch by epoch to one `FINISHED` run, `plots.png` uploaded at the end; a run stopped by SIGTERM after 50 epochs and resumed continued the same MLflow run, 500 distinct steps. Not exercised: non-finite metric values, a server dropping mid-run (unit-tested against a closed port). The Phase 5 and Phase 7 matrices are not yet exported (optional) | T1 |
| θ-map model and convention diff | | T2 |
| θ-map against the Phase 5 baseline | | T3 |

## Open items

Known work outside T2–T3, recorded so it is not only in old review documents. None
blocks Phase 8 or the programme's completion (design §13.16); each becomes a task only
when the author schedules it.

- **MLflow follow-ups (T1):** the `mlflow` clippy and test steps in CI (neither needs a
  network; CI does not compile the feature today); exporting the Phase 5 and Phase 7
  matrices; Burgers and Darcy run directories (reported `failed` today).
- **`train_components` has eight arguments** behind the repository's first `#[allow]`
  (`clippy::too_many_arguments`, T1). R1 of the 2026-10-24 review ("`train_fno` owns the
  run") would remove it.
- **Code-quality leftovers of the 2026-10-24 review**
  (`docs/code-quality/thermo-nuclear-repo-review-2026-10-24.md`, partly resolved): S1,
  `errors.json` and `prediction*.png` named in examples rather than in `RunDir`; S3, the
  `load_X` / `load_X_test` pairs of Darcy and Poisson, the run-write tail copied in three
  train examples and the predict → decode block copied four times. T2 touches the train
  and predict examples, so its brief decides whether to clear these first.
- **Deferred research:** D6's fixed optimisation-step budget for Phase 7 ("deferred to
  later research", decision 30); the aliasing row of Phase 5 T1 (the effect of the
  unfiltered R on f, needs `datagen`, skipped by the author's choice).

## Exit checklist

- [ ] T1–T3 merged, or a task dropped with the reason recorded in design §12.
- [x] Q1–Q3, Q7 chosen by the author and recorded in design §12 (2026-11-06).
- [x] Q4–Q6 confirmed by the author after T1 (#91), 2026-11-06.
- [ ] No result row cites MLflow alone.
- [ ] Every row of Results filled in.
- [ ] The "Current phase" bullet in `CLAUDE.md` updated for what comes next.

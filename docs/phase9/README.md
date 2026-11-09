# Phase 9: HPC (systems track) and the θ-map

As of 2026-11-09: T2–T4 **paused** (author, 2026-11-06). T1 done (#87); T2's code merged
(#88), its acceptance run blocked on the VM (see "Paused" below); T3 and T4 not started.
T5–T6 (the θ-map, moved here from Phase 8 on 2026-11-09) are not paused: T5's brief is
approved, decisions P8–P17 chosen by the author on 2026-11-09; T5 is next. Tasks
renumbered 2026-11-03 and 2026-11-04 (below). Design: `docs/design/2d-chebyshev-poisson-fno.md` (§10 Phase 9 and the old
Phase 5 rows it inherits, §13.11, decisions 19, 21 and 30).
`docs/CONVENTIONS.md` wins on any conflict.

Phase 9 makes training at production sizes fast and robust: know where the time goes,
survive interruption, and train one run on more than one GPU. It also carries the three
supplementary tasks of old Phase 5: example smoke runs in CI, the full-length flex run
(decision 21) and smooth display in `poisson_plot` (decision 19); the first and the last
closed at planning (below).

It is a systems track. **No speedup it delivers enters a conclusion of Phases 5–7**
(design §13.11), and a multi-GPU run must reproduce its single-GPU result on the same
seeds, within a stated round-off, before it is used.

Since 2026-11-09 it also carries **the θ-map (T5–T6)**: option 2 of design §6, old Phase 6
T3–T4, then Phase 8 T2–T3 (decision 30, amendment of 2026-11-09). These are research,
not systems work: the speedup rule above applies to T1–T4, and T6 compares error and
cost against the Phase 5 baseline on one device of one kind, like Phases 5–7. T5 runs on
the author's Mac and does not wait for the VM.

## Read first

- Design §13.11 (this phase), §10 "Phase 5: HPC (optional) — superseded" (the old task
  rows and supplementary tasks, kept as history), decisions 17, 19, 20, 21 and 30.
- `docs/phase5/README.md` (the baseline and its cost) and `docs/phase7/README.md` (the
  per-epoch cost fit).
- `CLAUDE.md`: Working agreement, Checks, Burn, Commands.

## Inputs from earlier phases

| Item | Value | Source |
| --- | --- | --- |
| Baseline run | stage 3 (n = 129, K = 64), s = 64, p = 0, modes [12, 12], width 32, 4 layers, batch 20, 500 epochs, Adam 1e-3 / wd 1e-4, cosine | Phase 5 T7 |
| Baseline cost (cuda, A100) | 1.69 s/epoch, about 851 s per run; per-epoch time flat from s = 16 to 64 | Phase 5 T5 |
| Per-epoch cost fit (cuda, A100, s = 64) | 0.117 s + 1.53 ms × N: a fixed cost, mostly the 200-sample test pass after every epoch, plus about 31 ms per training batch of 20 | Phase 7 T2 |
| Old HPC numbers | 129² at p = 4 on cuda: about 112 s/epoch, 15.6 h for 500 epochs; any p > 0 about 20× slower per epoch than p = 0 (Bluestein at non-power-of-two padded lengths) | Phase 4, decisions 17, 20 |
| Flex cost | stage 3, s = 64, p = 0, N = 25: 2.4 s/epoch (Phase 7 T1 smoke run); 65² at p = 4: about 520 s/epoch | Phase 7 T1, decision 17 |
| What a run saves | configs at start, one `metrics.csv` row per epoch (`EpochObserver`), weights, normalisers and `costs.json` at the end only; no optimiser or scheduler state | Phase 4s T10/T12, Phase 5 T4 |

## Facts verified at `8a545e5`

- **The observer sees the model only.** `training_loop` (`training/trainer.rs`) calls
  `on_epoch(&EpochMetrics, &FNO<R>)`; the optimiser and scheduler stay inside
  `TrainingComponents`. A checkpoint needs both, so T1 either widens the observer or
  saves inside the loop.
- **Burn can serialise the state T1 needs.** `ModuleOptimizer` has `to_record`,
  `load_record`, `save` and `load` (`burn-optim/src/optim/module/module_optimizer.rs`);
  `ModuleLrScheduler` has `to_record` / `load_record`, and the cosine scheduler's record
  is its `current_iter`.
- **The shuffle is seeded once.** `device_loader(.., shuffle: Some(seed))` builds one
  loader whose RNG advances every epoch. A resumed run at epoch k must see the epoch-k
  order, not epoch 0's, to match an uninterrupted run.
- **Burn has distributed training.** `burn::tensor::distributed` gives
  `DistributedContext::init(devices, config)` and `all_reduce(tensor, Sum | Mean,
  devices)`; `burn-train` has a DDP strategy (one thread and model replica per device,
  gradients all-reduced after backward). `all_reduce` works on cuda only (corrected
  in T2, cubecl `04676cd`): `ComputeClient::all_reduce` panics with "no transport
  between its devices" unless `server_comm_enabled` is set
  (`cubecl-runtime/src/client.rs:916-929, 973`), the default `Server::all_reduce` returns
  `no_device_transport` (`cubecl-runtime/src/server/base.rs:884-893`), and only
  cubecl-cuda sets the flag, when the NCCL library loads
  (`cubecl-cuda/src/runtime.rs:406`). Flex, metal and wgpu panic: multi-device code
  cannot run on flex, on metal or in CI.
- **NCCL is not a Burn feature at this pin, but cuda's collectives are NCCL.** No crate
  in the fork names NCCL; the collective goes through cubecl's `client.all_reduce`, which
  on cuda is `ncclAllReduce` on a dedicated communication stream (found in T2:
  `cubecl-cuda/src/compute/communication.rs:135-160`, `ncclAvg` for Mean, `ncclSum` for
  Sum; `compute/server.rs:642-680`), NCCL loaded at run time by `cudarc` (`dlopen`), so
  the machine needs `libnccl` installed. No host round trip.
- **The design asked for a hand-written data-parallel loop**, not the Learner's
  `MultiDevice` (old Phase 5 T3), because `train_epoch` is the canonical loop and
  `MultiDevice` splits the dataset, changing the effective batch.

## Planning decisions

Recorded in design §12, decision 30 (amendments of 2026-11-02 to 2026-11-06). P1, P4
and P7 are decided there; P2 and P3 are confirmed by the author (AHA-HH) for T1,
2026-11-02; P6 on 2026-11-03 and P5 on 2026-11-04; P4 amended 2026-11-04.

- **P1, profiling target: the Phase 5 baseline** (s = 64, p = 0, N = 1000, cuda A100),
  not the old 129² p = 4 run. The baseline is what every later run trains; the
  Bluestein cost at p > 0 is already measured (decisions 17, 20) and p = 0 is the
  chosen padding (D3).
- **P2, checkpoint cadence: every k epochs (default 25) plus the last epoch**, one
  checkpoint kept (overwritten), written atomically (temporary file then rename).
  Confirmed by the author, 2026-11-02. A failed write stops training (author,
  2026-11-02).
- **P3, resume must be exact on flex:** an interrupted-and-resumed run gives the same
  `metrics.csv` rows and weights as an uninterrupted one, bit for bit, on flex. On GPU,
  within a stated round-off. Confirmed by the author, 2026-11-02.
- **Shuffle across resume (T1): replay the shuffle.** The resumed loader's RNG is
  advanced k epochs, so the shuffle order is today's and old runs stay reproducible bit
  for bit; per-(seed, epoch) seeding is not adopted. Chosen by the author, 2026-11-02.
- **P4, multi-GPU: Burn's `all_reduce` inside a hand-written data-parallel step**, not
  the Learner's DDP strategy and not `GradientsAccumulator` across devices. Each batch
  of 20 is split across the devices, gradients are sum-reduced, and one optimiser step
  is taken, so the effective batch and the schedule are unchanged. **Amended by the
  author, 2026-11-04: sum, not mean.** The loss is the relative L² summed over the
  batch, so the summed shard gradients are the full-batch gradient; a mean would scale
  it by 1/D, which Adam's L2 weight decay does not cancel.
- **P5, hardware: a 2× A100 machine for T2 and T3.** Confirmed by the author,
  2026-11-04: a 2× A100 VM is available. Code is written on the author's Mac and the
  cuda checks and GPU runs are done on the VM. T1 and T4 do not need it.
- **P6, the flex run at p = 0, s = 64 on stage 3** (the baseline), recorded against the
  §11 threshold, rather than 65² at p = 4 (about 72 h). Confirmed by the author,
  2026-11-03.
- **P7, plot shading: flat cells kept** (chosen by the author, 2026-11-02; decision 19).
- **P8–P17, the θ-map (T5):** chosen by the author on 2026-11-09, all as recommended in
  `T5-theta-map.md` ("Decisions"): the even extension as an `FNOConfig` option (P8),
  once around the spectral block (P9), grid channels θ/π (P10), 12 modes with ~24 as a
  possible T6 arm (P11), `PoissonConfig::cgl` in the existing loaders (P12), the
  unweighted relative L² loss (P13), `theta` in the s argument (P14), `predict_poisson`
  and `poisson_resolution` in scope (P15), the review leftovers worked around (P16),
  Metal for acceptance if its parity case passes, else flex (P17).

## Tasks

Renumbered 2026-11-03 (author): the tasks closed at planning are removed, and the rest
are numbered in the order they run. Renumbered again 2026-11-04 (author), when the
flex run moved to the end of the phase. From the plan of 2026-11-02: T1 was T2, T2 was
T3, T3 was T4, T4 was T6. Branch `phase9/T2-checkpoint` and PR #87 keep the old name.

| Task | Brief | Delivers | Depends on |
| --- | --- | --- | --- |
| T1 | [T1-checkpoint.md](T1-checkpoint.md) | Periodic checkpoints (weights, optimiser, scheduler, epoch) and a run that resumes from them, exact on flex | – |
| T2 | [T2-multi-gpu.md](T2-multi-gpu.md) | Data-parallel training of one run on several GPUs with an unchanged effective batch; reproduces a single-GPU run within round-off | T1; a multi-GPU machine |
| T3 | [T3-scaling.md](T3-scaling.md) | One- against two-A100 scaling of the baseline: s/epoch, speedup, efficiency, where the time goes (with the trace and GPU utilisation from the closed profiling task) | T2 |
| T4 | [T4-flex-run.md](T4-flex-run.md) | The full-length baseline run on flex, cost stated first; the §11 threshold checked there | T1; last, by the author's choice |
| T5 | [T5-theta-map.md](T5-theta-map.md) | The θ-map model (option 2 of §6) and its convention diff (`CONVENTION_VERSION` bump) | Phase 5 T7, Phase 4s T16; not the VM |
| T6 | written when T5 merges | θ-map against the Phase 5 baseline at 65², 129² and across resolutions, same seeds and test sets | T5 |

T5–T6 joined on 2026-11-09 (author), as Phase 8 T2–T3 renumbered. They are independent
of T1–T4 and may run before them while the VM is down.

Task numbers are execution order (design §10). One task per branch and PR:
`phase9/T<k>-<name>`. Each task is worked from its brief alone, plus the files the brief
lists under "Read first".

## Closed at planning

Three tasks of the plan of 2026-11-02 closed that day without new work; their briefs are
removed (in git history at `a736dfc`) and design §12 decision 30 keeps the record.

- **Profiling (old T1), from existing data.** Closed 2026-11-02 from existing data, no new runs. cuda (A100), wall clock from `costs.json` / `metrics.csv`. Per epoch at N = 1000: about 1.53 s in 50 training batches (about 31 ms per batch of 20) and about 0.12 s fixed, mostly the 200-sample test pass (Phase 7 T2 fit 0.117 s + 1.53 ms × N, which predicts 1.65 s against 1.69 s measured). s/epoch is flat in the learning grid: 1.679 / 1.686 / 1.687 s at s = 16 / 32 / 64 (16× the grid points), 2.448 s at s = 128 (Phase 5 T5). **Finding:** up to s = 64 a training batch costs the same whatever its size, so the time is per-step overhead (kernel launches, synchronisation, host work), not arithmetic. Splitting a batch of 20 over two GPUs halves the arithmetic, not the overhead, so multi-GPU (T2) should expect little speedup at s = 64 and more at s = 128 or stage 4. **Not measured:** which overhead (no trace) and GPU utilisation; both move to scaling (T3), which profiles the 1- and 2-GPU runs anyway
- **Example smoke runs in CI (old T5), folded into T1.** Closed without a CI job (author, 2026-11-02). `train_poisson_smoke` (`training/trainer.rs`) already trains a Poisson FNO end to end through `train_fno` on flex, from synthetic `.npz` splits through `load_poisson_uniform` into a run directory, in every `cargo test` (CI's run-tests job). T1 extends it to stop and resume. **Not covered:** the `main`s of `train_poisson` and `predict_poisson` (argument parsing, the real dataset paths); CI only compiles them
- **Plot shading (old T7).** Flat cells kept (author, 2026-11-02): each CGL node stays one flat cell, so the plot shows the data as sampled and where the nodes are. Decision 19 closed; no code change

## Module layout

- `training/trainer` (T1, T2): the checkpoint hook in `training_loop`; the
  data-parallel step beside `train_epoch` (T2).
- `runs/` (T1): checkpoint files and their reader; the only place their names appear.
- `examples/train/poisson.rs` (T1, T2): resume and device-list arguments.
- No change to the model, the loss, the transfers, the hyperparameters or CONVENTIONS in
  T1–T4.
- `models/fno`, `data/loaders/poisson`, `metrics/poisson`, the train and predict
  examples, and CONVENTIONS with a version bump (T5): as set by T5's brief.

## Results

Filled in as tasks merge. Phase 9 is done when every row has a value.

| Item | Value | Source |
| --- | --- | --- |
| Checkpoint and resume, exact on flex | Exact on flex. `checkpoint.bin` (weights, Adam record, cosine record, next epoch, resumes, training time) every 25 epochs and after the last, one per run, temp file then rename; `train_poisson -- --resume runs/<dir>` reads the run's configs, truncates `metrics.csv` and replays the shuffle. Tests: 6 epochs at cadence 2, killed in epoch 3 and resumed: every row but `epoch_time_s` and every weight bit-identical on flex; on metal the same test passes with rows bit-identical and weights within 1e-5 relative (the tolerance tested, not a measured spread). Acceptance (flex, stage 2, s = 32, p = 0, N = 100, 6 epochs at cadence 2, temporary literals): SIGINT during epoch 5 after 4 rows, resumed from epoch 4; all 6 rows equal an uninterrupted run's but `epoch_time_s`, and `predict_poisson`'s `errors.json` (errors 2–4, all digits) identical; `costs.json`: `resumes` 1 | T1 |
| Multi-GPU run reproduces single-GPU, round-off stated | | T2 |
| Speedup and efficiency, 1 vs 2 A100 | | T3 |
| Flex run against the §11 threshold | | T4 |
| θ-map model and convention diff | | T5 |
| θ-map against the Phase 5 baseline | | T6 |

## Paused (2026-11-06)

T2–T4 paused by the author; T5–T6, added 2026-11-09, are not. Where each task stands and what restarts it:

- **T2:** code merged (#88), checked on flex and metal, type-checked on cuda. The first VM
  run (2× A100, 2026-11-05) failed before any training: the gradient test's
  `ncclCommInitRank` returned `NcclError`, and `NCCL_DEBUG=INFO` gave `CUDA driver
  version is insufficient for CUDA runtime version`: the VM's `libnccl` is built for a
  newer CUDA than its driver supports. Our code and cubecl are not at fault. **Restart:**
  on the VM, a Spack environment with `nccl ^cuda@X.Y`, X.Y no newer than `nvidia-smi`'s
  "CUDA Version" (author's choice of tooling, 2026-11-05); `spack env activate` before the
  run; then the gradient test with `NCCL_DEBUG=WARN`, the 2-GPU baseline run and
  `predict_poisson`, as in #88's runbook. Record driver, CUDA and NCCL versions with the
  result.
- **T3:** waits for T2's acceptance; runbook written then.
- **T4:** the 3-epoch timing is taken (about 29 s/epoch on the M2 Pro, about 4.1 h for
  500); the full run is last, by the author's choice.
- **T1 follow-up merged:** #89 (metal round-off in run-against-run test comparisons).

## Exit checklist

- [ ] T1–T6 merged, or a task dropped with the reason recorded in design §12.
- [x] P1–P7 confirmed by the author and recorded in design §12 (2026-11-06).
- [ ] No Phase 9 speedup (T1–T4) cited in a Phase 5–7 result.
- [x] P8–P17 (T5's decisions) chosen by the author and recorded in design §12 (2026-11-09).
- [ ] Every row of Results filled in.
- [ ] The "Current phase" bullet in `CLAUDE.md` updated for what comes next.

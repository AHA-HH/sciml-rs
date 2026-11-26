# CL4.2 3D baseline

Started 2026-11-26 on `laplace3d/cl4-2`. Roadmap: "CL4.2 3D baseline. Training at n = 65
over seeds." Proposal: §9 (model validation: errors 2–5 in 3D, mean and max over the test
split, seeds as in 2D Phase 5), §10 (L4), §3.3 (the input families differ), §7 (cost), §8
R3 (cuda parity before CL4.2); D4, D5, D8.

## Goal
A 3D Poisson baseline at n = 65 over seeds, errors 2–5 against the 3D floor, with 2D Phase 5
at the same s as context; first, the two small code changes the roadmap puts in this task's
first commit.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)
- "the results table with errors 2–5 against the 3D floor, and 2D Phase 5 at the same s as
  context (the input families differ, proposal §3.3)."
- "Before its first cuda run, CL0.2's cases run on cuda:
  `cargo test --release --no-default-features --features cuda --test gpu_parity`."
- First commit (CL4.1 review F3): "a 3D set lookup by n in `data/catalogue` (as `stage` for
  2D), used by `train_poisson3d` and `predict_poisson3d` in place of their two inline
  `GENERATED_3D` searches."
- First commit (L3 gate Q2), test and type only: "a CI test that writes a 3D split at n = 9
  and loads it back with the 3D loader, and `load_poisson3d_cheb_field` returning `Array4`."
- Proposal §9: "Model: errors 2–5 in 3D, mean and max over the test split, seeds as in 2D
  Phase 5."

## Read first
`src/neural_operators/data/catalogue.rs` (`stage`, `NoStage`, `GENERATED_3D`);
`examples/train/poisson3d.rs` (`set`), `examples/predict/poisson3d.rs`;
`data/loaders/poisson/cheb.rs`; `pde/poisson/dataset3d.rs` (`files_round_trip_3d`);
`docs/poisson2d/phase5/results.md` (seeds 42–44 at s = 32, 42–46 at s = 64; p = 0);
`docs/laplace3d/results/cl0-1.md` (cuda epoch times, p = 0).

## Changes
1. `data/catalogue`: `set_3d(n) -> Result<Poisson3dSet, NoSet3d>`, the 3D twin of `stage`
   over `GENERATED_3D`; `NoSet3d` names the sets' n. `train_poisson3d`'s private `set` and
   `predict_poisson3d`'s inline search call it.
2. `load_poisson3d_cheb_field` returns `Array4<f64>` (as the 2D one returns `Array3`); its
   two callers in tests and `predict_poisson3d` (which passes `.view()` to the generic
   `poisson_errors_nd`) follow.
3. `pde/poisson/dataset3d` test `split_3d_loads_back`: `generate_3d` at n = 9, both splits
   written by `write_split_3d`, loaded by `load_poisson3d_uniform` (s = 8) and
   `load_poisson3d_cheb_field`. Behind `datagen` (CI's `run-tests-datagen`).
4. The results report `docs/laplace3d/results/cl4-2.md`, after the runs.

Public API: `catalogue::set_3d`, `catalogue::NoSet3d`; `load_poisson3d_cheb_field`'s return
type. Conventions: none. `tests/gpu_parity.rs` is run, not edited, so the L0 gate Q2 tidy-up
stays with the next task that edits it.

## Tests that prove it
- `catalogue::tests::set_3d_finds_each_set_and_rejects_other_n`: each `GENERATED_3D` set
  found by n; n = 129 rejected with a message naming `[33, 65]`.
- `dataset3d::tests::split_3d_loads_back`: the cheb field equals `split.u` bit for bit
  (`Array4`); the uniform loader's raw targets equal `Transfers::new(9).to_uniform_all_nd`
  of `split.u` bit for bit (the loader's own transfer path), shapes `[N, 8, 8, 8(, 1)]`.
- Existing: `cheb_loader_3d_reads_raw_u`, `uniform3d` tests, unedited but for the type.
- Constraint 1: no 2D code is touched (catalogue's 2D items, the 2D loader and
  `metrics::poisson` unedited); the whole 2D suite passes.

## Checks
`/verify` (fmt, the CI clippy jobs, `cargo clippy --all-targets`, `cargo test`, `cargo doc`,
datagen clippy and tests), `CARGO_TARGET_DIR=~/sciml-rs-target/cl4-2`. Not backend-sensitive
on the Mac; the cuda parity cases run on the VM (below).

## VM runs
All on the VM at the author's go, started by the coordinating session; one run per GPU
(D8; 2 GPUs on one model are slower on this stack). Padding p = 0 (2D Phase 5's main arm and
CL0.1's timings; `train_poisson3d`'s first argument), n = 65, 500 epochs, D4's model.

Proposed matrix (seeds as in Phase 5, §9):

| s | seeds | h per run (CL0.1, cuda) | GPU-h |
| --- | --- | --- | --- |
| 32 | 42, 43, 44 | ~1.6 | ~4.8 |
| 64 | 42, 43, 44 (45, 46 only if the spread asks, as Phase 5's D4) | ~8.9 | ~26.7 |
| | | total | ~31.5 GPU-h; ~16 h wall on 2 GPUs |

Plus `predict_poisson3d` per run (minutes) and the parity tests (minutes).

Commands (after the VM is free of CO0.1 stage 3, `tmux ls` empty):

```sh
cd ~/sciml-rs && git fetch origin
git worktree add --detach ~/sciml-rs-wt/cl4-2 origin/laplace3d/cl4-2
cd ~/sciml-rs-wt/cl4-2
find ~/sciml-rs/datasets -mindepth 1 -maxdepth 1 ! -name README.md -exec ln -s {} datasets/ \;
source ~/sciml-gpu.sh
export CARGO_TARGET_DIR=~/sciml-rs-target/cl4-2   # one build directory per task
F="--release --no-default-features --features cuda"

# 1. CL0.2's cases on cuda, before any 3D cuda run (must pass; stop if not)
cargo test $F --test gpu_parity 2>&1 | tee ~/sciml-rs/logs/cl4-2-parity.log

# 2. Build once, then two queues, one per GPU. Distinct seeds within each GPU's queue
#    and a 120 s stagger between GPUs (RunDir collision, until optimisers CO0.4).
cargo build $F --example train_poisson3d --example predict_poisson3d
T=$CARGO_TARGET_DIR/release/examples
tmux new -d -s cl4-2-g0 "export CUDA_VISIBLE_DEVICES=0; for a in '32 42' '32 43' '64 42' '64 44'; do set -- \$a; $T/train_poisson3d 0 65 \$1 \$2; done 2>&1 | tee ~/sciml-rs/logs/cl4-2-g0.log"
sleep 120
tmux new -d -s cl4-2-g1 "export CUDA_VISIBLE_DEVICES=1; for a in '32 44' '64 43'; do set -- \$a; $T/train_poisson3d 0 65 \$1 \$2; done 2>&1 | tee ~/sciml-rs/logs/cl4-2-g1.log"

# 3. Check that every "run written to" line is unique:
grep -h "run written to" ~/sciml-rs/logs/cl4-2-g*.log | sort | uniq -d   # must print nothing

# 4. After both queues: evaluate each run (from the worktree, so cubek.toml applies)
for d in runs/poisson3d_*; do $T/predict_poisson3d $d; done 2>&1 | tee ~/sciml-rs/logs/cl4-2-predict.log
```

Queues: GPU 0 ~1.6 + 1.6 + 8.9 + 8.9 ≈ 21 h; GPU 1 ~1.6 + 8.9 ≈ 10.5 h. Balanced alternative
(GPU 1 takes seed 44 at s = 64 too): GPU 0 `'32 42' '32 43' '64 42'` ≈ 12.1 h, GPU 1
`'32 44' '64 43' '64 44'` ≈ 19.4 h; the plan uses this balanced one if the author agrees.
The binaries are run directly so the checkout's `cubek.toml` (TF32 off) is found from the
worktree's working directory.

What it decides: the L4 gate's baseline (errors 2–5 over seeds) and whether s = 64 needs
seeds 45, 46.

## Constraints and risks
- Constraint 1 (§1.4, D2): no 2D function edited; the 2D suite passes unchanged.
- R3: cuda parity of the 3D layer unverified until step 1; a failure stops the runs.
- The RunDir collision (optimisers CO0.1): distinct seeds per queue and a stagger; the
  uniqueness check of step 3.
- Cost: the s = 64 runs are ~9 h each; a crash mid-run resumes from the 25-epoch checkpoint
  (`train_poisson3d -- --resume runs/<dir>`).

## Pending runs
The whole VM matrix above, waiting for the author's go and a free VM.

## Open questions
1. The run matrix: s = 32 and 64, seeds 42–44 each, p = 0, ~31.5 GPU-h (~16 h wall,
   balanced queues)? Or a smaller first pass (s = 32 only, ~4.8 GPU-h)?
2. Padding p = 0 (Phase 5's main arm, CL0.1's timings) rather than `train_poisson3d`'s
   default p = 4 (+42 % cells at s = 32, +20 % at s = 64)?

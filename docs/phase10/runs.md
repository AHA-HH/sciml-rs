# Phase 10 run sheet

Every Phase 10 run, in order, on one A100 of the VM (G7; order amended 2026-11-11, decision
31: code first, runs at the end). Four stages, each ending at a decision point marked
**Stop**. Do not start a stage before the previous one's decision is recorded. All code is
on `main` (T1–T6, #98–#107); build once from the commit the runs cite.

## 0. Setup (once)

```sh
ssh cricket
tmux new -s p10                      # every run below inside tmux
cd ~/sciml-rs && git pull && git log --oneline -1   # the commit the runs cite
source ~/sciml-gpu.sh
mkdir -p logs
# datasets/poisson/ must hold stage 3 (n = 129): poisson_n129_K64_{train,test}.npz and sidecars
ls datasets/poisson/ | grep n129
CUDA="--no-default-features --features cuda"
cargo build --release $CUDA --example train_poisson --example predict_poisson \
  --example poisson_matrix --example poisson_conditioning
```

Two helpers, pasted into the shell. `gpurun` runs one command with its log under `logs/`
and samples GPU 0's memory beside it (T3's rule: `peak_gpu_memory` comes from
`nvidia-smi`, filled in by hand, the same way for Adam and GN). `setpeak` writes that peak
into the run's `costs.json`.

```sh
TP="cargo run --release $CUDA --example train_poisson --"
PP="cargo run --release $CUDA --example predict_poisson --"

gpurun() {   # gpurun <tag> <command ...>; returns the command's status
  local tag=$1; shift
  nvidia-smi > logs/$tag.gpu_before.txt          # the GPU unshared (Phase 9's rule)
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 -lms 500 \
    > logs/$tag.mem &
  local smi=$!
  # stop the sampler on Ctrl-C too, or it keeps appending to logs/$tag.mem
  trap "kill $smi 2>/dev/null" RETURN INT
  set -o pipefail
  "$@" 2>&1 | tee logs/$tag.log
  local status=$?
  set +o pipefail
  nvidia-smi > logs/$tag.gpu_after.txt
  if [ $status -ne 0 ]; then echo "FAILED ($status): see logs/$tag.log"; return $status; fi
  echo "peak $(sort -n logs/$tag.mem | tail -1) MiB"
}

setpeak() {  # setpeak <run_dir> <tag>
  python3 - "$1/costs.json" "$(sort -n logs/$2.mem | tail -1) MiB" <<'EOF'
import json, sys
p, m = sys.argv[1], sys.argv[2]
c = json.load(open(p)); c["peak_gpu_memory"] = m; json.dump(c, open(p, "w"), indent=2)
EOF
}
```

Each training run ends with `run written to <path>`, the run directory's full path
(`.../sciml-rs/runs/<name>_<unix_secs>`): the commands below write it as `runs/<dir>`, and
the full path works as well. A run that panics prints `FAILED` instead of a peak. Copy every run back to the Mac's main
checkout when its stage is done:
`rsync -a cricket:~/sciml-rs/runs/<dir> ~/Code/sciml-rs/runs/` (and `logs/` beside them).

## 1. T3's acceptance (about 10 minutes)

1. **cuda f64 (G7):**
   `cargo test --release $CUDA --test gauss_newton_f64 -- --ignored f64_oracle_and_adjoint_on_cuda 2>&1 | tee logs/t3_f64.log`.
   Record pass or fail and the printed errors. If it passes, T4's optional f64 GN run still
   needs code (`train_poisson` loads f32 only); decide then whether it is worth a task.
2. **A short GN run** of stage 3, s = 64, N = 100, subset 0. `train_poisson` has no epoch
   override, so the brief's "2 epochs" is a 60 s budget instead (it stops after the first
   epoch that reaches it):
   `gpurun t3_gn env OPTIMIZER=gauss-newton TIME_BUDGET_S=60 $TP 0 129 64 42 100 0`,
   then `$PP runs/<dir>`. Record `gn_steps.csv`'s first rows (ρ, λ, CG iterations,
   `adjoint_error`) and `costs.json`.
3. **The Adam check** (metrics against a Phase 7 run's, cuda round-off) is read from the
   first T5 Adam rerun (stage 3, seed 42) against Phase 7's
   `poisson_fno_n129_s64_p0_1793188648`: the same commit's Adam path, so no separate run.

**Stop:** T3's row in the README (VM part) and f64 yes / no; its Adam check is added in
stage 3, after the first Adam rerun.

## 2. T4's gate (about 10 minutes)

Comparator: Phase 7's Adam run at N = 100, subset 0, seed 42,
`poisson_fno_n129_s64_p0_N100_sub0_1793291675`, `training_time_s` 132.5.

**Before the runs**, confirm or change the gate (T4 brief): median adjoint error ≤ 1e-3
(f32); at least 80 % of steps accepted, ρ mostly in (¼, 2); CG's relative residual falling
in its first iterations on most steps.

```sh
gpurun t4_gn   env OPTIMIZER=gauss-newton GN_BATCH_SIZE=100 GN_PRECONDITIONER=none \
  TIME_BUDGET_S=132.5 $TP 0 129 64 42 100 0
gpurun t4_gn_d env OPTIMIZER=gauss-newton GN_BATCH_SIZE=100 GN_PRECONDITIONER=hutchinson \
  TIME_BUDGET_S=132.5 $TP 0 129 64 42 100 0
```

Both run names end in `_N100_sub0_gn`; tell them apart by `train_cfg.json`'s
`preconditioner` (`None` or `HutchinsonDiagonal`, inside `optimizer`). For each: `setpeak`, `$PP runs/<dir>`, copy back. On the Mac: the T4
table (error 4, final train L², updates, passes, wall time, peak memory; adjoint error,
acceptance rate, CG iterations, λ and Ritz range from `gn_steps.csv`) and the plots of Φ and
error 4 against updates and seconds, as the T4 brief lists.

**Stop:** go / no-go, recorded in T4's README row and paragraph, with the preconditioner
for GN200 and Hybrid (call it `<P>`, `none` or `hutchinson`). No-go on JVP noise: an
"exact FNO JVP" task comes before stage 3 (G2). No-go for another reason: the cause and a
proposal, then decide.

## 3. T5's matrix (about 4 hours of training)

Stage 3, s = 64, p = 0, N = 1000, seeds 42, 43, 44. Do not change the budget, seeds or
arms after seeing test errors.

**Adam reruns first** (about 15 minutes each):

```sh
for seed in 42 43 44; do gpurun t5_adam_$seed $TP 0 129 64 $seed; done
```

`setpeak` and `$PP` each. Check: error 4 of each within Phase 7's 1.041e-2–1.095e-2 (G6).
**Stop if one is not**, and report before any GN arm. The budget is their mean
`training_time_s`:

```sh
T=$(python3 -c 'import json,sys; t=[json.load(open(d+"/costs.json"))["training_time_s"] for d in sys.argv[1:]]; print(sum(t)/len(t))' runs/<adam42> runs/<adam43> runs/<adam44>)
echo $T
```

Also the T3 Adam check: `runs/<adam42>/metrics.csv` against Phase 7's
`poisson_fno_n129_s64_p0_1793188648/metrics.csv`, the first epochs.

**The GN arms** (each about `$T` seconds; 12 runs):

```sh
for seed in 42 43 44; do
  gpurun t5_gn20_$seed    env OPTIMIZER=gauss-newton GN_PRECONDITIONER=none       TIME_BUDGET_S=$T $TP 0 129 64 $seed
  gpurun t5_gn20d_$seed   env OPTIMIZER=gauss-newton GN_PRECONDITIONER=hutchinson TIME_BUDGET_S=$T $TP 0 129 64 $seed
  gpurun t5_gn200_$seed   env OPTIMIZER=gauss-newton GN_BATCH_SIZE=200 GN_PRECONDITIONER=<P> TIME_BUDGET_S=$T $TP 0 129 64 $seed
  gpurun t5_hybrid_$seed  env OPTIMIZER=hybrid GN_PRECONDITIONER=<P>              TIME_BUDGET_S=$T $TP 0 129 64 $seed
done
```

`setpeak` and `$PP` each; copy all 15 back. On the Mac:

```sh
cargo run --release --example poisson_matrix -- --by-optimizer docs/phase10/results.md runs/<15 dirs>
```

Then in `docs/phase10/results.md`, below the generated tables: H10a answered, and the
generalisation gap per arm. The T5 README row.

**Stop:** the best GN arm by mean error 4 (T6's second run).

## 4. T6's conditioning (time not yet measured; the cut-down smoke took 6.5 min on flex)

On the VM, three runs: one Adam seed (`<adam42>`), T5's best GN arm (its seed-42 run), and
T4's GN run with `<P>` (N = 100, a second point). Try `datagen` for the Chebyshev κ column;
if it does not build on the VM, drop it and the column reads "– (needs datagen)".

```sh
gpurun t6 cargo run --release $CUDA,datagen --example poisson_conditioning -- \
  docs/phase10/conditioning.md runs/<adam42> runs/<best_gn_42> runs/<t4_gn_P>
```

Copy `docs/phase10/conditioning.md` and its two plots back. In it, below the tables: H10b
answered (the Ritz values across s against κ's growth; the kernel diagonal against 1/μ_kl
and against Phase 6's POD eigenvalues of u at s = 64). The T6 README row.

## 5. Close

- README: every Results row filled; exit checklist ticked (T1–T6 merged, G6's regression
  check, H10a and H10b answered).
- `CLAUDE.md`'s "Current phase" bullet: Phase 10 done, and what comes next.
- Optional: `mlflow_export` of the Phase 10 runs (Phase 8).

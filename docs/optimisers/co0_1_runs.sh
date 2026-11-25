#!/usr/bin/env bash
# optimisers CO0.1: the reference arms on Burn 0.23 (docs/optimisers/tasks/CO0.1.md).
# Runs the given run specs one after another on one GPU, stage 3, s = 64, p = 0, N = 1000:
#   adam:<seed>                   Adam, 500 epochs (Phase 5 settings)
#   adam:<seed>:<epochs>          Adam, that many epochs (the long budget)
#   gn:<seed>:<cap>:<budget_s>    GN200-D at mu = 0, CG cap <cap>, stopped by the budget
#
# From the run's worktree, inside tmux, after `source ~/sciml-gpu.sh` and with
# CARGO_TARGET_DIR set (the plan's setup):
#   GPU=0 bash docs/optimisers/co0_1_runs.sh adam:42 gn:42:50:859.4
# GPU picks the device (default 0); it is also the device nvidia-smi samples.
#
# Each run's peak GPU memory (whole device, sampled every second) replaces "unavailable"
# in its costs.json, as in Phase 5 T5 and Phase 7. Logs go to ~/sciml-rs/logs/co0-1/;
# progress.txt there gets one line per run, "done <spec> <run dir> <s>s peak <MiB> MiB".
# A spec already marked done is skipped, so rerunning resumes; a failed run is logged and
# the list carries on.

set -u
GPU=${GPU:-0}
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) || exit 1
LOGDIR=${LOGDIR:-$HOME/sciml-rs/logs/co0-1}
PROGRESS=$LOGDIR/progress.txt
mkdir -p "$LOGDIR"
cd "$REPO" || exit 1
BIN=${CARGO_TARGET_DIR:-./target}/release/examples/train_poisson

cargo build --release --no-default-features --features cuda --example train_poisson || exit 1
echo "commit $(git rev-parse HEAD), GPU $GPU, started $(date -Is): $*" >> "$PROGRESS"

for spec in "$@"; do
  if grep -q "^done $spec " "$PROGRESS"; then
    echo "skip $spec (done)"
    continue
  fi
  IFS=: read -r arm seed a b <<< "$spec"
  case $arm in
    adam) vars=(${a:+EPOCHS=$a}) ;;
    gn) vars=(OPTIMIZER=gauss-newton GN_BATCH_SIZE=200 GN_PRECONDITIONER=hutchinson
             GN_WEIGHT_DECAY=0 GN_CG_MAX_ITERS="$a" TIME_BUDGET_S="$b") ;;
    *) echo "FAILED $spec: unknown arm" | tee -a "$PROGRESS"; continue ;;
  esac
  log=$LOGDIR/${spec//:/_}.log
  mem=$LOGDIR/${spec//:/_}.mem
  echo "start $spec $(date -Is)"

  nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > "$mem" &
  smi=$!
  start=$(date +%s)
  env "${vars[@]}" CUDA_VISIBLE_DEVICES="$GPU" "$BIN" 0 129 64 "$seed" > "$log" 2>&1
  status=$?
  kill "$smi" 2>/dev/null
  wait "$smi" 2>/dev/null
  secs=$(( $(date +%s) - start ))

  dir=$(sed -n 's/^run written to //p' "$log")
  if [ "$status" -ne 0 ] || [ -z "$dir" ]; then
    echo "FAILED $spec exit $status after ${secs}s, see $log" | tee -a "$PROGRESS"
    continue
  fi
  peak=$(sort -n "$mem" | tail -1)
  if [ -n "$peak" ]; then
    sed -i "s/\"peak_gpu_memory\": \"unavailable\"/\"peak_gpu_memory\": \"$peak MiB\"/" \
      "$dir/costs.json"
  fi
  echo "done $spec $dir ${secs}s peak ${peak:-?} MiB" | tee -a "$PROGRESS"
done
echo "finished $(date -Is)" >> "$PROGRESS"
grep "^FAILED" "$PROGRESS" || echo "no failures"

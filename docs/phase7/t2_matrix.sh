#!/usr/bin/env bash
# Phase 7 T2: the training matrix (docs/phase7/T2-matrix.md), 45 runs on one GPU.
# Stage 3, s = 64, p = 0; N in {25, 50, 100, 250, 500} x subset seeds {0, 1, 2} x seeds
# {42, 43, 44}; 500 epochs each (D6). Runs one after another, N = 25 first.
#
# Run it from anywhere, inside tmux, with the repo at the commit to train (Phase 7 used
# b9c45bf, T1 merged):
#   GPU=0 bash docs/phase7/t2_matrix.sh
# GPU picks the device (default 0); it is also the device nvidia-smi samples.
#
# Each run's peak GPU memory (whole device, sampled every second by nvidia-smi) is written
# into its costs.json, replacing "unavailable", as in Phase 5 T5. Logs go to t2_logs/
# beside the repo, outside it; progress to t2_logs/progress.txt, one line per run.
# Rerunning the script skips runs marked done, so it resumes after an interruption.
# A failed run is logged and the matrix carries on.

set -u
GPU=${GPU:-0}
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) || exit 1
LOGDIR=$(dirname "$REPO")/t2_logs
PROGRESS=$LOGDIR/progress.txt
mkdir -p "$LOGDIR"
cd "$REPO" || exit 1
BIN=./target/release/examples/train_poisson

cargo build --release --no-default-features --features cuda --example train_poisson || exit 1
echo "commit $(git rev-parse HEAD), GPU $GPU, started $(date -Is)" >> "$PROGRESS"

for N in 25 50 100 250 500; do
  for SUB in 0 1 2; do
    for SEED in 42 43 44; do
      tag="N${N}_sub${SUB}_seed${SEED}"
      if grep -q "^done $tag " "$PROGRESS"; then
        echo "skip $tag (done)"
        continue
      fi
      log=$LOGDIR/$tag.log
      mem=$LOGDIR/$tag.mem
      echo "start $tag $(date -Is)"

      nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > "$mem" &
      smi=$!
      start=$(date +%s)
      CUDA_VISIBLE_DEVICES=$GPU "$BIN" 0 129 64 "$SEED" "$N" "$SUB" > "$log" 2>&1
      status=$?
      kill "$smi" 2>/dev/null
      wait "$smi" 2>/dev/null
      secs=$(( $(date +%s) - start ))

      dir=$(sed -n 's/^run written to //p' "$log")
      if [ "$status" -ne 0 ] || [ -z "$dir" ]; then
        echo "FAILED $tag exit $status after ${secs}s, see $log" | tee -a "$PROGRESS"
        continue
      fi
      peak=$(sort -n "$mem" | tail -1)
      if [ -n "$peak" ]; then
        sed -i "s/\"peak_gpu_memory\": \"unavailable\"/\"peak_gpu_memory\": \"$peak MiB\"/" \
          "$dir/costs.json"
      fi
      echo "done $tag $dir ${secs}s peak ${peak:-?} MiB" | tee -a "$PROGRESS"
    done
  done
done
echo "finished $(date -Is)" >> "$PROGRESS"
grep -c "^done " "$PROGRESS" | xargs echo "runs done:"
grep "^FAILED" "$PROGRESS" || echo "no failures"

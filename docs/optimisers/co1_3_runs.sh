#!/usr/bin/env bash
# optimisers CO1.3: the Burgers 2 x 2 (docs/optimisers/tasks/CO1.3.md).
# Runs the given run specs one after another on one GPU, every one at PETScML's setting
# (RECIPE=petscml: n_train 1,000, s = 128, GELU, the squared loss on the Adam path):
#   a1:<seed>[:<epochs>]        AdamW, weight decay 1e-4 (PETScML's reference; D11)
#   a0:<seed>[:<epochs>]        AdamW, weight decay 0
#   c1:<seed>[:<epochs>]        our Adam (coupled L2, cosine), weight decay 1e-4 (D11's extra cells)
#   c0:<seed>[:<epochs>]        our Adam, weight decay 0
#   g1:<seed>:<budget_s>        Gauss-Newton (D12), full batch, mu = 1e-4
#   g0:<seed>:<budget_s>        Gauss-Newton, full batch, mu = 0
#   g0b200:<seed>:<budget_s>    Gauss-Newton, batch 200, mu = 0 (G0-200)
# The A and C cells run 500 epochs unless <epochs> is given (3000 at the long budget).
# Gauss-Newton runs stop at the budget; EPOCHS is set far above what any budget reaches,
# since a run also stops at EPOCHS.
#
# From the run's worktree, inside tmux, with the Spack environment and CARGO_TARGET_DIR
# set in the same command (a running tmux server does not pass the client's environment):
#   tmux new -d -s co13-g0 "source ~/sciml-gpu.sh; export CARGO_TARGET_DIR=~/sciml-rs-target/co1-3; \
#     cd ~/sciml-rs-wt/co1-3; GPU=0 bash docs/optimisers/co1_3_runs.sh a1:42 a1:43"
# GPU picks the device (default 0); it is also the device nvidia-smi samples.
#
# The script builds train_burgers, then runs a copy of it in $LOGDIR/bin-<commit>/, so
# another worktree building into the same CARGO_TARGET_DIR cannot swap the binary between
# specs. Every training variable is cleared before each run, so nothing in the
# environment changes a cell.
#
# Each run's peak GPU memory (whole device, sampled every second) replaces "unavailable"
# in its costs.json. Logs go to ~/sciml-rs/logs/co1-3/; progress.txt there gets one line
# per run, "done <spec> <run dir> <s>s peak <MiB> MiB". A spec already marked done is
# skipped, so rerunning resumes; a failed run is logged and the list carries on.
# Run directories are <name>_<unix_secs> and an existing one is reused (until optimisers
# CO0.4), so a run whose directory another done line already names is marked FAILED
# (collision) and must be run again.

set -u
GPU=${GPU:-0}
: "${CARGO_TARGET_DIR:?set CARGO_TARGET_DIR as in the plan setup}"
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) || exit 1
LOGDIR=${LOGDIR:-$HOME/sciml-rs/logs/co1-3}
PROGRESS=$LOGDIR/progress.txt
mkdir -p "$LOGDIR"
touch "$PROGRESS"
cd "$REPO" || exit 1

cargo build --release --no-default-features --features cuda --example train_burgers || exit 1
BINDIR=$LOGDIR/bin-$(git rev-parse --short HEAD)
mkdir -p "$BINDIR"
# two instances start together (one per GPU) and the first may already run the copy:
# overwriting a running binary fails ("Text file busy"), so copy only when it differs, to a
# temporary name, and rename it into place (the running process keeps the old inode)
src=$CARGO_TARGET_DIR/release/examples/train_burgers
if ! cmp -s "$src" "$BINDIR/train_burgers"; then
  cp "$src" "$BINDIR/train_burgers.tmp.$$" && mv -f "$BINDIR/train_burgers.tmp.$$" "$BINDIR/train_burgers" || exit 1
fi
BIN=$BINDIR/train_burgers
echo "commit $(git rev-parse HEAD), GPU $GPU, started $(date -Is): $*" >> "$PROGRESS"

for spec in "$@"; do
  if grep -qF "done $spec " "$PROGRESS"; then
    echo "skip $spec (done)"
    continue
  fi
  IFS=: read -r cell seed a extra <<< "$spec"
  case $cell in
    a1|a0|c1|c0)
      [ -n "$seed" ] && [ -z "$extra" ] || { echo "FAILED $spec: $cell takes <seed>[:<epochs>]" | tee -a "$PROGRESS"; continue; }
      case $cell in a*) opt=adamw ;; c*) opt=adam ;; esac
      case $cell in *1) wd=1e-4 ;; *0) wd=0 ;; esac
      vars=(OPTIMIZER=$opt WEIGHT_DECAY=$wd ${a:+EPOCHS=$a}) ;;
    g1|g0|g0b200)
      [ -n "$a" ] && [ -z "$extra" ] || { echo "FAILED $spec: $cell takes <seed>:<budget_s>" | tee -a "$PROGRESS"; continue; }
      case $cell in g1) mu=1e-4 ;; *) mu=0 ;; esac
      vars=(OPTIMIZER=gauss-newton GN_WEIGHT_DECAY=$mu TIME_BUDGET_S="$a" EPOCHS=100000)
      [ "$cell" = g0b200 ] && vars+=(GN_BATCH_SIZE=200) ;;
    *) echo "FAILED $spec: unknown cell" | tee -a "$PROGRESS"; continue ;;
  esac
  log=$LOGDIR/${spec//:/_}.log
  mem=$LOGDIR/${spec//:/_}.mem
  echo "start $spec $(date -Is)"

  nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > "$mem" &
  smi=$!
  start=$(date +%s)
  # PCI order, so CUDA's device $GPU is the one nvidia-smi -i $GPU samples
  env -u OPTIMIZER -u EPOCHS -u TIME_BUDGET_S -u WEIGHT_DECAY -u SUBSAMPLE -u SEED \
    -u GN_BATCH_SIZE -u GN_PRECONDITIONER -u GN_WEIGHT_DECAY -u GN_CG_MAX_ITERS \
    -u MLFLOW_EXPERIMENT_NAME -u BURN_DEVICE RECIPE=petscml SEED="$seed" "${vars[@]}" \
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" "$BIN" > "$log" 2>&1
  status=$?
  kill "$smi" 2>/dev/null
  wait "$smi" 2>/dev/null
  secs=$(( $(date +%s) - start ))

  dir=$(sed -n 's/^run written to //p' "$log")
  if [ "$status" -ne 0 ] || [ -z "$dir" ]; then
    echo "FAILED $spec exit $status after ${secs}s, see $log" | tee -a "$PROGRESS"
    continue
  fi
  if grep -q "^done .* $dir " "$PROGRESS"; then
    # both runs wrote into it: remove both done lines and run both again
    echo "FAILED $spec collision: $dir is another run's directory too, see $log" | tee -a "$PROGRESS"
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

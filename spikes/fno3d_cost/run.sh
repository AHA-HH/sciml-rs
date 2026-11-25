#!/usr/bin/env bash
# 3D CL0.1: runs `fno3d_cost` once per line of arguments on stdin
# (`<s> <modes> <batch> <n_train> <n_test>`) on backend $1 (flex, metal or cuda) and prints
# each run's output with its peak memory: macOS `/usr/bin/time -l` ("peak memory
# footprint", Metal buffers included) or, on cuda, the process's largest `nvidia-smi`
# reading (sampled every 200 ms; the caching allocator's reservation). A run that fails
# (out of memory) is reported and the next one starts.
# Usage: `bash spikes/fno3d_cost/run.sh <backend> <binary> < configs`
set -u
backend=$1
bin=$2
while read -r cfg; do
  [ -z "$cfg" ] && continue
  echo "== $backend $cfg"
  if [ "$backend" = cuda ]; then
    # shellcheck disable=SC2086
    BURN_DEVICE=cuda "$bin" $cfg &
    pid=$!
    peak=0
    while kill -0 "$pid" 2>/dev/null; do
      m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits |
        awk -F', ' -v p="$pid" '$1 == p { print $2 }')
      [ -n "$m" ] && [ "$m" -gt "$peak" ] && peak=$m
      sleep 0.2
    done
    wait "$pid"
    echo "exit $? peak_gpu_mib $peak"
  else
    # shellcheck disable=SC2086
    BURN_DEVICE=$backend /usr/bin/time -l "$bin" $cfg 2>&1 |
      grep -E '^(device|epoch)|panicked|peak memory footprint'
    echo "exit ${PIPESTATUS[0]}"
  fi
done

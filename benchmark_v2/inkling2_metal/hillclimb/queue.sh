#!/bin/zsh
# Serialized A/B queue: hc_bench.py on each config in order through the shared GPU lock, a progress line per run.
cd /Users/arthurzucker/Work/transformers/.claude/worktrees/inkling2-perf
H=benchmark_v2/inkling2_metal/hillclimb
i=0; start=$(date +%s)
for n in "$@"; do
  i=$((i+1))
  sleep ${COOLDOWN:-180}  # let the GPU cool between runs (sustained load throttles the M3 Max)
  PYTHONPATH=src:benchmark_v2/inkling2_metal python -u ~/Work/hub-kernels/opt/gpu_lock.py python -u $H/hc_bench.py $H/configs/$n.json > $H/runs-$n-$i.log 2>&1
  el=$(( $(date +%s) - start ))
  echo "[$i/$#] $n | $(grep -oE '"decode_tok_s": [0-9.]+|refusing.*|Error.*' $H/runs-$n-$i.log | tail -1) | elapsed ${el}s eta ~$(( el / i * ($# - i) ))s"
done
echo QUEUE_DONE

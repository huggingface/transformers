#!/usr/bin/env bash
set -uo pipefail

MODELS=(qwen3_moe qwen3_moe llama llama gpt_oss_mxfp4 gpt_oss_mxfp4)
LEGACY=(1 0 1 0 1 0)
GPU_PAIRS=(0,1 2,3 4,5 6,7)

run_worker() {
  local worker_id="$1"
  local job_id

  for ((job_id = worker_id; job_id < ${#MODELS[@]}; job_id += ${#GPU_PAIRS[@]})); do
    echo "[worker ${worker_id} gpu=${GPU_PAIRS[$worker_id]}] model=${MODELS[$job_id]} legacy=${LEGACY[$job_id]}"
    CUDA_VISIBLE_DEVICES="${GPU_PAIRS[$worker_id]}" \
      TRANSFORMERS_USE_LEGACY_TP="${LEGACY[$job_id]}" \
      MODEL_KIND="${MODELS[$job_id]}" \
      torchrun --master_port="$((29500 + job_id))" --nproc_per_node=2 bench_dtensor.py
  done
}

pids=()
for worker_id in "${!GPU_PAIRS[@]}"; do
  run_worker "$worker_id" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
exit "$failed"
#!/bin/bash
set -euo pipefail

SCRIPT="train_fsdp_tp.py"
LOG_FSDP_TP="log.txt"
LOG_FSDP_ONLY="ref.txt"

MODEL_NAME="${MODEL_NAME:-hf-internal-testing/tiny-random-MixtralForCausalLM}"
COMMON_ARGS="--model_name $MODEL_NAME --lr 3e-4 --seed 42"

rm -rf ./checkpoints_tp ./checkpoints_tp_resumed ./checkpoints_fsdp ./checkpoints_fsdp_resumed

echo "=== Phase 1: Train steps 0-9, save checkpoint ==="
echo "--- Launching FSDP+TP and FSDP-only in parallel ---"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 \
  $SCRIPT $COMMON_ARGS --fsdp_size 2 --tp_size 2 --enable_sp \
  --num_steps 10 --save_dir ./checkpoints_tp > "${LOG_FSDP_TP}.phase1" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=29501 \
  $SCRIPT $COMMON_ARGS --fsdp_size 2 \
  --num_steps 10 --save_dir ./checkpoints_fsdp > "${LOG_FSDP_ONLY}.phase1" 2>&1 &
PID2=$!

echo "FSDP+TP PID=$PID1 | FSDP-only PID=$PID2"
wait $PID1 && echo "Phase 1 FSDP+TP done" || { echo "Phase 1 FSDP+TP failed (exit $?)"; cat "${LOG_FSDP_TP}.phase1"; exit 1; }
wait $PID2 && echo "Phase 1 FSDP-only done" || { echo "Phase 1 FSDP-only failed (exit $?)"; cat "${LOG_FSDP_ONLY}.phase1"; exit 1; }

echo ""
echo "=== Phase 2: Resume from checkpoint, train steps 10-19, save ==="
echo "--- Launching FSDP+TP and FSDP-only in parallel ---"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 \
  $SCRIPT $COMMON_ARGS --fsdp_size 2 --tp_size 2 --enable_sp \
  --num_steps 10 --start_step 10 \
  --resume_dir ./checkpoints_tp --save_dir ./checkpoints_tp_resumed > "${LOG_FSDP_TP}.phase2" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=29501 \
  $SCRIPT $COMMON_ARGS --fsdp_size 2 \
  --num_steps 10 --start_step 10 \
  --resume_dir ./checkpoints_fsdp --save_dir ./checkpoints_fsdp_resumed > "${LOG_FSDP_ONLY}.phase2" 2>&1 &
PID2=$!

echo "FSDP+TP PID=$PID1 | FSDP-only PID=$PID2"
wait $PID1 && echo "Phase 2 FSDP+TP done" || { echo "Phase 2 FSDP+TP failed (exit $?)"; cat "${LOG_FSDP_TP}.phase2"; exit 1; }
wait $PID2 && echo "Phase 2 FSDP-only done" || { echo "Phase 2 FSDP-only failed (exit $?)"; cat "${LOG_FSDP_ONLY}.phase2"; exit 1; }

# Combine phase logs
cat "${LOG_FSDP_TP}.phase1" "${LOG_FSDP_TP}.phase2" > "$LOG_FSDP_TP"
cat "${LOG_FSDP_ONLY}.phase1" "${LOG_FSDP_ONLY}.phase2" > "$LOG_FSDP_ONLY"

echo ""
echo "=== Full Loss & Grad Diff (steps 0-19) ==="
git diff --no-index --color --word-diff=color "$LOG_FSDP_TP" "$LOG_FSDP_ONLY" || true
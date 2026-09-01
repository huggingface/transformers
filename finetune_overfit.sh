#!/bin/bash
# Minimalist TP/FSDP overfitting test on AWS Trainium or CUDA GPUs using the regular
# transformers Trainer (no SFTTrainer; LoRA optional via USE_LORA). Trains on a tiny fixed
# subset of examples for many steps to check that the loss goes to ~0, as a correctness check
# for TP or FSDP.

set -euo pipefail

# ---------------------------------------------------------------------------
# Parallelism: pick a variant with PARALLEL_MODE=tp|fsdp (default: fsdp).
# TP and FSDP2 are both enabled through the same mechanism -- `DistributedConfig`, which pre-shards
# the model into DTensor at from_pretrained time:
#   PARALLEL_MODE=tp   NUM_PROC=4 -> TP_SIZE=4
#   PARALLEL_MODE=fsdp NUM_PROC=4 -> FSDP_SIZE=4
# ---------------------------------------------------------------------------
PARALLEL_MODE=${PARALLEL_MODE:-tp}
NUM_PROC=${NUM_PROC:-4}
USE_LORA=${USE_LORA:-true}

case "$PARALLEL_MODE" in
    tp)
        export TP_SIZE=$NUM_PROC
        export FSDP_SIZE=1
        ;;
    fsdp)
        export TP_SIZE=1
        export FSDP_SIZE=$NUM_PROC
        ;;
    *)
        echo "Unknown PARALLEL_MODE: $PARALLEL_MODE (expected tp|fsdp)" >&2
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Neuron runtime environment (Trainium only -- no-op, skipped entirely on CUDA)
# ---------------------------------------------------------------------------
if ! command -v nvidia-smi &> /dev/null || ! nvidia-smi &> /dev/null; then
    export ON_NEURON_EAGER=1
    export NEURON_EAGER_MODEL_CACHE_SIZE=10000
    export OMP_NUM_THREADS=128
    export HF_DEACTIVATE_ASYNC_LOAD=1

    export TORCH_NEURONX_ENABLE_HOST_CC=1
    export TORCH_NEURONX_ENABLE_ASYNC_NRT=1
    #export NEURON_RT_NUM_CORES=1
fi

# ---------------------------------------------------------------------------
# Model / data / hyperparameters
# ---------------------------------------------------------------------------
MODEL_NAME=Qwen/Qwen3-1.7B
DATASET_NAME=trl-lib/Capybara
LORA_SUFFIX=""
if [ "$USE_LORA" = "true" ]; then
    LORA_SUFFIX="-lora"
fi
OUTPUT_DIR=Qwen3-1.7B-${PARALLEL_MODE}-Overfit${LORA_SUFFIX}

LEARNING_RATE=5.0e-4
NUM_TRAIN_EXAMPLES=16
MAX_STEPS=50
MAX_SEQ_LENGTH=1024
BATCH_SIZE=4
# Grad clipping disabled: Accelerate's clip_grad_norm_ can't batch mixed DTensor (TP-sharded)
# and plain Tensor (unsharded, e.g. norms/embeddings) grads together under native TP/FSDP loading.
MAX_GRAD_NORM=0

echo "=========================================="
echo "Plain Trainer parallelism overfitting test"
echo "  Model:           $MODEL_NAME"
echo "  Dataset:         $DATASET_NAME"
echo "  PARALLEL_MODE:   $PARALLEL_MODE"
echo "  NUM_PROC:        $NUM_PROC"
echo "  TP_SIZE:         $TP_SIZE"
echo "  FSDP_SIZE:       $FSDP_SIZE"
echo "  USE_LORA:        $USE_LORA"
echo "  Num examples:    $NUM_TRAIN_EXAMPLES"
echo "  Max steps:       $MAX_STEPS"
echo "  Batch:           $BATCH_SIZE"
echo "  Max seq len:     $MAX_SEQ_LENGTH"
echo "  Output dir:      $OUTPUT_DIR"
echo "=========================================="

if [ "$NUM_PROC" -eq 1 ]; then
    LAUNCHER="python"
else
    LAUNCHER="torchrun --nproc_per_node=${NUM_PROC}"
fi

$LAUNCHER \
    finetune_overfit.py \
    --model_name_or_path "$MODEL_NAME" \
    --use_lora $USE_LORA \
    --dataset_name "$DATASET_NAME" \
    --num_examples $NUM_TRAIN_EXAMPLES \
    --max_length $MAX_SEQ_LENGTH \
    --learning_rate $LEARNING_RATE \
    --gradient_checkpointing true \
    --bf16 true \
    --per_device_train_batch_size $BATCH_SIZE \
    --max_steps $MAX_STEPS \
    --max_grad_norm $MAX_GRAD_NORM \
    --eval_strategy no \
    --logging_steps 10 \
    --save_strategy no \
    --dataloader_num_workers 0 \
    --report_to trackio \
    --output_dir "$OUTPUT_DIR"

#!/bin/bash
# SFT with LoRA on AWS Trainium using TRL's SFTTrainer.
# Compare against sft_lora_finetune_custom.sh (custom training loop).

set -euo pipefail

# ---------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------
export TP_SIZE=${TP_SIZE:-2}
NUM_PROC=${NUM_PROC:-2}

# ---------------------------------------------------------------------------
# Neuron runtime environment
# ---------------------------------------------------------------------------
export TORCH_NEURONX_ENABLE_STABLEHLO=0
export ON_NEURON_EAGER=1
export TORCH_NEURONX_MLIR_ATEN_OPS=1
export ON_NEURON=1
export TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1
export NEURON_RT_MAP_HBM=0
export NEURON_RT_DBG_ZEROCOPY=0
export NEURON_EAGER_MODEL_CACHE_SIZE=128
export NEURON_RT_NUM_CORES=1
export OMP_NUM_THREADS=128
export HF_DEACTIVATE_ASYNC_LOAD=1

# Profiling
export NEURON_FRAMEWORK_DEBUG=1
export NEURON_RT_INSPECT_ENABLE=1
export NEURON_RT_INSPECT_OUTPUT_DIR="./profiler-trl"
export NEURON_RT_INSPECT_SYSTEM_PROFILE=1
export NEURON_RT_INSPECT_DEVICE_PROFILE=1

# ---------------------------------------------------------------------------
# Model / data / hyperparameters
# ---------------------------------------------------------------------------
MODEL_NAME=Qwen/Qwen3-1.7B
DATASET_NAME=trl-lib/Capybara
OUTPUT_DIR=Qwen3-1.7B-SFT-LoRA-trl

LEARNING_RATE=5.0e-4
NUM_EPOCHS=1
MAX_SEQ_LENGTH=1024
BATCH_SIZE=4
GRAD_ACCUM_STEPS=8

LORA_R=32
LORA_ALPHA=16
LORA_DROPOUT=0.0
LORA_TARGET_MODULES="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"

DP_SIZE=$(( NUM_PROC / TP_SIZE ))
echo "=========================================="
echo "TRL SFTTrainer"
echo "  Model:           $MODEL_NAME"
echo "  Dataset:         $DATASET_NAME"
echo "  NUM_PROC:        $NUM_PROC"
echo "  TP_SIZE:         $TP_SIZE"
echo "  DP_SIZE:         $DP_SIZE"
echo "  Batch:           $BATCH_SIZE x grad_accum $GRAD_ACCUM_STEPS"
echo "  Effective batch: $(( BATCH_SIZE * GRAD_ACCUM_STEPS * DP_SIZE ))"
echo "  Max seq len:     $MAX_SEQ_LENGTH"
echo "  LoRA r:          $LORA_R"
echo "  LoRA alpha:      $LORA_ALPHA"
echo "  Output dir:      $OUTPUT_DIR"
echo "=========================================="

torchrun --nproc_per_node="${NUM_PROC}" \
    sft_lora_finetune_hf.py \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_EPOCHS \
    --packing false \
    --bf16 true \
    --max_length $MAX_SEQ_LENGTH \
    --pad_to_multiple_of $MAX_SEQ_LENGTH \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --torch_compile \
    --eos_token '<|im_end|>' \
    --eval_strategy no \
    --logging_steps 10 \
    --save_steps 500 \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --use_peft true \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --lora_dropout $LORA_DROPOUT \
    --dataloader_num_workers 0 \
    --report_to none \
    --output_dir "$OUTPUT_DIR"

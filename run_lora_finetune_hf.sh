#!/bin/bash
# Launcher for HuggingFace LoRA finetuning on AWS Trainium.

set -euo pipefail

# ---------------------------------------------------------------------------
# Parallelism (override on the CLI: TP_SIZE=4 NUM_PROC=4 ./run_lora_finetune_hf.sh)
# ---------------------------------------------------------------------------
export TP_SIZE=${TP_SIZE:-4}
NUM_PROC=${NUM_PROC:-4}

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
export OMP_NUM_THREADS=128

# ---------------------------------------------------------------------------
# Model / data / hyperparameters
# ---------------------------------------------------------------------------
MODEL_NAME=Qwen/Qwen3-1.7B
DATASET_NAME=iamtarun/python_code_instructions_18k_alpaca
OUTPUT_DIR=Qwen3-1.7B-LoRA-Python-Coder

LEARNING_RATE=5.0e-4
NUM_EPOCHS=1
MAX_SEQ_LENGTH=1024
BATCH_SIZE=4
GRAD_ACCUM_STEPS=8
# torch.compile with Neuron backend actually REDUCES memory during training
# by optimizing the execution graph, even though it has compilation overhead
COMPILE=${COMPILE:-1}  # 1 = enable torch.compile (default)

# LoRA hyperparameters
LORA_R=32
LORA_ALPHA=16
LORA_DROPOUT=0.0
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

DP_SIZE=$(( NUM_PROC / TP_SIZE ))
echo "=========================================="
echo "HF LoRA FINETUNE"
echo "  Model:           $MODEL_NAME"
echo "  Dataset:         $DATASET_NAME"
echo "  NUM_PROC:        $NUM_PROC"
echo "  TP_SIZE:         $TP_SIZE"
echo "  DP_SIZE (FSDP):  $DP_SIZE"
echo "  Batch:           $BATCH_SIZE x grad_accum $GRAD_ACCUM_STEPS"
echo "  Effective batch: $(( BATCH_SIZE * GRAD_ACCUM_STEPS * DP_SIZE ))"
echo "  Max seq len:     $MAX_SEQ_LENGTH"
echo "  LoRA r:          $LORA_R"
echo "  LoRA alpha:      $LORA_ALPHA"
echo "  torch.compile:   $COMPILE"
echo "  Output dir:      $OUTPUT_DIR"
echo "=========================================="

torchrun --nproc_per_node="${NUM_PROC}" \
    /home/ubuntu/sft_lora_finetune_hf.py \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 500 \
    --bf16 \
    --trust_remote_code \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --lora_target_modules "$LORA_TARGET_MODULES" \
    $( [ "$COMPILE" = "1" ] && echo "--compile" )

echo "=========================================="
echo "Training completed. LoRA adapter saved to: $OUTPUT_DIR"
echo "=========================================="

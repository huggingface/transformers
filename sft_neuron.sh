#!/bin/bash
# Minimalist SFT launch script for Qwen/Qwen3.5-9B on AWS Trainium.
# Uses LoRA adapters and Tensor Parallelism (TP=2).

set -euo pipefail

# ---------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------
export TP_SIZE=2
NUM_PROC=$TP_SIZE   # one process per NeuronCore for TP-only setup

# ---------------------------------------------------------------------------
# Neuron runtime environment

export TORCH_NEURONX_ENABLE_STABLEHLO=0
export ON_NEURON_EAGER=1
export TORCH_NEURONX_MLIR_ATEN_OPS=1
export ON_NEURON=1

export TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1
export NEURON_RT_MAP_HBM=0
export NEURON_RT_DBG_ZEROCOPY=0
export NEURON_EAGER_MODEL_CACHE_SIZE=128
export NEURON_RT_NUM_CORES=1

export NEURON_FRAMEWORK_DEBUG=1
export NEURON_RT_INSPECT_ENABLE=1
export NEURON_RT_INSPECT_OUTPUT_DIR="./profiler"
export NEURON_RT_INSPECT_SYSTEM_PROFILE=1
export NEURON_RT_INSPECT_DEVICE_PROFILE=1

export OMP_NUM_THREADS=128

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
# MODEL_NAME=Qwen/Qwen3-0.6B
# DATASET_NAME=trl-lib/Capybara
MODEL_NAME=Qwen/Qwen3-1.7B
DATASET_NAME=iamtarun/python_code_instructions_18k_alpaca

BATCH_SIZE=4
GRAD_ACCUM_STEPS=8
MAX_SEQ_LENGTH=1024
COMPILE=${COMPILE:-1}  # 1 = enable torch.compile (default)

echo "Running SFT with the following configuration:"
echo "Model Name: $MODEL_NAME"
echo "Number of Processes: $NUM_PROC"
echo "Tensor Parallel Size: $TP_SIZE"

torchrun --nproc_per_node="${NUM_PROC}" \
    sft_neuron.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --learning_rate 5.0e-4 \
    --num_train_epochs 1 \
    --packing \
    --bf16 true \
    --max_length $MAX_SEQ_LENGTH \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --eos_token '<|im_end|>' \
    --eval_strategy no \
    --logging_steps 10 \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --use_peft true \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_target_parameters 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj' \
    --lora_dropout 0.0 \
    --report_to none \
    --torch_compile $COMPILE \
    --torch_compile_backend "neuron" \
    --output_dir $MODEL_NAME-SFT-LoRA

#!/bin/bash
LOG_DIR="./logs"
OUTPUT_DIR="./outputs"
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR
##    SETTINGS     ## 
# Defaul value
gpu_size=2
#Get the input arg
while getopts m:b:o:g: flag
do
    case "${flag}" in
        m) MODEL=${OPTARG};;
        g) gpu_size=${OPTARG};;
    esac
done

log_file="${LOG_DIR}/train.log"
## END OF SETTINGS ## 

export TRANSFORMERS_CACHE=/nas/huggingface_pretrained_models
export HF_DATASETS_CACHE=/nas/common_data/huggingface

args="
--length 20 \
--k 0 \
--p 0.95 \
--prefix '' \
--xlm_language '' \
--seed 42 \
"

# Model type
model_type=(
    "gpt2"
    "ctrl"
    "openai-gpt"
    "xlnet"
    "transfo-xl"
    "xlm"
)

## Using moreh device
moreh-switch-model --model $gpu_size

python run_generation.py \
  --model_type=openai-gpt \
  --model_name_or_path=$MODEL \
  --prompt="Once upon a time," \
  $args 
  2>&1 | tee $log_file
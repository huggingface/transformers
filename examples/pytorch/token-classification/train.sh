#!/bin/bash
LOG_DIR="./logs"
OUTPUT_DIR="./outputs"
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR
##    SETTINGS     ## 
# Defaul value
gpu_size=2
outputs=$OUTPUT_DIR
#Get the input arg
while getopts m:o:g: flag
do
    case "${flag}" in
        m) MODEL=${OPTARG};;
        g) gpu_size=${OPTARG};;
        o) outputs=${OPTARG}
    esac
done

log_file="${LOG_DIR}/train.log"
## END OF SETTINGS ## 

export TRANSFORMERS_CACHE=/nas/huggingface_pretrained_models
export HF_DATASETS_CACHE=/nas/common_data/huggingface

args="
--max_seq_length 128 \
--pad_to_max_length True \
--do_train \
--do_eval
--seed 42 \
"

## Using moreh device
moreh-switch-model --model $gpu_size

python run_ner.py \
  --model_name_or_path=$MODEL \
  --dataset_name conll2003 \
  --output_dir $outputs \
  $args 
  2>&1 | tee $log_file
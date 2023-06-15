#!/bin/bash
LOG_DIR="./logs"
OUTPUT_DIR="./outputs"
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR
##    SETTINGS     ## 
# Defaul value
output_dir="$OUTPUT_DIR"
BATCH_SIZE=8
gpu_size=2
#Get the input arg
while getopts m:b:o:g: flag
do
    case "${flag}" in
        m) MODEL=${OPTARG};;
        b) BATCH_SIZE=${OPTARG};;
        o) output_dir=${OPTARG};;
        g) gpu_size=${OPTARG};;
    esac
done
echo "MODEL: $MODEL ";
echo "BATCH SIZE: $BATCH_SIZE";
echo "OUTPUT DIR: $output_dir";
echo "GPU SIZE: $gpu_size";

log_file="${LOG_DIR}/train.log"
# MODEL=$1
# BATCH_SIZE=$2
# output_dir=${3:-"$OUTPUT_DIR/$MODEL"}
# gpu_size=${4:-2}
## END OF SETTINGS ## 

export TRANSFORMERS_CACHE=/nas/huggingface_pretrained_models
export HF_DATASETS_CACHE=/nas/common_data/huggingface

args="
--do_train \
--do_eval \
--learning_rate 3e-5 \
--num_train_epochs 2 \
--logging_strategy steps \
--logging_steps 100 \
--overwrite_output_dir \
--save_strategy epoch \
--save_total_limit 2 \
--seed 42
--predict_with_generate
"

## Using moreh device
moreh-switch-model --model $gpu_size

python run_summarization.py \
  --model_name_or_path $MODEL \
  --dataset_name xsum \
  --source_prefix "summarize: " \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --output_dir $output_dir \
  $args \
  2>&1 | tee $log_file
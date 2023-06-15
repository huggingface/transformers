#!/bin/bash
LOG_DIR="./logs"
OUTPUT_DIR="./outputs"
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR

export TRANSFORMERS_CACHE=/nas/huggingface_pretrained_models
export HF_DATASETS_CACHE=/nas/common_data/huggingface
export TASK_NAME=mrpc

##    SETTINGS     ## 
# Defaul value
output_dir="$OUTPUT_DIR"
BATCH_SIZE=8
gpu_size=2
#Get the input arg
while getopts m:b:o:g:t: flag
do
    case "${flag}" in
        m) MODEL=${OPTARG};;
        b) BATCH_SIZE=${OPTARG};;
        o) output_dir=${OPTARG};;
        g) gpu_size=${OPTARG};;
        t) TASK_NAME=${OPTARG};;
    esac
done
## END OF SETTINGS ## 

## task list ##
task_list=(
    "mrpc"
    "cola"
    "sst2"
    "stsb"
    "qqp"
    "mnli"
    "qnli"
    "rte"
    "wnli"
)

args="
--do_train \
--do_eval \
--learning_rate 3e-5 \
--num_train_epochs 2 \
--logging_strategy steps \
--logging_steps 100 \
--max_seq_length 384 \
--overwrite_output_dir \
--save_strategy epoch \
--save_total_limit 2 \
--seed 42
"

## Using moreh device
moreh-switch-model --model $gpu_size

python run_glue.py \
  --model_name_or_path $MODEL \
  --task_name $TASK_NAME \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --output_dir $output_dir \
  $args \
  2>&1 | tee $log_file
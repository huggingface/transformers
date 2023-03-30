#!/usr/bin/env bash

LOG_DIR="./logs"
OUTPUT_DIR="./outputs"
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR

model=${1:-"microsoft/resnet-50"}
bs=${2:-32}
model_name=${model#*/}
log_file="${LOG_DIR}/${model_name}.log"
output_dir="${OUTPUT_DIR}/${model_name}"

COMMON_ARGS="""
    --ignore_mismatched_sizes \
    --dataset_name beans \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --logging_strategy steps \
    --logging_steps 100 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 1 \
    --report_to tensorboard \
    --disable_tqdm True \
    --seed 42
"""

/usr/bin/env python3 run_image_classification.py \
    --model_name_or_path $model \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --output_dir $output_dir \
    $COMMON_ARGS \
    2>&1 | tee $log_file

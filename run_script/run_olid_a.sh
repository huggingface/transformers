#!/usr/bin/env bash
export DATA_DIR=data/olid/task_a
export TASK_NAME=MRPC
export OUTPUT=output/olid/task_a

python ./examples/run_glue_olid_a.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $DATA_DIR \
    --max_seq_length 64 \
    --per_gpu_eval_batch_size=64   \
    --per_gpu_train_batch_size=64   \
    --learning_rate 2e-5 \
    --num_train_epochs 2.0 \
    --output_dir $OUTPUT
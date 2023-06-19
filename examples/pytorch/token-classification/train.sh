#!/bin/bash
while getopts m:g: flag
do
    case "${flag}" in
        m) model=${OPTARG};;
        g) device_id=${OPTARG};;
    esac
done

LOG_DIR="./logs"
OUTPUT_DIR="./outputs"
log_file=$LOG_DIR/$model.log
output_dir=$OUTPUT_DIR/$model

mkdir -p "$(dirname $log_file)"
mkdir -p "$(dirname $output_dir)"

## Using moreh device
export MOREH_VISIBLE_DEVICE=$device_id

args="
--max_seq_length 128 \
--pad_to_max_length True \
--do_train \
--do_eval
--seed 42 \
"

python run_ner.py \
  --model_name_or_path=$MODEL \
  --dataset_name conll2003 \
  --output_dir $outputs \
  $args 
  2>&1 | tee $log_file
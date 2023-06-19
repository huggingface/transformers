#!/bin/bash

$model_type=openai-gpt
# Model type
model_type_lst=(
    "gpt2"
    "ctrl"
    "openai-gpt"
    "xlnet"
    "transfo-xl"
    "xlm"
)

while getopts m:g:t: flag
do
    case "${flag}" in
        m) model=${OPTARG};;
        g) device_id=${OPTARG};;
        t) model_type=${OPTARG};;
    esac
done

log_file=$LOG_DIR/$model.log
output_dir=$OUTPUT_DIR/$model

mkdir -p "$(dirname $log_file)"
mkdir -p "$(dirname $output_dir)"

## Using moreh device
export MOREH_VISIBLE_DEVICE=$device_id

args="
--length 20 \
--k 0 \
--p 0.95 \
--prefix '' \
--xlm_language '' \
--seed 42 \
"

python run_generation.py \
  --model_type=$model_type \
  --model_name_or_path=$model \
  --prompt="Once upon a time," \
  $args 
  2>&1 | tee $log_file
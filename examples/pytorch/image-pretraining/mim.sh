#!/usr/bin/env bash

model=${1:-"vit"}
output_dir="./${model}_outputs"
log_file="./${model}.log"

# tune batchsize in order to occupy about 70-80% GPU memory
declare -A a100_bs=( [vit]=128 [swin]=256 [swinv2]=256 [deit]=128 )
declare -A hac_bs=( [vit]=512 [swin]=512 [swinv2]=512 [deit]=512 )

if [[ $HOSTNAME =~ "haca100" ]]; then
    batch_size=${a100_bs[$model]}
elif [[ $HOSTNAME =~ "moreh-2004-vm" ]]; then
    batch_size=${hac_bs[$model]}
else
    batch_size=32
fi

COMMON_ARGS="""
    --overwrite_output_dir \
    --remove_unused_columns False \
    --label_names bool_masked_pos \
    --do_train \
    --do_eval \
    --learning_rate 2e-5 \
    --weight_decay 0.05 \
    --num_train_epochs 5 \
    --logging_strategy steps \
    --logging_steps 100 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 1 \
    --seed 42 \
    --report_to tensorboard \
    --disable_tqdm True \
"""

/usr/bin/env python3 run_mim.py \
    --model_type $model \
    --output_dir $output_dir \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    $COMMON_ARGS \
    2>&1 | tee $log_file

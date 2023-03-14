#!/usr/bin/env bash


/usr/bin/env python3 run_mim.py \
    --model_type vit \
    --output_dir ./outputs \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --label_names bool_masked_pos \
    --do_train \
    --do_eval \
    --learning_rate 2e-5 \
    --weight_decay 0.05 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --logging_strategy steps \
    --logging_steps 100 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 42 \
    --disable_tqdm True 2>&1 | tee mim.log

#!/usr/bin/env bash

output_dir="vitmae-outputs"

if [[ $HOSTNAME =~ "haca100" ]]; then
    batch_size=256
else
    batch_size=512
fi

/usr/bin/env python3 run_mae.py \
    --dataset_name cifar10 \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --label_names pixel_values \
    --mask_ratio 0.75 \
    --norm_pix_loss \
    --do_train \
    --do_eval \
    --base_learning_rate 1.5e-4 \
    --lr_scheduler_type cosine \
    --weight_decay 0.05 \
    --num_train_epochs 5 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --logging_strategy steps \
    --logging_steps 100 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 1 \
    --seed 42 \
    --report_to tensorboard \
    --disable_tqdm True 2>&1 | tee vitmae.log

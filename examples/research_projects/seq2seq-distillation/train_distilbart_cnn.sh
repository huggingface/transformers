#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"

export BS=32
export GAS=1

python finetune.py \
    --learning_rate=3e-5 \
    --fp16 \
    --gpus 1 \
    --do_train \
    --do_predict \
    --val_check_interval 0.25 \
    --n_val 500 \
    --num_train_epochs 2 \
    --freeze_encoder --freeze_embeds --data_dir cnn_dm \
    --max_target_length 142 --val_max_target_length=142 \
    --train_batch_size=$BS --eval_batch_size=$BS --gradient_accumulation_steps=$GAS \
    --model_name_or_path sshleifer/student_cnn_12_6 \
    --tokenizer_name facebook/bart-large \
    --warmup_steps 500 \
    --output_dir distilbart-cnn-12-6 \
    "$@"


#!/usr/bin/env bash


python finetune.py \
    --learning_rate=3e-5 \
    --do_train \
    --do_predict \
    --n_val 1000 \
    --val_check_interval 0.25 \
    --max_source_length 512 --max_target_length 56 \
    --freeze_embeds --max_target_length 56 --lab

    $@

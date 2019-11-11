#!/bin/bash

srun --mem=12G -c 2 --gres=gpu:1 -p nlp --pty bash

# train and eval on SQUAD
python ./run_squad.py \
    # bert, xlnet, or xlm
    --model_type bert \
    # for xlnet: xlnet-large-cased or xlnet-base-cased (dont set --do_lower_case)
    # for xlm: xlm-mlm-en-2048 (dont set --do_lower_case)
    --model_name_or_path bert-large-uncased-whole-word-masking \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file $SQUAD_DIR/train-v1.1.json \
    --predict_file $SQUAD_DIR/dev-v1.1.json \
    --learning_rate 3e-5 \
    --weight_decay 0 \
    --beta1 0.9 \
    --beta2 0.999 \
    --adam_epsilon 1e-8 \
    --lr_scheduler 'linear' \
    --num_train_epochs 2 \
    --max_steps -1 \ # if >0, overrides num_train_epochs
    --save_steps 30000 \
    --warmup_steps 0 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ../models/wwm_uncased_finetuned_squad/ \
    --gradient_accumulation_steps 1\ # 6 was used in bioASQ
    --per_gpu_eval_batch_size=3   \
    --per_gpu_train_batch_size=3   \

#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"


export BS=8
export GAS=4
export MAX_LEN=128
python finetune.py \
    --learning_rate=3e-5 \
    --do_train \
    --val_check_interval=0.25 \
    --adam_eps 1e-06 \
    --num_train_epochs 6 --src_lang en_XX --tgt_lang ro_RO \
    --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
    --train_batch_size=$BS --eval_batch_size=$BS --gradient_accumulation_steps=$GAS \
    --task translation \
    --warmup_steps 500 \
    --freeze_embeds \
    --model_name_or_path=facebook/mbart-large-cc25 \
	--label_smoothing 0.1  --freeze_embeds --gpus 1 --logger_name wandb --sortish_sampler \
	$@

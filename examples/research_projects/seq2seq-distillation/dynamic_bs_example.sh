#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"
export WANDB_PROJECT=dmar
export MAX_LEN=128
export m=sshleifer/student_marian_en_ro_6_1
python finetune.py \
  --learning_rate=3e-4 \
  --do_train \
  --fp16 \
  --data_dir wmt_en_ro \
  --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
  --freeze_encoder --freeze_embeds \
  --train_batch_size=48 --eval_batch_size=64 \
  --tokenizer_name $m --model_name_or_path $m --num_train_epochs=1 \
  --warmup_steps 500 --logger_name wandb --gpus 1 \
  --fp16_opt_level=O1 --task translation \
  "$@"

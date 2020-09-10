#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"
export WANDB_PROJECT=dmar
export MAX_LEN=128
export m=sshleifer/student_marian_en_ro_6_1
python distillation.py \
  --learning_rate=3e-4 \
  --do_train \
  --do_predict \
  --fp16 --no_teacher \
  --val_check_interval 0.25 \
  --data_dir $ENRO_DIR \
  --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
  --freeze_encoder --freeze_embeds \
  --train_batch_size=32 --eval_batch_size=64 \
  --tokenizer_name $m --model_name_or_path $m \
  --warmup_steps 500 --logger_name wandb --max_tokens_per_batch=16384 \
  --fp16_opt_level=O1 --task translation \
  "$@"

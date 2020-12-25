#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"
export WANDB_PROJECT=dmar
python distillation.py \
  --learning_rate=3e-4 \
  --do_train \
  --do_predict \
  --fp16 --no_teacher \
  --val_check_interval 0.25 \
  --data_dir $ENRO_DIR \
  --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
  --freeze_encoder --freeze_embeds \
  --train_batch_size=$BS --eval_batch_size=$BS \
  --tokenizer_name $m --model_name_or_path $m \
  --warmup_steps 500 --sortish_sampler --logger_name wandb \
  --gpus 1 --fp16_opt_level=O1 --task translation --num_sanity_val_steps=0 \
  "$@"

#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"
export BS=16
export GAS=2
python distillation.py \
  --learning_rate=3e-4 \
  --do_train \
  --do_predict \
  --fp16 \
  --val_check_interval 0.25 \
  --teacher Helsinki-NLP/opus-mt-en-ro --data_dir $ENRO_DIR \
  --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
  --student_decoder_layers 3 --student_encoder_layers 6 \
  --freeze_encoder --freeze_embeds \
  --model_name_or_path IGNORED \
  --alpha_hid=3. --length_penalty=0.5 \
  --train_batch_size=$BS --eval_batch_size=$BS --gradient_accumulation_steps=$GAS --num_train_epochs=6 \
  --tokenizer_name Helsinki-NLP/opus-mt-en-ro \
  --warmup_steps 500 \
  "$@"

#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"
export BS=16
export GAS=2
python distillation.py \
  --learning_rate=3e-4 \
  --do_train \
  --do_predict \
  --fp16 \
  --val_check_interval 0.1 --n_val 1000 \
  --teacher facebook/bart-large-xsum --data_dir $XSUM_DIR \
  --max_target_length=60 --val_max_target_length=60 --test_max_target_length=100 \
  --student_decoder_layers 6 --student_encoder_layers 12 \
  --freeze_encoder --freeze_embeds \
  --model_name_or_path IGNORED \
  --alpha_hid=3. --length_penalty=0.5 \
  --train_batch_size=$BS --eval_batch_size=$BS --gradient_accumulation_steps=$GAS --num_train_epochs=6 \
  --tokenizer_name facebook/bart-large \
  --warmup_steps 500 \
  --output_dir distilbart_xsum_12_6 \
  $@

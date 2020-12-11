#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"
python distillation.py \
  --teacher facebook/bart-large-xsum --data_dir xsum \
  --tokenizer_name facebook/bart-large-xsum \
  --student_decoder_layers 6 --student_encoder_layers 12 \
  --freeze_encoder --freeze_embeds \
  --learning_rate=3e-4 \
  --do_train \
  --do_predict \
  --fp16 --fp16_opt_level=O1 \
  --val_check_interval 0.1 --n_val 1000 --eval_beams 2 --length_penalty=0.5 \
  --max_target_length=60 --val_max_target_length=60 --test_max_target_length=100 \
  --model_name_or_path IGNORED \
  --alpha_hid=3. \
  --train_batch_size=16 --eval_batch_size=16 --gradient_accumulation_steps=2 \
  --sortish_sampler \
  --num_train_epochs=6 \
  --warmup_steps 500 \
  --output_dir distilbart_xsum_12_6 \
  "$@"

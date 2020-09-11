#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"

python finetune.py \
  --do_predict \
  --fp16 \
  --model_name_or_path sshleifer/distilbart-xsum-1-1 --data_dir xsum --test_max_target_length=512 \
  --train_batch_size=16 --eval_batch_size=16 \
  --tokenizer_name facebook/bart-large \
  --output_dir distilbart_xsum_1_1_eval --gpus 4

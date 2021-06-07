#!/usr/bin/env bash
python run_pretrain.py \
--output_dir="./wav2vec2-base" \
--num_train_epochs="1" \
--per_device_train_batch_size="8" \
--gradient_accumulation_steps="4" \
--save_total_limit="3" \
--save_steps="500" \
--logging_steps="50" \
--learning_rate="5e-3" \
--weight_decay="0.01" \
--warmup_steps="3000" \
--model_name_or_path="facebook/wav2vec2-base" \
--fp16 \
--dataset_name="librispeech_asr" \
--dataset_config_name="clean" \
--train_split_name="train.100" \
--preprocessing_num_workers="16" \
--group_by_length \
--gradient_checkpointing

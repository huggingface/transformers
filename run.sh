#!/usr/bin/env bash
export TOKENIZERS_PARALLELISM=0
MODEL_DIR="./norwegian-roberta-base"
NUM_GPUS="2"
python -m torch.distributed.launch --nproc_per_node ${NUM_GPUS} run_mlm.py \
											--output_dir="./runs" \
											--model_type="roberta" \
											--config_name="${MODEL_DIR}" \
											--tokenizer_name="${MODEL_DIR}" \
											--dataset_name="oscar" \
											--dataset_config_name="unshuffled_deduplicated_no" \
											--max_seq_length="128" \
											--pad_to_max_length \
											--weight_decay="0.01" \
											--per_device_train_batch_size="32" \
											--per_device_eval_batch_size="32" \
											--learning_rate="3e-4" \
											--warmup_steps="1000" \
											--overwrite_output_dir \
											--num_train_epochs="18" \
											--adam_beta1="0.9" \
											--adam_beta2="0.98" \
											--do_train \
											--do_eval \
											--logging_steps="500" \
											--evaluation_strategy="epoch" \
											--report_to="tensorboard" \
											--save_strategy="no" \

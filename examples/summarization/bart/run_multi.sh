#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

# OUTPUT_DIR=dbart/logs/dl1_last ./run_distiller.sh --n_val 500 --num_train_epochs 2 --student_decoder_layers 1

OUTPUT_DIR=dbart/logs/dl6_maxlen_140 ./run_distiller.sh --n_val 500 --num_train_epochs 2 --student_decoder_layers 6 \
  --max_target_length 140 --test_mtl 140 --val_mtl 140 \
  --gradient_accumulation_steps 3 \
  --train_batch_size=16 \
  --eval_batch_size=32 \

OUTPUT_DIR=dbart/logs/dl6_ckpt ./run_distiller.sh --n_val 500 --num_train_epochs 2 \
  --student_decoder_layers 6 --resume_from_checkpoint=dbart/logs/run_6l/checkpointepoch=0.ckpt


#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

OUTPUT_DIR=dbart/logs/dl1_last ./run_distiller.sh --n_val 500 --num_train_epochs 2 --student_decoder_layers 1
OUTPUT_DIR=dbart/logs/dl3 ./run_distiller.sh --n_val 500 --num_train_epochs 2 --student_decoder_layers 3
OUTPUT_DIR=dbart/logs/dl6_ckpt ./run_distiller.sh --n_val 500 --num_train_epochs 2 \
  --student_decoder_layers 6 --resume_from_checkpoint=dbart/logs/run_6l/checkpointepoch=0.ckpt


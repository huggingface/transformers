#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
OUTPUT_DIR=dbart/logs/1l_nv300 ./run_distiller.sh --n_val 300 --num_train_epochs 2 --student_decoder_layers 1
OUTPUT_DIR=dbart/logs/1l_nv300 ./run_distiller.sh --n_val 300 --num_train_epochs 2 --student_decoder_layers 1


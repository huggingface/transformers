#!/bin/bash
DATE=`date +%y%m%d_%H%M`

export SQUAD_DIR=~/work/datasets
export OUTPUT_DIR=./output-kor-cased-${DATE}

python examples/run_squad.py \
  --bert_model bert-base-multilingual-cased \
  --do_train \
  --do_predict \
  --train_file $SQUAD_DIR/kor_train.json \
  --predict_file $SQUAD_DIR/kor_dev.json \
  --train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 4.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $OUTPUT_DIR

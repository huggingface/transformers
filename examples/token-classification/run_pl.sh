#!/usr/bin/env bash

pip install -r ../requirements.txt

export MAX_LENGTH=128
export BERT_MODEL=bert-base-multilingual-cased
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SEED=1

export OUTPUT_DIR_NAME=germeval-model
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}
mkdir -p $OUTPUT_DIR

# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

python3 run_pl_ner.py --data_dir ./ \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--train_batch_size $BATCH_SIZE \
--seed $SEED \
--gpus 1 \
--do_train \
--do_predict

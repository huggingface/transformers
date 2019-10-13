#!/bin/bash

for ((i=0;i<4;i++));do
{
CUDA_VISIBLE_DEVICES=$i python ../examples/run_squad.py \
    --model_type bert \
    --model_name_or_path ../models/wwm_test \
    --do_eval \
    --do_lower_case \
    --train_file ../data/nq/all_squadformat_nq_train.json \
    --predict_file ../data/nq_sentence_selector/dev_1_piece/${i}_splited_test_squad2format.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --output_dir ../models/wwm_test \
    --per_gpu_eval_batch_size 32   \
    --per_gpu_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_steps 100000 \
    --version_2_with_negative \
    --task_name nq \
    --eval_gzip_dir data/nq_sentence_selector/dev_1_piece
    echo Done ${i} evaluate
} &
echo Start Eval
python ../examples/nq_split_eval.py --eval_gzip_dir ../data/nq_sentence_selector/dev_1_piece --input_prediction_dir ../models/wwm_test
done

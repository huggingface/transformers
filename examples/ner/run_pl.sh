# Require pytorch-lightning=0.6
export MAX_LENGTH=128
export BERT_MODEL=bert-base-multilingual-cased
export OUTPUT_DIR=germeval-model
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SAVE_STEPS=750
export SEED=1

python3 run_pl_ner.py --data_dir ./ \
--model_type bert \
--labels ./labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--train_batch_size 32 \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_predict

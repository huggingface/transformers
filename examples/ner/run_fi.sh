# download finer dataset
wget https://raw.githubusercontent.com/mpsilfve/finer-data/master/data/digitoday.2014.train.csv
wget https://raw.githubusercontent.com/mpsilfve/finer-data/master/data/digitoday.2014.dev.csv
wget https://raw.githubusercontent.com/mpsilfve/finer-data/master/data/digitoday.2015.test.csv

# directory for cache/intermediate results
export MODEL_INPUT_DIR=$DATA_DIR/transformers_ner_finer_uncased
# directory for output model
export OUTPUT_DIR=$MODEL_DIR/finer-uncased-model

export MAX_LENGTH=128
export BERT_MODEL=bert-base-finnish-uncased-v1

# first preprocessing
python preprocess_fi.py digitoday.2014.train.csv train.txt.tmp
python preprocess_fi.py digitoday.2014.dev.csv dev.txt.tmp
python preprocess_fi.py digitoday.2015.test.csv test.txt.tmp

# # for testing purpose, take very small input files
# mv train.txt.tmp train.txt.tmp.bk && head -1000 train.txt.tmp.bk > train.txt.tmp
# mv dev.txt.tmp dev.txt.tmp.bk && head -200 dev.txt.tmp.bk > dev.txt.tmp
# mv test.txt.tmp test.txt.tmp.bk && head -200 test.txt.tmp.bk > test.txt.tmp

# split according to BERT tokenizer
python preprocess.py train.txt.tmp $BERT_MODEL $MAX_LENGTH > $MODEL_INPUT_DIR/train.txt
python preprocess.py dev.txt.tmp $BERT_MODEL $MAX_LENGTH > $MODEL_INPUT_DIR/dev.txt
python preprocess.py test.txt.tmp $BERT_MODEL $MAX_LENGTH > $MODEL_INPUT_DIR/test.txt

cat $MODEL_INPUT_DIR/train.txt $MODEL_INPUT_DIR/dev.txt $MODEL_INPUT_DIR/test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > $MODEL_INPUT_DIR/labels.txt
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SAVE_STEPS=750
export SEED=1

python3 run_ner.py --data_dir $MODEL_INPUT_DIR \
--model_type bert \
--labels $MODEL_INPUT_DIR/labels.txt \
--model_name_or_path $BERT_MODEL \
--do_lower_case \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict

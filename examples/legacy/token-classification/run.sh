## The relevant files are currently on a shared Google
## drive at https://drive.google.com/drive/folders/1kC0I2UGl2ltrluI9NqDjaQJGw5iliw_J
## Monitor for changes and eventually migrate to use the `datasets` library

fetch () {
    curl -L "https://drive.google.com/uc?export=download&id=$1" |
    grep -v "^#" | cut -f 2,3 | tr '\t' ' '
}
fetch '1Jjhbal535VVz2ap4v4r_rN1UEHTdLK5P' > train.txt.tmp
fetch '1ZfRcQThdtAR5PPRjIDtrVP7BtXSCUBbm' > dev.txt.tmp
fetch '1u9mb7kNJHWQCWyweMDRMuTFoOHOfeBTH' > test.txt.tmp

export MAX_LENGTH=128
export BERT_MODEL=bert-base-multilingual-cased

prep () {
    python3 scripts/preprocess.py "$1" "$BERT_MODEL" "$MAX_LENGTH"
}
prep train.txt.tmp  > train.txt
prep dev.txt.tmp > dev.txt
prep test.txt.tmp > test.txt
cat train.txt dev.txt test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > labels.txt
export OUTPUT_DIR=germeval-model
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SAVE_STEPS=750
export SEED=1

python3 run_ner.py \
--task_type NER \
--data_dir . \
--labels ./labels.txt \
--model_name_or_path "$BERT_MODEL" \
--output_dir "$OUTPUT_DIR" \
--max_seq_length "$MAX_LENGTH" \
--num_train_epochs "$NUM_EPOCHS" \
--per_gpu_train_batch_size "$BATCH_SIZE" \
--save_steps "$SAVE_STEPS" \
--seed "$SEED" \
--do_train \
--do_eval \
--do_predict

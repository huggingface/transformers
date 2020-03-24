curl -L 'https://sites.google.com/site/germeval2014ner/data/NER-de-train.tsv?attredirects=0&d=1' \
| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > train.txt.tmp
curl -L 'https://sites.google.com/site/germeval2014ner/data/NER-de-dev.tsv?attredirects=0&d=1' \
| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > dev.txt.tmp
curl -L 'https://sites.google.com/site/germeval2014ner/data/NER-de-test.tsv?attredirects=0&d=1' \
| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > test.txt.tmp
 wget "https://raw.githubusercontent.com/stefan-it/fine-tuned-berts-seq/master/scripts/preprocess.py"
export MAX_LENGTH=128
export BERT_MODEL=bert-base-multilingual-cased
python3 preprocess.py train.txt.tmp $BERT_MODEL $MAX_LENGTH > $DATA_DIR/transformers_ner_test/train.txt
python3 preprocess.py dev.txt.tmp $BERT_MODEL $MAX_LENGTH > $DATA_DIR/transformers_ner_test/dev.txt
python3 preprocess.py test.txt.tmp $BERT_MODEL $MAX_LENGTH > $DATA_DIR/transformers_ner_test/test.txt
cat $DATA_DIR/transformers_ner_test/train.txt $DATA_DIR/transformers_ner_test/dev.txt $DATA_DIR/transformers_ner_test/test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > $DATA_DIR/transformers_ner_test/labels.txt
export OUTPUT_DIR=$MODEL_DIR/germeval-model
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SAVE_STEPS=750
export SEED=1

python3 run_ner.py --data_dir $DATA_DIR/transformers_ner_test \
--model_type bert \
--labels $DATA_DIR/transformers_ner_test/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict

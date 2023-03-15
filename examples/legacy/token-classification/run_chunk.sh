if ! [ -f ./dev.txt ]; then
  echo "Downloading CONLL2003 dev dataset...."
  curl -L -o ./dev.txt 'https://github.com/davidsbatista/NER-datasets/raw/master/CONLL2003/valid.txt'
fi

if ! [ -f ./test.txt ]; then
  echo "Downloading CONLL2003 test dataset...."
  curl -L -o ./test.txt 'https://github.com/davidsbatista/NER-datasets/raw/master/CONLL2003/test.txt'
fi

if ! [ -f ./train.txt ]; then
  echo "Downloading CONLL2003 train dataset...."
  curl -L -o ./train.txt 'https://github.com/davidsbatista/NER-datasets/raw/master/CONLL2003/train.txt'
fi

export MAX_LENGTH=200
export BERT_MODEL=bert-base-uncased
export OUTPUT_DIR=chunker-model
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SAVE_STEPS=750
export SEED=1

python3 run_ner.py \
--task_type Chunk \
--data_dir . \
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


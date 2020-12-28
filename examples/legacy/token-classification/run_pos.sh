if ! [ -f ./dev.txt ]; then
  echo "Download dev dataset...."
  curl -L -o ./dev.txt 'https://github.com/UniversalDependencies/UD_English-EWT/raw/master/en_ewt-ud-dev.conllu'
fi

if ! [ -f ./test.txt ]; then
  echo "Download test dataset...."
  curl -L -o ./test.txt 'https://github.com/UniversalDependencies/UD_English-EWT/raw/master/en_ewt-ud-test.conllu'
fi

if ! [ -f ./train.txt ]; then
  echo "Download train dataset...."
  curl -L -o ./train.txt 'https://github.com/UniversalDependencies/UD_English-EWT/raw/master/en_ewt-ud-train.conllu'
fi

export MAX_LENGTH=200
export BERT_MODEL=bert-base-uncased
export OUTPUT_DIR=postagger-model
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SAVE_STEPS=750
export SEED=1

python3 run_ner.py \
--task_type POS \
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


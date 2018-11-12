export BERT_BASE_DIR=./bert/uncased_L-12_H-768_A-12
export BERT_PYTORCH_DIR=./bert_pytorch
export SQUAD_DIR=./squad

python run_squad.py \
  --vocab_file $BERT_BASE_DIR/vocab.txt \
  --bert_config_file $BERT_BASE_DIR/bert_config.json \
  --init_checkpoint $BERT_PYTORCH_DIR/pytorch_model.bin \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ../debug_squad/
# Model types taken from https://huggingface.co/transformers/pretrained_models.html
export MODEL_TYPE=gpt2
# Specific pre-trained model used is `gpt2`
export MODEL_NAME=gpt2

# set env vars
export MAX_LENGTH=16
export NUM_EPOCHS=1
export OUTPUT_DIR="name_${MODEL_TYPE}"
export BATCH_SIZE=64
export SAVE_STEPS=300
export LOGGING_STEPS=50
export SEED=1

rm -rf $OUTPUT_DIR cached*
python3 run_seq2seq.py \
  --data_dir ./ \
  --warmup_steps 0 \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL_NAME \
  --output_dir $OUTPUT_DIR \
  --max_seq_length $MAX_LENGTH \
  --num_train_epochs $NUM_EPOCHS \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --per_gpu_eval_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps 1 \
  --save_steps $SAVE_STEPS \
  --logging_steps $LOGGING_STEPS \
  --optimizer lamb \
  --eval_all_checkpoints \
  --learning_rate 0.01 \
  --weight_decay 0.0 \
  --seed $SEED \
  --overwrite_output_dir \
  --overwrite_cache \
  --evaluate_during_training \
  --do_train \
  --do_eval \
  --do_predict

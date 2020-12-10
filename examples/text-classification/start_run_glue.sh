export TASK_NAME=MRPC
echo $TASK_NAME
DATE=$(date +%Y%m%d-%H:%M:%S)
nohup python ./run_glue.py  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ~/$TASK_NAME/  >${DATE}.log 2>&1 &

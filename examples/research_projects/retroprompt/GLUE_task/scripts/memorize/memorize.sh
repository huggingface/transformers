BS=32
MODEL="roberta-large"
MODEL_NAME="roberta_large"

for REAL_BS in 8
do

for TASK in "SST-2"
do

case $TASK in
    SST-2)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        DEMO_TEMPLATE=*sent_0*_It_was*mask*.
        MAPPING="{'0':'terrible','1':'great'}"
        MAX_LENGTH=128
        ;;
esac


DATA_DIR=data/original_data/glue/SST-2

CUDA_VISIBLE_DEVICES=0 python run.py \
  --task_name $TASK \
  --data_dir $DATA_DIR \
  --prompt true \
  --template $TEMPLATE \
  --mapping $MAPPING \
  --model_name_or_path $MODEL \
  --max_seq_length 80 \
  --per_device_eval_batch_size 32 \
  --seed 13 \
  --model_type "roberta" \
  --START 6000 \
  --LENGTH 1000 \
  --compute_mem


done
done
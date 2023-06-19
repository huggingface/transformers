while getopts m:b:g: flag
do
    case "${flag}" in
        m) model=${OPTARG};;
        b) batch_size=${OPTARG};;
        g) device_id=${OPTARG};;
    esac
done

LOG_DIR="./logs"
OUTPUT_DIR="./outputs"
log_file=$LOG_DIR/$model.log
output_dir=$OUTPUT_DIR/$model

mkdir -p "$(dirname $log_file)"
mkdir -p "$(dirname $output_dir)"

## Using moreh device
export MOREH_VISIBLE_DEVICE=$device_id

args="
--do_train \
--do_eval \
--learning_rate 3e-5 \
--num_train_epochs 2 \
--logging_strategy steps \
--logging_steps 100 \
--max_seq_length 384 \
--doc_stride 128 \
--overwrite_output_dir \
--save_strategy epoch \
--save_total_limit 2 \
--seed 42
"

python run_xnli.py \
  --model_name_or_path $model \
  --language de \
  --train_language en \
  --per_device_train_batch_size $batch_size \
  --per_device_eval_batch_size $batch_size \
  --output_dir $output_dir \
  $args \
  2>&1 | tee $log_file
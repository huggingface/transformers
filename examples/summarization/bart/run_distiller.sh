export OUTPUT_DIR_NAME=bart_sum
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../../":"${PYTHONPATH}"

python finetune.py \
--data_dir=/home/shleifer/transformers_fork/cnn_dm/ \
--teacher=bart-large-cnn \
--model_name_or_path=student \
--learning_rate=3e-5 \
--train_batch_size=4 \
--eval_batch_size=4 \
--output_dir=$OUTPUT_DIR \
--do_train \
--do_predict \
--n_gpu 1 \
--fp16 \
--output_dir $OUTPUT_DIR \
$@

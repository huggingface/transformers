export OUTPUT_DIR_NAME=t5
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

python finetune.py \
--data_dir=./cnn-dailymail/cnn_dm \
--model_name_or_path=t5-large \
--model_type=t5
--learning_rate=3e-5 \
--train_batch_size=4 \
--eval_batch_size=4 \
--output_dir=$OUTPUT_DIR \
--do_train  $@

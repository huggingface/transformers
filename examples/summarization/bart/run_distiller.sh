export CURRENT_DIR=${PWD}
#CNN_DIR = /home/shleifer/transformers_fork/examples/summarization/bart/cnn_dm

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../../":"${PYTHONPATH}"

python finetune.py \
--data_dir=$CNN_DIR \
--teacher=bart-large-cnn \
--model_name_or_path=student \
--learning_rate=3e-5 \
--output_dir=$OUTPUT_DIR \
--do_train \
--do_predict \
--n_gpu 1 \
--fp16 \
--val_check_interval 0.25 \
$@

export CURRENT_DIR=${PWD}
#CNN_DIR = /home/shleifer/transformers_fork/examples/summarization/bart/cnn_dm

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../../":"${PYTHONPATH}"

python finetune.py \
--data_dir=$CNN_DIR \
--no_teacher \
--do_predict \
--fp16 \
$@

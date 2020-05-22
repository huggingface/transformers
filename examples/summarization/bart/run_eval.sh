# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../../":"${PYTHONPATH}"

python finetune.py \
--data_dir=$CNN_DIR \
--no_teacher \
--do_predict \
--fp16 \
$@

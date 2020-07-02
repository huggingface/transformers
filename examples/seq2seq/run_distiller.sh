# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

python distillation.py \
--learning_rate=3e-4 \
--do_train \
--do_predict \
--fp16 \
--val_check_interval 0.1 \
$@

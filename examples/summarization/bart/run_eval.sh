# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../../":"${PYTHONPATH}"

python finetune.py \
--no_teacher \
--do_predict \
--fp16 \
$@

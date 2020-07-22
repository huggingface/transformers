# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

# the proper usage is documented in the README, you need to specify data_dir, output_dir and model_name_or_path
# NB You need to adjust the learning_rate, batch_size (for train, eval etc) and pass in n_tpu_cores as well.
# TPUs are very sensitive to these params.

python finetune.py \
    --learning_rate=3e-5 \
    --gpus 0 \
    --do_train \
    --do_predict \
    --n_val 1000 \
    --val_check_interval 0.1 \
    $@

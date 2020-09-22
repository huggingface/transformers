export PYTHONPATH="../":"${PYTHONPATH}"

export TPU_NUM_CORES=8

# the proper usage is documented in the README, you need to specify data_dir, output_dir and model_name_or_path
# run ./finetune_trainer.sh --help to see all the possible options
python xla_spawn.py --num_cores $TPU_NUM_CORES \
    finetune_trainer.py \
    --tpu_num_cores $TPU_NUM_CORES \
    --learning_rate=3e-5 \
    --fp16 \
    --do_train --do_eval --evaluate_during_training \
    --n_val 1000 \
    "$@"
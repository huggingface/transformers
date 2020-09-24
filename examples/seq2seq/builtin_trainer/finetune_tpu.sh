export TPU_NUM_CORES=8

# the proper usage is documented in the README, you need to specify data_dir, output_dir and model_name_or_path
# run ./builtin_trainer/finetune_tpu.sh --help to see all the possible options
python xla_spawn.py --num_cores $TPU_NUM_CORES \
    finetune_trainer.py \
    --learning_rate=3e-5 \
    --do_train --do_eval --evaluate_during_training \
    --prediction_loss_only \
    --n_val 1000 \
    "$@"

# the proper usage is documented in the README, you need to specify data_dir, output_dir and model_name_or_path
# run ./builtin_trainer/finetune.sh --help to see all the possible options
python finetune_trainer.py \
    --learning_rate=3e-5 \
    --fp16 \
    --do_train --do_eval --do_predict --evaluate_during_training \
    --predict_with_generate \
    --n_val 1000 \
    "$@"

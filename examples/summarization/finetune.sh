export OUTPUT_DIR=bart_cnn_finetune

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"


# --model_name_or_path=t5-base for t5

python -m pdb -c continue finetune.py \
    --data_dir=$CNN_DIR \
    --model_name_or_path=facebook/bart-large \
    --learning_rate=3e-5 \
    --train_batch_size=1\
    --eval_batch_size=1 \
    --fp16 \
    --gpus 1 \
    --output_dir $OUTPUT_DIR \
    --do_train \
    --do_predict \
    --n_val 1000 \
    --val_check_interval 0.1 \
    --sortish_sampler \
    --num_train_epochs 6 \
    --max_target_length=56 \
    $@

# # Install newest ptl.
# pip install -U git+http://github.com/PyTorchLightning/pytorch-lightning/
# Install example requirements
pip install -r ../requirements.txt

export TASK=mrpc
export DATA_DIR=./cached_glue_data
export MAX_LENGTH=128
export LEARNING_RATE=2e-5
export BERT_MODEL=bert-base-cased
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SEED=2
export GPUS=1
export NUM_WORKERS=4

# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

python3 -i run_pl_glue.py \
    --model_name_or_path $BERT_MODEL \
    --task $TASK \
    --data_dir $DATA_DIR \
    --max_seq_length  $MAX_LENGTH \
    --max_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --seed $SEED \
    --gpus $GPUS \
    --num_workers $NUM_WORKERS \
    --train_batch_size $BATCH_SIZE \
    --do_train \
    --do_predict

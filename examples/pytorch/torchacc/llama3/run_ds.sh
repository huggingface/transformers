set -ex

echo "Running a deepspeed job ..."

export USE_TORCH_XLA=0

[ -z "$RANK" ] && RANK=0
[ -z "$WORLD_SIZE" ] && WORLD_SIZE=1
[ -z "$MASTER_ADDR" ] && MASTER_ADDR=127.0.0.1
[ -z "$MASTER_PORT" ] && MASTER_PORT=9010

BS=1
SEQLEN=4096
NPROC_PER_NODE=8
PRECISION="bf16=true"
FSDP_CONFIG="llama3_fsdp_ds.json"
JOB_NAME="LLAMA3_FSDP_DEEPSPEED_GPU${NPROC_PER_NODE}_BS${BS}_SEQLEN${SEQLEN}_BF16_FA"


torchrun --nproc_per_node $NPROC_PER_NODE \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_port $MASTER_PORT \
    --master_addr $MASTER_ADDR \
    ../../language-modeling/run_clm.py \
    --num_train_epochs 2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --use_fast_tokenizer false \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size $BS \
    --do_train \
    --output_dir /tmp/test-clm \
    --overwrite_output_dir \
    --config_name ./Meta-Llama-3-8B/ \
    --tokenizer_name ./Meta-Llama-3-8B/ \
    --trust_remote_code true \
    --cache_dir ../cache \
    --block_size $SEQLEN \
    --optim adamw_torch \
    --save_strategy no \
    --logging_strategy steps \
    --gradient_checkpointing no \
    --logging_steps 100 \
    --$PRECISION \
    --deepspeed $FSDP_CONFIG 2>&1 | tee ./$JOB_NAME.log

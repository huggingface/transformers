#!/bin/bash

# Change for multinode config
MASTER_ADDR=$1
MASTER_PORT=1234
NNODES=$2
NODE_RANK=$3
N_GPUS=$4

DISTRIBUTED_ARGS="--nproc_per_node $N_GPUS --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT --use_env"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
  examples/run_pretraining_bert.py \

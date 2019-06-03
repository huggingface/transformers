#!/bin/bash

MASTER=192.168.170.99
NODES=(dgx01 dgx02 dgx04 dgx05)
NNODES=${#NODES[@]}
NGPUS=8

for i in "${!NODES[@]}"; do
    HOST=${NODES[$i]}
    CMD="cd pytorch-pretrained-BERT && bash scripts/launch_docker.sh \"bash scripts/run_pretraining_distributed.sh $MASTER $NNODES $i $NGPUS\""
    echo $MD
    ssh -tt $HOST $CMD &
done

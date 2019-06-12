#!/bin/bash

MASTER=dgx02
NODES=(dgx02 dgx04 dgx05 dgx06)
MASTER_IP=`dig +short ${MASTER}.elementai.net | tail -n 1`
NNODES=${#NODES[@]}
NGPUS=8

echo $MASTER_IP

for i in "${!NODES[@]}"; do
    HOST=${NODES[$i]}
    CMD="cd code/pytorch-pretrained-BERT && bash scripts/launch_docker.sh \"bash scripts/run_pretraining_distributed.sh $MASTER_IP $NNODES $i $NGPUS\""
    echo $CMD
    ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -tt $HOST $CMD &
done

#!/bin/bash

NODES=(dgx01 dgx02 dgx04 dgx05 dgx06)

for i in "${!NODES[@]}"; do
    HOST=${NODES[$i]}
    CMD="docker kill \$(docker ps -q --filter \"ancestor=images.borgy.elementai.net/hdvries/bert_pytorch\" --format=\"{{.ID}}\")"
    ssh -tt $HOST $CMD
done
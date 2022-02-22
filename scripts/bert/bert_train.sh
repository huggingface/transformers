#!/bin/bash
set -e 
set -x
pip3 install transformers datasets


cd ~ && git clone --branch bert-tf2 https://github.com/ROCmSoftwarePlatform/transformers
# Script to train the small 117M model
python3 transformers/scripts/bert/bert_train.py > log.txt
cat log.txt | tail -n 1
cat log.txt | tail -n 1 | awk '{ print "Accuracy: " $(NF) }'


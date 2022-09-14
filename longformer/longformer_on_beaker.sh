#!/bin/bash

export SCRIPTS=$(beaker dataset create -q .)
export INPUT_DATASET_ID="ds_drt127wv4aun"
export RESULT_SAVE_DIR="/runs"
export RESULT_SAVE_PREFIX="test"
export ARGS="$@"
export GPU_COUNT=8
export CPU_COUNT=32
export CLUSTER="ai2/on-prem-ai2-server3"
export RESULT_PATH=$RESULT_SAVE_DIR/$RESULT_SAVE_PREFIX

beaker experiment create -f experiment.yml

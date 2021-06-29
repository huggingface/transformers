#!/bin/bash
model_size=$1
echo $model_size
model_dir=$2
profile_dir=$3
rocprof --stats python3 gpt2_1step.py $model_size $model_dir
python3 gpt2_profile.py $profile_dir

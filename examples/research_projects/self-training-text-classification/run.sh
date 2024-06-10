# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

# Create a virtual environment
conda deactivate
conda update conda -y
conda update anaconda -y
pip install --upgrade pip
python3 -m pip install --user virtualenv
conda create -n strata python=3.9 -y
conda activate strata
# Install all necessary packages
pip install transformers
pip install -r requirements.txt

# Download and prepare data
WORK_DIR="/tmp/strata"
rm -rf "${WORK_DIR}" && mkdir -p "${WORK_DIR}"
wget https://storage.googleapis.com/gresearch/strata/demo.zip -P "${WORK_DIR}"
DEMO_ZIP_FILE="${WORK_DIR}/demo.zip"
unzip "${DEMO_ZIP_FILE}" -d "${WORK_DIR}" && rm "${DEMO_ZIP_FILE}"
DATA_DIR="${WORK_DIR}/demo/scitail-8"
OUTPUT_DIR="/tmp/output"
rm -rf "${OUTPUT_DIR}" && mkdir -p "${OUTPUT_DIR}"

# Specific hyperparameters
MODEL_NAME_OR_PATH="bert-base-uncased"
NUM_NODES=1
NUM_TRAINERS=4
LAUNCH_SCRIPT="torchrun --nnodes='${NUM_NODES}' --nproc_per_node='${NUM_TRAINERS}' python -c"
MAX_SELFTRAIN_ITERATIONS=100
TRAIN_FILE="train.csv"
INFER_FILE="infer.csv"
EVAL_FILE="eval_256.csv"
MAX_STEPS=100000

# Start self-training
${LAUNCH_SCRIPT} "
import os
from selftraining import selftrain

data_dir = '${DATA_DIR}'
parameters_dict = {
  'max_selftrain_iterations': ${MAX_SELFTRAIN_ITERATIONS},
  'model_name_or_path': '${MODEL_NAME_OR_PATH}',
  'output_dir': '${OUTPUT_DIR}',
  'train_file': os.path.join(data_dir, '${TRAIN_FILE}'),
  'infer_file': os.path.join(data_dir, '${INFER_FILE}'),
  'eval_file': os.path.join(data_dir, '${EVAL_FILE}'),
  'eval_strategy': 'steps',
  'task_name': 'scitail',
  'label_list': ['entails', 'neutral'],
  'per_device_train_batch_size': 32,
  'per_device_eval_batch_size': 8,
  'max_length': 128,
  'learning_rate': 2e-5,
  'max_steps': ${MAX_STEPS},
  'eval_steps': 1,
  'early_stopping_patience': 50,
  'overwrite_output_dir': True,
  'do_filter_by_confidence': False,
  'do_filter_by_val_performance': True,
  'finetune_on_labeled_data': False,
  'seed': 42,
}

selftrain(**parameters_dict)
"

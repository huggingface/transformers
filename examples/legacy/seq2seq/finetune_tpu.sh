# Copyright 2020 The HuggingFace Team. All rights reserved.
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

export TPU_NUM_CORES=8

# the proper usage is documented in the README, you need to specify data_dir, output_dir and model_name_or_path
# run ./finetune_tpu.sh --help to see all the possible options
python xla_spawn.py --num_cores $TPU_NUM_CORES \
    finetune_trainer.py \
    --learning_rate=3e-5 \
    --do_train --do_eval \
    --eval_strategy steps \
    --prediction_loss_only \
    --n_val 1000 \
    "$@"

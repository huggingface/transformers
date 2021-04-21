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

export WANDB_PROJECT=distil-marian
export BS=64
export m=sshleifer/student_marian_en_ro_6_3
export MAX_LEN=128
export TPU_NUM_CORES=8

python xla_spawn.py --num_cores $TPU_NUM_CORES \
    finetune_trainer.py \
    --tokenizer_name $m --model_name_or_path $m \
    --data_dir $ENRO_DIR \
    --output_dir marian_en_ro_6_3 --overwrite_output_dir \
    --learning_rate=3e-4 \
    --warmup_steps 500 \
    --per_device_train_batch_size=$BS --per_device_eval_batch_size=$BS \
    --freeze_encoder --freeze_embeds \
    --num_train_epochs=6 \
    --save_steps 500 --eval_steps 500 \
    --logging_first_step --logging_steps 200 \
    --max_source_length $MAX_LEN --max_target_length $MAX_LEN \
    --val_max_target_length $MAX_TGT_LEN --test_max_target_length $MAX_TGT_LEN \
    --do_train --do_eval \
    --evaluation_strategy steps \
    --prediction_loss_only \
    --task translation --label_smoothing_factor 0.1 \
    "$@"

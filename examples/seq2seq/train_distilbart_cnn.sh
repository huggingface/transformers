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

export WANDB_PROJECT=distilbart-trainer
export BS=32
export m=sshleifer/student_cnn_12_6
export tok=facebook/bart-large
export MAX_TGT_LEN=142

python finetune_trainer.py \
    --model_name_or_path $m --tokenizer_name $tok \ 
    --data_dir cnn_dm \
    --output_dir distilbart-cnn-12-6 --overwrite_output_dir \
    --learning_rate=3e-5 \
    --warmup_steps 500 --sortish_sampler \
    --fp16 \
    --n_val 500 \
    --gradient_accumulation_steps=1 \
    --per_device_train_batch_size=$BS --per_device_eval_batch_size=$BS \
    --freeze_encoder --freeze_embeds \
    --num_train_epochs=2 \
    --save_steps 3000 --eval_steps 3000 \
    --logging_first_step \
    --max_target_length 56 --val_max_target_length $MAX_TGT_LEN --test_max_target_length $MAX_TGT_LEN \
    --do_train --do_eval --do_predict \
    --evaluation_strategy steps \
    --predict_with_generate --sortish_sampler \
    "$@"

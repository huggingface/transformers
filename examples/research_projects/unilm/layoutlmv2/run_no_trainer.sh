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

accelerate launch run_layoutlmv2_no_trainer.py \
        --model_name_or_path microsoft/layoutlmv2-base-uncased \
        --processor_name microsoft/layoutlmv2-base-uncased \
        --output_dir /tmp/test-layoutlmv2 \
        --dataset_name nielsr/funsd \
        --max_steps 1000 \
        --warmup_ratio 0.1 \
        --fp16 \
        --model_revision no_ocr

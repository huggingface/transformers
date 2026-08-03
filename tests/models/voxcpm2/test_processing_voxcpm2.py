# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from transformers import DacFeatureExtractor
from transformers.models.voxcpm2.processing_voxcpm2 import VoxCPM2Processor

from .test_tokenization_voxcpm2 import get_tiny_voxcpm2_tokenizer


def get_tiny_voxcpm2_processor() -> VoxCPM2Processor:
    tokenizer = get_tiny_voxcpm2_tokenizer()
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "<|audio_start|>",
                "<|audio_prompt_start|>",
                "<|audio_prompt_end|>",
            ]
        }
    )
    feature_extractor = DacFeatureExtractor(sampling_rate=16000, hop_length=1, return_attention_mask=True)
    return VoxCPM2Processor(feature_extractor, tokenizer, audio_patch_size=4)

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import unittest

from transformers.models.llava_onevision1_5.convert_llava_onevision1_5_weights_to_hf import build_config


class LlavaOnevision1_5ConversionTest(unittest.TestCase):
    def test_preserves_generation_token_ids(self):
        original_config = {
            "image_token_id": 98,
            "video_token_id": 97,
            "text_config": {
                "vocab_size": 100,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "hidden_act": "silu",
                "max_position_embeddings": 128,
                "initializer_range": 0.02,
                "rms_norm_eps": 1e-6,
                "use_cache": True,
                "rope_theta": 10000.0,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "use_sliding_window": False,
                "sliding_window": None,
                "max_window_layers": 2,
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
            },
            "vision_config": {
                "depth": 2,
                "hidden_size": 32,
                "hidden_act": "gelu",
                "intermediate_size": 64,
                "num_heads": 4,
                "in_channels": 3,
                "patch_size": 4,
                "spatial_merge_size": 2,
                "temporal_patch_size": 1,
                "text_hidden_size": 32,
                "layer_norm_eps": 1e-5,
                "initializer_range": 0.02,
            },
        }
        config = build_config(original_config)
        self.assertEqual(config.text_config.pad_token_id, 0)
        self.assertEqual(config.text_config.bos_token_id, 1)
        self.assertEqual(config.text_config.eos_token_id, 2)

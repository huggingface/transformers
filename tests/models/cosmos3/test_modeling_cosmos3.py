# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Cosmos3 model."""

import copy
import unittest

from transformers import AutoConfig, Cosmos3Config, is_torch_available
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import WeightRenaming, rename_source_key
from transformers.testing_utils import require_torch


if is_torch_available():
    from transformers import AutoModel, AutoModelForImageTextToText, Cosmos3ForConditionalGeneration, Cosmos3Model


def get_tiny_cosmos3_config():
    return Cosmos3Config(
        text_config={
            "vocab_size": 99,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "pad_token_id": 0,
            "rope_parameters": {
                "rope_type": "default",
                "mrope_section": [16, 8, 8],
                "mrope_interleaved": True,
                "rope_theta": 10000,
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 16,
            "spatial_merge_size": 1,
            "temporal_patch_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0],
        },
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
        tie_word_embeddings=False,
        pad_token_id=0,
    )


class Cosmos3ConfigTest(unittest.TestCase):
    def test_auto_config_mapping(self):
        config = AutoConfig.for_model("cosmos3_omni")

        self.assertIsInstance(config, Cosmos3Config)
        self.assertEqual(config.model_type, "cosmos3_omni")


class Cosmos3ConversionMappingTest(unittest.TestCase):
    def test_checkpoint_conversion_mapping_targets_unified_checkpoint_namespaces(self):
        mapping = get_checkpoint_conversion_mapping("cosmos3_omni")
        renamings = [entry for entry in mapping if isinstance(entry, WeightRenaming)]

        self.assertEqual(
            rename_source_key("model.layers.0.self_attn.q_proj.weight", renamings, [])[0],
            "model.language_model.layers.0.self_attn.q_proj.weight",
        )
        self.assertEqual(
            rename_source_key("blocks.0.norm1.weight", renamings, [])[0],
            "model.visual.blocks.0.norm1.weight",
        )
        self.assertEqual(
            rename_source_key("merger.mlp.0.weight", renamings, [])[0],
            "model.visual.merger.mlp.0.weight",
        )

        already_nested_key = "model.language_model.layers.0.self_attn.q_proj.weight"
        self.assertEqual(rename_source_key(already_nested_key, renamings, [])[0], already_nested_key)


@require_torch
class Cosmos3ModelTest(unittest.TestCase):
    def test_auto_model_mappings(self):
        config = get_tiny_cosmos3_config()

        self.assertIsInstance(AutoModel.from_config(copy.deepcopy(config)), Cosmos3Model)
        self.assertIsInstance(
            AutoModelForImageTextToText.from_config(copy.deepcopy(config)), Cosmos3ForConditionalGeneration
        )

    def test_unified_checkpoint_unexpected_keys_are_ignored(self):
        self.assertIn(r"_moe_gen", Cosmos3Model._keys_to_ignore_on_load_unexpected)
        self.assertIn(r"^llm2sound\.", Cosmos3ForConditionalGeneration._keys_to_ignore_on_load_unexpected)
        self.assertIn(r"^lm_head\.weight$", Cosmos3Model._keys_to_ignore_on_load_unexpected)
        self.assertNotIn(r"^lm_head\.weight$", Cosmos3ForConditionalGeneration._keys_to_ignore_on_load_unexpected)

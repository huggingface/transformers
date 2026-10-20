# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from transformers import is_torch_available
from transformers.quantizers.quantizers_utils import should_convert_module
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch


class SkipListTest(unittest.TestCase):
    def test_a_module_name_covers_its_subtree_on_a_name_boundary(self):
        """`"model.layers.1"` once also skipped layers 10-19, and `"fc1"` any name ending in it."""
        skip = ["model.layers.1", "fc1", "lm_head"]
        for name in ("model.layers.1", "model.layers.1.mlp.up_proj", "model.decoder.layers.23.fc1", "lm_head"):
            self.assertFalse(should_convert_module(name, skip), name)
        for name in ("model.layers.10", "model.layers.12.mlp.up_proj", "model.decoder.layers.23.xfc1", "lm_head_2"):
            self.assertTrue(should_convert_module(name, skip), name)

    def test_a_regex_matches_from_the_start(self):
        skip = ["model.layers.*.self_attn", r"^model\.audio_tower(?:\.|$)", ".*router$"]
        for name in (
            "model.layers.3.self_attn.q_proj",
            "model.audio_tower.layers.1.proj",
            "model.layers.3.mlp.router",
        ):
            self.assertFalse(should_convert_module(name, skip), name)
        for name in ("model.layers.3.mlp.up_proj", "model.audio_towers.proj", "lm_head"):
            self.assertTrue(should_convert_module(name, skip), name)

    @require_torch
    def test_a_checkpoints_skip_list_covers_the_modules_it_names_once_renamed(self):
        """A skip-list ships in the checkpoint's module names; Llava renames `language_model.model.*` and
        `vision_tower.*` under `model.` on load, and the skip-list follows them there."""
        from transformers import LlavaConfig, LlavaForConditionalGeneration
        from transformers.quantizers.base import HfQuantizer

        config = LlavaConfig(
            text_config={"model_type": "llama", "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
                         "num_attention_heads": 2, "num_key_value_heads": 2, "vocab_size": 64},
            vision_config={"model_type": "clip_vision_model", "hidden_size": 16, "intermediate_size": 32,
                           "num_hidden_layers": 1, "num_attention_heads": 2, "image_size": 8, "patch_size": 4},
        )  # fmt: skip
        with torch.device("meta"):
            model = LlavaForConditionalGeneration(config)
        skip = HfQuantizer.get_modules_to_not_convert(model, ["vision_tower", "language_model.model.layers.1"])
        for name in (
            "model.vision_tower.vision_model.encoder.layers.0.mlp.fc1",
            "model.language_model.layers.1.self_attn.q_proj",
        ):
            self.assertFalse(should_convert_module(name, skip), name)
        self.assertTrue(should_convert_module("model.language_model.layers.0.self_attn.q_proj", skip))

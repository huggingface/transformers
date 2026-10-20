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
    def test_a_layer_index_is_matched_whole(self):
        """`"model.layers.1"` once also skipped layers 10-19; a name still covers its sub-tree and the names ending in
        it, as Hub skip-lists rely on (`"gate"` covers Qwen2-MoE's `shared_expert_gate`)."""
        skip = ["model.layers.1", "fc1", "gate"]
        for name in (
            "model.layers.1",
            "model.layers.1.mlp.up_proj",
            "model.decoder.layers.23.fc1",
            "mlp.shared_expert_gate",
        ):
            self.assertFalse(should_convert_module(name, skip), name)
        for name in ("model.layers.10", "model.layers.12.mlp.up_proj", "model.layers.3.mlp.up_proj"):
            self.assertTrue(should_convert_module(name, skip), name)

    def test_a_name_does_not_cover_a_longer_sibling(self):
        """Qwen3.8-27B-FP8 skips its router `mlp.gate`, which must not also skip the quantized `mlp.gate_proj`."""
        skip = ["model.layers.0.mlp.gate"]
        self.assertFalse(should_convert_module("model.layers.0.mlp.gate", skip))
        self.assertTrue(should_convert_module("model.layers.0.mlp.gate_proj", skip))

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

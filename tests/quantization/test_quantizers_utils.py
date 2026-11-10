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

from transformers.quantizers.quantizers_utils import should_convert_module


class SkipListTest(unittest.TestCase):
    def test_should_convert_module(self):
        # Should convert by default
        self.assertTrue(should_convert_module("model", None))
        self.assertTrue(should_convert_module("model", []))

        # Should not convert if in exclusion list
        modules_to_not_convert = ["model.layers.*.self_attn", "lm_head"]
        self.assertFalse(should_convert_module("lm_head", modules_to_not_convert))
        self.assertTrue(should_convert_module("experts", modules_to_not_convert))
        self.assertFalse(should_convert_module("model.layers.3.self_attn.q_proj", modules_to_not_convert))

        # A match from the start ends at a module boundary
        modules_to_not_convert = ["model.layers.1", "model.layers.0.mlp.gate", r"^model\.audio_tower(?:\.|$)"]
        self.assertFalse(should_convert_module("model.layers.1.mlp.up_proj", modules_to_not_convert))
        self.assertTrue(should_convert_module("model.layers.10.mlp.up_proj", modules_to_not_convert))
        self.assertFalse(should_convert_module("model.layers.0.mlp.gate", modules_to_not_convert))
        self.assertTrue(should_convert_module("model.layers.0.mlp.gate_proj", modules_to_not_convert))
        # A user regex that brings its own anchor and boundary still works.
        self.assertFalse(should_convert_module("model.audio_tower.layers.1.proj", modules_to_not_convert))
        self.assertTrue(should_convert_module("model.audio_towers.proj", modules_to_not_convert))

        # A name still covers the names ending in it, as Hub skip-lists rely on
        self.assertFalse(should_convert_module("model.layers.0.mlp.shared_expert_gate", ["gate"]))

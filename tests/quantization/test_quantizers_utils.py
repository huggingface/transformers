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
    @require_torch
    def test_a_checkpoints_skip_list_covers_the_modules_it_names_once_renamed(self):
        """Check that a skip-list of checkpoint names covers the modules Llava renames under `model.` on load."""
        from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration
        from transformers.quantizers.base import HfQuantizer

        text_config = LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=64,
        )
        vision_config = CLIPVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=8,
            patch_size=4,
        )
        config = LlavaConfig(text_config=text_config, vision_config=vision_config)
        with torch.device("meta"):
            model = LlavaForConditionalGeneration(config)
        skip = HfQuantizer.get_modules_to_not_convert(model, ["vision_tower", "language_model.model.layers.1"])
        for name in (
            "model.vision_tower.vision_model.encoder.layers.0.mlp.fc1",
            "model.language_model.layers.1.self_attn.q_proj",
        ):
            self.assertFalse(should_convert_module(name, skip), name)
        self.assertTrue(should_convert_module("model.language_model.layers.0.self_attn.q_proj", skip))


@require_torch
class AdaptQuantizationConfigTest(unittest.TestCase):
    def test_the_model_config_adapts_the_quantization_config(self):
        """Check that the model config adapts both a checkpoint's and a passed quantization config."""
        from transformers import LlamaConfig
        from transformers.quantizers.auto import get_hf_quantizer
        from transformers.utils.quantization_config import Mxfp4Config

        adapted = []

        class AdaptingConfig(LlamaConfig):
            def adapt_quantization_config(self, quantization_config):
                adapted.append(type(quantization_config))
                return quantization_config

        for checkpoint in (None, {"quant_method": "mxfp4"}):
            with self.subTest(pre_quantized=checkpoint is not None):
                adapted.clear()
                config = AdaptingConfig()
                config.quantization_config = checkpoint
                get_hf_quantizer(config, Mxfp4Config(dequantize=True), None, True, {})
                self.assertEqual(adapted, [Mxfp4Config])

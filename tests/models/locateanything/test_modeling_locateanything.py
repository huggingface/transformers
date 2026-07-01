# Copyright 2026 NVIDIA and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch LocateAnything model."""

import unittest

from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    LocateAnythingConfig,
    LocateAnythingVisionConfig,
    is_torch_available,
)
from transformers.models.qwen2 import Qwen2Config
from transformers.testing_utils import require_torch, torch_device


if is_torch_available():
    import torch

    from transformers import LocateAnythingForConditionalGeneration, LocateAnythingModel


class LocateAnythingModelTester:
    """Builds a tiny LocateAnything model and matching dummy inputs (CPU friendly)."""

    def __init__(self, parent):
        self.parent = parent
        self.patch_size = 14
        self.grid_h = 4
        self.grid_w = 4
        self.merge = 2
        self.image_token_id = 150
        self.vocab_size = 200
        self.text_hidden_size = 64

    def get_config(self):
        vision_config = LocateAnythingVisionConfig(
            patch_size=self.patch_size,
            init_pos_emb_height=8,
            init_pos_emb_width=8,
            num_attention_heads=4,
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            merge_kernel_size=(self.merge, self.merge),
        )
        text_config = Qwen2Config(
            vocab_size=self.vocab_size,
            hidden_size=self.text_hidden_size,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            eos_token_id=7,
        )
        return LocateAnythingConfig(
            vision_config=vision_config,
            text_config=text_config,
            image_token_id=self.image_token_id,
            box_start_token_id=151,
            box_end_token_id=152,
            ref_start_token_id=153,
            ref_end_token_id=154,
            text_mask_token_id=155,
            coord_start_token_id=160,
            coord_end_token_id=180,
            null_token_id=181,
            switch_token_id=182,
            none_token_id=10,
        )

    def prepare_inputs(self, config):
        num_patches = self.grid_h * self.grid_w
        num_image_tokens = (self.grid_h // self.merge) * (self.grid_w // self.merge)
        pixel_values = torch.randn(num_patches, 3, self.patch_size, self.patch_size, device=torch_device)
        image_grid_hws = torch.tensor([[self.grid_h, self.grid_w]], dtype=torch.long, device=torch_device)
        prompt = [1, 2, 3] + [config.image_token_id] * num_image_tokens + [4, 5, 6]
        input_ids = torch.tensor([prompt], dtype=torch.long, device=torch_device)
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_hws": image_grid_hws,
        }


@require_torch
class LocateAnythingModelTest(unittest.TestCase):
    def setUp(self):
        self.model_tester = LocateAnythingModelTester(self)

    def test_config_subconfigs(self):
        config = self.model_tester.get_config()
        self.assertIsInstance(config.vision_config, LocateAnythingVisionConfig)
        self.assertIsInstance(config.text_config, Qwen2Config)
        # Round-trips through a dict.
        config_dict = config.to_dict()
        restored = LocateAnythingConfig.from_dict(config_dict)
        self.assertEqual(restored.image_token_id, config.image_token_id)
        self.assertEqual(restored.block_size, config.block_size)

    def test_attn_implementation_propagates_to_text_config(self):
        config = self.model_tester.get_config()
        # Instantiation propagates the attn implementation down to the text config.
        LocateAnythingForConditionalGeneration(config)
        self.assertEqual(
            config.text_config._attn_implementation,
            config._attn_implementation,
        )

    def test_forward(self):
        config = self.model_tester.get_config()
        model = LocateAnythingForConditionalGeneration(config).to(torch_device).eval()
        inputs = self.model_tester.prepare_inputs(config)
        with torch.no_grad():
            out = model(**inputs)
        self.assertEqual(
            out.logits.shape,
            (1, inputs["input_ids"].shape[1], config.text_config.vocab_size),
        )
        self.assertIsNotNone(out.image_hidden_states)

    def test_text_only_forward(self):
        config = self.model_tester.get_config()
        model = LocateAnythingForConditionalGeneration(config).to(torch_device).eval()
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long, device=torch_device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        self.assertEqual(out.logits.shape, (1, 6, config.text_config.vocab_size))

    def test_model_without_lm_head(self):
        config = self.model_tester.get_config()
        model = LocateAnythingModel(config).to(torch_device).eval()
        inputs = self.model_tester.prepare_inputs(config)
        with torch.no_grad():
            out = model(**inputs)
        self.assertEqual(
            out.last_hidden_state.shape,
            (1, inputs["input_ids"].shape[1], config.text_config.hidden_size),
        )

    def test_generate_modes(self):
        config = self.model_tester.get_config()
        model = LocateAnythingForConditionalGeneration(config).to(torch_device).eval()
        inputs = self.model_tester.prepare_inputs(config)
        for mode in ("fast", "slow", "hybrid"):
            generated = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                image_grid_hws=inputs["image_grid_hws"],
                generation_mode=mode,
                max_new_tokens=12,
                do_sample=False,
            )
            self.assertEqual(generated.shape[0], 1)
            self.assertGreaterEqual(generated.shape[1], inputs["input_ids"].shape[1])

    def test_generate_invalid_mode(self):
        config = self.model_tester.get_config()
        model = LocateAnythingForConditionalGeneration(config).to(torch_device).eval()
        inputs = self.model_tester.prepare_inputs(config)
        with self.assertRaises(ValueError):
            model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                image_grid_hws=inputs["image_grid_hws"],
                generation_mode="not-a-mode",
            )

    def test_auto_model_from_config(self):
        config = self.model_tester.get_config()
        auto_config = AutoConfig.for_model(
            "locateanything",
            vision_config=config.vision_config.to_dict(),
            text_config=config.text_config.to_dict(),
        )
        self.assertIsInstance(auto_config, LocateAnythingConfig)
        model = AutoModelForImageTextToText.from_config(auto_config)
        self.assertIsInstance(model, LocateAnythingForConditionalGeneration)

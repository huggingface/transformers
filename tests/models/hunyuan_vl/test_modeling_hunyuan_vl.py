# Copyright (C) 2026 THL A29 Limited, a Tencent company and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch HunYuanVL model."""

import unittest

from transformers import (
    HunYuanVLConfig,
    HunYuanVLForCausalLM,
    HunYuanVLForConditionalGeneration,
    HunYuanVLTextConfig,
    HunYuanVLVisionConfig,
    is_torch_available,
)
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch


def ids_tensor(shape, vocab_size):
    return torch.randint(low=0, high=vocab_size, size=tuple(shape), dtype=torch.long)


def floats_tensor(shape, scale=1.0):
    return torch.rand(tuple(shape), dtype=torch.float32) * scale


class HunYuanVLVisionText2TextModelTester:
    """Build a tiny HunYuanVL config plus matching multimodal inputs for unit tests."""

    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=32,
        num_channels=3,
        patch_size=16,
        image_size=64,
        image_token_id=5,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.image_token_id = image_token_id
        self.device = "cpu"
        self.num_image_patches = (image_size // patch_size) ** 2
        self.grid_hw = image_size // patch_size
        # HunYuanVL inserts an extra column per row (newline) and 2 begin/end tokens.
        self.num_image_placeholder_tokens = self.grid_hw * (self.grid_hw + 1) + 2
        self.text_config = {
            "vocab_size": 256,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "hidden_act": "silu",
            "max_position_embeddings": 128,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "head_dim": 16,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
        }
        self.vision_config = {
            "num_channels": num_channels,
            "patch_size": patch_size,
            "temporal_patch_size": 1,
            "spatial_merge_size": 1,
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "hidden_act": "silu",
            "out_hidden_size": 64,
            "text_hidden_size": 64,
            "max_image_size": image_size,
            "min_image_size": image_size,
            "anyres_vit_max_image_size": image_size,
            "max_vit_seq_len": self.num_image_patches,
        }

    def get_config(self):
        return HunYuanVLConfig(
            attn_implementation="eager",
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_id=self.image_token_id,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        pixel_values = floats_tensor(
            [self.batch_size * self.num_image_patches, self.num_channels * self.patch_size * self.patch_size]
        )
        image_grid_thw = torch.tensor(
            [[1, self.grid_hw, self.grid_hw]] * self.batch_size,
            device=self.device,
        )

        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size)
        attention_mask = torch.ones_like(input_ids, device=self.device)
        input_ids = input_ids.to(self.device)
        input_ids[input_ids == self.image_token_id] = config.text_config.pad_token_id
        input_ids[:, : self.num_image_placeholder_tokens] = self.image_token_id
        # HunYuanVL uses 4 position-id channels in multimodal mode: text, width, height, and temporal.
        position_ids = torch.arange(self.seq_length, device=self.device).view(1, 1, -1).expand(self.batch_size, 4, -1)

        return config, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values.to(self.device),
            "image_grid_thw": image_grid_thw,
            "position_ids": position_ids,
        }


@require_torch
class HunYuanVLModelTest(unittest.TestCase):
    """Lightweight CPU model tests for `HunYuanVLForConditionalGeneration`.

    These tests intentionally avoid the heavy `ModelTesterMixin` machinery so the suite stays runnable in
    minimal environments (no GPU, no full Transformers test infrastructure). Coverage focuses on:

    - The forward path produces the expected logits shape on multimodal inputs.
    - Mismatched image / placeholder counts raise a clear error.
    - Text-only forward and text-only generate continue to work without any pixel inputs.
    - The expected backbone (`HunYuanVLTextModel`) is wired in as `model`.
    """

    def setUp(self):
        self.model_tester = HunYuanVLVisionText2TextModelTester(self)

    def test_config_classes(self):
        config = self.model_tester.get_config()
        self.assertIsInstance(config, HunYuanVLConfig)
        self.assertIsInstance(config.text_config, HunYuanVLTextConfig)
        self.assertIsInstance(config.vision_config, HunYuanVLVisionConfig)

    def test_forward_uses_text_backbone(self):
        config, _ = self.model_tester.prepare_config_and_inputs()
        model = HunYuanVLForConditionalGeneration(config).to(self.model_tester.device)
        self.assertEqual(model.model.__class__.__name__, "HunYuanVLTextModel")

    def test_forward_with_image_placeholders(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = HunYuanVLForConditionalGeneration(config).to(self.model_tester.device)
        model.eval()

        with torch.no_grad():
            outputs = model(**inputs_dict)

        self.assertEqual(
            outputs.logits.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, config.text_config.vocab_size),
        )

    def test_forward_raises_on_mismatched_image_tokens(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = HunYuanVLForConditionalGeneration(config).to(self.model_tester.device)
        model.eval()

        bad_inputs = dict(inputs_dict)
        bad_inputs["pixel_values"] = bad_inputs["pixel_values"][: -self.model_tester.num_image_patches]
        bad_inputs["image_grid_thw"] = bad_inputs["image_grid_thw"][:1]

        with self.assertRaisesRegex(ValueError, "Image features and image tokens do not match"):
            with torch.no_grad():
                model(**bad_inputs)

    def test_forward_supports_text_only_inputs(self):
        config = self.model_tester.get_config()
        model = HunYuanVLForConditionalGeneration(config).to(self.model_tester.device)
        model.eval()

        input_ids = torch.tensor([[config.bos_token_id, 5, 6, config.eos_token_id]], device=self.model_tester.device)
        attention_mask = torch.ones_like(input_ids, device=self.model_tester.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        self.assertEqual(outputs.logits.shape, (1, input_ids.shape[1], config.text_config.vocab_size))

    def test_generate_supports_text_only_inputs(self):
        config = self.model_tester.get_config()
        model = HunYuanVLForConditionalGeneration(config).to(self.model_tester.device)
        model.eval()

        input_ids = torch.tensor([[config.bos_token_id, 5, 6]], device=self.model_tester.device)
        attention_mask = torch.ones_like(input_ids, device=self.model_tester.device)

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=2,
                do_sample=False,
            )

        self.assertEqual(generated.shape[0], input_ids.shape[0])
        self.assertGreaterEqual(generated.shape[1], input_ids.shape[1] + 1)

    def test_for_causal_lm_text_only(self):
        config = self.model_tester.get_config()
        model = HunYuanVLForCausalLM(config).to(self.model_tester.device)
        model.eval()

        input_ids = torch.tensor([[config.bos_token_id, 5, 6]], device=self.model_tester.device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        self.assertEqual(outputs.logits.shape, (1, input_ids.shape[1], config.text_config.vocab_size))

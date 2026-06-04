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
"""Testing suite for the PyTorch Zaya1-VL model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers import (
        Zaya1VLConfig,
        Zaya1VLForConditionalGeneration,
        Zaya1VLTextConfig,
    )


def _tiny_config():
    return Zaya1VLConfig(
        text_config=Zaya1VLTextConfig(
            vocab_size=128,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            moe_intermediate_size=16,
            num_experts=2,
            router_hidden_size=4,
            tie_word_embeddings=False,
        ),
        tie_word_embeddings=False,
        image_token_id=127,
        vision_start_token_id=126,
        vision_end_token_id=125,
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 16,
            "num_heads": 4,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 2,
            "out_hidden_size": 32,
            "fullatt_block_indexes": [0],
            "window_size": 4,
        },
    )


@require_torch
class Zaya1VLModelTest(unittest.TestCase):
    def test_image_forward(self):
        config = _tiny_config()
        model = Zaya1VLForConditionalGeneration(config).eval()

        input_ids = torch.tensor([[2, config.image_token_id, 5]])
        attention_mask = torch.ones_like(input_ids)
        pixel_values = torch.randn(
            4, 3 * config.vision_config.temporal_patch_size * config.vision_config.patch_size**2
        )
        image_grid_thw = torch.tensor([[1, 2, 2]])

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_router_logits=True,
            )

        self.assertEqual(outputs.logits.shape, (1, 3, config.text_config.vocab_size))
        self.assertEqual(len(outputs.router_logits), config.text_config.num_hidden_layers)
        self.assertEqual(outputs.router_logits[0].shape, (3, config.text_config.num_experts + 1))

    def test_image_generation(self):
        config = _tiny_config()
        model = Zaya1VLForConditionalGeneration(config).eval()

        input_ids = torch.tensor([[2, config.image_token_id, 5]])
        attention_mask = torch.ones_like(input_ids)
        pixel_values = torch.randn(
            4, 3 * config.vision_config.temporal_patch_size * config.vision_config.patch_size**2
        )
        image_grid_thw = torch.tensor([[1, 2, 2]])

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=2,
                do_sample=False,
            )

        self.assertEqual(generated_ids.shape, (1, input_ids.shape[-1] + 2))


if __name__ == "__main__":
    unittest.main()

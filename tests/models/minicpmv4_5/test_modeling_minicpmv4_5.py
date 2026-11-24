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
"""Testing suite for the PyTorch MiniCPM-V 4.5 model."""

import unittest

from transformers import MiniCPMV4_5Config, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...test_modeling_common import floats_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import MiniCPMV4_5ForConditionalGeneration, MiniCPMV4_5Model


class MiniCPMV4_5VisionText2TextModelTester(VLMModelTester):
    base_model_class = MiniCPMV4_5Model
    config_class = MiniCPMV4_5Config
    conditional_generation_class = MiniCPMV4_5ForConditionalGeneration

    small_config_overrides = {
        "batch_size": 2,
        "image_token_id": 100,
        "image_size": 32,
        "patch_size": 8,  # each test image is one 4x4-patch slice
        "resampler_n_latents": 4,  # ... which the resampler turns into 4 tokens
        "resampler_n_heads": 4,
        "num_image_tokens": 4,
        "vocab_size": 256,
        "hidden_size": 128,
        "intermediate_size": 37,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "hidden_act": "silu",
        "max_position_embeddings": 512,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "vision_hidden_act": "gelu_pytorch_tanh",
        "vision_intermediate_size": 37,
    }

    def create_pixel_values(self, batch_size: int | None = None):
        # One slice per image, as a single row of 4 x 4 = 16 patches: (num_slices, channels, patch_size, 16 * patch_size)
        batch_size = batch_size if batch_size is not None else self.batch_size
        num_patches = (self.image_size // self.patch_size) ** 2
        return floats_tensor([batch_size, self.num_channels, self.patch_size, num_patches * self.patch_size])

    def get_additional_inputs(self, config, input_ids, pixel_values, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        side = self.image_size // self.patch_size
        # Every other slice is a smaller (side - 1) x side grid padded to side x side, so attention tests cover padding.
        pixel_attention_mask = torch.ones(batch_size, side * side, dtype=torch.bool, device=torch_device)
        pixel_attention_mask[1::2, (side - 1) * side :] = False
        rows, cols = torch.meshgrid(
            torch.arange(side, device=torch_device), torch.arange(side, device=torch_device), indexing="ij"
        )
        grid = torch.stack([rows, cols], dim=-1).reshape(1, side * side, 2).repeat(batch_size, 1, 1)
        patch_position_ids = grid * pixel_attention_mask[..., None]
        return {"pixel_attention_mask": pixel_attention_mask, "patch_position_ids": patch_position_ids}


@require_torch
class MiniCPMV4_5ModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = MiniCPMV4_5VisionText2TextModelTester

    def test_reverse_loading_mapping(self):
        # As in llava, the base model has no `model.` prefix to reverse the `^llm\.model\.` rename onto
        super().test_reverse_loading_mapping(skip_base_model=True)

    def _image_features_prepare_config_and_inputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        inputs_dict = {
            key: value for key, value in inputs_dict.items() if key.startswith("pixel") or key == "patch_position_ids"
        }
        return config, inputs_dict

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
"""Testing suite for the PyTorch LLaVA-OneVision-1.5 model."""

import copy
import unittest

import pytest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        LlavaOnevision1_5Config,
        LlavaOnevision1_5ForConditionalGeneration,
        LlavaOnevision1_5Model,
        LlavaOnevision1_5VisionConfig,
        LlavaOnevision1_5VisionModel,
        Qwen3Config,
    )


class LlavaOnevision1_5ModelTester(VLMModelTester):
    base_model_class = LlavaOnevision1_5Model
    config_class = LlavaOnevision1_5Config
    text_config_class = Qwen3Config
    vision_config_class = LlavaOnevision1_5VisionConfig
    conditional_generation_class = LlavaOnevision1_5ForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_token_id", 3)
        kwargs.setdefault("video_token_id", 4)
        kwargs.setdefault("vision_start_token_id", 5)
        kwargs.setdefault("vision_end_token_id", 6)
        kwargs.setdefault("image_size", 16)
        kwargs.setdefault("patch_size", 4)
        kwargs.setdefault("num_image_tokens", 4)
        kwargs.setdefault("hidden_act", "gelu")
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("head_dim", 8)
        kwargs.setdefault("depth", 2)
        kwargs.setdefault("num_heads", 4)
        kwargs.setdefault("spatial_merge_size", 2)
        kwargs.setdefault("temporal_patch_size", 1)
        kwargs.setdefault("out_hidden_size", 32)
        kwargs.setdefault("layer_norm_eps", 1e-5)
        kwargs.setdefault("fullatt_block_indexes", (0, 1))
        super().__init__(parent, **kwargs)
        self.in_channels = self.num_channels

    def create_pixel_values(self):
        grid_size = self.image_size // self.patch_size
        return torch.rand(
            self.batch_size * grid_size**2,
            self.num_channels * self.patch_size**2,
            device=torch_device,
        )

    def get_additional_inputs(self, config, input_ids, pixel_values):
        grid_size = self.image_size // self.patch_size
        return {"image_grid_thw": torch.tensor([[1, grid_size, grid_size]] * self.batch_size, device=torch_device)}

    def place_image_tokens(self, input_ids, config):
        input_ids = super().place_image_tokens(input_ids, config)
        input_ids = torch.roll(input_ids, shifts=1, dims=1)
        input_ids[:, 0] = config.vision_start_token_id
        return input_ids


@require_torch
class LlavaOnevision1_5ModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = LlavaOnevision1_5ModelTester
    test_all_params_have_gradient = False

    def test_reverse_loading_mapping(self):
        super().test_reverse_loading_mapping(skip_base_model=True)

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, inputs_dict = super().prepare_config_and_inputs_for_generate(batch_size)
        grid_size = self.model_tester.image_size // self.model_tester.patch_size
        _, full_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        inputs_dict["pixel_values"] = full_inputs["pixel_values"][: batch_size * grid_size**2]
        return config, inputs_dict

    @pytest.mark.xfail(reason="Reentrant checkpointing cannot reuse the learned CLS positional embedding graph.")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        super().test_training_gradient_checkpointing_use_reentrant_true()

    @unittest.skip(reason="The generic fullgraph test cannot split flattened vision patches by batch dimension.")
    def test_generate_compile_model_forward_fullgraph(self):
        pass

    @unittest.skip(reason="LLaVA-OneVision-1.5 does not support assisted decoding with multimodal inputs.")
    def test_assisted_decoding_matches_greedy_search_0_random(self):
        pass

    @unittest.skip(reason="LLaVA-OneVision-1.5 does not support assisted decoding with multimodal inputs.")
    def test_assisted_decoding_matches_greedy_search_1_same(self):
        pass

    @unittest.skip(reason="LLaVA-OneVision-1.5 does not support assisted decoding with multimodal inputs.")
    def test_assisted_decoding_sample(self):
        pass

    def test_mismatching_num_image_tokens(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            curr_input_dict = copy.deepcopy(input_dict)
            model(**curr_input_dict)

            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][-1:]
            curr_input_dict["image_grid_thw"] = curr_input_dict["image_grid_thw"][-1:]
            with self.assertRaises(ValueError):
                model(**curr_input_dict)

    def test_variable_resolution_eager_attention(self):
        config = self.model_tester.get_config()
        config._attn_implementation = "eager"
        patch_dim = self.model_tester.num_channels * self.model_tester.patch_size**2
        grid_thw = torch.tensor([[1, 4, 4], [1, 2, 4]], device=torch_device)
        pixel_values = torch.rand(24, patch_dim, device=torch_device)

        model = LlavaOnevision1_5ForConditionalGeneration(config).to(torch_device).eval()
        outputs = model.get_image_features(pixel_values, grid_thw)
        self.assertEqual([features.shape[0] for features in outputs.pooler_output], [4, 2])


@require_torch
class LlavaOnevision1_5VisionModelTest(unittest.TestCase):
    all_model_classes = (LlavaOnevision1_5VisionModel,) if is_torch_available() else ()

    def test_forward(self):
        config = LlavaOnevision1_5VisionConfig(
            depth=2,
            hidden_size=32,
            intermediate_size=37,
            num_heads=4,
            patch_size=4,
            spatial_merge_size=2,
            temporal_patch_size=1,
            out_hidden_size=32,
            layer_norm_eps=1e-5,
            fullatt_block_indexes=(0, 1),
        )
        model = LlavaOnevision1_5VisionModel(config).to(torch_device).eval()
        outputs = model(torch.rand(16, 48, device=torch_device), grid_thw=torch.tensor([[1, 4, 4]]))
        self.assertEqual(outputs.last_hidden_state.shape, (16, 32))
        self.assertEqual(outputs.pooler_output.shape, (4, 32))

# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Qwen3-VL model."""

import copy
import unittest

import pytest

from transformers import (
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    is_torch_available,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig, Qwen3VLVisionConfig
from transformers.testing_utils import (
    require_torch,
    torch_device,
)

from ...test_modeling_common import floats_tensor, ids_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch


class Qwen3VLVisionText2TextModelTester(VLMModelTester):
    base_model_class = Qwen3VLModel
    config_class = Qwen3VLConfig
    text_config_class = Qwen3VLTextConfig
    vision_config_class = Qwen3VLVisionConfig
    conditional_generation_class = Qwen3VLForConditionalGeneration

    # Qwen3 VL-specific configuration
    image_token_id = 3
    video_token_id = 4
    vision_start_token_id = 5
    vision_end_token_id = 6

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_size", 16)
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("num_image_tokens", 32)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 2)
        super().__init__(parent, **kwargs)

        # Override head_dim (base class computes it, but Qwen3 VL needs specific value)
        self.head_dim = 8

        # Qwen3 VL-specific vision config attributes
        self.depth = 2
        self.vision_hidden_act = "gelu_pytorch_tanh"
        self.out_hidden_size = self.hidden_size
        self.vision_hidden_size = self.hidden_size
        self.vision_intermediate_size = self.hidden_size
        self.num_heads = 4
        self.spatial_merge_size = 1
        self.temporal_patch_size = 2
        self.num_position_embeddings = 16
        self.deepstack_visual_indexes = [0, 1]
        self.rope_parameters = {"rope_type": "default", "mrope_section": [16, 8, 8], "mrope_interleaved": True, "rope_theta": 10000}

    def create_pixel_values(self):
        # Qwen3VL expects flattened patches: (total_patches, channels * patch_size^2 * temporal_patch_size)
        return floats_tensor(
            [
                self.batch_size * (self.image_size**2) // (self.patch_size**2),
                self.num_channels * (self.patch_size**2) * self.temporal_patch_size,
            ]
        )

    def place_image_tokens(self, input_ids, config):
        # Place image tokens with vision_start_token_id prefix
        input_ids = input_ids.clone()
        # Clear any accidental special tokens first
        input_ids[:, -1] = self.pad_token_id
        input_ids[input_ids == self.video_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[input_ids == self.vision_start_token_id] = self.pad_token_id
        # Place image tokens with vision_start_token_id prefix
        input_ids[:, -1] = self.image_token_id
        input_ids[:, -2] = self.vision_start_token_id
        return input_ids

    def get_additional_inputs(self, config, input_ids, pixel_values):
        return {
            "image_grid_thw": torch.tensor([[1, 1, 1]] * self.batch_size, device=torch_device),
        }

    def get_config(self):
        # Qwen3VLConfig expects text_config and vision_config as dicts, not config objects
        return self.config_class(
            text_config=self.get_text_config().to_dict(),
            vision_config=self.get_vision_config().to_dict(),
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            pad_token_id=self.pad_token_id,
        )


@require_torch
class Qwen3VLModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Qwen3VLVisionText2TextModelTester

    @pytest.mark.xfail(reason="This architecture seems to not compute gradients for some layer.")
    def test_training_gradient_checkpointing(self):
        super().test_training_gradient_checkpointing()

    @pytest.mark.xfail(reason="This architecture seems to not compute gradients for some layer.")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        super().test_training_gradient_checkpointing_use_reentrant_false()

    @pytest.mark.xfail(reason="This architecture seems to not compute gradients for some layer.")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        super().test_training_gradient_checkpointing_use_reentrant_true()

    def test_mismatching_num_image_tokens(self):
        # Override the base test because we need to slice image_grid_thw too
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            _ = model(**input_dict)  # successful forward with no modifications
            curr_input_dict = copy.deepcopy(input_dict)

            # remove one image but leave the image token in text
            patch_size = config.vision_config.patch_size
            one_img_length = (self.model_tester.image_size**2) // (patch_size**2)
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][-one_img_length:, ...]
            curr_input_dict["image_grid_thw"] = curr_input_dict["image_grid_thw"][-1:, ...]
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # simulate multi-image case by concatenating inputs where each has exactly one image/image-token
            input_ids = curr_input_dict["input_ids"][:1]
            pixel_values = curr_input_dict["pixel_values"][:one_img_length]
            image_grid_thw = curr_input_dict["image_grid_thw"][:1]
            input_ids = torch.cat([input_ids, input_ids], dim=0)

            # one image and two image tokens raise an error
            with self.assertRaises(ValueError):
                _ = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )

            # two images and two image tokens don't raise an error
            pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
            image_grid_thw = torch.cat([image_grid_thw, image_grid_thw], dim=0)
            _ = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

    def test_video_forward(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        B = self.model_tester.batch_size
        C = config.vision_config.in_channels
        T = config.vision_config.temporal_patch_size
        P = config.vision_config.patch_size

        input_ids = ids_tensor([B, self.model_tester.seq_length], self.model_tester.vocab_size)

        F = 4
        patch_H = self.model_tester.image_size // P
        patch_W = self.model_tester.image_size // P
        patch_T = F // T
        patches_per_video = patch_T * patch_H * patch_W
        pathed_per_frame = patch_H * patch_W
        pixel_values_videos = floats_tensor(
            [
                # first dim: batch_size * num_patches
                B * patches_per_video,
                # second dim: in_channels * temporal_patch_size * patch_size^2
                C * T * (P**2),
            ]
        )

        # qwen3vl use timestamps for video, so split it into patch_T sub-videos
        video_grid_thw = torch.tensor([[1, patch_H, patch_W] for _ in range(patch_T)] * B)

        # sanity check
        self.assertEqual(pixel_values_videos.shape[0], video_grid_thw.prod(dim=1).sum().item())

        # Insert video token sequence
        input_ids[:, -1] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.video_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.image_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.vision_start_token_id] = self.model_tester.pad_token_id
        input_ids[:, self.model_tester.num_image_tokens] = self.model_tester.video_token_id

        insertion_point = self.model_tester.num_image_tokens

        self.assertLessEqual((B * patches_per_video) + insertion_point, self.model_tester.seq_length)
        for b in range(B):
            # each frame is separated by a vision_start_token_id
            for frame_idx in range(patch_T):
                input_ids[b, insertion_point + frame_idx * (pathed_per_frame + 1)] = (
                    self.model_tester.vision_start_token_id
                )
                input_ids[
                    b,
                    insertion_point + frame_idx * (pathed_per_frame + 1) + 1 : insertion_point
                    + (frame_idx + 1) * (pathed_per_frame + 1),
                ] = self.model_tester.video_token_id

        for model_class in self.all_model_classes:
            # TODO:we should remove this because we use timestamps for video
            model = model_class(config).to(torch_device)
            outputs = model(
                input_ids=input_ids,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )
            self.assertIsNotNone(outputs)

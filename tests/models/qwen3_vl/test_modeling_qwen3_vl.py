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
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel


if is_torch_available():
    import torch


class Qwen3VLVisionText2TextModelTester(VLMModelTester):
    base_model_class = Qwen3VLModel
    config_class = Qwen3VLConfig
    text_config_class = Qwen3VLTextConfig
    vision_config_class = Qwen3VLVisionConfig
    conditional_generation_class = Qwen3VLForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_token_id", 3)
        kwargs.setdefault("video_token_id", 4)
        kwargs.setdefault("vision_start_token_id", 5)
        kwargs.setdefault("vision_end_token_id", 6)
        kwargs.setdefault("image_size", 16)
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("num_image_tokens", 32)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("head_dim", 8)
        kwargs.setdefault("depth", 2)
        kwargs.setdefault("vision_hidden_act", "gelu_pytorch_tanh")
        kwargs.setdefault("num_heads", 4)
        kwargs.setdefault("spatial_merge_size", 1)
        kwargs.setdefault("temporal_patch_size", 2)
        kwargs.setdefault("num_position_embeddings", 16)
        kwargs.setdefault("deepstack_visual_indexes", [0, 1])
        kwargs.setdefault(
            "rope_parameters",
            {
                "rope_type": "default",
                "mrope_section": [16, 8, 8],
                "mrope_interleaved": True,
                "rope_theta": 10000,
            },
        )
        super().__init__(parent, **kwargs)

        # These can be inferred from existing properties and don't get separate kwargs
        self.out_hidden_size = self.hidden_size
        self.vision_hidden_size = self.hidden_size
        self.vision_intermediate_size = self.hidden_size

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
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == self.image_token_id] = 1
        return {
            "image_grid_thw": torch.tensor([[1, 1, 1]] * self.batch_size, device=torch_device),
            "mm_token_type_ids": mm_token_type_ids,
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

            model.base_model.rope_deltas = None
            # simulate multi-image case by concatenating inputs where each has exactly one image/image-token
            input_ids = curr_input_dict["input_ids"][:1]
            pixel_values = curr_input_dict["pixel_values"][:one_img_length]
            image_grid_thw = curr_input_dict["image_grid_thw"][:1]
            mm_token_type_ids = curr_input_dict["mm_token_type_ids"][:1]
            input_ids = torch.cat([input_ids, input_ids], dim=0)

            # one image and two image tokens raise an error
            with self.assertRaises(ValueError):
                _ = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    mm_token_type_ids=torch.cat([mm_token_type_ids, mm_token_type_ids], dim=0),
                )

            model.base_model.rope_deltas = None
            # two images and two image tokens don't raise an error
            pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
            image_grid_thw = torch.cat([image_grid_thw, image_grid_thw], dim=0)
            mm_token_type_ids = torch.cat(
                [curr_input_dict["mm_token_type_ids"][:1], curr_input_dict["mm_token_type_ids"][:1]], dim=0
            )
            _ = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
            )

    def test_image_forward(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        B = self.model_tester.batch_size
        C = config.vision_config.in_channels
        T = config.vision_config.temporal_patch_size
        P = config.vision_config.patch_size
        num_images = 2

        input_ids = ids_tensor([B, self.model_tester.seq_length], self.model_tester.vocab_size)
        input_ids[:, -1] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.video_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.image_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.vision_start_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.vision_end_token_id] = self.model_tester.pad_token_id

        # For this tiny config, each image corresponds to one patch token.
        patches_per_image = 1
        pixel_values = floats_tensor(
            [
                B * num_images * patches_per_image,
                C * T * (P**2),
            ]
        )
        image_grid_thw = torch.tensor([[1, 1, 1]] * (B * num_images))
        self.assertEqual(pixel_values.shape[0], image_grid_thw.prod(dim=1).sum().item())

        insertion_point = 0
        tokens_per_image = 3  # vision_start + image_token + vision_end
        required_seq_length = insertion_point + num_images * tokens_per_image
        self.assertLessEqual(required_seq_length, input_ids.shape[1])

        for b in range(B):
            for image_idx in range(num_images):
                image_start = insertion_point + image_idx * tokens_per_image
                input_ids[b, image_start] = self.model_tester.vision_start_token_id
                input_ids[b, image_start + 1] = self.model_tester.image_token_id
                input_ids[b, image_start + 2] = self.model_tester.vision_end_token_id

        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == self.model_tester.image_token_id] = 1

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
            )
            self.assertIsNotNone(outputs)

    def test_video_forward(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        B = self.model_tester.batch_size
        C = config.vision_config.in_channels
        T = config.vision_config.temporal_patch_size
        P = config.vision_config.patch_size

        input_ids = ids_tensor([B, self.model_tester.seq_length], self.model_tester.vocab_size)

        F = 4
        num_video = 2
        frame_timestamp_tokens = 5
        patch_H = self.model_tester.image_size // P
        patch_W = self.model_tester.image_size // P
        patch_T = F // T
        patches_per_video = patch_T * patch_H * patch_W
        pathed_per_frame = patch_H * patch_W
        pixel_values_videos = floats_tensor(
            [
                # first dim: batch_size * num_patches
                B * num_video * patches_per_video,
                # second dim: in_channels * temporal_patch_size * patch_size^2
                C * T * (P**2),
            ]
        )

        video_grid_thw = torch.tensor([[patch_T, patch_H, patch_W]] * (B * num_video))

        # sanity check
        self.assertEqual(pixel_values_videos.shape[0], video_grid_thw.prod(dim=1).sum().item())

        # Insert video token sequence
        input_ids[:, -1] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.video_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.image_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.vision_start_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.vision_end_token_id] = self.model_tester.pad_token_id

        insertion_point = 0
        tokens_per_frame = frame_timestamp_tokens + 1 + pathed_per_frame + 1
        tokens_per_video = patch_T * tokens_per_frame
        required_seq_length = insertion_point + num_video * tokens_per_video
        if required_seq_length > input_ids.shape[1]:
            pad_extension = torch.full(
                (B, required_seq_length - input_ids.shape[1]),
                self.model_tester.pad_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            input_ids = torch.cat([input_ids, pad_extension], dim=1)
        timestamp_start_token_id = self.model_tester.vision_end_token_id + 1
        self.assertLessEqual(timestamp_start_token_id + frame_timestamp_tokens, self.model_tester.vocab_size)
        timestamp_token_ids = torch.arange(
            timestamp_start_token_id,
            timestamp_start_token_id + frame_timestamp_tokens,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )

        self.assertLessEqual(required_seq_length, input_ids.shape[1])
        for b in range(B):
            for video_idx in range(num_video):
                video_start = insertion_point + video_idx * tokens_per_video
                for frame_idx in range(patch_T):
                    frame_start = video_start + frame_idx * tokens_per_frame
                    input_ids[b, frame_start : frame_start + frame_timestamp_tokens] = timestamp_token_ids

                    vision_start_pos = frame_start + frame_timestamp_tokens
                    input_ids[b, vision_start_pos] = self.model_tester.vision_start_token_id

                    frame_token_start = vision_start_pos + 1
                    frame_token_end = frame_token_start + pathed_per_frame
                    input_ids[b, frame_token_start:frame_token_end] = self.model_tester.video_token_id

                    input_ids[b, frame_token_end] = self.model_tester.vision_end_token_id

        # build mm_token_type_ids
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == self.model_tester.video_token_id] = 2

        for model_class in self.all_model_classes:
            # TODO:we should remove this because we use timestamps for video
            model = model_class(config).to(torch_device)
            outputs = model(
                input_ids=input_ids,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
            )
            self.assertIsNotNone(outputs)


@require_torch
class Qwen3VLTextModelPositionIdsTest(unittest.TestCase):
    """Regression tests for text_position_ids extraction (PR #44158)."""

    def get_text_config(self):
        return Qwen3VLTextConfig(
            vocab_size=99,
            hidden_size=32,
            intermediate_size=37,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            hidden_act="silu",
            max_position_embeddings=512,
            rope_parameters={"rope_type": "default", "mrope_section": [16, 8, 8], "mrope_interleaved": True},
        )

    def _make_vision_position_ids(self, batch_size, seq_len):
        """Create 3D vision position_ids (temporal=0, height=arange, width=arange)."""
        pos = torch.zeros(3, batch_size, seq_len, dtype=torch.long, device=torch_device)
        pos[1] = torch.arange(seq_len, device=torch_device).unsqueeze(0).expand(batch_size, -1)
        pos[2] = torch.arange(seq_len, device=torch_device).unsqueeze(0).expand(batch_size, -1)
        return pos

    def test_3d_vision_position_ids_no_cache(self):
        config = self.get_text_config()
        model = Qwen3VLTextModel(config).to(torch_device).eval()

        batch_size, seq_len = 2, 10
        input_ids = ids_tensor([batch_size, seq_len], config.vocab_size).to(torch_device)
        vision_position_ids = self._make_vision_position_ids(batch_size, seq_len)

        with torch.no_grad():
            output = model(input_ids=input_ids, position_ids=vision_position_ids, use_cache=False)
        self.assertEqual(output.last_hidden_state.shape, (batch_size, seq_len, config.hidden_size))

    def test_3d_vision_position_ids_produce_finite_output(self):
        config = self.get_text_config()
        model = Qwen3VLTextModel(config).to(torch_device).eval()

        batch_size, seq_len = 2, 8
        input_ids = ids_tensor([batch_size, seq_len], config.vocab_size).to(torch_device)
        vision_position_ids = self._make_vision_position_ids(batch_size, seq_len)

        with torch.no_grad():
            output_3d = model(input_ids=input_ids, position_ids=vision_position_ids, use_cache=False)
            output_none = model(input_ids=input_ids, position_ids=None, use_cache=False)

        self.assertTrue(torch.isfinite(output_3d.last_hidden_state).all())
        self.assertTrue(torch.isfinite(output_none.last_hidden_state).all())

    def test_4d_position_ids_forward(self):
        config = self.get_text_config()
        model = Qwen3VLTextModel(config).to(torch_device).eval()

        batch_size, seq_len = 2, 8
        input_ids = ids_tensor([batch_size, seq_len], config.vocab_size).to(torch_device)

        text_pos = torch.arange(seq_len, device=torch_device).unsqueeze(0).expand(batch_size, -1)
        spatial_pos = torch.arange(seq_len, device=torch_device).unsqueeze(0).expand(batch_size, -1)
        zero_pos = torch.zeros(batch_size, seq_len, dtype=torch.long, device=torch_device)
        position_ids_4d = torch.stack([text_pos, zero_pos, spatial_pos, spatial_pos], dim=0)

        with torch.no_grad():
            output = model(input_ids=input_ids, position_ids=position_ids_4d, use_cache=False)
        self.assertEqual(output.last_hidden_state.shape, (batch_size, seq_len, config.hidden_size))
        self.assertTrue(torch.isfinite(output.last_hidden_state).all())

    def test_use_cache_true_vs_false_with_vision_position_ids(self):
        """use_cache should not affect output when 3D vision position_ids are provided."""
        config = self.get_text_config()
        model = Qwen3VLTextModel(config).to(torch_device).eval()

        batch_size, seq_len = 1, 12
        input_ids = ids_tensor([batch_size, seq_len], config.vocab_size).to(torch_device)
        vision_position_ids = self._make_vision_position_ids(batch_size, seq_len)

        with torch.no_grad():
            output_cached = model(input_ids=input_ids, position_ids=vision_position_ids.clone(), use_cache=True)
            output_no_cache = model(input_ids=input_ids, position_ids=vision_position_ids.clone(), use_cache=False)

        torch.testing.assert_close(
            output_cached.last_hidden_state, output_no_cache.last_hidden_state, atol=1e-5, rtol=1e-5
        )

    def test_2d_position_ids_forward(self):
        config = self.get_text_config()
        model = Qwen3VLTextModel(config).to(torch_device).eval()

        batch_size, seq_len = 2, 8
        input_ids = ids_tensor([batch_size, seq_len], config.vocab_size).to(torch_device)
        position_ids_2d = torch.arange(seq_len, device=torch_device).unsqueeze(0).expand(batch_size, -1)

        with torch.no_grad():
            output = model(input_ids=input_ids, position_ids=position_ids_2d, use_cache=False)
        self.assertEqual(output.last_hidden_state.shape, (batch_size, seq_len, config.hidden_size))
        self.assertTrue(torch.isfinite(output.last_hidden_state).all())

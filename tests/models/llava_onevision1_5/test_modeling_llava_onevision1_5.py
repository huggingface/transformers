# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from transformers import (
    LlavaOnevision1_5Config,
    LlavaOnevision1_5ForConditionalGeneration,
    LlavaOnevision1_5Model,
    is_torch_available,
)
from transformers.testing_utils import require_torch, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch


class LlavaOnevision1_5VisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        num_channels=3,
        ignore_index=-100,
        image_token_id=4,
        video_token_id=5,
        is_training=True,
        text_config={
            "vocab_size": 99,
            "hidden_size": 32,
            "intermediate_size": 37,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "hidden_act": "silu",
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
            "pad_token_id": 2,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "hidden_act": "gelu",
            "intermediate_size": 37,
            "num_heads": 4,
            "patch_size": 4,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "out_hidden_size": 32,
            "layer_norm_eps": 1e-5,
        },
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.text_config = text_config
        self.vision_config = vision_config
        self.pad_token_id = text_config["pad_token_id"]

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = batch_size
        self.num_channels = num_channels
        # A 4x4 patch grid (spatial_merge_size=2) merges down to a single 2x2 merged token per image.
        self.grid_h = 4
        self.grid_w = 4
        self.num_image_tokens = (self.grid_h // vision_config["spatial_merge_size"]) * (
            self.grid_w // vision_config["spatial_merge_size"]
        )
        self.seq_length = seq_length + self.num_image_tokens

    def get_config(self):
        return LlavaOnevision1_5Config(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        patch_size = config.vision_config.patch_size
        pixel_values = floats_tensor(
            [
                self.batch_size * self.grid_h * self.grid_w,
                self.num_channels * (patch_size**2),
            ]
        )
        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[input_ids == self.video_token_id] = self.pad_token_id
        input_ids[:, : self.num_image_tokens] = self.image_token_id

        inputs_dict = {
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor([[1, self.grid_h, self.grid_w]] * self.batch_size, device=torch_device),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class LlavaOnevision1_5ForConditionalGenerationModelTest(
    ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    """
    Model tester for `LlavaOnevision1_5ForConditionalGeneration`.
    """

    all_model_classes = (
        (
            LlavaOnevision1_5Model,
            LlavaOnevision1_5ForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "image-text-to-text": LlavaOnevision1_5ForConditionalGeneration,
            "any-to-any": LlavaOnevision1_5ForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    _is_composite = True

    def setUp(self):
        self.model_tester = LlavaOnevision1_5VisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LlavaOnevision1_5Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_mismatching_num_image_tokens(self):
        """
        Tests that VLMs throw an error with explicit message saying what is wrong
        when the number of images doesn't match the number of image tokens in the text.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            curr_input_dict = copy.deepcopy(input_dict)
            _ = model(**curr_input_dict)  # successful forward with no modifications

            # remove one image but leave the image token in text
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][-1:, ...]
            curr_input_dict["image_grid_thw"] = curr_input_dict["image_grid_thw"][-1:, ...]
            with self.assertRaisesRegex(ValueError, "Image features and image tokens do not match"):
                _ = model(**curr_input_dict)

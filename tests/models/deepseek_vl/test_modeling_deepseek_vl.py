# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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
"""Testing suite for the PyTorch DeepseekVL model."""

import re
import tempfile
import unittest
from functools import reduce

import numpy as np
import requests
from huggingface_hub import hf_hub_download

from transformers import (
    AutoProcessor,
    DeepseekVLConfig,
    DeepseekVLForConditionalGeneration,
    DeepseekVLModel,
    is_torch_available,
    is_vision_available,
)
from transformers.models.auto import get_values
from transformers.models.auto.modeling_auto import MODEL_FOR_BACKBONE_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_torch_available():
    import torch


if is_vision_available():
    from PIL import Image


class DeepseekVLModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=25,
        num_channels=3,
        initializer_range=0.02,
        is_training=True,
        use_cache=False,
        text_config={
            "num_hidden_layers": 2,
            "vocab_size": 99,
            "hidden_size": 16,
            "intermediate_size": 37,
            "max_position_embeddings": 512,
            "num_attention_heads": 4,
            "pad_token_id": 1,
        },
        use_high_res_vision=True,
        low_res_vision_config={
            "num_hidden_layers": 1,
            "hidden_size": 16,
            "intermediate_size": 37,
            "image_size": 32,
            "patch_size": 8,
            "hidden_act": "gelu",
            "vision_use_head": False,
            "num_attention_heads": 4,
        },
        high_res_vision_config={
            "num_hidden_layers": 1,
            "global_attn_indexes": [0],
            "hidden_size": 16,
            "intermediate_size": 37,
            "image_size": 128,
            "patch_size": 32,
            "num_attention_heads": 4,
        }
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_channels = num_channels
        self.initializer_range = initializer_range
        self.is_training = is_training
        self.use_cache = use_cache

        self.text_config = text_config
        self.use_high_res_vision = use_high_res_vision
        self.low_res_vision_config = low_res_vision_config
        self.high_res_vision_config = high_res_vision_config
        self.low_res_vision_config['num_channels'] = self.num_channels
        self.high_res_vision_config['num_channels'] = self.num_channels

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.image_size = high_res_vision_config["image_size"]
        self.num_image_tokens = low_res_vision_config["image_size"] // low_res_vision_config["patch_size"]
        self.pad_token_id = text_config["pad_token_id"]
        self.image_token_index = self.vocab_size - 1

    def get_config(self):
        return DeepseekVLConfig(
            text_config=self.text_config,
            use_high_res_vision=self.use_high_res_vision,
            low_res_vision_config=self.low_res_vision_config,
            high_res_vision_config=self.high_res_vision_config,
            image_token_index=self.image_token_index,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()

        # create text and vision inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 2) + 1
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.num_channels,
                self.image_size,
                self.image_size,
            ]
        )
        # fill image_tokens
        input_ids[:, : self.num_image_tokens] = self.image_token_index

        return config, input_ids, attention_mask, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class DeepseekVLModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (DeepseekVLModel, DeepseekVLForConditionalGeneration) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "image-text-to-text": DeepseekVLForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    _is_composite = True
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = DeepseekVLModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DeepseekVLConfig, has_text_modality=False)
